"""Decision-level A/B measurement.

Whole-tournament measurement cannot see a change that touches 1% of
decisions: the signal is buried under every hand the change never touched.
This harness measures ONLY the hands where old and new behaviour differ.

Pairing.  Each sample is one hand played twice from an identical state -- same
stacks, level, button, deck and opponents -- once with the hero on the old
feature set and once on the new.  Every bot here is deterministic given the
deck, so until the hero's two versions choose differently the hands are
identical, and a hand where they never diverge contributes exactly zero.  All
the variance left is the variance of the divergent spots themselves.

Runouts.  A divergent spot is then replayed over several fresh boards (hole
cards fixed), both arms on the same board each time, so the comparison
averages over card luck instead of riding one river.

Two currencies, reported side by side:
  chips   hero's stack change in big blinds
  equity  hero's tournament equity after the hand (Malmuth-Harville with the
          given payouts), in percentage points of the prize pool
An ICM change is SUPPOSED to give up chips to gain equity; judging it in
bb/100 penalises it by construction.

Stack states are drawn from a spread that looks like the back half of a
freezeout: 2-5 players left, unequal stacks, blinds such that the average
stack is 8-45bb.
"""

import math
import os
import random
import time
from multiprocessing import Pool

from poker.adapter import StrategyBot
from poker.bots import CallingStation, Maniac, Nit, OverFolder
from poker.engine import Table, DEFAULT_LEVELS
from poker.placement import equity as placement_equity
from poker.strategy import V1_FEATURES, FEATURES

TOTAL_CHIPS = 25000


class Recorder(object):
    """Wraps a bot and records every (action, amount) it takes, with timing."""

    def __init__(self, bot):
        self.bot = bot
        self.name = bot.name
        self.log = []
        self.times = []

    def act(self, hand, seat, legal):
        t = time.monotonic()
        out = self.bot.act(hand, seat, legal)
        self.times.append(time.monotonic() - t)
        self.log.append((hand.street, out[0], out[1]))
        return out


def opponent(kind, seed):
    if kind == "v1":
        return StrategyBot(features=V1_FEATURES, name="v1")
    if kind == "v2":
        return StrategyBot(name="v2")
    return {"station": CallingStation, "nit": Nit, "overfolder": OverFolder,
            "maniac": lambda: Maniac(seed)}[kind]()


FIELDS = {
    # who the hero plays against; "mixed" draws each seat independently
    "mixed": ["v1", "v1", "overfolder", "nit", "station", "maniac"],
    "v1": ["v1"],
    "self": ["v2"],
    "archetypes": ["overfolder", "nit", "station", "maniac"],
}


def random_spot(rng, field="mixed"):
    """A plausible late-tournament table: stacks, level, button, hero seat, field."""
    alive = rng.choice([2, 3, 3, 4, 4, 5, 5, 5])
    seats = sorted(rng.sample(range(5), alive))
    weights = [rng.expovariate(1.0) + 0.15 for _ in seats]
    total = sum(weights)
    stacks = [0] * 5
    for s, w in zip(seats, weights):
        stacks[s] = max(int(TOTAL_CHIPS * w / total), 1)
    # choose the level that puts the average stack at the drawn depth
    depth = rng.uniform(8.0, 45.0)
    avg = TOTAL_CHIPS / float(alive)
    level = min(range(len(DEFAULT_LEVELS)),
                key=lambda i: abs(avg / DEFAULT_LEVELS[i][1] - depth))
    button = rng.choice(seats)
    hero = rng.choice(seats)
    kinds = FIELDS[field]
    opp = [rng.choice(kinds) for _ in range(5)]
    return {"stacks": stacks, "level": level, "button": button, "hero": hero,
            "opponents": opp, "seed": rng.randrange(1 << 30)}


def play(spot, hero_features, deck, payouts):
    """Play one hand from `spot`; return (hero chip delta in bb, hero equity, log, times)."""
    bots = []
    recorder = None
    for i in range(5):
        if i == spot["hero"]:
            recorder = Recorder(StrategyBot(features=hero_features, name="hero"))
            bots.append(recorder)
        else:
            bots.append(opponent(spot["opponents"][i], spot["seed"] + i))
    table = Table(bots, starting_stack=1, seed=spot["seed"])
    for i, seat in enumerate(table.seats):
        seat.stack = spot["stacks"][i]
    table.button = spot["button"]
    table.play_hand(level_index=spot["level"], deck=deck)
    bb = DEFAULT_LEVELS[spot["level"]][1]
    end = [s.stack for s in table.seats]
    delta = (end[spot["hero"]] - spot["stacks"][spot["hero"]]) / float(bb)
    eq = placement_equity(end, payouts, spot["hero"])
    return delta, eq, recorder.log, recorder.times


def _reboard(deck, alive, rng):
    """Same hole cards, fresh board: shuffle only the cards after the holes."""
    head = deck[:2 * alive]
    tail = deck[2 * alive:]
    rng.shuffle(tail)
    return head + tail


def measure_spot(args):
    """One spot: detect divergence, then average both arms over runouts."""
    spot, old, new, payouts, runouts = args
    rng = random.Random(spot["seed"])
    deck = list(range(52))
    rng.shuffle(deck)
    alive = len([s for s in spot["stacks"] if s > 0])

    d_old, e_old, log_old, t_old = play(spot, old, deck, payouts)
    d_new, e_new, log_new, t_new = play(spot, new, deck, payouts)
    times = t_old + t_new
    diverged = log_old != log_new
    first_street = None
    if diverged:
        for a, b in zip(log_old + [None], log_new + [None]):
            if a != b:
                first_street = (a or b)[0]
                break
    if not diverged:
        return {"diverged": False, "times": times}

    chips, eqs = [d_new - d_old], [e_new - e_old]
    for _ in range(runouts - 1):
        board = _reboard(deck, alive, rng)
        a = play(spot, old, board, payouts)
        b = play(spot, new, board, payouts)
        chips.append(b[0] - a[0])
        eqs.append(b[1] - a[1])
        times += a[3] + b[3]
    return {"diverged": True, "street": first_street,
            "chips": sum(chips) / len(chips), "equity": sum(eqs) / len(eqs),
            "times": times, "spot": spot, "old": log_old, "new": log_new}


def _mean_se(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / float(n)
    if n < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (n - 1.0)
    return m, math.sqrt(v / n)


def compare_decisions(old, new, spots=2000, payouts=(0.5, 0.3, 0.2), runouts=8,
                      field="mixed", seed=1, workers=None, log_path=None):
    """A/B two feature sets on the spots where they disagree.

    Returns a summary dict.  Per-spot results are appended to `log_path` as
    they finish, so a long run that is killed still leaves its data.
    """
    import json
    # the strategy reads the payout structure from the environment; set it
    # here so the workers (forked below) price ICM the way we measure it
    os.environ["BOT_PAYOUTS"] = ",".join(str(x) for x in payouts)
    rng = random.Random(seed)
    jobs = [(random_spot(rng, field), old, new, list(payouts), runouts)
            for _ in range(spots)]
    results = []
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    pool = Pool(workers)
    try:
        for r in pool.imap_unordered(measure_spot, jobs, chunksize=4):
            results.append(r)
            if log_path and r["diverged"]:
                with open(log_path, "a") as fh:
                    fh.write(json.dumps({k: r[k] for k in
                                         ("street", "chips", "equity", "spot", "old", "new")}) + "\n")
    finally:
        pool.close()
        pool.join()

    div = [r for r in results if r["diverged"]]
    chips_m, chips_se = _mean_se([r["chips"] for r in div])
    eq_m, eq_se = _mean_se([100.0 * r["equity"] for r in div])
    times = sorted(t for r in results for t in r["times"])
    p95 = times[int(0.95 * (len(times) - 1))] if times else 0.0
    return {
        "spots": len(results),
        "diverged": len(div),
        "rate": len(div) / float(max(len(results), 1)),
        "chips_bb": chips_m, "chips_se": chips_se,
        "equity_pp": eq_m, "equity_se": eq_se,
        "by_street": dict((s, len([r for r in div if r["street"] == s]))
                          for s in ("preflop", "flop", "turn", "river")),
        "decision_p95_s": p95,
        "decision_max_s": times[-1] if times else 0.0,
        "decisions": len(times),
    }


def only(name):
    """v1 plus exactly one v2 feature -- for gating features one at a time."""
    out = dict(V1_FEATURES)
    out[name] = True
    return out


ALL_V2 = dict((k, True) for k in FEATURES)


def gate(spots=600, runouts=8, fields=("mixed", "v1", "archetypes", "self"),
         features=None, payouts=(0.5, 0.3, 0.2), out_dir="data/gate_v2", seed=1,
         include_all=True):
    """Every v2 feature alone against v1, then all of them, in each field.

    Writes one summary line per (feature, field) to out_dir/summary.jsonl as
    it finishes, and every divergent spot to its own file.
    """
    import json
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    arms = [(name, only(name)) for name in (features or list(FEATURES))]
    if include_all:
        arms.append(("all", ALL_V2))
    for field in fields:
        for name, new in arms:
            started = time.time()
            res = compare_decisions(
                V1_FEATURES, new, spots=spots, payouts=payouts, runouts=runouts,
                field=field, seed=seed,
                log_path=os.path.join(out_dir, "%s-%s.jsonl" % (name, field)))
            res.update({"feature": name, "field": field, "payouts": list(payouts),
                        "runouts": runouts, "seconds": round(time.time() - started, 1)})
            with open(os.path.join(out_dir, "summary.jsonl"), "a") as fh:
                fh.write(json.dumps(res) + "\n")
            print("%-13s %-10s diverged %4d/%d  chips %+6.3f +/- %.3f bb   "
                  "equity %+6.3f +/- %.3f pp   p95 %.2fs  (%ds)" % (
                      name, field, res["diverged"], res["spots"], res["chips_bb"],
                      res["chips_se"], res["equity_pp"], res["equity_se"],
                      res["decision_p95_s"], res["seconds"]), flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="decision-level v1 vs v2 gate")
    ap.add_argument("--spots", type=int, default=600)
    ap.add_argument("--runouts", type=int, default=8)
    ap.add_argument("--fields", default="mixed,v1,archetypes,self")
    ap.add_argument("--features", default="")
    ap.add_argument("--payouts", default="0.5,0.3,0.2")
    ap.add_argument("--out", default="data/gate_v2")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-all", action="store_true", help="skip the all-features arm")
    a = ap.parse_args()
    gate(spots=a.spots, runouts=a.runouts, fields=a.fields.split(","),
         features=[f for f in a.features.split(",") if f] or None,
         payouts=[float(x) for x in a.payouts.split(",")], out_dir=a.out, seed=a.seed,
         include_all=not a.no_all)
