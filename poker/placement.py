"""Tournament equity: what a stack is actually worth.

The solvers currently maximise chips.  In a freezeout that is the wrong
objective -- busting ends the tournament, and doubling up does not double
what you are playing for.  This module answers "given these stacks, how
likely is each player to finish 1st..5th", which converts a chip amount into
tournament equity once a payout vector is applied.

Two implementations behind one interface:

  malmuth_harville   closed form, no training, no dependencies.  The baseline
                     the learned model must beat, and a perfectly shippable
                     answer on its own.
  LearnedPlacement   an MLP fitted to self-play outcomes, which can in
                     principle capture what the closed form ignores: blind
                     pressure, position relative to the button, and the fact
                     that short stacks bust faster than chip counts imply.

Deliberately payout-agnostic: both return a distribution over finishing
positions, and the payout vector is applied at the end.  That way winner-take-
all, top-two and top-three all come from the same model, and we do not need to
know the league's payouts before building it.
"""

import itertools
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODEL_PATH = os.path.join(DATA_DIR, "placement.pt")


def malmuth_harville(stacks):
    """Placement distribution by the standard ICM recursion.

    Probability of a given finishing order is the product, over each place in
    turn, of that player's share of the chips still unplaced.  Exact by
    enumeration for five players (120 orderings).
    """
    live = [(i, s) for i, s in enumerate(stacks) if s > 0]
    n = len(stacks)
    dist = [[0.0] * n for _ in range(n)]

    if not live:
        return dist
    if len(live) == 1:
        dist[live[0][0]][0] = 1.0
        for i, s in enumerate(stacks):
            if s <= 0:
                dist[i][n - 1] = 1.0
        return dist

    total = float(sum(s for _i, s in live))
    for order in itertools.permutations(live):
        p = 1.0
        remaining = total
        for place, (_idx, s) in enumerate(order):
            if remaining <= 0:
                p = 0.0
                break
            p *= s / remaining
            remaining -= s
        if p <= 0:
            continue
        for place, (idx, _s) in enumerate(order):
            dist[idx][place] += p

    # players already out fill the bottom places
    busted = [i for i, s in enumerate(stacks) if s <= 0]
    for k, i in enumerate(busted):
        dist[i][n - 1 - k] = 1.0
    return dist


def equity(stacks, payouts=None, index=None, model=None):
    """Tournament equity as a share of the prize pool.

    payouts defaults to winner-take-all, which is the right default for a
    friends' freezeout unless told otherwise.
    """
    n = len(stacks)
    payouts = payouts or ([1.0] + [0.0] * (n - 1))
    dist = (model.distribution(stacks) if model is not None
            else malmuth_harville(stacks))
    values = []
    for i in range(n):
        v = sum(dist[i][p] * (payouts[p] if p < len(payouts) else 0.0)
                for p in range(n))
        values.append(v)
    return values if index is None else values[index]



def bubble_factor(stacks, hero, villain, payouts=None, model=None, risk=None):
    """How much more a chip lost costs hero than a chip won from `villain` gains.

        BF = (EV_begin - EV_lose) / (EV_win - EV_begin)

    It is PAIRWISE: the same hero has a different bubble factor against every
    opponent, because what matters is who the chips move to.  Losing to the
    other big stack creates a monster; losing to a short stack merely feeds
    them.  Under chip EV (and under winner-take-all ICM, where equity is linear
    in chips) it is exactly 1.

    `risk` defaults to the full effective stack between the two -- the all-in
    confrontation.  Returns float('inf') when winning is worth nothing to hero
    (e.g. hero is already locked into first place).
    """
    eff = min(stacks[hero], stacks[villain]) if risk is None else risk
    if eff <= 0 or hero == villain:
        return 1.0
    begin = equity(stacks, payouts, hero, model)
    win = list(stacks)
    win[hero] += eff
    win[villain] -= eff
    lose = list(stacks)
    lose[hero] -= eff
    lose[villain] += eff
    gain = equity(win, payouts, hero, model) - begin
    loss = begin - equity(lose, payouts, hero, model)
    if gain <= 1e-12:
        return float("inf")
    return loss / gain


def required_equity(bf):
    """Share of pots hero must win for an even-money all-in to break even."""
    if bf == float("inf"):
        return 1.0
    return bf / (bf + 1.0)


def risk_premium(bf):
    """Equity needed beyond the chip-EV 50%."""
    return required_equity(bf) - 0.5


def call_threshold(value_fn, index, fold, win, lose):
    """Equity at which calling breaks even for player `index`, in any currency.

    `fold`, `win` and `lose` are the complete stack vectors after each
    outcome; `value_fn(stacks, index)` prices one.  Under chip EV this reduces
    to plain pot odds, risk / (pot + risk).  Under ICM it equals

        BF * risk / (pot + BF * risk)

    with BF the bubble factor of this particular confrontation -- the pot odds
    are kept and the bubble factor is applied to the risk side.  Computing it
    from the outcome vectors directly avoids having to define "risk" and "pot"
    separately for blinds, antes and dead money.
    """
    v_fold = value_fn(fold, index)
    v_win = value_fn(win, index)
    v_lose = value_fn(lose, index)
    span = v_win - v_lose
    if span <= 1e-12:
        return 1.0
    return min(max((v_fold - v_lose) / span, 0.0), 1.0)

class ICM(object):
    """Closed-form model, wrapped so it is interchangeable with the learned one."""

    name = "malmuth-harville"

    def distribution(self, stacks):
        return malmuth_harville(stacks)


def load(path=MODEL_PATH):
    """Return the learned model if one has been trained and beaten the baseline."""
    if not os.path.exists(path):
        return ICM()
    try:
        from poker.placement_net import LearnedPlacement
        return LearnedPlacement.load(path)
    except Exception:
        return ICM()
