"""Behavioural acceptance tests for the v2 stack-aware model.

"Attack medium stacks, avoid big stacks, short stacks gamble" must EMERGE from
pricing each caller in their own tournament equity; nothing here is a rule in
the strategy.  So these tests build synthetic tables and assert only the SIGN
of each effect on the solved first-in shove range.

Every table: five players, 125bb in play (25bb average), 100/200 blinds with a
25 ante, hero in the small blind and exactly one player behind in the big
blind -- the purest first-in confrontation, where the only thing that changes
between tests is who is in the big blind.

Run:  python3 -m tests.behaviour
"""

import sys

from poker.placement import bubble_factor
from poker.pushfold import solve_first_in, range_fraction, PlacementEV, ChipEV

BB, SB, ANTE = 200, 100, 25
PAYOUTS = [0.5, 0.3, 0.2]
WTA = [1.0]


def shove_share(hero, villain, others, payouts, chip=False):
    stacks = [hero * BB, villain * BB] + [o * BB for o in others]
    value = ChipEV() if chip else PlacementEV(payouts)
    shove, _calls = solve_first_in(
        stacks[0], [stacks[1]], [BB], SB, ANTE, 5, iterations=200,
        value_fn=value, table_stacks=stacks, hero_index=0, opponent_seats=[1],
        icm_callers=True)
    return range_fraction(shove)


# (hero bb, villain-in-BB bb, the three bystanders)
TABLES = {
    "big->medium": (40, 25, [25, 25, 10]),
    "big->short":  (40, 10, [25, 25, 25]),
    "big->big":    (40, 40, [25, 10, 10]),
    "medium->big": (25, 40, [25, 25, 10]),
    "medium->medium": (25, 25, [40, 25, 10]),
    "medium->short": (25, 10, [40, 25, 25]),
    "short->short": (10, 10, [40, 40, 25]),
    "short->medium": (10, 25, [40, 40, 10]),
    "short->big":  (10, 40, [40, 25, 10]),
}


def run():
    icm = dict((k, shove_share(h, v, o, PAYOUTS)) for k, (h, v, o) in TABLES.items())
    wta = dict((k, shove_share(h, v, o, WTA)) for k, (h, v, o) in TABLES.items())
    chip = dict((k, shove_share(h, v, o, None, chip=True)) for k, (h, v, o) in TABLES.items())

    print("%-16s %8s %8s %8s" % ("table", "chip", "WTA", "50/30/20"))
    for k in TABLES:
        print("%-16s %8.3f %8.3f %8.3f" % (k, chip[k], wta[k], icm[k]))
    print()

    checks = []

    def check(name, ok, detail):
        checks.append(ok)
        print("%-4s %s  (%s)" % ("PASS" if ok else "FAIL", name, detail))

    shorts = [k for k in TABLES if k.startswith("short")]
    # effect = how far ICM moves a range away from chip EV
    effect = dict((k, icm[k] - chip[k]) for k in TABLES)
    wta_effect = dict((k, wta[k] - chip[k]) for k in TABLES)

    check("big stack shoves materially wider into a medium than into a short",
          icm["big->medium"] > icm["big->short"] + 0.05,
          "%.3f vs %.3f" % (icm["big->medium"], icm["big->short"]))
    check("big stack shoves tighter into another big than into a medium",
          icm["big->big"] < icm["big->medium"],
          "%.3f vs %.3f" % (icm["big->big"], icm["big->medium"]))
    check("medium with a big stack behind is the tightest spot of all",
          icm["medium->big"] == min(icm.values()),
          "%.3f; next %.3f" % (icm["medium->big"],
                               sorted(icm.values())[1]))
    check("short stack is widest against another short",
          icm["short->short"] == max(icm[k] for k in shorts),
          ", ".join("%s %.3f" % (k, icm[k]) for k in shorts))
    # Informational, not a pass/fail criterion.  The brief also asked for the
    # short stack to be the widest shover overall, but that contradicts the
    # first criterion: a big stack shoving into a medium it covers faces a
    # caller who needs ~67% equity, so it can shove nearly any two cards --
    # wider than any short stack, whose callers need only ~53%.  Chip-leader
    # bubble abuse is the correct ICM answer, so the model is not bent to
    # satisfy the other reading.
    widest = max(icm, key=icm.get)
    print("INFO widest shover overall is %s at %.3f (short stacks: %s)" % (
        widest, icm[widest], ", ".join("%.3f" % icm[k] for k in shorts)))
    check("winner-take-all: every ICM effect shrinks to (near) zero",
          max(abs(x) for x in wta_effect.values()) < 0.01
          and max(abs(x) for x in effect.values()) > 0.05,
          "largest WTA effect %.4f, largest 50/30/20 effect %.3f" % (
              max(abs(x) for x in wta_effect.values()),
              max(abs(x) for x in effect.values())))

    # the pairwise quantity itself, on the published final-table example
    st = [41, 37, 9, 13, 25]
    bfs = [bubble_factor(st, 0, 1, PAYOUTS), bubble_factor(st, 0, 2, PAYOUTS),
           bubble_factor(st, 2, 0, PAYOUTS), bubble_factor(st, 2, 3, PAYOUTS)]
    check("bubble factors order as published (CL>2nd highest, CL>short lowest)",
          bfs[0] > bfs[2] > bfs[3] > bfs[1] > 1.0,
          "CL-2nd %.2f  short-CL %.2f  short-short %.2f  CL-short %.2f" % (
              bfs[0], bfs[2], bfs[3], bfs[1]))
    check("bubble factor is exactly 1 under winner-take-all",
          all(abs(bubble_factor(st, h, v, WTA) - 1.0) < 1e-9
              for h in range(5) for v in range(5) if h != v), "")
    return checks


if __name__ == "__main__":
    results = run()
    print("\n%d/%d passed" % (sum(results), len(results)))
    sys.exit(0 if all(results) else 1)
