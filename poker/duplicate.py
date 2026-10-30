"""Duplicate-deal evaluation.

Poker's variance is large enough that naive comparison is useless: 25
tournaments per arm gave a standard error of ~13.5 percentage points, wide
enough to hide any effect worth having.  Two fixes, both applied here:

  rotation   the same deck is replayed with the bots rotated through every
             seat, so card luck is dealt to everyone equally and cancels
  hand-level a tournament yields one noisy bit; a hand yields a chip count.
             Measuring per hand converts every decision into a datapoint.

`compare_hands` is the workhorse -- use it for any gate decision.
`compare_tournaments` measures what we actually care about (placement) but
needs far more samples to say anything.
"""

import math
import random
from collections import defaultdict

from poker.engine import Table, DEFAULT_LEVELS


def _mean_se(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / float(n)
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1.0)
    return mean, math.sqrt(var / n)


def compare_hands(make_bots, deals=2000, stack=5000, level=4, seed=0):
    """Play one hand per rotation on identical decks; report chips won per hand.

    `make_bots` is called fresh for every rotation and must return the five
    bots in a fixed identity order.  Bots are rotated through seats, so each
    identity sees every position with the same cards.
    """
    sb, bb, ante = DEFAULT_LEVELS[min(level, len(DEFAULT_LEVELS) - 1)]
    rng = random.Random(seed)
    names = [b.name for b in make_bots()]
    n = len(names)
    per_deal = defaultdict(list)

    for d in range(deals):
        deck = list(range(52))
        rng.shuffle(deck)
        totals = dict((nm, 0) for nm in names)
        for rot in range(n):
            bots = make_bots()
            seated = [bots[(i + rot) % n] for i in range(n)]
            table = Table(seated, starting_stack=stack, seed=d * 97 + rot)
            table.button = 0
            hand = table.play_hand(level_index=level, deck=deck)
            if hand is None:
                continue
            for seat_i, seat in enumerate(table.seats):
                identity = names[(seat_i + rot) % n]
                totals[identity] += seat.stack - stack
        for nm in names:
            per_deal[nm].append(totals[nm] / float(n))

    out = {}
    for nm in names:
        mean, se = _mean_se(per_deal[nm])
        out[nm] = {
            "chips_per_hand": mean,
            "se": se,
            "bb_per_100": 100.0 * mean / bb,
            "bb_per_100_se": 100.0 * se / bb,
            "hands": len(per_deal[nm]) * n,
        }
    return out


def compare_tournaments(make_bots, trials=200, stack=5000, hands_per_level=12,
                        max_hands=600, payouts=None, seed=0):
    """Full freezeouts on matched deck seeds, rotated through seats.

    Reports expected payout share, which is the quantity the bot should be
    maximising -- not chips, and not raw win rate.
    """
    names = [b.name for b in make_bots()]
    n = len(names)
    payouts = payouts or [1.0, 0.0, 0.0, 0.0, 0.0]
    per_trial = defaultdict(list)

    for t in range(trials):
        share = dict((nm, 0.0) for nm in names)
        for rot in range(n):
            bots = make_bots()
            seated = [bots[(i + rot) % n] for i in range(n)]
            table = Table(seated, starting_stack=stack, seed=t * 131 + rot)
            hands = 0
            while len(table.survivors()) > 1 and hands < max_hands:
                if table.play_hand(level_index=min(hands // hands_per_level,
                                                   len(DEFAULT_LEVELS) - 1)) is None:
                    break
                hands += 1
            order = sorted(range(n), key=lambda i: -table.seats[i].stack)
            for place, seat_i in enumerate(order):
                identity = names[(seat_i + rot) % n]
                share[identity] += payouts[min(place, len(payouts) - 1)]
        for nm in names:
            per_trial[nm].append(share[nm] / float(n))

    out = {}
    for nm in names:
        mean, se = _mean_se(per_trial[nm])
        out[nm] = {"payout_share": mean, "se": se, "tournaments": trials * n}
    return out


def report(results, key="bb_per_100", se_key="bb_per_100_se"):
    lines = ["%-14s %12s %10s" % ("bot", key, "+/- 1 se")]
    lines.append("-" * 38)
    for nm, r in sorted(results.items(), key=lambda kv: -kv[1].get(key, 0)):
        lines.append("%-14s %12.2f %10.2f" % (nm, r.get(key, 0), r.get(se_key, r.get("se", 0))))
    return "\n".join(lines)
