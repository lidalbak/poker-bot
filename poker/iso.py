"""Suit isomorphism.

Suits carry no strategic meaning until they form flushes, so many situations
that look different are identical.  Canonicalising collapses them, which is
what makes exhaustive precomputation affordable: 22,100 raw flops become
1,755 distinct ones, and runtime equity caches hit far more often.
"""

from itertools import permutations, combinations

from poker.cards import rank_of, suit_of, make_card

_PERMS = tuple(permutations(range(4)))


def _apply(perm, cards):
    return tuple(sorted(make_card(rank_of(c), perm[suit_of(c)]) for c in cards))


def canonical_flop(flop):
    """Canonical representative of a three-card flop."""
    return min(_apply(p, flop) for p in _PERMS)


def canonical_key(hole, board=()):
    """Canonical key for a full situation.

    Street boundaries are preserved (flop / turn / river are kept apart) because
    a card arriving on the turn is not interchangeable with one on the flop.
    """
    hole = tuple(hole)
    board = tuple(board)
    flop, turn, river = board[:3], board[3:4], board[4:5]
    best = None
    for p in _PERMS:
        key = (_apply(p, hole), _apply(p, flop), _apply(p, turn), _apply(p, river))
        if best is None or key < best:
            best = key
    return best


def all_canonical_flops():
    """Every strategically distinct flop.  Returns a sorted list (1,755 of them)."""
    seen = set()
    for flop in combinations(range(52), 3):
        seen.add(canonical_flop(flop))
    return sorted(seen)


# --- board texture ----------------------------------------------------------

def flop_texture(flop):
    """Coarse texture label used as a policy feature.

    Deliberately small: a handful of buckets the opponent model can actually
    accumulate enough observations in (see the pooling requirement in the
    playbook).
    """
    ranks = sorted((rank_of(c) for c in flop), reverse=True)
    suits = [suit_of(c) for c in flop]

    if len(set(suits)) == 1:
        suitedness = "mono"
    elif len(set(suits)) == 2:
        suitedness = "twotone"
    else:
        suitedness = "rainbow"

    if ranks[0] == ranks[1] == ranks[2]:
        pairing = "trips"
    elif ranks[0] == ranks[1] or ranks[1] == ranks[2]:
        pairing = "paired"
    else:
        pairing = "unpaired"

    spread = ranks[0] - ranks[2]
    if pairing == "unpaired" and spread <= 4:
        connectivity = "connected"
    elif pairing == "unpaired" and spread <= 7:
        connectivity = "semi"
    else:
        connectivity = "disconnected"

    if ranks[0] >= 10:        # ace/king/queen/jack high
        height = "high"
    elif ranks[0] >= 6:
        height = "mid"
    else:
        height = "low"

    return "%s-%s-%s-%s" % (height, pairing, suitedness, connectivity)
