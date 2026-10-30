"""Fast 7-card hand evaluator.

evaluate(cards) -> int.  Higher is better; values are directly comparable
across hands.  No table files, no dependencies: the only precomputation is an
8192-entry straight lookup built at import (a few milliseconds).

Encoding: category * 16^5 + five 4-bit tiebreak ranks, most significant first.
"""

from poker.cards import rank_of, suit_of

HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)

CATEGORY_NAMES = [
    "high card", "pair", "two pair", "trips", "straight",
    "flush", "full house", "quads", "straight flush",
]

_SHIFT = [16 ** 4, 16 ** 3, 16 ** 2, 16, 1]
_CAT = 16 ** 5


def _encode(category, tiebreaks):
    v = category * _CAT
    for i, r in enumerate(tiebreaks[:5]):
        v += r * _SHIFT[i]
    return v


def _build_straight_table():
    """mask of 13 rank bits -> rank of the straight's high card, or -1."""
    table = [-1] * (1 << 13)
    for mask in range(1 << 13):
        best = -1
        # wheel: A-2-3-4-5, high card is the five (rank index 3)
        if mask & (1 << 12) and (mask & 0b1111) == 0b1111:
            best = 3
        for high in range(4, 13):
            window = 0b11111 << (high - 4)
            if mask & window == window:
                best = high
        table[mask] = best
    return table


_STRAIGHT = _build_straight_table()


def evaluate(cards):
    """cards: iterable of 5..7 card ints."""
    rank_counts = [0] * 13
    suit_counts = [0] * 4
    suit_masks = [0] * 4
    rank_mask = 0

    for c in cards:
        r = rank_of(c)
        s = suit_of(c)
        rank_counts[r] += 1
        suit_counts[s] += 1
        suit_masks[s] |= 1 << r
        rank_mask |= 1 << r

    # --- flush family ---
    for s in range(4):
        if suit_counts[s] >= 5:
            fmask = suit_masks[s]
            sf = _STRAIGHT[fmask]
            if sf >= 0:
                return _encode(STRAIGHT_FLUSH, [sf])
            top = []
            for r in range(12, -1, -1):
                if fmask & (1 << r):
                    top.append(r)
                    if len(top) == 5:
                        break
            return _encode(FLUSH, top)

    # --- rank-count families ---
    quads = trips = []
    quads = [r for r in range(12, -1, -1) if rank_counts[r] == 4]
    trips = [r for r in range(12, -1, -1) if rank_counts[r] == 3]
    pairs = [r for r in range(12, -1, -1) if rank_counts[r] == 2]
    singles = [r for r in range(12, -1, -1) if rank_counts[r] == 1]

    if quads:
        q = quads[0]
        kickers = [r for r in range(12, -1, -1) if r != q and rank_counts[r] > 0]
        return _encode(QUADS, [q, kickers[0]])

    if trips and (len(trips) > 1 or pairs):
        t = trips[0]
        pair = trips[1] if len(trips) > 1 else pairs[0]
        if pairs and pairs[0] > pair:
            pair = pairs[0]
        return _encode(FULL_HOUSE, [t, pair])

    st = _STRAIGHT[rank_mask]
    if st >= 0:
        return _encode(STRAIGHT, [st])

    if trips:
        t = trips[0]
        kickers = [r for r in singles if r != t][:2]
        return _encode(TRIPS, [t] + kickers)

    if len(pairs) >= 2:
        hi, lo = pairs[0], pairs[1]
        rest = [r for r in range(12, -1, -1) if r not in (hi, lo) and rank_counts[r] > 0]
        return _encode(TWO_PAIR, [hi, lo, rest[0]])

    if pairs:
        p = pairs[0]
        return _encode(PAIR, [p] + singles[:3])

    return _encode(HIGH_CARD, singles[:5])


def category_of(value):
    return value // _CAT


def describe(value):
    return CATEGORY_NAMES[category_of(value)]
