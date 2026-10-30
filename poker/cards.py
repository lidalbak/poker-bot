"""Card representation.

A card is an int 0..51.  rank = card >> 2  (0 = deuce .. 12 = ace),
suit = card & 3.  This packing makes suit-isomorphism work (iso.py) cheap
and keeps everything in the integer domain for the evaluator.
"""

RANKS = "23456789TJQKA"
SUITS = "cdhs"

NUM_CARDS = 52
DECK = tuple(range(NUM_CARDS))


def make_card(rank, suit):
    """rank 0..12, suit 0..3 -> card int."""
    return (rank << 2) | suit


def rank_of(card):
    return card >> 2


def suit_of(card):
    return card & 3


def card_str(card):
    return RANKS[card >> 2] + SUITS[card & 3]


def cards_str(cards):
    return " ".join(card_str(c) for c in cards)


def parse_card(text):
    text = text.strip()
    if len(text) != 2:
        raise ValueError("bad card: %r" % (text,))
    r = RANKS.find(text[0].upper())
    s = SUITS.find(text[1].lower())
    if r < 0 or s < 0:
        raise ValueError("bad card: %r" % (text,))
    return make_card(r, s)


def parse_cards(text):
    return [parse_card(t) for t in text.replace(",", " ").split()]


# --- 169 canonical starting hands -------------------------------------------

def hand_class(c1, c2):
    """Map two hole cards onto one of the 169 strategically distinct classes.

    Returns a string like 'AKs', 'AKo', 'TT'.  This is the preflop lookup key:
    suits do not matter preflop beyond suited-vs-offsuit.
    """
    r1, r2 = rank_of(c1), rank_of(c2)
    hi, lo = (r1, r2) if r1 >= r2 else (r2, r1)
    if hi == lo:
        return RANKS[hi] * 2
    suited = suit_of(c1) == suit_of(c2)
    return RANKS[hi] + RANKS[lo] + ("s" if suited else "o")


def all_hand_classes():
    """All 169 classes, in a stable order."""
    out = []
    for hi in range(12, -1, -1):
        for lo in range(hi, -1, -1):
            if hi == lo:
                out.append(RANKS[hi] * 2)
            else:
                out.append(RANKS[hi] + RANKS[lo] + "s")
                out.append(RANKS[hi] + RANKS[lo] + "o")
    return out


def hand_class_combos(hc):
    """Every concrete two-card combo belonging to a class."""
    hi = RANKS.index(hc[0])
    lo = RANKS.index(hc[1])
    combos = []
    if len(hc) == 2:  # pocket pair
        for s1 in range(4):
            for s2 in range(s1 + 1, 4):
                combos.append((make_card(hi, s1), make_card(lo, s2)))
    elif hc[2] == "s":
        for s in range(4):
            combos.append((make_card(hi, s), make_card(lo, s)))
    else:
        for s1 in range(4):
            for s2 in range(4):
                if s1 != s2:
                    combos.append((make_card(hi, s1), make_card(lo, s2)))
    return combos
