"""Shove-or-fold equilibrium for the first-in situation.

In a structure this fast most decisive hands are "folded to me, all-in or
fold".  That game is small enough to solve exactly for the stacks actually at
the table, which is better than a generic chart: charts assume a stack
distribution the real table never has.

Solved by iterated best response.  Hero shoves or folds; each player behind
calls or folds; the first caller plays it heads-up (multiway calls are rare
enough that ignoring them costs little and keeps the fixed point clean).

Final-stack expectations, with H hero's stack and E the effective all-in:
    fold          -> H - posted
    all fold      -> H + D - posted
    b calls       -> H - E + q*(2E + O)
where D is everything posted and O is what players other than hero and the
caller left behind.
"""

from poker.cards import all_hand_classes
from poker.equity import equity_vs_range, equities_vs_range, COMBO_COUNT
from poker.placement import equity as placement_equity, call_threshold

CLASSES = all_hand_classes()
TOTAL_COMBOS = float(sum(COMBO_COUNT[c] for c in CLASSES))


class ChipEV(object):
    """Value a final stack as chips.  The original, and wrong for a freezeout."""

    def __call__(self, stacks, hero):
        return stacks[hero]


class PlacementEV(object):
    """Value a final stack as tournament equity.

    The reason this matters: equity is CONCAVE in chips.  Doubling up is worth
    much less than twice as much, and busting costs everything, so the value of
    a gamble must be averaged over outcomes AFTER converting each one to
    equity -- never by converting the average stack.  Getting that order wrong
    reproduces chip-EV exactly.
    """

    def __init__(self, payouts=None, model=None):
        self.payouts = payouts
        self.model = model

    def __call__(self, stacks, hero):
        return placement_equity(stacks, self.payouts, index=hero, model=self.model)


def _outcome_values(value_fn, table_stacks, hero, opponents, dead_total,
                    hero_posted, posted, ante, eff_list, other_list):
    """Value every distinct end state of this spot, once.

    The stack vectors that can result do not depend on hero's cards -- only the
    win probability does.  Precomputing them here keeps the fictitious-play
    loop cheap even though each valuation enumerates finishing orders.
    """
    base = list(table_stacks)

    folded = list(base)
    folded[hero] -= hero_posted + ante
    v_fold = value_fn(folded, hero)

    allfold = list(base)
    allfold[hero] += dead_total - hero_posted - ante
    for k, seat in enumerate(opponents):
        allfold[seat] -= posted[k] + ante
    v_allfold = value_fn(allfold, hero)

    wins, loses = [], []
    for k, seat in enumerate(opponents):
        eff, other = eff_list[k], other_list[k]
        w = list(base)
        w[hero] += eff + other
        w[seat] -= eff
        for j, s2 in enumerate(opponents):
            if j != k:
                w[s2] -= posted[j] + ante
        wins.append(value_fn(w, hero))

        l = list(base)
        l[hero] -= eff
        l[seat] += eff + other
        for j, s2 in enumerate(opponents):
            if j != k:
                l[s2] -= posted[j] + ante
        loses.append(value_fn(l, hero))
    return v_fold, v_allfold, wins, loses


def _aware_threshold(value_fn, index, fold, win, lose, awareness):
    """A caller's price, between chip odds (awareness 0) and full ICM (1).

    The equilibrium assumes opponents feel their own ICM pressure.  A
    hastily built league bot may not -- a chip-EV caller facing a shove it
    "should" fold calls anyway -- and shoving wide into a player who does not
    know they are supposed to fold is expensive.  Awareness is the hedge.
    """
    icm = call_threshold(value_fn, index, fold, win, lose)
    if awareness >= 1.0:
        return icm
    chip = call_threshold(ChipEV(), index, fold, win, lose)
    return chip + max(awareness, 0.0) * (icm - chip)


def _conserved_outcomes(value_fn, pre_stacks, hero, opponents, posted,
                       hero_posted, ante, awareness=1.0):
    """Every end state of a first-in shove, with chips accounted exactly.

    `pre_stacks` are the stacks BEFORE this hand's antes and blinds.  Every
    player dealt in pays the ante, blinds are what they are, the matched
    all-in is capped by the shorter stack, and whoever wins the pot gets all
    of it.  The sum of every outcome vector equals the sum of pre_stacks.

    This matters more than it looks.  The v1 accounting (_outcome_values)
    lets bystanders' antes appear from nowhere and lets the pot vanish when
    hero folds.  Under ICM, where a stack is worth its share of a total,
    those few hundred phantom chips shift the fold/shove comparison by as much
    as the blinds being fought over -- enough that winner-take-all ICM, which
    is exactly linear in chips, did not reproduce chip EV.

    Returns hero's (v_fold, v_allfold, v_win[k], v_lose[k]) and each caller's
    break-even equity priced in that caller's own tournament equity.
    """
    n = len(pre_stacks)
    dealt = [i for i in range(n) if pre_stacks[i] > 0]
    blind = [0] * n
    blind[hero] = hero_posted
    for k, seat in enumerate(opponents):
        blind[seat] = posted[k]

    base = list(pre_stacks)
    for i in dealt:
        base[i] -= min(ante, pre_stacks[i]) + blind[i]
    antes = sum(min(ante, pre_stacks[i]) for i in dealt)
    blinds = sum(blind)

    # hero folds: the pot goes to the player behind with the largest blind --
    # who exactly wins it barely moves hero's equity, but it must go somewhere
    folded = list(base)
    taker = max(opponents, key=lambda s: blind[s]) if opponents else hero
    folded[taker] += antes + blinds

    allfold = list(base)
    allfold[hero] += antes + blinds

    v_win, v_lose, thresholds = [], [], []
    for k, seat in enumerate(opponents):
        # street bets on top of the ante; the shorter stack caps the match
        cap_h = pre_stacks[hero] - min(ante, pre_stacks[hero])
        cap_c = pre_stacks[seat] - min(ante, pre_stacks[seat])
        matched = min(cap_h, cap_c)
        dead = antes + blinds - blind[hero] - blind[seat]
        pot = 2 * matched + dead

        hero_wins = list(base)
        caller_wins = list(base)
        # undo the blinds of the two players involved: their bets are now
        # the matched amount, not the blind
        for vec in (hero_wins, caller_wins):
            vec[hero] += blind[hero] - matched
            vec[seat] += blind[seat] - matched
        hero_wins[hero] += pot
        caller_wins[seat] += pot

        v_win.append(value_fn(hero_wins, hero))
        v_lose.append(value_fn(caller_wins, hero))
        thresholds.append(_aware_threshold(value_fn, seat, allfold, caller_wins, hero_wins,
                                           awareness))

    return (value_fn(folded, hero), value_fn(allfold, hero), v_win, v_lose,
            thresholds)


def range_fraction(rng):
    """Share of all starting hands in a range, weighted by combos."""
    return sum(COMBO_COUNT[c] * w for c, w in rng.items()) / TOTAL_COMBOS


def range_text(rng, limit=18):
    inside = [c for c in CLASSES if rng.get(c, 0) > 0.5]
    head = inside[:limit]
    tail = "" if len(inside) <= limit else " ...+%d" % (len(inside) - limit)
    return " ".join(head) + tail


def solve_first_in(hero_stack, opp_stacks, posted, hero_posted, ante, players,
                   iterations=200, value_fn=None, table_stacks=None, hero_index=None,
                   opponent_seats=None, icm_callers=False, awareness=1.0):
    """Solve one first-in spot by fictitious play.

    Plain best-response iteration cycles here -- hero shoves wide, the blind
    widens its calls, hero tightens, the blind tightens, forever.  Fictitious
    play (best-respond to the running AVERAGE, then fold the response into that
    average) converges instead, and the equilibrium is genuinely mixed at the
    margin, so ranges carry weights rather than being pure sets.

    hero_stack   hero's chips before posting
    opp_stacks   chips of each player still to act, in action order
    posted       what each of those players already put in (blind only)
    hero_posted  hero's own blind
    ante         per-player ante
    players      how many players posted an ante this hand
    icm_callers  v2 model: table_stacks are PRE-HAND stacks, outcomes are
                 accounted with exact chip conservation, and each caller's
                 decision is priced in their own tournament equity rather than
                 chip odds.  Off, the solve is exactly v1.
    """
    dead_total = hero_posted + sum(posted) + ante * players

    shove = dict((c, 1.0) for c in CLASSES)
    calls = [dict((c, 1.0) for c in CLASSES) for _ in opp_stacks]

    eff_list = [min(hero_stack, s) for s in opp_stacks]
    other_list = [dead_total - hero_posted - posted[i] for i in range(len(opp_stacks))]
    thresholds = []
    for i, stack in enumerate(opp_stacks):
        pot = 2.0 * eff_list[i] + other_list[i]
        thresholds.append((eff_list[i] - posted[i]) / pot if pot > 0 else 1.0)
    if value_fn is None or table_stacks is None:
        # chip EV: the closed forms the solver used before
        v_fold = hero_stack - hero_posted
        v_allfold = hero_stack + dead_total - hero_posted - ante
        v_win = [hero_stack - e + (2 * e + o) for e, o in zip(eff_list, other_list)]
        v_lose = [hero_stack - e for e in eff_list]
    else:
        hero_index = 0 if hero_index is None else hero_index
        seats = opponent_seats
        if not seats:
            seats = [i for i in range(len(table_stacks)) if i != hero_index]
        # must line up one-to-one with the players still to act; seats beyond
        # that have already folded and their antes are inside dead_total
        seats = list(seats)[:len(opp_stacks)]
        if len(seats) < len(opp_stacks):
            raise ValueError("opponent_seats must cover every player still to act")
        if icm_callers:
            v_fold, v_allfold, v_win, v_lose, thresholds = _conserved_outcomes(
                value_fn, table_stacks, hero_index, seats, posted, hero_posted, ante,
                awareness)
        else:
            v_fold, v_allfold, v_win, v_lose = _outcome_values(
                value_fn, table_stacks, hero_index, seats, dead_total, hero_posted,
                posted, ante, eff_list, other_list)

    for t in range(iterations):
        weight = 1.0 / (t + 2.0)

        # --- callers best-respond to hero's average shoving range ---
        vs_shove = equities_vs_range(shove)
        for i, _stack in enumerate(opp_stacks):
            threshold = thresholds[i]
            rng = calls[i]
            for k, c in enumerate(CLASSES):
                br = 1.0 if vs_shove[k] > threshold else 0.0
                rng[c] = rng[c] * (1.0 - weight) + br * weight

        # --- hero best-responds to the average calling ranges ---
        call_prob = [range_fraction(r) for r in calls]
        vs_calls = [equities_vs_range(r) for r in calls]
        for k, c in enumerate(CLASSES):
            ev = 0.0
            survive = 1.0
            for i, _stack in enumerate(opp_stacks):
                q = vs_calls[i][k]
                # value each outcome, then average -- not the other way round
                ev += survive * call_prob[i] * (q * v_win[i] + (1.0 - q) * v_lose[i])
                survive *= (1.0 - call_prob[i])
            ev += survive * v_allfold
            br = 1.0 if ev > v_fold else 0.0
            shove[c] = shove[c] * (1.0 - weight) + br * weight

    return shove, calls


def solve_table(stacks, sb, bb, ante, button=0):
    """Solve every first-in position at a table.

    `stacks` is in seat order; `button` indexes it.  Returns position label ->
    (shove range, calling ranges of the players behind).
    """
    n = len(stacks)
    order = [(button + 1 + i) % n for i in range(n)]  # SB, BB, then the rest
    if n == 2:
        labels = ["SB(BTN)", "BB"]
        seats = [button, (button + 1) % n]
    else:
        seats = [order[i] for i in range(n)]
        labels = ["SB", "BB"] + ["UTG", "HJ", "CO", "MP", "LJ"][:n - 3] + ["BTN"]
        seats = seats[:2] + seats[2:]
        # act order after the blinds is UTG..BTN, so reorder for the solve
        seats = seats[2:] + seats[:2]
        labels = labels[2:] + labels[:2]

    blind_of = {}
    for i, s in enumerate(seats):
        blind_of[s] = 0
    if n == 2:
        blind_of[button] = sb
        blind_of[(button + 1) % n] = bb
    else:
        blind_of[order[0]] = sb
        blind_of[order[1]] = bb

    out = {}
    for i, seat in enumerate(seats):
        behind = seats[i + 1:]
        if not behind:
            continue
        shove, calls = solve_first_in(
            hero_stack=stacks[seat],
            opp_stacks=[stacks[b] for b in behind],
            posted=[blind_of[b] for b in behind],
            hero_posted=blind_of[seat],
            ante=ante,
            players=n,
        )
        out[labels[i]] = (shove, calls, [labels[seats.index(b)] for b in behind])
    return out


def range_of_top(fraction, reference):
    """The strongest `fraction` of hands, measured against `reference`.

    Used to model an opponent from a single observed number: if they fold to
    shoves 80% of the time, their calling range is the top 20% by equity.
    """
    scored = sorted(CLASSES, key=lambda c: -equity_vs_range(c, reference))
    out = dict((c, 0.0) for c in CLASSES)
    taken = 0.0
    target = max(0.0, min(1.0, fraction)) * TOTAL_COMBOS
    for c in scored:
        if taken >= target:
            break
        out[c] = 1.0
        taken += COMBO_COUNT[c]
    return out


def exploit_shove_range(hero_stack, opp_stacks, posted, hero_posted, ante, players,
                        fold_probs, reference):
    """Best response to opponents who fold to shoves at OBSERVED rates.

    This is the aggressive end of the dial.  It is never played straight --
    the strategy blends it with the equilibrium range and caps the blend --
    because it is only as good as the fold rates it was handed.
    """
    dead_total = hero_posted + sum(posted) + ante * players
    calls = []
    for i in range(len(opp_stacks)):
        calls.append(range_of_top(1.0 - fold_probs[i], reference))
    call_prob = [1.0 - f for f in fold_probs]

    shove = {}
    for c in CLASSES:
        ev = 0.0
        survive = 1.0
        for i, stack in enumerate(opp_stacks):
            eff = min(hero_stack, stack)
            other = dead_total - hero_posted - posted[i]
            q = equity_vs_range(c, calls[i])
            ev += survive * call_prob[i] * (hero_stack - eff + q * (2 * eff + other))
            survive *= (1.0 - call_prob[i])
        ev += survive * (hero_stack + dead_total - hero_posted - ante)
        shove[c] = 1.0 if ev > hero_stack - hero_posted else 0.0
    return shove


def blend(equilibrium, exploit, dial):
    """Mix at the level of per-hand frequencies, not whole strategies.

    Picking one strategy for a whole hand inherits the exploitative version's
    full exploitability even when it only plays half the time; blending the
    action frequencies at each decision does not.
    """
    dial = max(0.0, min(1.0, dial))
    return dict((c, equilibrium.get(c, 0.0) * (1.0 - dial) + exploit.get(c, 0.0) * dial)
                for c in CLASSES)


def _reshove_outcomes(value_fn, pre_stacks, contrib, hero, raiser):
    """End states of hero re-shoving over an open, chips accounted exactly.

    `contrib` is what each seat has already put in this hand, antes included.
    Returns the four stack vectors: hero folds, raiser folds, hero wins the
    all-in, raiser wins it.
    """
    base = [p - c for p, c in zip(pre_stacks, contrib)]
    pot_now = sum(contrib)
    hero_folds = list(base)
    hero_folds[raiser] += pot_now
    raiser_folds = list(base)
    raiser_folds[hero] += pot_now

    matched = min(pre_stacks[hero], pre_stacks[raiser])
    dead = pot_now - contrib[hero] - contrib[raiser]
    pot = 2 * matched + dead
    hero_wins = list(base)
    raiser_wins = list(base)
    for vec in (hero_wins, raiser_wins):
        vec[hero] = pre_stacks[hero] - matched
        vec[raiser] = pre_stacks[raiser] - matched
    hero_wins[hero] += pot
    raiser_wins[raiser] += pot
    return hero_folds, raiser_folds, hero_wins, raiser_wins


def solve_vs_raise(hero_stack, raiser_stack, raise_to, dead, hero_posted, ante,
                   opener_range, iterations=150, value_fn=None, pre_stacks=None,
                   contrib=None, hero_index=None, raiser_index=None, awareness=1.0):
    """Short stack facing an open raise: shove or fold.

    This spot dominates the back half of a fast tournament and it is not the
    same game as first-in.  There is already a raise in the middle (so more
    dead money to win), the opener has shown strength (so a narrower range to
    run into), and they get a price on calling because their raise is already
    committed.  Solving it properly rather than guessing is worth real chips.

    Solved as a two-player game by fictitious play, exactly as first-in is:
    hero's shoving range and the opener's calling range are mutual best
    responses, with the opener's calls restricted to hands they would have
    opened with.

    Given `value_fn` with `pre_stacks` (before this hand) and `contrib` (what
    each seat has put in so far), both sides are priced in their own
    tournament equity: hero's shove against staying out, and the opener's
    call against folding what they already raised.  Without them the solve is
    the v1 chip-EV one.
    """
    eff = min(hero_stack, raiser_stack)
    other = max(dead - hero_posted - raise_to, 0)

    shove = dict((c, 1.0) for c in CLASSES)
    call = dict((c, opener_range.get(c, 0.0)) for c in CLASSES)

    opener_mass = sum(COMBO_COUNT[c] * opener_range.get(c, 0.0) for c in CLASSES)
    if opener_mass <= 0:
        opener_mass = 1.0

    pot_if_called = 2.0 * eff + other
    threshold = (eff - raise_to) / pot_if_called if pot_if_called > 0 else 1.0

    priced = (value_fn is not None and pre_stacks is not None and contrib is not None
              and hero_index is not None and raiser_index is not None)
    if priced:
        h_fold, r_fold, h_win, r_win = _reshove_outcomes(
            value_fn, pre_stacks, contrib, hero_index, raiser_index)
        v_fold = value_fn(h_fold, hero_index)
        v_steal = value_fn(r_fold, hero_index)
        v_win = value_fn(h_win, hero_index)
        v_lose = value_fn(r_win, hero_index)
        threshold = _aware_threshold(value_fn, raiser_index, r_fold, r_win, h_win,
                                     awareness)

    for t in range(iterations):
        weight = 1.0 / (t + 2.0)

        vs_shove = dict(zip(CLASSES, equities_vs_range(shove)))
        for c in CLASSES:
            if opener_range.get(c, 0.0) <= 0:
                call[c] = 0.0
                continue
            br = 1.0 if vs_shove[c] > threshold else 0.0
            call[c] = call[c] * (1.0 - weight) + br * opener_range[c] * weight

        call_mass = sum(COMBO_COUNT[c] * call[c] for c in CLASSES)
        p_call = min(max(call_mass / opener_mass, 0.0), 1.0)
        vs_call = equities_vs_range(call) if call_mass > 0 else [0.5] * len(CLASSES)
        q = dict(zip(CLASSES, vs_call))

        for c in CLASSES:
            if priced:
                ev = (p_call * (q[c] * v_win + (1.0 - q[c]) * v_lose)
                      + (1.0 - p_call) * v_steal)
                br = 1.0 if ev > v_fold else 0.0
            else:
                ev = (p_call * (hero_stack - eff + q[c] * pot_if_called)
                      + (1.0 - p_call) * (hero_stack + dead - hero_posted - ante))
                br = 1.0 if ev > hero_stack - hero_posted else 0.0
            shove[c] = shove[c] * (1.0 - weight) + br * weight

    return shove, call
