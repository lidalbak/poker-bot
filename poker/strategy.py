"""The decision policy.

Three regimes, because that is what actually changes in this structure:

  short   first-in spots are solved as a shove-or-fold game for the stacks
          actually at the table, rather than read off a chart
  middle  tight-aggressive opening ranges, shove over aggression
  deep    open ranges plus equity-driven postflop play

v1 picked the regime by stack in big blinds (15 / 30).  v2 picks it by
Harrington's effective M, which also sees the antes and the table shrinking;
see `regime`.  v2 also prices every all-in confrontation pairwise -- by the
bubble factor between hero and that particular opponent -- and models a
raiser's range from THEIR stack, since a short stack's shove is desperation,
not strength.  Each v2 change sits behind its own flag (FEATURES) so it can be
measured against v1 in isolation and shipped disabled if it fails its gate.

Postflop is deliberately thin.  In a structure this fast, most decisive hands
end preflop, and the time is better spent on the spots that recur.
"""

import os
import random

from poker.cards import hand_class
from poker.evaluator import evaluate
from poker.protocol import FOLD, CHECK, CALL, RAISE, clamp_raise, safest
from poker.equity import equity_vs_range
from poker.placement import bubble_factor, call_threshold
from poker.pushfold import (solve_first_in, solve_vs_raise, exploit_shove_range,
                            blend, range_of_top, range_fraction, _reshove_outcomes,
                            CLASSES, ChipEV, PlacementEV)

SHORT_BB = 15.0
MIDDLE_BB = 30.0

# Effective-M zone boundaries (Harrington).  Red and below is shove-or-fold;
# Orange keeps opening but never flat-calls.  With no ante, five-handed, these
# land exactly on v1's 15bb / 30bb; antes and a shrinking table pull them
# deeper, which is the point.
SHORT_M = 5.0
MIDDLE_M = 10.0
# Effective M divides by table size, so heads-up a 40bb stack reads as "red".
# Push/fold that deep is wrong, so the M regime is capped in big blinds.
def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


SHORT_CAP_BB = _env_float("BOT_SHORT_CAP_BB", 25.0)
MIDDLE_CAP_BB = _env_float("BOT_MIDDLE_CAP_BB", 50.0)
# Orange zone facing a raise: shove-or-fold instead of flat calling.
ORANGE_SHOVE = os.environ.get("BOT_ORANGE_SHOVE", "1") != "0"

# A raiser this shallow who is all-in (or nearly) is modelled by the solved
# shove range for their depth and seat, not by a positional opening range.
SHOVE_MODEL_MAX_BB = 30.0
NEARLY_ALL_IN = 0.85
# Postflop, the bubble factor is applied only when a call risks at least this
# share of hero's chips; below it ICM barely moves the answer and the
# estimate only adds noise.
LARGE_RISK = 0.30

# v2 changes, each separately switchable.  Environment variable -> default.
# Defaults are the outcome of the decision-level gate (docs/V2-GATE.md):
# a change ships on only if it was not measurably worse in any opponent field.
#   stack_ranges  on   +1.14 +/- 0.36 pp equity per spot vs v1 bots
#   bf_sizing     on   +0.06 +/- 0.05 pp pooled; small, never negative
#   icm_callers   OFF  loses to opponents who do not fold by ICM (-2.0 pp vs
#                      the archetypes); the equilibrium assumes they do
#   m_regime      OFF  push/fold to 25bb with antes lost 1.3 pp in every
#                      field; capped at 15bb it is inert
#   bf_postflop   on   rare (6 in 10,000 spots) but every one positive
#   exact_icm     on   fixes v1's chip-leaking ICM accounting; tighter-only
#                      (widen on reads alone): +1.85 +/- 0.69 pp pooled on a
#                      fresh seed, positive in all four fields
FEATURES = {
    "m_regime":     ("BOT_M_REGIME", "0"),
    "icm_callers":  ("BOT_ICM_CALLERS", "0"),
    "exact_icm":    ("BOT_EXACT_ICM", "1"),
    "stack_ranges": ("BOT_STACK_RANGES", "1"),
    "bf_sizing":    ("BOT_BF_SIZING", "1"),
    # Off: recorded as unmeasurable at its gate, and inert under winner-take-all
    # anyway (bubble factor is 1.0 when only first place pays).
    "bf_postflop":  ("BOT_BF_POSTFLOP", "0"),
    # Regime from the EFFECTIVE stack rather than hero's own. Only bites when
    # hero is the bigger stack: a 50bb hero facing one 6bb player is playing a
    # 6bb hand, not a 50bb one, and should attack accordingly. Unlike the
    # bubble-factor features this is pure arithmetic, so it still applies under
    # winner-take-all. Default off until it passes its gate.
    "eff_stack":    ("BOT_EFF_STACK", "0"),
    # Live games: 3rd twice, folding ~88% of the time it faced a raise and
    # opening 11% from early seats. Wider opens and a real continuing range.
    "aggro":        ("BOT_AGGRO", "1"),
}
V1_FEATURES = dict((k, False) for k in FEATURES)


def exact_on(sit):
    """Use the chip-conserving v2 solves (real stacks behind, exact accounting)?

    exact_icm is the fix for v1's ICM accounting bug; icm_callers implies it."""
    return feature(sit, "exact_icm") or feature(sit, "icm_callers")


def awareness_for(sit):
    """Caller ICM awareness actually used: the configured value when
    icm_callers is on, otherwise 0 -- callers priced by chip odds, as in v1,
    which is the assumption that held up against bots that do not fold by ICM."""
    return caller_awareness(sit) if feature(sit, "icm_callers") else 0.0


def caller_awareness(sit):
    """How ICM-aware to assume the players behind are, 0 (chip EV) .. 1.

    Per-bot override via sit.features["awareness"], else BOT_CALLER_AWARENESS.
    """
    override = getattr(sit, "features", None)
    if override and "awareness" in override:
        return float(override["awareness"])
    try:
        return float(os.environ.get("BOT_CALLER_AWARENESS", "1"))
    except ValueError:
        return 1.0


def feature(sit, name):
    """Is a v2 change on for this decision?  Per-bot override, then environment.

    BOT_V2=0 turns every v2 change off at once, reproducing v1 exactly.
    """
    override = getattr(sit, "features", None)
    if override and name in override:
        return bool(override[name])
    if os.environ.get("BOT_V2", "1") == "0":
        return False
    env, default = FEATURES[name]
    return os.environ.get(env, default) != "0"

# Payout structure of the league.  Winner-take-all is the default because that
# is the common friends' freezeout, and because under it tournament equity and
# chips nearly coincide -- so an unknown payout costs us very little.  Set
# BOT_PAYOUTS="0.5,0.3,0.2" once the real structure is known; the survival
# effect is large under pay jumps and almost nil without them.
def _payouts():
    raw = os.environ.get("BOT_PAYOUTS", "")
    if not raw:
        return None
    try:
        vals = [float(x) for x in raw.split(",") if x.strip()]
        total = sum(vals)
        return [v / total for v in vals] if total > 0 else None
    except ValueError:
        return None


USE_ICM = os.environ.get("BOT_ICM", "1") != "0"


def value_function(use_icm=None):
    on = USE_ICM if use_icm is None else use_icm
    return PlacementEV(_payouts()) if on else ChipEV()


def table_state(sit):
    """Stack vector, hero's index, and the seats still to act -- for the solver."""
    live = [p for p in sit.players if not p.get("eliminated")]
    stacks = [(p.get("stack") or 0) + (p.get("bet") or 0) for p in live]
    ids = [p.get("id") for p in live]
    if sit.you_id not in ids:
        return None, None, None
    hero = ids.index(sit.you_id)
    order = action_order(ids, sit.button_id, sit.street)
    after = order[order.index(sit.you_id) + 1:] if sit.you_id in order else []
    seats = [ids.index(pid) for pid in after if pid in ids]
    return stacks, hero, seats

def dealt_in(sit):
    """Players still in the tournament, i.e. dealt into this hand."""
    return [p for p in sit.players if not p.get("eliminated")]


def m_ratio(chips, sb, bb, ante, players):
    """Harrington's M: how many orbits the stack survives doing nothing."""
    orbit = (sb or 0) + (bb or 0) + (ante or 0) * max(players, 1)
    return chips / float(orbit) if orbit > 0 else float("inf")


def effective_m(chips, sb, bb, ante, players):
    """M scaled to table size.  Five-handed the blinds come round twice as
    fast as ten-handed, so the same M buys half as many hands."""
    return m_ratio(chips, sb, bb, ante, players) * players / 10.0


def zone(eff_m):
    if eff_m >= 20:
        return "green"
    if eff_m >= 10:
        return "yellow"
    if eff_m >= 6:
        return "orange"
    if eff_m >= 1:
        return "red"
    return "dead"


def hero_m(sit):
    players = len(dealt_in(sit)) or max(sit.num_active, 2)
    return effective_m(sit.stack + sit.bet + (sit.ante or 0), sit.small_blind,
                       sit.big_blind, sit.ante, players)


def effective_depth_bb(sit):
    """How deep this hand can actually get, in big blinds.

    min(hero, the largest stack still to act). If everyone behind is short, the
    hand is short no matter how many chips hero is sitting on; if anyone behind
    covers hero, it can go the full depth of hero's stack.
    """
    mine = sit.effective_bb
    live = [p for p in sit.players or []
            if not p.get("folded") and not p.get("eliminated")
            and p.get("id") != sit.you_id]
    if not live:
        return mine
    # Every live opponent counts, not just those still to act: a player who has
    # already called and covers hero can still take hero's whole stack, so
    # treating the hand as short because the ONE player behind is short would
    # shove a big stack into a big caller.
    bb = sit.big_blind or 1
    deepest = max(((p.get("stack") or 0) + (p.get("bet") or 0)) / float(bb) for p in live)
    return min(mine, deepest)


def regime(sit):
    """'short', 'middle' or 'deep'.

    v1 switched on stack in big blinds, which ignores the two things this
    structure has in abundance: antes (up to a full big blind's worth per
    orbit late on) and a shrinking table.  Effective M sees both.  The big
    blind count still feeds the solvers -- they need chip amounts -- and caps
    the regimes, because effective M over-corrects short-handed.
    """
    eff = effective_depth_bb(sit) if feature(sit, "eff_stack") else sit.effective_bb
    if not feature(sit, "m_regime"):
        if eff <= SHORT_BB:
            return "short"
        return "middle" if eff <= MIDDLE_BB else "deep"
    m = hero_m(sit)
    if m <= SHORT_M and eff <= SHORT_CAP_BB:
        return "short"
    if m <= MIDDLE_M and eff <= MIDDLE_CAP_BB:
        return "middle"
    return "deep"


def pre_hand_stacks(sit):
    """Every dealt-in player's chips before this hand's ante and blind.

    The v2 solvers account for chips exactly from the start of the hand, so
    they need what each player had before posting, plus what each has already
    put in.  Returns (ids, pre_stacks, contributions) or None.
    """
    live = dealt_in(sit)
    if not live:
        return None
    ids = [p.get("id") for p in live]
    ante = sit.ante or 0
    pre, contrib = [], []
    for p in live:
        behind = p.get("stack") or 0
        bet = p.get("bet") or 0
        # The host also reports `contribution`.  Whether it includes the ante
        # has not been seen live (every traced hand was ante-free), so take
        # whichever is larger: right in both cases preflop.
        put_in = max(p.get("contribution") or 0, bet + (ante if behind + bet > 0 else 0))
        pre.append(behind + put_in)
        contrib.append(put_in)
    return ids, pre, contrib


def current_stacks(sit):
    """Chips behind plus the current street bet, for every dealt-in player."""
    live = dealt_in(sit)
    return ([p.get("id") for p in live],
            [(p.get("stack") or 0) + (p.get("bet") or 0) for p in live])


# Opening ranges by seats left to act, used once stacks are too deep to shove.
_OPEN_AGGRO = {
    0: "22+ A2s+ K2s+ Q2s+ J4s+ T5s+ 95s+ 84s+ 74s+ 63s+ 53s 43s A2o+ K5o+ Q7o+ J7o+ T7o+ 97o+ 87o 76o",
    1: "22+ A2s+ K2s+ Q5s+ J7s+ T7s+ 96s+ 86s+ 75s+ 64s+ 54s A2o+ K8o+ Q9o+ J9o+ T9o 98o",
    2: "22+ A2s+ K5s+ Q8s+ J8s+ T8s+ 97s+ 87s 76s 65s A7o+ A5o KTo+ QTo+ JTo",
    3: "22+ A2s+ K8s+ Q9s+ J9s+ T9s 98s 87s A9o+ A5o KJo+ QJo",
    4: "22+ A5s+ A2s K9s+ QTs+ JTs T9s ATo+ KQo",
}

_OPEN = {
    0: "22+ A2s+ K2s+ Q4s+ J6s+ T6s+ 96s+ 85s+ 75s+ 64s+ 54s A2o+ K7o+ Q8o+ J8o+ T8o+ 98o",
    1: "22+ A2s+ K5s+ Q7s+ J8s+ T8s+ 97s+ 87s 76s A5o+ K9o+ Q9o+ J9o+ T9o",
    2: "22+ A2s+ K8s+ Q9s+ J9s+ T9s 98s 87s A8o+ KTo+ QTo+ JTo",
    3: "33+ A7s+ A5s A4s KTs+ QTs+ JTs T9s ATo+ KJo+ QJo",
    4: "55+ A9s+ A5s KTs+ QJs AJo+ KQo",
}

_RANK_ORDER = "23456789TJQKA"
_cache = {}


def _expand(spec):
    """Expand a compact range string like '22+ A2s+ KTo+' into a class set."""
    out = set()
    for token in spec.split():
        plus = token.endswith("+")
        token = token.rstrip("+")
        if len(token) == 2 and token[0] == token[1]:
            lo = _RANK_ORDER.index(token[0])
            hi = 12 if plus else lo
            for r in range(lo, hi + 1):
                out.add(_RANK_ORDER[r] * 2)
        elif len(token) == 3:
            hi_r = _RANK_ORDER.index(token[0])
            lo_r = _RANK_ORDER.index(token[1])
            suit = token[2]
            top = hi_r - 1 if plus else lo_r
            for r in range(lo_r, top + 1):
                if r < hi_r:
                    out.add(_RANK_ORDER[hi_r] + _RANK_ORDER[r] + suit)
    return out


OPEN_RANGES = dict((k, _expand(v)) for k, v in _OPEN.items())
OPEN_RANGES_AGGRO = dict((k, _expand(v)) for k, v in _OPEN_AGGRO.items())

# Facing a raise. The v1 pair below continues with 11.9% of hands, which folds
# away nearly every spot that is not a premium -- five-handed that is most of
# the game. The aggro sets add value 3-bets, a real in-position calling range,
# and proper big-blind defence where the pot is laying a price.
_AGGRO_3BET      = "99+ AJs+ AQo+ KQs"
_AGGRO_CALL_IP   = "22+ A2s+ K9s+ Q9s+ J9s+ T9s 98s 87s 76s ATo+ KJo+ QJo"
_AGGRO_DEFEND_BB = ("22+ A2s+ K2s+ Q5s+ J7s+ T7s+ 96s+ 86s+ 75s+ 64s+ 54s "
                    "A2o+ K7o+ Q9o+ J9o+ T9o 98o")


def open_size_bb(eff_bb, behind, ante, limpers=0, min_bf=1.0):
    """How large to open, in big blinds.

    A single fixed size is readable and leaves money behind.  Three things
    genuinely move it:

      depth     deep stacks can afford a larger raise because there is room to
                play after it; short stacks want the cheapest price that still
                folds out the field, since a large open commits them anyway
      position  the more players still to act, the more you must charge them
                to continue -- so early opens are larger and the button is
                cheapest
      antes     dead money in the middle makes a steal profitable at a smaller
                price, so opens shrink once antes start

    Limpers add one big blind each, because there is already money to win and
    a raise must be big enough to charge them for it.

    ICM pressure behind is the fourth: a player whose bubble factor against
    us is high needs far more equity to continue, so a smaller steal folds
    them just as reliably.  `min_bf` is the LOWEST bubble factor among the
    players still to act -- the one most willing to fight sets the price.
    """
    if eff_bb <= 20:
        size = 2.0
    elif eff_bb <= 40:
        size = 2.2
    else:
        size = 2.5

    size += 0.1 * max(0, min(behind, 5) - 1)   # earlier position, larger open
    if ante:
        size -= 0.2                            # dead money, cheaper steal
    size += 1.0 * limpers
    size -= min(0.4, 0.5 * max(min_bf - 1.0, 0.0))
    return max(2.0, size)


def count_limpers(sit):
    """Players who have matched the big blind without raising."""
    bb = sit.big_blind or 1
    if sit.current_bet > bb:
        return 0
    n = 0
    for p in sit.players:
        if p.get("id") == sit.you_id or p.get("folded") or p.get("eliminated"):
            continue
        if (p.get("bet") or 0) >= bb and p.get("id") not in (
                sit.raw and (sit.raw.get("state") or {}).get("hand", {}).get("bigBlindId"),):
            n += 1
    return n


def action_order(ids, button_id, street):
    """Seats in the order they act on this street.

    Preflop the blinds act LAST (order runs UTG..BTN, SB, BB); on every later
    street they act first.  Getting this backwards silently disables the
    push/fold solver for the button, which is exactly where it matters most.
    """
    if not ids:
        return []
    try:
        btn = ids.index(button_id)
    except ValueError:
        btn = len(ids) - 1
    n = len(ids)
    rotated = [ids[(btn + 1 + i) % n] for i in range(n)]   # SB, BB, ..., BTN
    if street != "preflop":
        return rotated
    if n == 2:
        return [ids[btn], ids[(btn + 1) % n]]              # heads-up: button is SB, acts first
    return rotated[2:] + rotated[:2]                       # UTG..BTN, then SB, BB


def live_action_order(sit):
    """Determine blind positions before filtering players who folded.

    A five-handed hand with three folds is still ordered five-handed.
    """
    ids = [p.get("id") for p in dealt_in(sit)]
    live = {p.get("id") for p in dealt_in(sit) if not p.get("folded")}
    return [pid for pid in action_order(ids, sit.button_id, sit.street) if pid in live]


def seats_behind(sit):
    order = live_action_order(sit)
    if sit.you_id not in order:
        return max(sit.num_active - 1, 0)
    return len(order) - 1 - order.index(sit.you_id)


def opener_range_for(sit):
    """What range did the player who raised most likely open with?

    Keyed off how many players were still to act behind them: an early-position
    raise is far stronger than a button raise, and treating them alike is how
    short stacks get their money in badly.
    """
    order = action_order([p.get("id") for p in dealt_in(sit)],
                         sit.button_id, sit.street)
    raiser = None
    for p in sit.players:
        if p.get("id") != sit.you_id and (p.get("bet") or 0) >= sit.current_bet > 0:
            raiser = p.get("id")
    behind = 2
    if raiser in order:
        behind = len(order) - 1 - order.index(raiser)
    spec = OPEN_RANGES.get(min(max(behind, 0), 4), OPEN_RANGES[2])
    return dict((c, 1.0) for c in spec)


def find_raiser(sit):
    """The player whose bet we are facing, or None.

    Also reports whether the spot is clean: exactly one voluntary raiser and
    nobody else yet in for more than a blind.  Anything multiway falls back to
    the v1 heuristics rather than a model built for heads-up.
    """
    raiser = None
    others_in = 0
    bb = sit.big_blind or 1
    for p in sit.players:
        if p.get("id") == sit.you_id or p.get("folded") or p.get("eliminated"):
            continue
        bet = p.get("bet") or 0
        if bet >= sit.current_bet > 0 and raiser is None:
            raiser = p
        elif bet > bb or (bet >= sit.current_bet > bb):
            others_in += 1
    return raiser, others_in == 0


def is_all_in(player, bb):
    """All-in, or so nearly that the chips behind cannot matter.

    The host's snapshot carries `allIn`; the event stream does not, and the
    chip test below covers "nearly" and any host that omits the flag."""
    if player.get("allIn"):
        return True
    bet = player.get("bet") or 0
    behind = player.get("stack") or 0
    total = bet + behind
    return total > 0 and (behind <= 0 or bet >= NEARLY_ALL_IN * total or behind < bb)


_shover_cache = {}


def shover_range(sit, raiser_id, book=None):
    """The range a rational player shoves with at THEIR depth, from THEIR seat.

    A 6bb stack shoves about half of all hands; a 20bb stack about 15%.
    Modelling a short stack's all-in with a positional opening range --
    premiums -- is exactly how a big stack folds itself out of profitable
    calls.  So solve the first-in game from the raiser's chair: their stack,
    the players who were behind them, and the whole table's ICM.

    A confirmed read bends the result a little toward observed aggression;
    see Profile.range_dial for why only a little.
    """
    state = pre_hand_stacks(sit)
    if state is None or raiser_id not in state[0]:
        return None
    ids, pre, _contrib = state
    bb = sit.big_blind or 1
    order = action_order(ids, sit.button_id, "preflop")
    if raiser_id not in order:
        return None
    behind_ids = order[order.index(raiser_id) + 1:]
    if not behind_ids:
        return None
    # who posted what when the raiser acted: blinds act last preflop
    blinds = {}
    if len(ids) == 2:
        blinds[order[0]], blinds[order[1]] = sit.small_blind, bb
    else:
        blinds[order[-2]], blinds[order[-1]] = sit.small_blind, bb
    r = ids.index(raiser_id)
    seats = [ids.index(pid) for pid in behind_ids]
    # half-big-blind resolution: invisible to the solve, but lets spots repeat
    grain = bb / 2.0
    rounded = [int(round(x / grain) * grain) for x in pre]
    on = _icm_on(sit)
    key = (tuple(rounded), r, tuple(seats), sit.small_blind, bb, sit.ante, on,
           exact_on(sit), awareness_for(sit))
    if key not in _shover_cache:
        kw = {}
        if exact_on(sit):
            kw = dict(value_fn=value_function(on), table_stacks=rounded, hero_index=r,
                      opponent_seats=seats, icm_callers=True,
                      awareness=awareness_for(sit))
        shove, _calls = solve_first_in(
            hero_stack=rounded[r],
            opp_stacks=[rounded[i] for i in seats],
            posted=[blinds.get(pid, 0) for pid in behind_ids],
            hero_posted=blinds.get(raiser_id, 0),
            ante=sit.ante or 0,
            players=len(ids),
            iterations=150,
            **kw
        )
        _shover_cache[key] = shove
    rng = _shover_cache[key]
    if book is not None:
        prof = book.profiles.get(raiser_id)
        dial = prof.range_dial() if prof is not None else 0.0
        if dial > 0:
            uniform = dict((c, 1.0) for c in CLASSES)
            width = min(max(range_fraction(rng) * prof.raise_width, 0.0), 1.0)
            # weights feed an equity calculation as probabilities, so blending
            # is safe here -- unlike a blended ACTION, nothing is thresholded
            rng = blend(rng, range_of_top(width, uniform), dial)
    return rng


def _icm_on(sit):
    use_icm = getattr(sit, "use_icm", None)
    return USE_ICM if use_icm is None else use_icm


def villain_model(sit, book=None):
    """(raiser, range, kind) for the raise hero faces.

    kind is 'shove' when the range was solved from the raiser's stack, and
    'open' for the v1 positional opening range.
    """
    raiser, clean = find_raiser(sit)
    if raiser is None:
        return None, opener_range_for(sit), "open"
    bb = sit.big_blind or 1
    depth = ((raiser.get("stack") or 0) + (raiser.get("bet") or 0)) / float(bb)
    if (feature(sit, "stack_ranges") and clean and sit.street == "preflop"
            and is_all_in(raiser, bb) and depth <= SHOVE_MODEL_MAX_BB):
        rng = shover_range(sit, raiser.get("id"), book)
        if rng is not None and range_fraction(rng) > 0:
            return raiser, rng, "shove"
    return raiser, opener_range_for(sit), "open"


def call_all_in(sit, cls, raiser, villain_range):
    """Call a preflop all-in when equity clears THIS confrontation's price.

    The price is the bubble-factor-adjusted pot odds against this particular
    villain, computed from the exact end states: fold (they take the pot),
    call and win, call and lose.  Under chip EV it is plain pot odds.
    Returns True / False, or None when the spot cannot be priced.
    """
    state = pre_hand_stacks(sit)
    if state is None:
        return None
    ids, pre, contrib = state
    if sit.you_id not in ids or raiser.get("id") not in ids:
        return None
    h, r = ids.index(sit.you_id), ids.index(raiser.get("id"))
    h_fold, _r_fold, h_win, r_win = _reshove_outcomes(None, pre, contrib, h, r)
    needed = call_threshold(value_function(_icm_on(sit)), h, h_fold, h_win, r_win)
    return equity_vs_range(cls, villain_range) > needed


def min_bf_behind(sit):
    """Lowest bubble factor, against hero, among players still to act."""
    ids, stacks = current_stacks(sit)
    if sit.you_id not in ids:
        return 1.0
    payouts = _payouts() if _icm_on(sit) else None
    if not payouts:
        return 1.0
    h = ids.index(sit.you_id)
    order = live_action_order(sit)
    if sit.you_id not in order:
        return 1.0
    behind = [ids.index(pid) for pid in order[order.index(sit.you_id) + 1:] if pid in ids]
    if not behind:
        return 1.0
    return min(bubble_factor(stacks, i, h, payouts) for i in behind)


_reshove_cache = {}


def cached_reshove_range(sit, eff, bb, book=None):
    """Solve (and remember) the shove-or-fold response to an open raise."""
    raise_to = sit.current_bet
    if raise_to <= 0 or bb <= 0:
        return None
    priced = None
    if exact_on(sit) or feature(sit, "stack_ranges"):
        priced = _priced_reshove(sit, bb, book)
        if priced is not None:
            return priced
    key = (int(round(eff)), int(round(raise_to / float(bb))),
           int(round(sit.pot / float(bb))), sit.num_active)
    if key in _reshove_cache:
        return _reshove_cache[key]
    stack = sit.stack + sit.bet
    raiser_stack = stack
    for p in sit.players:
        if p.get("id") != sit.you_id and (p.get("bet") or 0) >= raise_to:
            raiser_stack = (p.get("stack") or 0) + (p.get("bet") or 0)
            break
    shove, _call = solve_vs_raise(
        hero_stack=stack,
        raiser_stack=raiser_stack,
        raise_to=raise_to,
        dead=max(sit.pot, raise_to + bb),
        hero_posted=sit.bet,
        ante=sit.ante,
        opener_range=opener_range_for(sit),
        iterations=120,
    )
    _reshove_cache[key] = shove
    return shove


def _priced_reshove(sit, bb, book=None):
    """v2 re-shove solve: the opener's range from their stack, both sides
    priced in tournament equity, chips accounted from the start of the hand."""
    raiser, clean = find_raiser(sit)
    state = pre_hand_stacks(sit)
    if raiser is None or not clean or state is None:
        return None
    ids, pre, contrib = state
    if sit.you_id not in ids or raiser.get("id") not in ids:
        return None
    h, r = ids.index(sit.you_id), ids.index(raiser.get("id"))
    _r, opener, _kind = villain_model(sit, book)
    on = _icm_on(sit)
    key = ("v2", tuple(int(round(x * 2.0 / bb)) for x in pre),
           tuple(int(round(x * 2.0 / bb)) for x in contrib), h, r, on,
           exact_on(sit), awareness_for(sit),
           tuple(sorted(c for c, w in opener.items() if w > 0.5)))
    if key in _reshove_cache:
        return _reshove_cache[key]
    kw = {}
    if exact_on(sit):
        kw = dict(value_fn=value_function(on), pre_stacks=pre, contrib=contrib,
                  hero_index=h, raiser_index=r, awareness=awareness_for(sit))
    stack = sit.stack + sit.bet
    shove, _call = solve_vs_raise(
        hero_stack=stack,
        raiser_stack=(raiser.get("stack") or 0) + (raiser.get("bet") or 0),
        raise_to=sit.current_bet,
        dead=max(sit.pot, sit.current_bet + bb),
        hero_posted=sit.bet,
        ante=sit.ante,
        opener_range=opener,
        iterations=120,
        **kw
    )
    _reshove_cache[key] = shove
    return shove


def cached_shove_range(eff_bb, behind, sb, bb, ante, players, sit=None):
    """Solve (and remember) the shove-or-fold equilibrium for this shape.

    Under placement EV the answer depends on the whole stack distribution, not
    just our own depth, so the cache key carries the rounded stack vector too.
    """
    use_icm = getattr(sit, "use_icm", None) if sit is not None else None
    on = USE_ICM if use_icm is None else use_icm
    stacks = hero = seats = None
    if sit is not None and on:
        stacks, hero, seats = table_state(sit)
    shape = tuple(int(round(x / float(bb))) for x in stacks) if stacks else ()
    key = (int(round(eff_bb)), behind, players, shape, hero, bool(on))
    if key in _cache:
        return _cache[key]
    stack = int(round(eff_bb * bb))
    posted = [0] * max(behind - 2, 0) + [sb, bb]
    posted = posted[-behind:] if behind else []
    kw = {}
    if stacks and seats and len(seats) >= behind:
        kw = dict(value_fn=value_function(on), table_stacks=stacks,
                  hero_index=hero, opponent_seats=seats[:behind])
    shove, calls = solve_first_in(
        hero_stack=stack,
        opp_stacks=[stack] * behind,
        posted=posted,
        hero_posted=0,
        ante=ante,
        players=players,
        iterations=200,
        **kw
    )
    _cache[key] = (shove, calls)
    return _cache[key]


def first_in_range(sit):
    """v2 first-in solve: the real stacks behind, not copies of our own.

    v1 solved against opponents assumed to hold exactly hero's stack, so the
    only thing that could vary by seat was the ICM valuation of hero's own
    outcomes.  Here each player behind has their real stack and blind, chips
    are accounted from the start of the hand, and each caller decides in
    their own tournament equity -- which is what lets "shove the medium
    stacks, not the short ones" come out of the solve.
    """
    state = pre_hand_stacks(sit)
    if state is None:
        return None
    ids, pre, contrib = state
    if sit.you_id not in ids:
        return None
    order = live_action_order(sit)
    if sit.you_id not in order:
        return None
    behind_ids = [pid for pid in order[order.index(sit.you_id) + 1:] if pid in ids]
    if not behind_ids:
        return None
    bb = sit.big_blind or 1
    h = ids.index(sit.you_id)
    seats = [ids.index(pid) for pid in behind_ids]
    ante = sit.ante or 0
    grain = bb / 2.0
    rounded = [int(round(x / grain) * grain) for x in pre]
    posted = [contrib[i] - (ante if pre[i] > 0 else 0) for i in seats]
    hero_posted = contrib[h] - (ante if pre[h] > 0 else 0)
    on = _icm_on(sit)
    aware = awareness_for(sit)
    key = ("v2", tuple(rounded), h, tuple(seats), tuple(posted), hero_posted, ante, on, aware)
    if key in _cache:
        return _cache[key][0]
    shove, calls = solve_first_in(
        hero_stack=rounded[h],
        opp_stacks=[rounded[i] for i in seats],
        posted=posted,
        hero_posted=hero_posted,
        ante=ante,
        players=len(ids),
        iterations=200,
        value_fn=value_function(on),
        table_stacks=rounded,
        hero_index=h,
        opponent_seats=seats,
        icm_callers=True,
        awareness=aware,
    )
    _cache[key] = (shove, calls)
    return shove


def equity_vs_random(hole, board, opponents=1, samples=600, rng=None):
    """Monte Carlo equity against `opponents` random holdings."""
    rng = rng or random.Random(12345)
    known = set(hole) | set(board)
    deck = [c for c in range(52) if c not in known]
    wins = 0.0
    need = 5 - len(board)
    for _ in range(samples):
        pool = deck[:]
        rng.shuffle(pool)
        idx = 0
        extra = pool[idx:idx + need]
        idx += need
        mine = evaluate(list(hole) + list(board) + extra)
        best_other = -1
        tie = False
        for _o in range(opponents):
            oh = pool[idx:idx + 2]
            idx += 2
            score = evaluate(oh + list(board) + extra)
            if score > best_other:
                best_other = score
        if mine > best_other:
            wins += 1.0
        elif mine == best_other:
            wins += 0.5
    return wins / samples


def decide(sit, book=None):
    """Return (action, raiseTo).  `book` is the opponent Book, or None."""
    legal = sit.legal
    actions = legal.get("actions") or []
    if not sit.hole:
        return safest(legal), None

    if sit.street == "preflop":
        return _preflop(sit, legal, actions, book)
    return _postflop(sit, legal, actions)


def _preflop(sit, legal, actions, book=None):
    cls = hand_class(sit.hole[0], sit.hole[1])
    eff = effective_depth_bb(sit) if feature(sit, "eff_stack") else sit.effective_bb
    behind = seats_behind(sit)
    bb = sit.big_blind or 1
    mode = regime(sit)

    # --- facing an all-in from a stack we can model: price the call ---
    # Any depth.  A short stack's shove is a wide, desperate range, and the
    # price of calling it depends on who we would be busting (or doubling).
    if not sit.first_in and feature(sit, "stack_ranges"):
        raiser, rng, kind = villain_model(sit, book)
        if kind == "shove":
            verdict = call_all_in(sit, cls, raiser, rng)
            if verdict is not None:
                if verdict:
                    return (CALL if CALL in actions else CHECK), None
                return (CHECK if CHECK in actions else FOLD), None

    # --- short: solve the shove-or-fold game ---
    if mode == "short":
        if sit.first_in and behind > 0:
            shove = first_in_range(sit) if exact_on(sit) else None
            if shove is None:
                shove, _calls = cached_shove_range(eff, behind, sit.small_blind, bb,
                                                   sit.ante, max(sit.num_active, 2), sit)
            elif not feature(sit, "icm_callers"):
                # Widen only on evidence.  Where the exact solve is TIGHTER than
                # v1 it is pricing hero's own bust risk, which we know; where it
                # is WIDER it is betting that the players behind fold, which
                # the gate showed they often do not.  So take the tighter of
                # the two per hand, and leave widening to confirmed reads
                # (_apply_reads, next line).
                v1, _calls = cached_shove_range(eff, behind, sit.small_blind, bb,
                                                sit.ante, max(sit.num_active, 2), sit)
                shove = dict((c, min(shove.get(c, 0.0), v1.get(c, 0.0))) for c in CLASSES)
            shove = _apply_reads(sit, book, shove, behind, eff, bb)
            # The blended weight is a FREQUENCY, so it must be sampled.
            # Thresholding it at 0.5 collapses the mixture back into a pure
            # strategy -- and since the dial caps at 0.5, every hand the
            # exploit range wanted to add would land at exactly the threshold
            # and be discarded, making the whole opponent head inert.
            if _roll(sit) < shove.get(cls, 0.0) and RAISE in actions:
                return clamp_raise(legal, legal.get("allInTo"))
            return (CHECK if CHECK in actions else FOLD), None
        # Facing a raise while short: solve it rather than guess.
        reshove = cached_reshove_range(sit, eff, bb, book)
        if reshove is not None:
            return _play_reshove(reshove, cls, legal, actions)
        # only if the spot could not be modelled at all
        if cls in _expand("TT+ AQs+ AKo") and RAISE in actions:
            return clamp_raise(legal, legal.get("allInTo"))
        return (CHECK if CHECK in actions else FOLD), None

    # --- middle and deep: open or respond ---
    table = OPEN_RANGES_AGGRO if feature(sit, "aggro") else OPEN_RANGES
    opening = table.get(min(behind, 4), table[4])
    if sit.first_in:
        if cls in opening and RAISE in actions:
            min_bf = min_bf_behind(sit) if feature(sit, "bf_sizing") else 1.0
            size = open_size_bb(eff, behind, sit.ante, count_limpers(sit), min_bf)
            target = int(round(size * bb))
            # An open costing a third of the stack has already committed us;
            # shoving outright is better than raising and then facing a
            # decision we have no chips left to make.
            if target >= 0.30 * (sit.stack + sit.bet) and mode != "deep":
                return clamp_raise(legal, legal.get("allInTo"))
            return clamp_raise(legal, target)
        return (CHECK if CHECK in actions else FOLD), None

    # Orange zone: no flat calls.  Facing a raise it is shove or fold, and
    # that game has a solver.
    if (mode == "middle" and feature(sit, "m_regime") and ORANGE_SHOVE
            and zone(hero_m(sit)) in ("orange", "red", "dead")):
        reshove = cached_reshove_range(sit, eff, bb, book)
        if reshove is not None:
            return _play_reshove(reshove, cls, legal, actions)

    if feature(sit, "aggro"):
        premium = _expand(_AGGRO_3BET)
        # closing the action from the blind gets a price; otherwise we still
        # have players behind and need a tighter continuing range
        good = _expand(_AGGRO_DEFEND_BB if behind == 0 else _AGGRO_CALL_IP)
    else:
        premium = _expand("TT+ AQs+ AKo")
        good = _expand("77+ A9s+ KQs AJo+")
    if cls in premium and RAISE in actions:
        target = legal.get("minRaiseTo", 0) + (legal.get("raiseStep") or bb)
        if mode != "deep":
            return clamp_raise(legal, legal.get("allInTo"))
        return clamp_raise(legal, target)
    # Defend a closing big blind against one small open at an affordable price.
    # Equity realization is a conservative heuristic, not an all-in valuation.
    ids = [p.get("id") for p in dealt_in(sit)]
    pre_order = action_order(ids, sit.button_id, "preflop")
    raiser, clean = find_raiser(sit)
    call = legal.get("callAmount") or 0
    if (CALL in actions and pre_order and sit.you_id == pre_order[-1]
            and sit.bet <= bb and sit.num_active == 2 and clean
            and raiser is not None and not is_all_in(raiser, bb)
            and bb < sit.current_bet <= 3 * bb and 0 < call <= 0.10 * sit.stack):
        equity = equity_vs_range(cls, opener_range_for(sit))
        price = call / float(sit.pot + call)
        suited = cls.endswith("s")
        pair = len(cls) == 2
        realization = 0.80 if suited or pair else 0.70
        if equity * realization > price + 0.04:
            return CALL, None
    if cls in good:
        return (CALL if CALL in actions else CHECK), None
    return (CHECK if CHECK in actions else FOLD), None


def _play_reshove(reshove, cls, legal, actions):
    if reshove.get(cls, 0.0) > 0.5 and RAISE in actions:
        return clamp_raise(legal, legal.get("allInTo"))
    if reshove.get(cls, 0.0) > 0.5 and CALL in actions:
        return CALL, None
    return (CHECK if CHECK in actions else FOLD), None


def _roll(sit):
    """Deterministic per-decision randomiser.

    Seeded from the hand and our own cards so the same spot always resolves the
    same way -- reproducible in the harness, and it cannot be induced to
    re-roll by an opponent repeating an action.
    """
    seed = (hash((sit.hand_id, sit.you_id, tuple(sorted(sit.hole)), sit.street))
            & 0x7fffffff)
    return random.Random(seed).random()


def _apply_reads(sit, book, equilibrium, behind, eff, bb):
    """Turn the dial toward exploitation, as far as the evidence allows."""
    if book is None:
        return equilibrium
    behind_ids = live_action_order(sit)
    if sit.you_id not in behind_ids:
        return equilibrium
    behind_ids = behind_ids[behind_ids.index(sit.you_id) + 1:]
    if not behind_ids:
        return equilibrium

    profiles = [book.get(pid) for pid in behind_ids]
    dial = min([p.dial() for p in profiles]) if profiles else 0.0
    if dial <= 0.0:
        return equilibrium

    stack = int(round(eff * bb))
    posted = ([0] * max(len(behind_ids) - 2, 0) + [sit.small_blind, bb])[-len(behind_ids):]
    exploit = exploit_shove_range(
        hero_stack=stack,
        opp_stacks=[stack] * len(behind_ids),
        posted=posted,
        hero_posted=0,
        ante=sit.ante,
        players=max(sit.num_active, 2),
        fold_probs=[p.fold_to_shove for p in profiles],
        reference=equilibrium,
    )
    return blend(equilibrium, exploit, dial)


def _postflop(sit, legal, actions):
    opponents = max(sit.num_active - 1, 1)
    eq = equity_vs_random(sit.hole, sit.board, opponents=opponents, samples=400)
    call_amount = legal.get("callAmount", 0)
    pot = max(sit.pot, 1)
    pot_odds = call_amount / float(pot + call_amount) if call_amount else 0.0

    if call_amount == 0:
        if eq > 0.62 and RAISE in actions:
            target = legal.get("raiseBase", 0) + max(1, int(pot * 0.6) // (legal.get("raiseStep") or 1)) * (legal.get("raiseStep") or 1)
            return clamp_raise(legal, target)
        return (CHECK if CHECK in actions else FOLD), None

    needed = pot_odds
    if feature(sit, "bf_postflop"):
        needed = _postflop_price(sit, call_amount, pot_odds)

    if eq > needed + 0.12 and RAISE in actions and eq > 0.70:
        target = legal.get("raiseBase", 0) + max(1, int(pot * 0.7) // (legal.get("raiseStep") or 1)) * (legal.get("raiseStep") or 1)
        return clamp_raise(legal, target)
    if eq > needed + 0.02:
        return (CALL if CALL in actions else CHECK), None
    return (CHECK if CHECK in actions else FOLD), None


def _postflop_price(sit, call_amount, pot_odds):
    """Equity a postflop call must clear, with ICM where it matters.

    Only for calls that risk a large share of hero's chips: there the bubble
    factor against the bettor raises the price, keeping the pot odds and
    weighting the risk side.  Small calls keep plain pot odds -- ICM barely
    moves them and an estimated adjustment would only add noise.
    """
    chips = (sit.stack or 0) + (sit.bet or 0)
    if chips <= 0 or call_amount < LARGE_RISK * chips:
        return pot_odds
    payouts = _payouts() if _icm_on(sit) else None
    if not payouts:
        return pot_odds
    bettor = None
    for p in sit.players:
        if p.get("id") == sit.you_id or p.get("folded") or p.get("eliminated"):
            continue
        if (p.get("bet") or 0) >= sit.current_bet and bettor is None:
            bettor = p
    live = dealt_in(sit)
    ids = [p.get("id") for p in live]
    if bettor is None or sit.you_id not in ids:
        return pot_odds
    h, v = ids.index(sit.you_id), ids.index(bettor.get("id"))
    behind = [p.get("stack") or 0 for p in live]
    pot = sit.pot
    call = min(call_amount, sit.stack or 0)
    # if hero is short of a full call, the uncovered part of the bet goes back
    excess = max((bettor.get("bet") or 0) - ((sit.bet or 0) + call), 0)
    fold = list(behind)
    fold[v] += pot
    win = list(behind)
    win[h] += pot - excess
    win[v] += excess
    lose = list(behind)
    lose[h] -= call
    lose[v] += pot + call
    needed = call_threshold(PlacementEV(payouts), h, fold, win, lose)
    return needed
