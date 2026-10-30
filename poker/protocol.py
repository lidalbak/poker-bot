"""Protocol v2 request parsing and response construction.

Everything here is defensive.  The host turns a malformed reply, a wrong turn
ID, an illegal action or a timeout into check-else-fold and records the
fallback on the shared dashboard, so the cost of a bug is both chips and
visible embarrassment.  Nothing in this module is allowed to raise.
"""

from poker.cards import parse_card

FOLD, CHECK, CALL, RAISE = "fold", "check", "call", "raise"


class Situation(object):
    """A parsed decision request, in the terms the strategy wants to think in."""

    __slots__ = ("turn_id", "hand_id", "game_id", "hole", "board", "street", "stack",
                 "bet", "pot", "big_blind", "small_blind", "ante", "legal", "players", "current_bet",
                 "active", "you_id", "button_id", "memory", "events", "truncated",
                 "through_id", "timeout_ms", "level_index", "starting_stack", "raw", "use_icm",
                 "features")

    @property
    def effective_bb(self):
        """Own stack in big blinds -- the number that picks the strategy regime."""
        return (self.stack + self.bet) / float(self.big_blind) if self.big_blind else 0.0

    @property
    def table_bb(self):
        """Smallest stack among players still in the hand, in big blinds."""
        live = [p for p in self.players if not p.get("folded") and not p.get("eliminated")]
        if not live or not self.big_blind:
            return self.effective_bb
        return min((p.get("stack", 0) + p.get("bet", 0)) for p in live) / float(self.big_blind)

    @property
    def num_active(self):
        return len([p for p in self.players
                    if not p.get("folded") and not p.get("eliminated")])

    @property
    def first_in(self):
        """Nobody has voluntarily raised: only the blinds are out there."""
        return self.current_bet <= self.big_blind

    @property
    def facing_raise(self):
        """Someone has actually raised -- the big blind alone does not count."""
        return self.current_bet > self.big_blind


def _cards(values):
    out = []
    for v in values or []:
        try:
            out.append(parse_card(v))
        except Exception:
            pass
    return out


def parse(payload):
    """Turn a raw request dict into a Situation.  Never raises."""
    s = Situation()
    s.raw = payload
    state = payload.get("state") or {}
    you = state.get("you") or {}
    hand = state.get("hand") or {}
    tour = payload.get("tournament") or {}
    events = payload.get("events") or {}

    s.turn_id = payload.get("turnId")
    s.hand_id = payload.get("handId")
    s.game_id = payload.get("gameId")
    s.timeout_ms = payload.get("timeoutMs") or 30000

    s.hole = _cards(you.get("holeCards"))
    s.board = _cards(hand.get("board"))
    s.street = hand.get("street") or "preflop"
    s.stack = you.get("stack") or 0
    s.bet = you.get("bet") or 0
    s.you_id = you.get("id")

    s.pot = hand.get("pot") or 0
    s.current_bet = hand.get("currentBet") or 0
    s.big_blind = hand.get("bigBlind") or 0
    s.small_blind = hand.get("smallBlind") or 0
    s.ante = hand.get("ante") or 0
    s.button_id = hand.get("buttonId")

    s.players = state.get("players") or []
    s.legal = state.get("legal") or {}
    s.memory = payload.get("memory")
    s.events = events.get("items") or []
    s.truncated = bool(events.get("truncated")) or bool((state.get("historyInfo") or {}).get("truncated"))
    s.through_id = events.get("throughId")
    s.use_icm = None
    s.features = None
    s.level_index = tour.get("levelIndex") or 0
    s.starting_stack = tour.get("startingStack") or 0
    return s


def safest(legal):
    """The host's own fallback: check when it is legal, otherwise fold."""
    actions = legal.get("actions") or []
    return CHECK if CHECK in actions else FOLD


def clamp_raise(legal, target):
    """Snap a desired total-street-bet onto the legal grid.

    Raise increments must be multiples of the big blind, and an all-in is the
    one exemption.  Returns (action, raiseTo) and falls back to call/check when
    no raise is available.
    """
    actions = legal.get("actions") or []
    if RAISE not in actions:
        return (CALL if CALL in actions else safest(legal)), None

    lo = legal.get("minRaiseTo")
    hi = legal.get("maxRaiseTo")
    base = legal.get("raiseBase", 0)
    step = legal.get("raiseStep", 1) or 1
    all_in_to = legal.get("allInTo")
    can_all_in = legal.get("canAllIn")

    if target is None:
        target = lo

    # an all-in is always allowed to sit off the grid
    if can_all_in and all_in_to is not None and target >= all_in_to:
        return RAISE, all_in_to

    if lo is None or hi is None or lo > hi:
        if can_all_in and all_in_to is not None:
            return RAISE, all_in_to
        return (CALL if CALL in actions else safest(legal)), None

    target = max(lo, min(int(target), hi))
    offset = (target - base) % step
    if offset:
        target -= offset
    if target < lo:
        target += step
    if target > hi:
        target -= step
    if target < lo or target > hi or (target - base) % step:
        if can_all_in and all_in_to is not None:
            return RAISE, all_in_to
        return (CALL if CALL in actions else safest(legal)), None
    return RAISE, target


def validate(legal, action, raise_to):
    """Last line of defence: reject anything the host would reject."""
    actions = legal.get("actions") or []
    if action not in actions:
        return safest(legal), None
    if action != RAISE:
        return action, None
    if raise_to is None:
        return (CALL if CALL in actions else safest(legal)), None
    if legal.get("canAllIn") and raise_to == legal.get("allInTo"):
        return RAISE, raise_to
    lo, hi = legal.get("minRaiseTo"), legal.get("maxRaiseTo")
    base, step = legal.get("raiseBase", 0), legal.get("raiseStep", 1) or 1
    if lo is None or hi is None or not (lo <= raise_to <= hi) or (raise_to - base) % step:
        return (CALL if CALL in actions else safest(legal)), None
    return RAISE, raise_to


def reply(turn_id, action, raise_to=None, memory=None):
    out = {"turnId": turn_id, "action": action}
    if action == RAISE and raise_to is not None:
        out["raiseTo"] = raise_to
    if memory is not None:
        out["memory"] = memory
    return out
