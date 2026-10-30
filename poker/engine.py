"""Five-handed freezeout engine mirroring the poker-night host rules.

Deliberately a faithful copy of the league's rules rather than generic poker,
because the tuning harness is only useful if it reproduces the real game:

  * five seats, freezeout, no rebuys
  * per-player antes, escalating blind levels
  * raiseTo is the TOTAL street commitment, excluding ante
  * raise increments must be multiples of the current big blind
  * all-ins are exempt from the multiple rule and a short all-in does not
    reopen betting unless it reaches the full-raise threshold
  * timeout / error / illegal action becomes check-if-legal-else-fold
"""

import random
from collections import namedtuple

from poker.evaluator import evaluate

PREFLOP, FLOP, TURN, RIVER = "preflop", "flop", "turn", "river"
STREETS = (PREFLOP, FLOP, TURN, RIVER)

DEFAULT_LEVELS = [
    (25, 50, 0), (50, 100, 10), (75, 150, 15), (100, 200, 25),
    (150, 300, 40), (200, 400, 50), (300, 600, 75), (400, 800, 100),
    (600, 1200, 150), (1000, 2000, 250), (1500, 3000, 400),
    (2000, 4000, 500), (4000, 8000, 1000),
]

Legal = namedtuple("Legal", "actions call_amount min_raise_to max_raise_to raise_base raise_step all_in_to can_all_in")


class Seat(object):
    __slots__ = ("id", "bot", "stack", "bet", "contribution", "folded", "all_in", "hole")

    def __init__(self, sid, bot, stack):
        self.id = sid
        self.bot = bot
        self.stack = stack
        self.bet = 0
        self.contribution = 0
        self.folded = False
        self.all_in = False
        self.hole = ()

    def reset_hand(self):
        self.bet = 0
        self.contribution = 0
        self.folded = False
        self.all_in = False
        self.hole = ()

    @property
    def eliminated(self):
        return self.stack <= 0 and self.contribution == 0


class Hand(object):
    """One dealt hand.  Produced by Table.play_hand; carries the log used for stats."""

    def __init__(self, seats, button, sb, bb, ante, rng):
        self.seats = seats
        self.button = button
        self.sb, self.bb, self.ante = sb, bb, ante
        self.rng = rng
        self.board = []
        self.street = PREFLOP
        self.pot_dead = 0            # antes and folded chips already in the middle
        self.current_bet = 0
        self.min_raise_inc = bb
        self.events = []
        self.result = {}

    # -- helpers ------------------------------------------------------------

    def live(self):
        return [s for s in self.seats if not s.folded and s.stack + s.contribution > 0]

    def contenders(self):
        return [s for s in self.seats if not s.folded]

    def can_act(self):
        return [s for s in self.contenders() if not s.all_in and s.stack > 0]

    def total_pot(self):
        return sum(s.contribution for s in self.seats)

    def log(self, action, seat=None, amount=None):
        self.events.append({
            "id": len(self.events),
            "street": self.street,
            "action": action,
            "playerId": seat.id if seat else None,
            "amount": amount,
        })

    # -- betting ------------------------------------------------------------

    def legal_for(self, seat):
        to_call = min(self.current_bet - seat.bet, seat.stack)
        actions = ["fold"]
        if self.current_bet <= seat.bet:
            actions.append("check")
        else:
            actions.append("call")

        all_in_to = seat.bet + seat.stack
        max_raise_to = all_in_to
        # raiseTo must sit on the big-blind grid above the current bet
        base = self.current_bet
        step = self.bb
        floor = self.current_bet + self.min_raise_inc
        k = -(-(floor - base) // step) if floor > base else 1
        min_raise_to = base + max(k, 1) * step
        can_all_in = all_in_to > self.current_bet and seat.stack > 0
        if min_raise_to <= max_raise_to or can_all_in:
            actions.append("raise")
        return Legal(actions, to_call, min_raise_to, max_raise_to, base, step, all_in_to, can_all_in)

    def _commit(self, seat, amount):
        amount = min(amount, seat.stack)
        seat.stack -= amount
        seat.bet += amount
        seat.contribution += amount
        if seat.stack == 0:
            seat.all_in = True
        return amount

    def apply(self, seat, action, raise_to=None):
        """Apply an action, normalising anything illegal to check-else-fold."""
        legal = self.legal_for(seat)

        if action == "raise":
            if raise_to is None:
                action = "call" if "call" in legal.actions else "check"
            else:
                on_grid = (raise_to - legal.raise_base) % legal.raise_step == 0
                full = legal.min_raise_to <= raise_to <= legal.max_raise_to and on_grid
                shove = legal.can_all_in and raise_to == legal.all_in_to
                if not (full or shove):
                    action = "call" if "call" in legal.actions else "check"

        if action not in legal.actions:
            action = "check" if "check" in legal.actions else "fold"

        if action == "fold":
            seat.folded = True
            self.log("fold", seat)
        elif action == "check":
            self.log("check", seat)
        elif action == "call":
            paid = self._commit(seat, self.current_bet - seat.bet)
            self.log("call", seat, paid)
        else:
            increment = raise_to - self.current_bet
            self._commit(seat, raise_to - seat.bet)
            reopens = increment >= self.min_raise_inc
            if reopens:
                self.min_raise_inc = increment
            self.current_bet = max(self.current_bet, seat.bet)
            self.log("raise", seat, seat.bet)
            return reopens
        return False

    def betting_round(self, first_idx, decide):
        actors = self.can_act()
        if len(actors) <= 1 and self.current_bet == 0:
            return
        n = len(self.seats)
        acted = set()
        may_raise = set(s.id for s in self.seats)
        idx = first_idx
        guard = 0
        while True:
            guard += 1
            if guard > 400:
                break
            seat = self.seats[idx % n]
            idx += 1
            if seat.folded or seat.all_in or seat.stack == 0:
                if self._round_closed(acted):
                    break
                continue
            if seat.bet == self.current_bet and seat.id in acted:
                if self._round_closed(acted):
                    break
                continue

            action, amount = decide(self, seat, seat.id in may_raise)
            reopened = self.apply(seat, action, amount)
            acted.add(seat.id)
            if reopened:
                acted = {seat.id}
                may_raise = set(s.id for s in self.seats)
            if len(self.contenders()) == 1:
                break
            if self._round_closed(acted):
                break

    def _round_closed(self, acted):
        for s in self.seats:
            if s.folded or s.all_in or s.stack == 0:
                continue
            if s.id not in acted or s.bet < self.current_bet:
                return False
        return True

    def end_street(self):
        for s in self.seats:
            s.bet = 0
        self.current_bet = 0
        self.min_raise_inc = self.bb

    # -- showdown -----------------------------------------------------------

    def settle(self):
        """Split the pot into main/side pots by contribution level and award each."""
        payouts = dict((s.id, 0) for s in self.seats)
        contributions = dict((s.id, s.contribution) for s in self.seats)
        remaining = [s for s in self.seats if s.contribution > 0]
        levels = sorted(set(c for c in contributions.values() if c > 0))

        prev = 0
        for level in levels:
            pot = 0
            for s in remaining:
                take = min(max(contributions[s.id] - prev, 0), level - prev)
                pot += take
            eligible = [s for s in self.seats
                        if not s.folded and contributions[s.id] >= level]
            if pot <= 0:
                prev = level
                continue
            if not eligible:
                eligible = [s for s in self.seats if contributions[s.id] >= level]
            best = None
            winners = []
            for s in eligible:
                score = evaluate(list(s.hole) + self.board)
                if best is None or score > best:
                    best, winners = score, [s]
                elif score == best:
                    winners.append(s)
            share, extra = divmod(pot, len(winners))
            # odd chips go to the first winner left of the button
            order = sorted(winners, key=lambda s: (self.seats.index(s) - self.button - 1) % len(self.seats))
            for i, w in enumerate(order):
                payouts[w.id] += share + (1 if i < extra else 0)
            prev = level

        for s in self.seats:
            s.stack += payouts[s.id]
        self.result = payouts
        return payouts


class Table(object):
    def __init__(self, bots, starting_stack=5000, levels=None, seed=None):
        assert len(bots) == 5, "the league is exactly five seats"
        self.levels = levels or DEFAULT_LEVELS
        self.starting_stack = starting_stack
        self.seats = [Seat("seat-%d" % (i + 1), b, starting_stack) for i, b in enumerate(bots)]
        self.button = 0
        self.rng = random.Random(seed)
        self.hand_number = 0

    def survivors(self):
        return [s for s in self.seats if s.stack > 0]

    def play_hand(self, level_index=0, deck=None):
        sb, bb, ante = self.levels[min(level_index, len(self.levels) - 1)]
        alive = self.survivors()
        if len(alive) < 2:
            return None

        for s in self.seats:
            s.reset_hand()

        hand = Hand(self.seats, self.button, sb, bb, ante, self.rng)
        self.hand_number += 1

        if deck is None:
            deck = list(range(52))
            self.rng.shuffle(deck)
        else:
            deck = list(deck)
        pos = 0
        for s in alive:
            s.hole = (deck[pos], deck[pos + 1])
            pos += 2
        board_cards = deck[pos:pos + 5]

        # antes then blinds
        for s in alive:
            if ante:
                hand._commit(s, min(ante, s.stack))
        for s in self.seats:
            s.bet = 0  # antes are not part of the street bet

        order = [s for s in self._order_from(self.button) if s in alive]
        heads_up = len(alive) == 2
        if heads_up:
            sb_seat, bb_seat = order[0], order[1]
        else:
            sb_seat, bb_seat = order[1], order[2]
        hand._commit(sb_seat, min(sb, sb_seat.stack))
        hand._commit(bb_seat, min(bb, bb_seat.stack))
        hand.current_bet = max(sb_seat.bet, bb_seat.bet)
        hand.min_raise_inc = bb

        first = self.seats.index(bb_seat) + 1 if not heads_up else self.seats.index(sb_seat)
        hand.betting_round(first, self._decide)

        postflop_first = self.seats.index(sb_seat) if not heads_up else self.seats.index(bb_seat)
        for street, count in ((FLOP, 3), (TURN, 1), (RIVER, 1)):
            if len(hand.contenders()) <= 1:
                break
            hand.end_street()
            hand.street = street
            hand.board.extend(board_cards[:count])
            board_cards = board_cards[count:]
            if len(hand.can_act()) > 1:
                hand.betting_round(postflop_first, self._decide)

        while len(hand.board) < 5 and len(hand.contenders()) > 1:
            hand.board.append(board_cards.pop(0))

        hand.settle()
        self._advance_button()
        return hand

    def _order_from(self, button_idx):
        n = len(self.seats)
        return [self.seats[(button_idx + i) % n] for i in range(n)]

    def _advance_button(self):
        n = len(self.seats)
        for step in range(1, n + 1):
            nxt = (self.button + step) % n
            if self.seats[nxt].stack > 0:
                self.button = nxt
                return

    @staticmethod
    def _decide(hand, seat, may_raise):
        legal = hand.legal_for(seat)
        try:
            action, amount = seat.bot.act(hand, seat, legal)
        except Exception:
            action, amount = ("check" if "check" in legal.actions else "fold"), None
        if action == "raise" and not may_raise:
            action = "call" if "call" in legal.actions else "check"
        return action, amount
