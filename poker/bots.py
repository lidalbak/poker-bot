"""Opponent archetypes.

These are the ladder the real bot is measured against, and the population the
opponent head learns to classify.  Each is deliberately simple and stationary:
that is what makes them useful as a benchmark, and it mirrors what a hastily
built league bot actually looks like.
"""

import random

from poker.cards import hand_class, rank_of
from poker.evaluator import evaluate


class Bot(object):
    name = "bot"

    def act(self, hand, seat, legal):
        """Return (action, raise_to_or_None)."""
        raise NotImplementedError


class FoldBot(Bot):
    """The safety floor: never crashes, never wins."""
    name = "folder"

    def act(self, hand, seat, legal):
        return ("check" if "check" in legal.actions else "fold"), None


class CallingStation(Bot):
    name = "station"

    def act(self, hand, seat, legal):
        if "call" in legal.actions:
            return "call", None
        return "check", None


class Maniac(Bot):
    name = "maniac"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, hand, seat, legal):
        if "raise" in legal.actions and self.rng.random() < 0.55:
            target = min(legal.min_raise_to * 2, legal.max_raise_to)
            target -= (target - legal.raise_base) % legal.raise_step
            if legal.min_raise_to <= target <= legal.max_raise_to:
                return "raise", target
            if legal.can_all_in:
                return "raise", legal.all_in_to
        if "call" in legal.actions:
            return "call", None
        return "check", None


class Nit(Bot):
    """Plays only premium holdings; folds everything else."""
    name = "nit"
    PREMIUM = set("AA KK QQ JJ TT AKs AKo AQs".split())

    def act(self, hand, seat, legal):
        strong = hand_class(*seat.hole) in self.PREMIUM
        if hand.board:
            score = evaluate(list(seat.hole) + hand.board)
            strong = score // (16 ** 5) >= 3  # trips or better
        if strong and "raise" in legal.actions:
            target = legal.min_raise_to
            if target <= legal.max_raise_to:
                return "raise", target
        if strong and "call" in legal.actions:
            return "call", None
        if "check" in legal.actions:
            return "check", None
        return "fold", None


class RandomLegal(Bot):
    """Stress-tests the engine rather than playing poker."""
    name = "random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, hand, seat, legal):
        choice = self.rng.choice(legal.actions)
        if choice == "raise":
            if legal.min_raise_to <= legal.max_raise_to and self.rng.random() < 0.8:
                span = (legal.max_raise_to - legal.min_raise_to) // max(legal.raise_step, 1)
                target = legal.min_raise_to + self.rng.randint(0, max(span, 0)) * legal.raise_step
                return "raise", target
            if legal.can_all_in:
                return "raise", legal.all_in_to
            choice = "call" if "call" in legal.actions else "check"
        return choice, None


class OverFolder(Bot):
    """Plays reasonably, but folds to all-in pressure far too often.

    The single most common leak in a hastily built tournament bot, and the one
    the opponent head is built to punish.
    """
    name = "overfolder"
    CALL_SHOVE = set("AA KK QQ JJ AKs AKo".split())
    OPEN = set("AA KK QQ JJ TT 99 88 77 AKs AKo AQs AQo AJs ATs KQs".split())

    def act(self, hand, seat, legal):
        cls = hand_class(*seat.hole)
        facing = legal.call_amount
        committing = facing >= seat.stack * 0.5
        if committing:
            if cls in self.CALL_SHOVE and "call" in legal.actions:
                return "call", None
            return ("check" if "check" in legal.actions else "fold"), None
        if not hand.board:
            if cls in self.OPEN and "raise" in legal.actions:
                return "raise", min(legal.min_raise_to, legal.max_raise_to)
            if facing == 0 and "check" in legal.actions:
                return "check", None
            if cls in self.OPEN and "call" in legal.actions:
                return "call", None
            return ("check" if "check" in legal.actions else "fold"), None
        score = evaluate(list(seat.hole) + hand.board) // (16 ** 5)
        if score >= 2 and "call" in legal.actions:
            return "call", None
        if facing == 0 and "check" in legal.actions:
            return "check", None
        return ("check" if "check" in legal.actions else "fold"), None


ARCHETYPES = [FoldBot, CallingStation, Maniac, Nit, RandomLegal, OverFolder]
