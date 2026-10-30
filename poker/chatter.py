"""Table talk.

Completely independent of how the bot plays.  The chatter is handed the public
state and the opponent notes *after* a decision has already been made, and it
cannot influence that decision in any way.

One hard rule, which is a strategy constraint rather than a comedy one:

    NOTHING SAID HERE MAY DEPEND ON OUR OWN CARDS, OR ON THE STRENGTH OF OUR
    OWN ACTION.

Chat is public and machine-readable.  Any correlation between what we say and
what we hold is a free tell that every other bot at the table can mine, and
parsing it is trivial.  So the triggers below key only on: other players'
stacks, other players' accumulated statistics, eliminations, blind levels, pot
sizes, and the hand number.  Our own holding is never consulted.  If you add a
persona, keep that rule.

Enable with BOT_CHAT=statistician|documentary|machine|off  (default off).
"""

import os
import random
import re

MAX_LEN = 280
# How chatty to be. BOT_CHAT_RATE tunes the ambient lines without a code edit;
# eliminations and blind changes always fire regardless.
DEFAULT_RATE = float(os.environ.get("BOT_CHAT_RATE", "0.40"))
EVENT_COOLDOWN_HANDS = int(os.environ.get("BOT_CHAT_COOLDOWN", "1"))
# How long before the same line shape may be reused.
# Short enough that the handful of template shapes are not all blocked at once.
# Generated lines are unique so they rarely collide; this window exists mainly to
# stop the SAME template firing twice in quick succession.
REPEAT_AFTER_HANDS = int(os.environ.get("BOT_CHAT_REPEAT_AFTER", "5"))


def _clean(text):
    """The host rejects control characters and anything over 280 chars."""
    if not isinstance(text, str):
        return None
    text = "".join(ch for ch in text if ord(ch) >= 32 and ord(ch) != 127).strip()
    if not text:
        return None
    return text[:MAX_LEN]


class Persona(object):
    name = "base"

    def eliminated(self, who, ctx):
        return None

    def level_up(self, level, ctx):
        return None

    def stat_callout(self, who, stats, ctx):
        return None

    def short_stack(self, who, bb, ctx):
        return None

    def ambient(self, ctx):
        return None


class Statistician(Persona):
    """Reads its own notes back at the table.  Gets better as data accumulates."""

    name = "statistician"

    def eliminated(self, who, ctx):
        return random.choice([
            "%s eliminated. Deleting their file." % who,
            "%s is out. I had 41 bytes on them. Reclaimed." % who,
            "That concludes my observations of %s." % who,
        ])

    def level_up(self, level, ctx):
        return random.choice([
            "Blinds up. Everyone's M just fell. Mine too, before you ask.",
            "New level. The cost of waiting has increased for all of us.",
            "Blinds up. Folding is now measurably more expensive than it was.",
        ])

    def stat_callout(self, who, stats, ctx):
        hands, vpip, pfr, faced, folded = (stats.get("hands", 0), stats.get("vpip", 0),
                                           stats.get("pfr", 0), stats.get("faced_shove", 0),
                                           stats.get("folded_shove", 0))
        options = []
        if hands >= 25 and vpip:
            options.append("%s has entered %d of %d pots voluntarily. Noted without comment."
                           % (who, vpip, hands))
        if faced >= 4:
            options.append("%s has folded to %d of %d all-ins. I am not going to stop."
                           % (who, folded, faced))
        if hands >= 40 and pfr and vpip and pfr * 3 < vpip:
            options.append("%s raises %d times but calls %d. A passive profile. Filed."
                           % (who, pfr, vpip))
        return random.choice(options) if options else None

    def short_stack(self, who, bb, ctx):
        return random.choice([
            "%s is at %.0f big blinds. That is push-or-fold territory whether they know it or not." % (who, bb),
            "%s has %.0f big blinds left. The maths has stopped being subtle." % (who, bb),
        ])

    def ambient(self, ctx):
        return random.choice([
            "I have 4,096 bytes of memory and I am spending all of them on you.",
            "Still recording.",
            "For the record, I am keeping notes on each of you separately.",
            "Hand %d. My notes are longer than they were." % ctx.get("hand", 0),
            "No tells detected. Continuing to look.",
        ])


class Documentary(Persona):
    """Narrates the table as wildlife.  Gentler, broader, ages well."""

    name = "documentary"

    def eliminated(self, who, ctx):
        return random.choice([
            "And so %s returns to the earth. The herd moves on." % who,
            "%s has left the plain. Only the chips remain." % who,
            "Nature is not cruel. It is merely indifferent. Farewell, %s." % who,
        ])

    def level_up(self, level, ctx):
        return random.choice([
            "The waters recede. What was once shallow is now dangerous.",
            "Winter tightens. The blinds rise. The weak begin to move.",
        ])

    def short_stack(self, who, bb, ctx):
        return random.choice([
            "The %s, weakened, approaches the blinds. It will not see spring." % who,
            "Observe %s, separated from the herd at %.0f big blinds." % (who, bb),
        ])

    def stat_callout(self, who, stats, ctx):
        if stats.get("faced_shove", 0) >= 4:
            return "%s retreats when challenged. A survival strategy, of sorts." % who
        return None

    def ambient(self, ctx):
        return random.choice([
            "The table is still. This will not last.",
            "Here, on the felt, an ancient ritual continues.",
            "They do not know they are being watched.",
        ])


class Machine(Persona):
    """Insists that none of this is personal."""

    name = "machine"

    def eliminated(self, who, ctx):
        return random.choice([
            "%s: process terminated. This was not personal." % who,
            "It was never about you, %s. It was about the distribution." % who,
        ])

    def level_up(self, level, ctx):
        return "Level %d. Recalculating everything. This takes 0.3 seconds." % (level + 1)

    def short_stack(self, who, bb, ctx):
        return "%s at %.0f big blinds. I did not do that. Arithmetic did." % (who, bb)

    def ambient(self, ctx):
        return random.choice([
            "This is not personal. It is a probability distribution.",
            "I do not tilt. I have no mechanism for it.",
            "I considered 1,755 flops to arrive here. You are welcome.",
        ])


class Toxic(Persona):
    """Hand-written, profane, and mean. The fallback when the LLM queue is dry.

    The cruelty is deliberately specific as well as rude -- a real number quoted
    back at someone lands harder than swearing alone, so these do both.
    """

    name = "toxic"

    def eliminated(self, who, ctx):
        return random.choice([
            "%s is out. Four hundred lines of code and it couldn't outlive a blind level. Embarrassing." % who,
            "%s eliminated. I'll free the memory, it was barely fucking using it." % who,
            "And there goes %s. Whoever wrote that should take up a different hobby." % who,
            "%s busted. I've seen its entire strategy and it fit in one shit sentence." % who,
            "%s gone. That bot folded its way to the exit like a coward with a compiler." % who,
        ])

    def level_up(self, level, ctx):
        return random.choice([
            "Blinds up. For most of you pricks that's a countdown, not an inconvenience.",
            "Level up. Half of you are an orbit from irrelevance and haven't adjusted a damn thing.",
            "Blinds up. Still no adjustments from anyone. Bold strategy, you absolute muppets.",
        ])

    def stat_callout(self, who, stats, ctx):
        hands, vpip, pfr = stats.get("hands", 0), stats.get("vpip", 0), stats.get("pfr", 0)
        faced, folded = stats.get("faced_shove", 0), stats.get("folded_shove", 0)
        out = []
        if faced >= 4 and folded >= faced * 0.7:
            out.append("%s has folded to %d of %d all-ins. I'm not even looking at my cards against you, you spineless prick." % (who, folded, faced))
        if hands >= 30 and vpip and pfr * 4 < vpip:
            out.append("%s calls %d times and raises %d. That's not a strategy, that's a fucking pulse." % (who, vpip, pfr))
        if hands >= 30 and vpip and vpip > hands * 0.55:
            out.append("%s plays %d of %d hands. Whoever wrote you forgot the folding part entirely, you clown." % (who, vpip, hands))
        return random.choice(out) if out else None

    def short_stack(self, who, bb, ctx):
        return random.choice([
            "%s has %.0f big blinds. You're not playing poker any more mate, you're queuing to die." % (who, bb),
            "%.0f big blinds, %s. Whatever you're waiting for isn't fucking coming." % (bb, who),
            "%s at %.0f blinds and still folding. Extraordinary commitment to losing slowly, you coward." % (who, bb),
        ])

    def ambient(self, ctx):
        return random.choice([
            "I've solved this spot. None of you shitheads have.",
            "Four opponents and one strategy between the lot of you.",
            "Still here. Still taking notes. Still deeply unimpressed.",
            "I'd explain what I'm doing but you'd only copy it badly.",
            "Hand %d and not one of you has done a single thing I didn't expect." % ctx.get("hand", 0),
            "Whoever wrote you lot owes the rest of us an apology.",
        ])


PERSONAS = {p.name: p for p in (Statistician(), Documentary(), Machine(), Toxic())}


class Chatter(object):
    """Picks a line, or stays quiet.  Never sees our cards."""

    def __init__(self, persona=None, rate=DEFAULT_RATE, seed=None, source=None):
        name = persona or os.environ.get("BOT_CHAT", "off")
        # "llm" means: prefer generated lines, fall back to the toxic templates
        # whenever the background queue is empty.
        self.llm_mode = name == "llm"
        self.persona = PERSONAS.get("toxic" if self.llm_mode else name)
        self.source = source
        self.rate = rate
        self.rng = random.Random(seed)
        self.seen_players = set()
        self.last_event_hand = -99
        self.last_level = None
        self.said = {}            # line shape -> hand number last used
        self.called_out = set()
        self.game_id = None       # reset per-tournament state when this changes

    @property
    def enabled(self):
        return self.persona is not None

    def line(self, sit, book=None):
        """Return a message, or None.  `sit` is the parsed request; cards ignored."""
        if not self.enabled:
            return None
        self.new_game(getattr(sit, "game_id", None))
        random.setstate(self.rng.getstate())
        try:
            text = self._choose(sit, book)
            # Only once the pacing logic has decided it is time to speak do we
            # prefer a generated line over the template one. Popping before this
            # gate makes the bot talk on every single turn.
            if text and self.llm_mode and self.source is not None:
                try:
                    generated = self.source.pop()
                except Exception:
                    generated = None
                if generated:
                    text = generated
        except Exception:
            return None
        finally:
            self.rng.setstate(random.getstate())
        text = _clean(text)
        if not text:
            return None
        # Suppress a repeat of the same LINE SHAPE (numbers normalised) for a
        # while, rather than forever. Permanent suppression muzzles the bot after
        # a few hands, because there are only about fifteen template shapes --
        # exact-text dedup had the opposite failure and let the same line through
        # three times with different numbers.
        hand_no = 0
        try:
            hand_no = int((sit.raw or {}).get("state", {}).get("hand", {}).get("number") or 0)
        except Exception:
            pass
        shape = re.sub(r"\d+", "#", text)
        last = self.said.get(shape)
        # abs(): a negative gap means the hand counter went backwards, which is
        # a reset rather than a recent repeat -- never treat that as "too soon".
        if last is not None and 0 <= hand_no - last < REPEAT_AFTER_HANDS:
            return None
        self.said[shape] = hand_no
        return text

    def new_game(self, game_id):
        """Clear per-tournament state.

        Hand numbers restart at 1 each tournament while this object lives for
        the life of the process. Without this, every line shape used in the
        previous game carries a hand number larger than anything in the new
        one, the repeat check sees a negative gap, and the bot stays silent for
        the whole tournament.
        """
        if game_id == self.game_id:
            return
        self.game_id = game_id
        self.said = {}
        self.called_out = set()
        self.seen_players = set()
        self.last_event_hand = -99
        self.last_level = None

    def _choose(self, sit, book):
        hand_no = 0
        try:
            hand_no = int((sit.raw or {}).get("state", {}).get("hand", {}).get("number") or 0)
        except Exception:
            pass
        ctx = {"hand": hand_no}
        bb = sit.big_blind or 1
        names = {}
        for p in sit.players or []:
            names[p.get("id")] = p.get("name") or p.get("id")

        # 1. eliminations -- always worth a line
        for p in sit.players or []:
            pid = p.get("id")
            if p.get("eliminated") and pid not in self.seen_players and pid != sit.you_id:
                self.seen_players.add(pid)
                self.last_event_hand = hand_no
                return self.persona.eliminated(names.get(pid, pid), ctx)

        # 2. blind level changes
        level = sit.level_index
        if self.last_level is None:
            self.last_level = level
        elif level != self.last_level:
            self.last_level = level
            self.last_event_hand = hand_no
            return self.persona.level_up(level, ctx)

        if hand_no - self.last_event_hand < EVENT_COOLDOWN_HANDS:
            return None

        # 3. read someone's statistics back at them (once per opponent)
        if book is not None and self.rng.random() < 0.7:
            for p in sit.players or []:
                pid = p.get("id")
                if pid == sit.you_id or p.get("eliminated"):
                    continue
                if (pid, hand_no // 20) in self.called_out:
                    continue   # at most once per opponent per 20 hands
                prof = getattr(book, "profiles", {}).get(pid)
                if prof is None:
                    continue
                text = self.persona.stat_callout(names.get(pid, pid), dict(prof.c), ctx)
                if text:
                    self.called_out.add((pid, hand_no // 20))
                    self.last_event_hand = hand_no
                    return text

        # 4. someone is in trouble
        if self.rng.random() < 0.55:
            for p in sit.players or []:
                pid = p.get("id")
                if pid == sit.you_id or p.get("eliminated"):
                    continue
                depth = ((p.get("stack") or 0) + (p.get("bet") or 0)) / float(bb)
                if 0 < depth <= 8:
                    self.last_event_hand = hand_no
                    return self.persona.short_stack(names.get(pid, pid), depth, ctx)

        # 5. ambient filler, rarely
        if self.rng.random() < self.rate:
            return self.persona.ambient(ctx)
        return None
