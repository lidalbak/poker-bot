"""Opponent modelling: pooled counters, shrinkage, and a bounded dial.

Three design decisions, all forced by the format:

1.  A tournament is ~100 hands, so per-situation statistics never confirm.
    Observations are POOLED into a handful of buckets instead.
2.  The host clears its 4KB memory between tournaments, so the real profile
    lives in our own store on disk and accumulates across the whole league.
    Each tournament starts from that prior rather than from nothing.
3.  Reads move a DIAL, never a switch.  The response is blended with the
    equilibrium and capped, because a best response to a wrong model loses
    more than a slightly-too-tight one ever will.
"""

import json
import os

STORE_PATH = os.environ.get("BOT_PROFILES", "data/opponents.json")

# population priors, used until an opponent has earned their own numbers
PRIOR = {"vpip": 0.42, "pfr": 0.24, "fold_to_shove": 0.62, "aggression": 0.45}
PRIOR_WEIGHT = 20.0          # observations of "someone average" we start from
MIN_CONFIRM = 10             # never act on fewer real observations than this
MAX_DIAL = 0.5               # hard cap on how far we deviate from equilibrium

FIELDS = ("hands", "vpip", "pfr", "faced_shove", "folded_shove", "aggr", "passive")


class Profile(object):
    """Counters for one opponent, poolable and mergeable across tournaments."""

    def __init__(self, data=None):
        self.c = dict((f, 0) for f in FIELDS)
        if data:
            if isinstance(data, list):
                for i, f in enumerate(FIELDS):
                    if i < len(data):
                        self.c[f] = int(data[i])
            else:
                for f in FIELDS:
                    self.c[f] = int(data.get(f, 0))

    def to_list(self):
        return [self.c[f] for f in FIELDS]

    def merge(self, other):
        for f in FIELDS:
            self.c[f] += other.c[f]
        return self

    def rate(self, made, of, prior_key):
        """Beta-Binomial posterior mean, shrunk toward the population prior."""
        a = PRIOR[prior_key] * PRIOR_WEIGHT + self.c[made]
        b = (1.0 - PRIOR[prior_key]) * PRIOR_WEIGHT + max(self.c[of] - self.c[made], 0)
        return a / (a + b)

    @property
    def fold_to_shove(self):
        return self.rate("folded_shove", "faced_shove", "fold_to_shove")

    @property
    def confirmed_shove_obs(self):
        return self.c["faced_shove"]

    def is_deterministic(self, made, of):
        """Pure behaviour in a pool: a bot that always does the same thing.

        Against a stochastic opponent a frequency needs ~1000 observations to
        pin down.  Against a deterministic one, a handful settles it -- and
        every opponent here is a bot.
        """
        n, k = self.c[of], self.c[made]
        return n >= 6 and (k == 0 or k == n)

    @property
    def raise_width(self):
        """Preflop raising rate relative to the population, shrunk.

        Used as the observed TENDENCY behind a shove: someone who raises twice
        as often as the field is assumed to shove about twice as wide.  That
        is a proxy -- shoves are not counted separately -- which is one more
        reason the read is only ever a small modifier.
        """
        return self.rate("pfr", "hands", "pfr") / PRIOR["pfr"]

    def range_dial(self):
        """Weight for bending a modelled RANGE toward the read, 0..MAX_DIAL/2.

        Half the cap of the fold-to-shove dial.  The stack-conditioned range
        is sound from first principles; this read has never been shown to
        improve results and its live data path is unverified, so it may nudge
        the range but never replace it.
        """
        n = self.c["hands"]
        if n < 3 * MIN_CONFIRM:
            return 0.0
        deviation = abs(self.raise_width - 1.0)
        if deviation < 0.25:
            return 0.0
        evidence = min(n / 120.0, 1.0)
        return min(MAX_DIAL / 2.0, evidence * min(deviation, 1.0) * MAX_DIAL / 2.0)

    def dial(self):
        """How far we may deviate from equilibrium against this opponent, 0..MAX_DIAL."""
        n = self.confirmed_shove_obs
        if n < MIN_CONFIRM and not self.is_deterministic("folded_shove", "faced_shove"):
            return 0.0
        deviation = abs(self.fold_to_shove - PRIOR["fold_to_shove"])
        if deviation < 0.12:
            return 0.0
        evidence = min(n / 40.0, 1.0)
        if self.is_deterministic("folded_shove", "faced_shove"):
            evidence = max(evidence, 0.6)
        return min(MAX_DIAL, evidence * min(deviation / 0.3, 1.0) * MAX_DIAL)


class Book(object):
    """Every opponent profile, with the durable store behind it."""

    def __init__(self, path=STORE_PATH):
        self.path = path
        self.profiles = {}
        self.seen_events = set()
        self.stacks = {}      # last seen chips behind, per player
        self.me = None        # our own seat, never profiled

    # -- persistence -------------------------------------------------------

    def load(self):
        try:
            with open(self.path) as fh:
                raw = json.load(fh)
            for name, data in (raw.get("players") or {}).items():
                self.profiles[name] = Profile(data)
        except Exception:
            pass
        return self

    def save(self):
        try:
            directory = os.path.dirname(self.path)
            if directory and not os.path.isdir(directory):
                os.makedirs(directory)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump({"players": dict((k, v.to_list())
                                           for k, v in self.profiles.items())}, fh)
            os.replace(tmp, self.path)
        except Exception:
            pass

    def get(self, name):
        if name not in self.profiles:
            self.profiles[name] = Profile()
        return self.profiles[name]

    def note_snapshot(self, players, me=None):
        """Record chips behind for each seat, from the snapshot in each request.

        Events carry no stack information and no all-in flag, so a shove can
        only be recognised by comparing a raise against what that player had
        left -- which we know solely from these snapshots.
        """
        if me:
            self.me = me
            self.profiles.pop(me, None)
        for p in players or []:
            pid = p.get("id")
            if pid:
                self.stacks[pid] = (p.get("stack") or 0) + (p.get("bet") or 0)

    # -- the host's 4KB memory field --------------------------------------

    def to_memory(self, limit=4096):
        """Compact enough to survive the byte cap with room to spare."""
        payload = {"v": 1, "p": dict((k, v.to_list()) for k, v in self.profiles.items())}
        blob = json.dumps(payload, separators=(",", ":"))
        while len(blob.encode("utf-8")) > limit and payload["p"]:
            payload["p"].pop(sorted(payload["p"])[-1])
            blob = json.dumps(payload, separators=(",", ":"))
        return payload

    def from_memory(self, memory):
        if not isinstance(memory, dict):
            return self
        for name, data in (memory.get("p") or {}).items():
            existing = self.profiles.get(name)
            incoming = Profile(data)
            # the host's memory is authoritative for THIS tournament; the disk
            # store already holds earlier ones, so take the larger count
            if existing is None or sum(incoming.to_list()) > sum(existing.to_list()):
                self.profiles[name] = incoming
        return self

    # -- ingestion ---------------------------------------------------------

    def ingest(self, events, truncated=False):
        """Fold a batch of public events into the counters.

        Events repeat whenever we decline to commit memory, so everything is
        deduplicated by id.  When the host flags truncation we have holes, and
        a hole must not be read as "they did nothing".
        """
        by_hand = {}
        for ev in events or []:
            eid = ev.get("id")
            if eid is not None:
                if eid in self.seen_events:
                    continue
                self.seen_events.add(eid)
            by_hand.setdefault(ev.get("handNumber"), []).append(ev)

        for _hand, items in by_hand.items():
            shove_by = None
            counted_hand = set()
            for ev in items:
                pid = ev.get("playerId")
                action = ev.get("action")
                street = ev.get("street") or "preflop"
                if not pid or not action or pid == self.me:
                    if pid == self.me and action == "raise":
                        # our own shove still starts a measurable spot
                        amount = ev.get("amount") or 0
                        behind = self.stacks.get(pid) or 0
                        shove_by = pid if behind and amount >= behind * 0.85 else shove_by
                    continue
                prof = self.get(pid)
                if pid not in counted_hand and action in ("fold", "call", "raise", "check"):
                    prof.c["hands"] += 1
                    counted_hand.add(pid)
                if action in ("call", "raise") and street == "preflop":
                    prof.c["vpip"] += 1
                if action == "raise" and street == "preflop":
                    prof.c["pfr"] += 1
                if action == "raise":
                    prof.c["aggr"] += 1
                elif action == "call":
                    prof.c["passive"] += 1

                # Anyone acting after a shove is facing one.  There is no flag
                # for this in the event stream, so it is inferred from the
                # raise total against the last known chips behind.
                if shove_by is not None and shove_by != pid and action in ("fold", "call"):
                    prof.c["faced_shove"] += 1
                    if action == "fold":
                        prof.c["folded_shove"] += 1

                if action == "raise":
                    amount = ev.get("amount") or 0
                    behind = self.stacks.get(pid) or 0
                    if behind and amount >= behind * 0.85:
                        shove_by = pid
        return self

    def observe_shove_response(self, player_id, folded):
        """Direct observation, used by the harness and by our own hand tracking."""
        prof = self.get(player_id)
        prof.c["faced_shove"] += 1
        if folded:
            prof.c["folded_shove"] += 1
