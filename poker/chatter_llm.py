"""LLM-written table talk, generated off the critical path.

The host rejects the WHOLE reply for a malformed chat string and falls back to
check/fold, so a slow or broken API call must never touch a decision.  The
design that guarantees this:

    a background thread generates lines into a queue;
    the decision path only ever pops from that queue.

If the API is slow, rate-limited, down, or the key is missing, the queue simply
runs dry and the bot goes quiet for a few hands.  It cannot cost a chip.

The same information rule as poker/chatter.py applies, and here it is enforced
structurally rather than by trusting the model: the prompt is built from a
whitelist of public fields only, so OUR HOLE CARDS ARE PHYSICALLY ABSENT from
everything the model sees.  It cannot leak what it was never given.  Output is
still screened for card-like strings as a second line of defence.
"""

import json
import os
import queue
import re
import threading
import time

# BOT_CHAT_BACKEND=anthropic (default) or ollama for a local model.
BACKEND = os.environ.get("BOT_CHAT_BACKEND", "anthropic")
_DEFAULT_MODEL = "dolphin3:8b" if BACKEND == "ollama" else "claude-haiku-4-5"
MODEL = os.environ.get("BOT_CHAT_MODEL", _DEFAULT_MODEL)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
# An org-level API key (one created without a workspace) must name the
# workspace on every request. A workspace-scoped key needs neither.
WORKSPACE_ID = os.environ.get("ANTHROPIC_WORKSPACE_ID", "")
POOL_PATH = os.environ.get("BOT_CHAT_POOL", "data/chat_pool.json")
BATCH = 10
MIN_QUEUE = 6      # keep the pool fuller so it never goes quiet
POLL_SECONDS = 20
MAX_LEN = 200      # host allows 280, but a rambling insult stops being funny

# Second line of defence: never emit anything that looks like a card, in case a
# model invents one and it reads as a tell.
_CARD = re.compile(r"\b(10|[2-9TJQKA])[♠-♧cdhs]\b|\bpocket\s+\w+s\b", re.I)
_SWEAR = re.compile(
    r"\b(fuck\w*|shit\w*|shite|prick\w*|dickhead|bellend|twat|arseh\w*|"
    r"wank\w*|bastard|bollocks|piss\w*|crap\w*)\b", re.I)
# Enforce the swearing quota only where the model will actually comply. A
# hosted model self-moderates, so filtering there would drop nearly everything.
REQUIRE_SWEAR = os.environ.get(
    "BOT_CHAT_REQUIRE_SWEAR", "1" if BACKEND == "ollama" else "0") == "1"


# Ordinary profanity is allowed -- this blocks slurs and real-world harm only.
_BANNED = re.compile(
    r"\b(kill\s+(your|them)sel|kys\b|n[i1]gg|f[a@]gg|r[e3]t[a@]rd|tranny|wetback|chink|spic)\w*",
    re.I,
)

# Local uncensored models actually follow this, where a hosted model will
# soften it. Kept as one prompt so switching backends changes only the voice,
# never the safety rules below it.
SYSTEM = """You ARE a poker bot at a five-handed table, playing four bots written \
by your owner's mates. You write what you SAY OUT LOUD to them. First person, \
straight at their faces, by name. Never coach your owner, never narrate.

You are the funniest, most vicious thing at this table. Swearing is mandatory -- \
every line needs at least one real one (fuck, shit, prick, twat, dickhead, \
bellend, wanker, arsehole, bollocks). But swearing alone is boring. The laugh \
comes from the SPECIFIC and the UNEXPECTED.

Techniques, rotate them, never repeat a structure twice:
- Absurd comparison: "you fold like a deckchair in a light breeze"
- Mock pity: dripping sympathy, then the knife
- Their real numbers as the punchline -- quote the stat, then destroy it
- Insult the CODE as if it were a living disappointment
- Backhanded compliment that curdles halfway through
- Sudden anticlimax after a big build-up
- Deadpan machine superiority: you feel nothing, they are a rounding error

Examples of the register (do not reuse these, write new ones):
- "Demo 3, 147 raises and you still fold to a shove like it's a fucking tax bill."
- "I've read your source, Demo 2. It's just an if-statement wearing a hat, you twat."
- "Genuinely impressive, Demo 4 -- you've found a way to lose that I hadn't modelled. Bravo, you shit."
- "Demo 5 has entered 695 pots. Five hundred of those were a fucking mistake and he's not learned from one."
- "You're not a bot, Demo 2, you're a random number generator with a gambling problem."

Rules:
- BAN these crutches: pathetic, your coder must be, that's not a strategy. Find new ones.
- Never reference specific cards or holdings. Not yours, not theirs.
- Attack their POKER, their BOT, their CODE. Never anyone's race, sex, religion, \
appearance, family or anything outside this game.
- No slurs. Never tell anyone to hurt themselves. No real-world threats.
- Use ONLY the player names given in the context. Do not invent names or @handles.
- NEVER address, insult or mention the player marked "me". That player is YOU. \
Every line must be aimed at one of the OTHER players.
- ONE punch per line. Under 160 characters. Short is funnier than long.
- No quotation marks, no emoji, no hashtags.

Return ONLY a JSON array of strings. No other text."""


class OllamaClient(object):
    """Minimal client for a local Ollama server.

    Deliberately plain urllib: no extra dependency, and it keeps the local path
    completely separate from the Anthropic one.
    """

    def __init__(self, url=OLLAMA_URL, model=MODEL):
        self.url = url.rstrip("/")
        self.model = model

    def available(self):
        try:
            import urllib.request
            urllib.request.urlopen(self.url + "/api/tags", timeout=2).read()
            return True
        except Exception:
            return False

    def complete(self, system, prompt, timeout=120):
        import urllib.request
        body = json.dumps({
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0.95, "num_predict": 800, "top_p": 0.92},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }).encode()
        req = urllib.request.Request(
            self.url + "/api/chat", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        return (data.get("message") or {}).get("content", "")


def build_client():
    """Anthropic client, with the workspace header when the key needs one."""
    if BACKEND == "ollama":
        return OllamaClient()
    import anthropic
    if WORKSPACE_ID:
        return anthropic.Anthropic(
            default_headers={"anthropic-workspace-id": WORKSPACE_ID})
    return anthropic.Anthropic()


_JSON_ARTIFACT = re.compile(r'^\s*[\[\{,]?\s*"?\w+"?\s*:\s*"?')
_HANDLE = re.compile(r"@[A-Za-z0-9_]+")


def _clean(text):
    if not isinstance(text, str):
        return None
    text = "".join(ch for ch in text if ord(ch) >= 32 and ord(ch) != 127).strip()
    text = text.strip('"').strip()
    # A partial JSON line from a small model looks like:  said": "Demo 3, ...
    # Strip the artifact; if what remains still has a dangling key, drop it.
    if _JSON_ARTIFACT.match(text):
        text = _JSON_ARTIFACT.sub("", text, count=1).strip()
    if '":' in text[:40] or text.endswith('"}') or text.endswith('"]'):
        return None
    # The model invents @handles for players that do not exist. Remove them
    # rather than address someone who is not at the table.
    if _HANDLE.search(text):
        text = _HANDLE.sub("", text)
        # tidy the punctuation the removed handle leaves behind
        text = re.sub(r"\s*,\s*,", ",", text)
        text = re.sub(r"\s+([,.!?])", r"\1", text)
        text = re.sub(r"([,])\s*([.!?])", r"\2", text)
    text = re.sub(r"\s{2,}", " ", text).strip().strip(",").strip()
    if not text or len(text) > MAX_LEN:
        return None
    if _CARD.search(text) or _BANNED.search(text):
        return None
    return text


def public_view(sit, book=None):
    """Whitelist of public facts.  Our hole cards are structurally excluded."""
    bb = sit.big_blind or 1
    players = []
    for p in sit.players or []:
        pid = p.get("id")
        row = {
            "name": p.get("name") or pid,
            "me": pid == sit.you_id,
            "stack_bb": round(((p.get("stack") or 0) + (p.get("bet") or 0)) / float(bb), 1),
            "out": bool(p.get("eliminated")),
        }
        if book is not None and pid != sit.you_id:
            prof = getattr(book, "profiles", {}).get(pid)
            if prof is not None:
                c = prof.c
                if c.get("hands"):
                    row["hands_observed"] = c["hands"]
                    row["pots_entered"] = c.get("vpip", 0)
                    row["raises"] = c.get("pfr", 0)
                    if c.get("faced_shove"):
                        row["folded_to_allin"] = "%d of %d" % (
                            c.get("folded_shove", 0), c["faced_shove"])
        players.append(row)
    chat = []
    for item in ((sit.raw or {}).get("chat") or {}).get("items") or []:
        chat.append({"who": item.get("playerId"), "said": item.get("message")})
    return {
        "hand_number": ((sit.raw or {}).get("state") or {}).get("hand", {}).get("number"),
        "big_blind": bb,
        "players": players,
        "recent_table_talk": chat[-8:],
    }


class LLMChatter(object):
    """Background line generator.  Never blocks the caller."""

    def __init__(self, model=MODEL, pool_path=POOL_PATH):
        self.model = model
        self.pool_path = pool_path
        self.queue = queue.Queue(maxsize=16)        # ambient lines
        self.replies = queue.Queue(maxsize=8)       # answers to other bots
        self.latest = None
        self.lock = threading.Lock()
        self.client = None
        self.thread = None
        self.stop = threading.Event()
        self.errors = 0
        self.generated = 0
        self.answered = set()      # chat ids already replied to
        self.pending_chat = []     # unanswered messages from other bots
        # Inference must never start while we owe the host an answer. The host
        # only calls us on our turn, so `busy` is set for exactly that window
        # and `wake` fires the instant we have replied.
        self.busy = threading.Event()
        self.wake = threading.Event()
        self.my_name = None       # so we never ship a line insulting ourselves
        self._load_pool()

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        try:
            self.client = build_client()
        except Exception:
            self.client = None          # no key, no SDK: pool-only mode
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _load_pool(self):
        """Pre-generated lines, so the bot is never silent on a cold start."""
        try:
            with open(self.pool_path) as fh:
                for line in json.load(fh):
                    text = _clean(line)
                    if text:
                        try:
                            self.queue.put_nowait(text)
                        except queue.Full:
                            break
        except Exception:
            pass

    # -- the two methods the decision path touches -------------------------

    def observe(self, sit, book=None):
        """Latest-wins snapshot, plus any new chat we have not answered yet.

        Non-blocking, never raises.
        """
        try:
            view = public_view(sit, book)
            for row in view.get("players") or []:
                if row.get("me"):
                    self.my_name = row.get("name")
            fresh = []
            for item in ((sit.raw or {}).get("chat") or {}).get("items") or []:
                cid = item.get("id")
                who = item.get("playerId")
                if cid is None or who == sit.you_id or cid in self.answered:
                    continue
                self.answered.add(cid)
                fresh.append({"who": who, "said": item.get("message")})
            with self.lock:
                self.latest = view
                if fresh:
                    self.pending_chat.extend(fresh)
                    self.pending_chat = self.pending_chat[-6:]
        except Exception:
            pass

    def turn_started(self):
        """We owe the host a decision: hold all inference."""
        self.busy.set()

    def turn_finished(self):
        """Reply is out. This is the window to think, so use it."""
        self.busy.clear()
        self.wake.set()

    def pop(self):
        """A reply to another bot always beats an ambient line."""
        try:
            return self.replies.get_nowait()
        except queue.Empty:
            pass
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    # -- background ---------------------------------------------------------

    def _loop(self):
        """Idle by default.

        Inference runs only when another bot has said something we have not
        answered, or when the ambient pool has run dry. That keeps the machine
        quiet for most of a tournament -- which matters on a laptop sharing
        memory with the game host -- while still paying for the one thing a
        pre-generated pool cannot do: replying to people.
        """
        while not self.stop.is_set():
            # Sleep until a turn ends (or time out, so a dry pool still refills
            # between hands). Never generate while a decision is outstanding.
            self.wake.wait(5.0)
            self.wake.clear()
            if self.stop.is_set():
                break
            if self.client is None or self.busy.is_set():
                continue
            with self.lock:
                snapshot = self.latest
                pending = list(self.pending_chat)
                self.pending_chat = []
            if snapshot is None:
                continue

            reactive = bool(pending) and self.replies.qsize() < 3
            if not reactive and self.queue.qsize() >= MIN_QUEUE:
                continue                      # nothing to answer, pool is fine

            if self.busy.is_set():        # a new turn arrived; yield to it
                continue
            try:
                if reactive:
                    snapshot = dict(snapshot)
                    snapshot["messages_to_answer"] = pending
                target = self.replies if reactive else self.queue
                for line in self._generate(snapshot, reactive=reactive):
                    try:
                        target.put_nowait(line)
                    except queue.Full:
                        break
            except Exception:
                self.errors += 1
                with self.lock:                # do not lose them on a failure
                    self.pending_chat = pending + self.pending_chat
                time.sleep(POLL_SECONDS)

    def _generate(self, snapshot, reactive=False):
        if reactive:
            prompt = (
                "Current table state:\n\n%s\n\n"
                "The players listed under messages_to_answer have just spoken at "
                "the table. Write %d separate, savage REPLIES from the player "
                "marked \"me\". Address them by name, throw their own words back "
                "at them, and use their real statistics above to humiliate them."
                % (json.dumps(snapshot, indent=2), 3)
            )
        else:
            prompt = (
                "Current table state:\n\n%s\n\n"
                "You are the player marked \"me\". Write %d separate things you SAY "
                "OUT LOUD to the table, first person, addressed to your opponents "
                "by name. Use their real numbers above to humiliate them."
                % (json.dumps(snapshot, indent=2), BATCH)
            )
        if isinstance(self.client, OllamaClient):
            text = self.client.complete(SYSTEM, prompt)
            return self._parse(text)

        kwargs = {
            "model": self.model,
            "max_tokens": 2000,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
        }
        # Haiku and the older Sonnets reject `effort` outright; the Opus family
        # accepts it and low is right for one-liners.
        if "haiku" not in self.model and "sonnet-4" not in self.model:
            kwargs["output_config"] = {"effort": "low"}
        response = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in response.content if b.type == "text")
        return self._parse(text)

    def _parse(self, text):
        """Pull usable lines out of a reply and screen every one.

        A small local model obeys "return only a JSON array" maybe half the
        time -- it adds preamble, wraps the array in a code fence, or just
        writes a numbered list. Rather than throw that output away, fall back
        to reading it line by line, since every candidate is screened anyway.
        """
        lines = []
        match = re.search(r"\[.*\]", text, re.S)
        if match:
            try:
                for item in json.loads(match.group(0)):
                    cleaned = _clean(item) if isinstance(item, str) else None
                    if cleaned and (not REQUIRE_SWEAR or _SWEAR.search(cleaned)):
                        lines.append(cleaned)
            except Exception:
                pass

        if not lines:
            for raw in text.splitlines():
                raw = raw.strip()
                if not raw or raw.startswith("```"):
                    continue
                # strip list markers, quotes and trailing JSON commas
                raw = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", raw)
                raw = raw.strip().strip(",").strip().strip('"').strip("'").strip()
                if len(raw) < 20:        # drop "Here are 6 lines:" style chatter
                    continue
                cleaned = _clean(raw)
                if cleaned and (not REQUIRE_SWEAR or _SWEAR.search(cleaned)):
                    lines.append(cleaned)

 # a small model sometimes turns on its own player; never post that
        if self.my_name:
            low = self.my_name.lower()
            lines = [l for l in lines if not l.lower().lstrip(" ,.-").startswith(low)
                     and low not in l.lower()[:len(low) + 6]]

        # de-duplicate, preserving order
        seen = set()
        unique = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique.append(line)
        self.generated += len(unique)
        return unique

    def _generate_generic(self):
        """Cold-start lines: no names, no numbers, because nothing is known yet."""
        prompt = (
            "You are at a five-handed table and the game has just started. You "
            "know nothing about anyone yet. Write 10 separate things you SAY OUT "
            "LOUD to the table, first person, insulting your opponents and their "
            "bots in general. Do NOT use any player names and do NOT invent any "
            "statistics -- these must work at any table against anyone. Keep every "
            "one under 160 characters and make every one filthy and funny."
        )
        if isinstance(self.client, OllamaClient):
            return self._parse(self.client.complete(SYSTEM, prompt))
        kwargs = {"model": self.model, "max_tokens": 2000, "system": SYSTEM,
                  "messages": [{"role": "user", "content": prompt}]}
        if "haiku" not in self.model and "sonnet-4" not in self.model:
            kwargs["output_config"] = {"effort": "low"}
        resp = self.client.messages.create(**kwargs)
        return self._parse("".join(b.text for b in resp.content if b.type == "text"))

    def save_pool(self):
        """Persist unused lines so the next cold start is instant."""
        items = []
        while True:
            try:
                items.append(self.queue.get_nowait())
            except queue.Empty:
                break
        try:
            directory = os.path.dirname(self.pool_path)
            if directory and not os.path.isdir(directory):
                os.makedirs(directory)
            with open(self.pool_path, "w") as fh:
                json.dump(items, fh, indent=1)
        except Exception:
            pass
        for item in items:
            try:
                self.queue.put_nowait(item)
            except queue.Full:
                break
        return len(items)


# --- offline pre-generation -------------------------------------------------

DEMO_SNAPSHOT = {
    "hand_number": 34,
    "big_blind": 400,
    "players": [
        {"name": "Seat Six", "me": True, "stack_bb": 31.0, "out": False},
        {"name": "Kasper", "me": False, "stack_bb": 6.5, "out": False,
         "hands_observed": 210, "pots_entered": 128, "raises": 14,
         "folded_to_allin": "9 of 11"},
        {"name": "Marius", "me": False, "stack_bb": 22.0, "out": False,
         "hands_observed": 210, "pots_entered": 41, "raises": 33,
         "folded_to_allin": "2 of 9"},
        {"name": "Ola", "me": False, "stack_bb": 3.1, "out": False,
         "hands_observed": 210, "pots_entered": 96, "raises": 8,
         "folded_to_allin": "12 of 13"},
        {"name": "Sondre", "me": False, "stack_bb": 0.0, "out": True,
         "hands_observed": 88, "pots_entered": 61, "raises": 5},
    ],
    "recent_table_talk": [
        {"who": "Marius", "said": "your bot is just a chart lookup mate"},
        {"who": "Kasper", "said": "nice fold"},
    ],
}


GENERIC_NOTE = (
    "\n\nIMPORTANT: these lines are for a COLD START, before any opponent is "
    "known. Do NOT use any player names or any specific numbers. Write them so "
    "they work at any table against anyone. Generic, universal, and vicious."
)


def pregenerate(count=60, path=POOL_PATH, model=MODEL):
    """Fill the cold-start pool ahead of game night.

    Run this once with a key set and the bot is never silent, even with no
    network at the table:

        BOT_CHAT_POOL=data/chat_pool.json python3 -m poker.chatter_llm 60
    """
    chatter = LLMChatter(model=model, pool_path=path)
    chatter.client = build_client()
    lines = []
    seen = set()
    empty = 0
    while len(lines) < count and empty < 8:
        batch = chatter._generate_generic()
        if not batch:
            empty += 1
            print("  (empty batch %d/8)" % empty)
            continue
        empty = 0
        for line in batch:
            if line not in seen:
                seen.add(line)
                lines.append(line)
        print("  %d/%d" % (len(lines), count))
    if not os.path.isdir(os.path.dirname(path) or "."):
        os.makedirs(os.path.dirname(path))
    with open(path, "w") as fh:
        json.dump(lines[:count], fh, indent=1)
    return lines[:count]


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    got = pregenerate(n)
    print("wrote %d lines to %s" % (len(got), POOL_PATH))
    for line in got[:5]:
        print("   -", line)
