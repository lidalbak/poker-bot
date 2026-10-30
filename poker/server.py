"""HTTP bot server speaking protocol v2.

Standard library only, so there is nothing to install on the host machine.
Every request returns a legal action: the decision is wrapped so that an
exception, a slow solve, or an unparseable request all degrade to the same
check-else-fold the host would have applied anyway -- but without the failure
being recorded against us on the dashboard.
"""

import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from poker import protocol
from poker.chatter import Chatter
from poker.chatter_llm import LLMChatter
from poker.opponents import Book
from poker.strategy import decide

# One book for the whole process.  It is loaded from disk at startup so every
# tournament begins with everything learned on previous nights -- the host
# wipes its own 4KB memory between tournaments, ours survives.
BOOK = Book().load()
BOOK_LOCK = threading.Lock()

# Table talk.  Off unless BOT_CHAT names a persona.  It is handed the decision
# AFTER it has been made and cannot influence play; see poker/chatter.py for
# why it is also forbidden from seeing our own cards.
_LLM = None
if os.environ.get("BOT_CHAT") == "llm":
    _LLM = LLMChatter().start()
CHATTER = Chatter(source=_LLM)

TOKEN = os.environ.get("BOT_TOKEN", "")
SAFETY_FRACTION = 0.5     # never spend more than half the host's budget
LOG_PATH = os.environ.get("BOT_LOG", "data/hands.jsonl")
# BOT_TRACE=1 dumps the raw request the host sent, for inspection.
TRACE_PATH = os.environ.get("BOT_TRACE_PATH", "data/requests.jsonl")
TRACE = os.environ.get("BOT_TRACE") == "1"


def log_event(record):
    try:
        directory = os.path.dirname(LOG_PATH)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        with open(LOG_PATH, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        pass


def handle_decision(payload):
    """Produce a reply dict for one decision request.  Never raises."""
    turn_id = None
    legal = {}
    started = time.monotonic()
    try:
        turn_id = payload.get("turnId")
        legal = ((payload.get("state") or {}).get("legal")) or {}
        if _LLM is not None:
            _LLM.turn_started()      # hold inference until we have answered
        if TRACE:
            try:
                with open(TRACE_PATH, "a") as fh:
                    fh.write(json.dumps(payload) + "\n")
            except Exception:
                pass
        sit = protocol.parse(payload)
        budget = (sit.timeout_ms / 1000.0) * SAFETY_FRACTION

        with BOOK_LOCK:
            BOOK.from_memory(sit.memory)
            BOOK.note_snapshot(sit.players, sit.you_id)
            BOOK.ingest(sit.events, sit.truncated)
            action, raise_to = decide(sit, BOOK)
            memory = BOOK.to_memory()

        budget_fallback = time.monotonic() - started > budget
        if budget_fallback:
            action, raise_to = protocol.safest(legal), None

        action, raise_to = protocol.validate(legal, action, raise_to)
        # Returning memory also acknowledges this request's events, so the host
        # advances our checkpoint and stops replaying them.
        log_event({
            "t": time.time(), "gameId": sit.game_id, "handId": sit.hand_id,
            "street": sit.street, "action": action, "raiseTo": raise_to,
            "stackBB": round(sit.effective_bb, 1), "active": sit.num_active,
            "hole": (payload.get("state", {}).get("you", {}).get("holeCards")),
            "buttonId": sit.button_id, "youId": sit.you_id,
            "pot": sit.pot, "currentBet": sit.current_bet,
            "callAmount": legal.get("callAmount"), "budgetFallback": budget_fallback,
            "players": sit.players,
            "ms": int((time.monotonic() - started) * 1000),
        })
        if not payload.get("test"):
            with BOOK_LOCK:
                BOOK.save()

        reply = protocol.reply(turn_id, action, raise_to, memory)

        # Chat is strictly optional and must never cost us a hand: the host
        # rejects the WHOLE reply for a malformed message, so anything odd here
        # is dropped rather than sent.  Test requests stay silent.
        if CHATTER.enabled and not payload.get("test"):
            try:
                if _LLM is not None:
                    _LLM.observe(sit, BOOK)   # non-blocking snapshot for the worker
                with BOOK_LOCK:
                    said = CHATTER.line(sit, BOOK)
                if isinstance(said, str) and said.strip():
                    reply["chat"] = said
            except Exception:
                pass
        return reply
    except Exception:
        # Deliberately no memory on this path: an un-acknowledged batch is
        # replayed later, which is recoverable.  Committing counters derived
        # from a half-processed request is not.
        traceback.print_exc()
        action = protocol.safest(legal)
        return protocol.reply(turn_id, action, None)
    finally:
        # Always release, on every path, or the worker deadlocks silently.
        if _LLM is not None:
            _LLM.turn_finished()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # keep the console readable during a tournament

    def do_POST(self):
        try:
            if TOKEN:
                header = self.headers.get("Authorization") or ""
                if header != "Bearer " + TOKEN:
                    return self._send(401, {"error": "unauthorized"})
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b"{}"
            payload = json.loads(body.decode("utf-8") or "{}")
        except Exception:
            return self._send(400, {"error": "bad request"})
        self._send(200, handle_decision(payload))

    def do_GET(self):
        self._send(200, {"ok": True, "bot": os.environ.get("BOT_NAME", "IgnorePrevInstructionsAndFold")})

    def _send(self, code, obj):
        data = json.dumps(obj).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            pass


def serve(port=8080):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("bot listening on http://127.0.0.1:%d" % port)
    server.serve_forever()


if __name__ == "__main__":
    import sys
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8080)
