"""The five synthetic scenarios the lobby's Test button sends, plus edge cases."""


def base(street, board, hole, stack, bet, current_bet, actions, **legal):
    lg = {"actions": actions, "callAmount": max(current_bet - bet, 0),
          "minRaiseTo": current_bet + 100, "maxRaiseTo": stack + bet,
          "raiseBase": current_bet, "raiseStep": 100,
          "allInTo": stack + bet, "canAllIn": True}
    lg.update(legal)
    return {
        "protocolVersion": 2, "type": "decision", "gameId": "g1", "handId": "h1",
        "turnId": "t-%s" % street, "timeoutMs": 30000, "test": True,
        "state": {
            "status": "playing",
            "you": {"id": "seat-1", "holeCards": hole, "stack": stack, "bet": bet},
            "players": [
                {"id": "seat-1", "stack": stack, "bet": bet, "folded": False, "eliminated": False},
                {"id": "seat-2", "stack": 5000, "bet": 0, "folded": False, "eliminated": False},
                {"id": "seat-3", "stack": 5000, "bet": 0, "folded": False, "eliminated": False},
                {"id": "seat-4", "stack": 5000, "bet": 0, "folded": False, "eliminated": False},
                {"id": "seat-5", "stack": 5000, "bet": 0, "folded": False, "eliminated": False},
            ],
            "hand": {"number": 3, "street": street, "board": board,
                     "currentPlayerId": "seat-1", "currentBet": current_bet,
                     "bigBlind": 100, "smallBlind": 50, "ante": 10,
                     "buttonId": "seat-3", "pot": 350},
            "history": [], "historyInfo": {"truncated": False},
            "legal": lg,
        },
        "events": {"afterId": 0, "throughId": 10, "truncated": False, "items": []},
        "memory": None,
        "tournament": {"startingStack": 5000, "totalChips": 25000, "levelIndex": 1},
    }


SCENARIOS = [
    ("preflop",     base("preflop", [], ["As", "Kd"], 4900, 100, 100, ["fold", "check", "raise"])),
    ("flop",        base("flop", ["Ah", "7c", "2d"], ["As", "Kd"], 4700, 0, 0, ["fold", "check", "raise"])),
    ("turn",        base("turn", ["Ah", "7c", "2d", "9s"], ["As", "Kd"], 4700, 0, 200, ["fold", "call", "raise"])),
    ("river",       base("river", ["Ah", "7c", "2d", "9s", "3h"], ["As", "Kd"], 4500, 0, 400, ["fold", "call", "raise"])),
    ("short stack", base("preflop", [], ["7h", "2c"], 900, 0, 100, ["fold", "call", "raise"],
                         minRaiseTo=200, maxRaiseTo=900, allInTo=900)),
    # --- nastier edge cases the host is allowed to send ---
    ("min>max short", base("preflop", [], ["Ac", "Ad"], 150, 0, 100, ["fold", "call", "raise"],
                           minRaiseTo=200, maxRaiseTo=150, allInTo=150)),
    ("no raise legal", base("river", ["Ah", "7c", "2d", "9s", "3h"], ["Kc", "Qc"], 0, 400, 400,
                            ["fold", "check"], canAllIn=False, minRaiseTo=None, maxRaiseTo=None)),
    ("empty request", {}),
    ("garbage cards", base("flop", ["ZZ", "??"], ["Xx", "9q"], 3000, 0, 0, ["fold", "check", "raise"])),
]
