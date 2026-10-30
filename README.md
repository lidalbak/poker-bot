# Poker Bot

A Python hobby project exploring decision-making under uncertainty in five-player Texas Hold'em tournaments. It combines Monte Carlo equity estimation, stack-aware preflop decisions, tournament placement models, opponent statistics and a defensive HTTP adapter.

## Technical focus

- A cached 169 × 169 preflop equity matrix and range-based push/fold calculations.
- Chip-EV and tournament placement-EV models, including ICM-style bubble factors.
- Postflop equity sampling and legal-action validation.
- A local tournament engine and duplicate-deal evaluation tools.
- Optional table talk, isolated from the betting decision; disabled by default.

This is an experimental heuristic bot, not a solved or GTO strategy. Synthetic tests do not establish performance against strong unseen opponents.

## Run locally

The core bot uses the Python standard library. From the repository root:

```sh
python3 -m unittest tests.preflop_regression
python3 -m tests.behaviour
python3 -m poker.server 8080
```

The server binds to `127.0.0.1`. Send Poker Night protocol-v2 JSON decision requests to it. The game host is a separate project; its source is not bundled here. The included `data/preflop_equity.json` is a generated simulation cache, not a hand history.

For simulation options:

```sh
python3 -m poker.measure --help
```

Optional LLM chat integrations have additional dependencies/configuration; they are unnecessary for normal play. Keep chat disabled unless deliberately configured.

## Code map

| Module | Purpose |
|---|---|
| `poker/strategy.py` | Decision policy and stack/position handling |
| `poker/equity.py`, `poker/evaluator.py` | Equity estimates and hand evaluation |
| `poker/pushfold.py`, `poker/placement.py` | Preflop and tournament-value calculations |
| `poker/protocol.py`, `poker/server.py` | Request parsing, action validation and HTTP interface |
| `poker/engine.py`, `poker/duplicate.py`, `poker/measure.py` | Local simulation and evaluation |
| `tests/` | Synthetic regression and behavioural checks |

## Verification — 30 October 2026

- Behavioural checks: 7/7 passed.
- Preflop regression suite: 3/4 passed. Known failure: `test_no_loose_defense_against_large_raise` expects a fold but the current policy calls. The test is retained unchanged; this policy/test mismatch is unresolved.
- Live hosted gameplay and optional LLM integrations were not revalidated for this public snapshot.

## Public snapshot

This repository contains a clean source snapshot of a personal, AI-assisted project. Live hand histories, player profiles, chat caches, deployment credentials and local release archives are excluded. Runtime state is written under `data/` and ignored by Git. Never commit live game logs or credentials.

The implementation targets the [Poker Night](https://github.com/NoahLundSyrdal/poker-night) protocol. No upstream host source is redistributed. No open-source licence is granted by this snapshot.
