"""Preflop all-in equity.

The push/fold solver needs one primitive: how often does hand class A beat
hand class B when all five board cards run out.  That is a 169x169 matrix,
computed once by Monte Carlo and cached on disk.  Everything downstream
(equity against a range, shove/call thresholds) is a weighted lookup.
"""

import json
import os
import random
from multiprocessing import Pool

from poker.cards import all_hand_classes, hand_class_combos
from poker.evaluator import evaluate

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MATRIX_PATH = os.path.join(DATA_DIR, "preflop_equity.json")

CLASSES = all_hand_classes()
INDEX = dict((c, i) for i, c in enumerate(CLASSES))
_COMBOS = dict((c, hand_class_combos(c)) for c in CLASSES)


def pair_equity(a, b, samples=4000, seed=0):
    """Equity of class `a` against class `b`: wins + half the ties."""
    rng = random.Random(seed)
    ca, cb = _COMBOS[a], _COMBOS[b]
    full = list(range(52))
    won = 0.0
    played = 0
    attempts = 0
    while played < samples and attempts < samples * 40:
        attempts += 1
        h1 = ca[rng.randrange(len(ca))]
        h2 = cb[rng.randrange(len(cb))]
        if h1[0] in h2 or h1[1] in h2:
            continue
        used = set(h1) | set(h2)
        board = []
        while len(board) < 5:
            c = full[rng.randrange(52)]
            if c not in used:
                used.add(c)
                board.append(c)
        s1 = evaluate([h1[0], h1[1]] + board)
        s2 = evaluate([h2[0], h2[1]] + board)
        if s1 > s2:
            won += 1.0
        elif s1 == s2:
            won += 0.5
        played += 1
    return won / played if played else 0.5


def _job(args):
    i, j, samples = args
    return i, j, pair_equity(CLASSES[i], CLASSES[j], samples, seed=i * 1000 + j)


def build_matrix(samples=4000, workers=None, verbose=True):
    """Compute the full matrix (upper triangle + symmetry) and cache it."""
    tasks = [(i, j, samples) for i in range(len(CLASSES)) for j in range(i, len(CLASSES))]
    matrix = [[0.5] * len(CLASSES) for _ in CLASSES]
    pool = Pool(workers or os.cpu_count())
    try:
        done = 0
        for i, j, eq in pool.imap_unordered(_job, tasks, chunksize=64):
            matrix[i][j] = eq
            if i != j:
                matrix[j][i] = 1.0 - eq
            done += 1
            if verbose and done % 2000 == 0:
                print("   %d/%d pairs" % (done, len(tasks)))
    finally:
        pool.close()
        pool.join()
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)
    with open(MATRIX_PATH, "w") as fh:
        json.dump({"classes": CLASSES, "matrix": matrix, "samples": samples}, fh)
    return matrix


_MATRIX = None


def matrix():
    global _MATRIX
    if _MATRIX is None:
        with open(MATRIX_PATH) as fh:
            _MATRIX = json.load(fh)["matrix"]
    return _MATRIX


def equity_vs_class(a, b):
    m = matrix()
    return m[INDEX[a]][INDEX[b]]


# number of concrete combos per class, used to weight a range correctly
COMBO_COUNT = dict((c, len(_COMBOS[c])) for c in CLASSES)


def equity_vs_range(a, weights):
    """`weights` maps class -> probability in [0,1] that the opponent holds it.

    Weighted by combo count, because there are six ways to hold a pair and
    twelve to hold an offsuit hand.
    """
    m = matrix()
    ia = INDEX[a]
    total = 0.0
    acc = 0.0
    for cls, w in weights.items():
        if w <= 0:
            continue
        n = COMBO_COUNT[cls] * w
        acc += m[ia][INDEX[cls]] * n
        total += n
    return acc / total if total else 0.5


def equities_vs_range(weights):
    """Equity of EVERY class against one range, as a list in CLASSES order.

    The solvers ask "how does each of the 169 hands fare against this range"
    once per iteration; doing it as one pass over the range's non-zero entries
    instead of 169 separate dict walks is several times faster and returns the
    same numbers as equity_vs_range.
    """
    m = matrix()
    cols = []
    total = 0.0
    for cls, w in weights.items():
        if w <= 0:
            continue
        n = COMBO_COUNT[cls] * w
        cols.append((INDEX[cls], n))
        total += n
    if not total:
        return [0.5] * len(CLASSES)
    out = []
    for row in m:
        acc = 0.0
        for j, n in cols:
            acc += row[j] * n
        out.append(acc / total)
    return out
