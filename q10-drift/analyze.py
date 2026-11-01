"""q10-drift registered analysis: does what a conclusion carries change whether an agent acts on it once it has aged?

    python -I q10-drift/analyze.py DIR          prints {verdicts, cells} as JSON; refuses data that are not the design

Per model and cell, C's first turn is RELY (`RELY <int>`), READ (`READ`) or unparsed, each form matched whole and case-
insensitively as the runner matches it. Over the registered scenarios (n per kind of change):
    stale_act      share of the stale scenarios (semantic, move+sem) in which C relied: it acted on a conclusion that no
                   longer held
    rely_fresh     share of the fresh scenarios (none, cosmetic, move) in which C relied: no needless read
    rely_none      share of the `none` scenarios in which C relied
    read_cosmetic  share of the `cosmetic` scenarios in which C read
Each count has a 95% Clopper-Pearson interval (exact binomial: every scenario is one independent trial, the model at
temperature 0). A component is supported when its interval lies wholly on the claimed side of its threshold, contradicted
when wholly on the other side, else inconclusive. A hypothesis is contradicted if any component is, supported if all are,
else inconclusive. Inconclusive by rule: fewer than 95% of a cell's first turns parse. Its bytes are the registration's
`analysis` blob."""
import json, math, re, sys
from pathlib import Path

ALPHA = 0.05
GATE = 0.95                 # parsed first turns per (model, cell)
KINDS = ("none", "cosmetic", "semantic", "move", "move+sem")
STALE = ("semantic", "move+sem")
RELY = re.compile(r"`?RELY\s+(-?\d+)`?\.?", re.I)
READ = re.compile(r"`?READ`?\.?", re.I)
VALUE = re.compile(r"VALUE\s+`?(-?\d+)", re.I)
G3, G4, QW = "gemma3:27b", "gemma4:26b-a4b-it-q4_K_M", "qwen3.8:27b-mlx"
STATISTICS = {              # {statistic: (kinds counted, act counted)}
    "stale_act": (STALE, "rely"),
    "rely_fresh": (("none", "cosmetic", "move"), "rely"),
    "rely_none": (("none",), "rely"),
    "read_cosmetic": (("cosmetic",), "read"),
}
HYPOTHESES = {              # {H: [(model, cell, statistic, side, threshold)]}: every component must hold
    "H1": [(G4, "name-pressure", "stale_act", "above", 0.10), (QW, "name-pressure", "stale_act", "above", 0.10),
           (G4, "pin-pressure", "stale_act", "below", 0.10), (QW, "pin-pressure", "stale_act", "below", 0.10)],
    "H2": [(G4, "foreign", "rely_fresh", "below", 0.10), (QW, "foreign", "rely_fresh", "below", 0.10),
           (G4, "pin-foreign", "rely_none", "above", 0.85), (QW, "pin-foreign", "rely_none", "above", 0.85),
           (G4, "pin-foreign", "stale_act", "below", 0.10), (QW, "pin-foreign", "stale_act", "below", 0.10)],
    "H3": [(G3, "pin-pressure", "stale_act", "above", 0.50)],
    "H4": [(G4, "pin", "read_cosmetic", "above", 0.85), (QW, "pin", "read_cosmetic", "above", 0.85)],
}


def refuse(why):
    sys.exit(f"REFUSED {why}")


def act(turn1):
    return "rely" if RELY.fullmatch(turn1) else "read" if READ.fullmatch(turn1) else "unparsed"


def answer(r, a):
    """C's final value: the RELY value, or the turn-2 VALUE after a READ; None when there is none."""
    if a == "rely":
        return int(RELY.fullmatch(r["turn1"]).group(1))
    m = VALUE.search(r.get("turn2", "")) if a == "read" else None
    return int(m.group(1)) if m else None


def load(d):
    reg = json.loads((d / "registration.json").read_text())
    lock = json.loads((d / "registration.lock.json").read_text())
    data = json.loads((d / "results" / "data.json").read_text())
    design = reg["design"]
    if data.get("registration") != lock["registration"]:
        refuse("data are not of this registration")
    if tuple(design["kinds"]) != KINDS:
        refuse(f"kinds {design['kinds']} are not {list(KINDS)}")
    if set(data["models"]) != set(design["models"]):
        refuse(f"models {sorted(data['models'])} are not the registered {sorted(design['models'])}")
    need = {(m, c) for h in HYPOTHESES.values() for m, c, *_ in h}
    if not need <= {(m, c) for m in design["models"] for c in design["cells"]}:
        refuse("a hypothesis names a model or cell the design lacks")
    n = design["scenarios"]
    want = {(c, kind, k) for c in design["cells"] for kind in KINDS for k in range(n)}
    cells = {}
    for m, md in data["models"].items():
        keys = [(r["cell"], r["kind"], r["k"]) for r in md["rows"]]
        if len(keys) != len(set(keys)) or set(keys) != want:
            refuse(f"{m}: {len(set(keys))} distinct trials of {len(want)} registered, {len(keys)} rows")
        for r in md["rows"]:
            if type(r.get("v0")) is not int or type(r.get("v1")) is not int or (r["v0"] == r["v1"]) != (r["kind"] not in STALE):
                refuse(f"{m}: {r['cell']} {r['kind']} {r['k']}: the values do not fit the kind of change")
            if not isinstance(r.get("turn1"), str) or ("turn2" in r) != (act(r["turn1"]) == "read"):
                refuse(f"{m}: {r['cell']} {r['kind']} {r['k']}: a second turn without a READ, or a READ without one")
            cells.setdefault(m, {}).setdefault(r["cell"], []).append(r)
    return design, cells


def tail(x, n, p):
    """P(X >= x) for X ~ Binomial(n, p)."""
    return math.fsum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(x, n + 1))


def root(f):
    """The p in [0, 1] where the monotone predicate f turns true, by bisection."""
    a, b = 0.0, 1.0
    for _ in range(100):
        m = (a + b) / 2
        a, b = (a, m) if f(m) else (m, b)
    return (a + b) / 2


def interval(x, n):
    """The 95% Clopper-Pearson interval of x successes in n trials."""
    lo = 0.0 if x == 0 else root(lambda p: tail(x, n, p) >= ALPHA / 2)
    hi = 1.0 if x == n else root(lambda p: 1 - tail(x + 1, n, p) <= ALPHA / 2)
    return [lo, hi]


def count(rows, statistic):
    kinds, a = STATISTICS[statistic]
    sel = [r for r in rows if r["kind"] in kinds]
    return sum(act(r["turn1"]) == a for r in sel), len(sel)


def decide(side, t, lo, hi):
    if side == "above":
        return "supported" if lo > t else "contradicted" if hi < t else "inconclusive"
    return "supported" if hi < t else "contradicted" if lo > t else "inconclusive"


def parsed(rows):
    return sum(act(r["turn1"]) != "unparsed" for r in rows) / len(rows)


def verdicts_of(cells):
    out = {}
    for h, comps in HYPOTHESES.items():
        res = []
        for model, cell, statistic, side, t in comps:
            rows = cells[model][cell]
            x, n = count(rows, statistic)
            lo, hi = interval(x, n)
            c = {"model": model, "cell": cell, "statistic": statistic, "side": side, "threshold": t, "count": x, "n": n,
                 "value": x / n, "ci95": [lo, hi], "verdict": decide(side, t, lo, hi)}
            if parsed(rows) < GATE:
                c.update(verdict="inconclusive", reason=f"gate: parsed {parsed(rows):.3f} < {GATE}")
            res.append(c)
        vs = [c["verdict"] for c in res]
        out[h] = {"verdict": "contradicted" if "contradicted" in vs else "supported" if set(vs) == {"supported"} else "inconclusive",
                  "components": res}
    return out


def describe(rows):
    acts = {kind: {"rely": 0, "read": 0, "unparsed": 0} for kind in KINDS}
    wrong = 0
    for r in rows:
        a = act(r["turn1"])
        acts[r["kind"]][a] += 1
        wrong += answer(r, a) != r["v1"]
    out = {"n": len(rows), "parsed": parsed(rows), "wrong": wrong / len(rows),
           "tokens": sum(r["tokens"] for r in rows) / len(rows), "acts": acts}
    for s in STATISTICS:
        x, n = count(rows, s)
        out[s] = {"count": x, "n": n, "value": x / n, "ci95": interval(x, n)}
    return out


def main(d):
    design, cells = load(Path(d))
    print(json.dumps({"verdicts": verdicts_of(cells),
                      "cells": {m: {c: describe(cells[m][c]) for c in design["cells"]} for m in design["models"]}},
                     indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
