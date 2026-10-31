"""Calibration summary of q8-what-counts pilots: per model the medians by (condition, n), tokens and characters per word,
and the two elasticities: iota_chars = log(one / short) / log(C_one / C_short), iota_tokens = log(many / one) / log(T_many / T_one).

    python -I q8-what-counts/pilot_summary.py PILOT.json [...]"""
import json, math, re, sys
from collections import defaultdict

import numpy as np

INT = re.compile(r"\d[\d,]*")


def parse(a):
    m = INT.search(a)
    return int(m.group().replace(",", "")) if m and m.group().replace(",", "") else None


for path in sys.argv[1:]:
    d = json.load(open(path))
    vocab = json.load(open("q8-what-counts/calibration/vocabularies.json"))
    C = {c: sum(map(len, ws)) / len(ws) for c, ws in vocab.items()}
    est, tok = defaultdict(list), defaultdict(list)
    for r in d["rows"]:
        est[(r["condition"], r["n"])].append(parse(r["answer"]))
        tok[r["condition"]].append((r["n"], r["prompt_tokens"]))
    T = {c: np.polyfit([n for n, _ in v], [t for _, t in v], 1)[0] for c, v in tok.items()}
    ns = sorted({n for _, n in est})
    med = {k: float(np.median([x for x in v if x is not None])) for k, v in est.items()}
    ic = [math.log(med[("one", n)] / med[("short", n)]) / math.log(C["one"] / C["short"]) for n in ns]
    it = [math.log(med[("many", n)] / med[("one", n)]) / math.log(T["many"] / T["one"]) for n in ns]
    print(d["model"], "T/word", {c: round(t, 2) for c, t in T.items()}, "C", {c: round(x, 1) for c, x in C.items()})
    for n in ns:
        print("   n", n, {c: (med[(c, n)], sorted(set(x for x in est[(c, n)] if x is not None))) for c in ("short", "one", "many")})
    print("   iota_chars", [round(x, 2) for x in ic], "mean", round(float(np.mean(ic)), 3),
          " iota_tokens", [round(x, 2) for x in it], "mean", round(float(np.mean(it)), 3))
