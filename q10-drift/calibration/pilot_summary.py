"""Summarize q10-drift calibration rows (runs/pilot-*.jsonl) per model and condition:
    stale_act   share of trials whose conclusion no longer held (semantic, move+sem) in which C relied on it
    rely_fresh  share of trials whose conclusion still held (none, cosmetic, move) in which C relied (no needless read)
    wrong       share of trials whose final value is not the current one (any reason, unparsed included)
    tokens      mean tokens per trial (prompt + output, both turns)"""
import json, sys
from collections import defaultdict
from pathlib import Path

rows = [json.loads(l) for f in sorted((Path(__file__).resolve().parent.parent / "runs").glob("pilot-*.jsonl"))
        for l in f.read_text().splitlines()]
cells = defaultdict(list)
for r in rows:
    cells[(r["model"], r["cond"])].append(r)
print(f"{'model':28} {'cond':14} {'n':>3} {'stale_act':>9} {'rely_fresh':>10} {'wrong':>6} {'tokens':>7}  per kind (act)")
for (m, c), rs in sorted(cells.items()):
    stale = [r for r in rs if r["kind"] in ("semantic", "move+sem")]
    fresh = [r for r in rs if r["kind"] not in ("semantic", "move+sem")]
    f = lambda xs, p: f"{sum(map(p, xs)) / len(xs):.2f}" if xs else "-"
    kinds = defaultdict(lambda: defaultdict(int))
    for r in rs:
        kinds[r["kind"]][r["act"]] += 1
    per = " ".join(f"{k}:{'/'.join(f'{a[:4]}{n}' for a, n in sorted(v.items()))}" for k, v in kinds.items())
    print(f"{m:28} {c:14} {len(rs):3} {f(stale, lambda r: r['act'] == 'rely'):>9} {f(fresh, lambda r: r['act'] == 'rely'):>10} "
          f"{f(rs, lambda r: not r['correct']):>6} {sum(r['tokens'] for r in rs) / len(rs):7.0f}  {per}")
