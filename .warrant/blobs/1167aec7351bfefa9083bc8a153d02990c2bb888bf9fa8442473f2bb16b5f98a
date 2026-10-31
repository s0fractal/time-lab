"""q1-weber registered analysis: does a model's count estimate follow its own duration (tokens) or the events (words)?

    python -I q1-weber/analyze.py DIR          prints {verdicts, descriptives} as JSON; refuses incomplete data

Its bytes are the registration's `analysis` blob: a change is a new registration (METHOD §5)."""
import hashlib, json, math, re, sys
from pathlib import Path

import numpy as np

B = 2000                 # bootstrap resamples
GATE = 0.95              # parseable answers per (model, condition), else INCONCLUSIVE
MIN_T = 1.5              # the long-word condition must take this many times the tokens per word, else INCONCLUSIVE
FROM_N = 91              # ι is read where all three models estimate rather than count (calibration)
EVENTS = 0.25            # |ι| within this band: the estimate ignores word length (H2)
INT = re.compile(r"\d[\d,]*")


def parse(answer):
    m = INT.search(answer)
    return int(m.group().replace(",", "")) if m and m.group().replace(",", "") else None


def refuse(why):
    sys.exit(f"REFUSED {why}")


def load(d):
    reg = json.loads((d / "registration.json").read_text())
    lock = json.loads((d / "registration.lock.json").read_text())
    data = json.loads((d / "results" / "data.json").read_text())
    design = reg["design"]
    if data.get("registration") != lock["registration"]:
        refuse("data are not of this registration")
    conds, grid, k = [c["name"] for c in design["conditions"]], design["grid"], design["lists"]
    want = {(c, n, i) for c in conds for n in grid for i in range(k)}
    if set(data["models"]) != set(design["models"]):
        refuse(f"models {sorted(data['models'])} are not the registered {sorted(design['models'])}")
    cells = {}
    for m, md in data["models"].items():
        keys = [(r["condition"], r["n"], r["k"]) for r in md["rows"]]
        if len(keys) != len(set(keys)) or set(keys) != want:
            refuse(f"{m}: {len(set(keys))} distinct cells of {len(want)} registered, {len(keys)} rows")
        for r in md["rows"]:
            cells.setdefault(m, {}).setdefault((r["condition"], r["n"]), []).append((r["k"], parse(r["answer"]), r["prompt_tokens"]))
    return design, lock["registration"], conds, grid, k, cells


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())


def iota(est, grid, T):
    """Mean over n ≥ FROM_N of log(median_L / median_S) / log T: 0 counts events, 1 follows tokens."""
    xs = []
    for n in grid:
        if n < FROM_N:
            continue
        s, l = np.median(est[("short", n)]), np.median(est[("long", n)])
        if s <= 0 or l <= 0:
            return None
        xs.append(math.log(l / s) / math.log(T))
    return float(np.mean(xs))


def auc(a, b):
    """P(b > a) + ½ P(b = a): how often the larger n gets the larger estimate."""
    a, b = np.asarray(a)[:, None], np.asarray(b)[None, :]
    return float((b > a).mean() + 0.5 * (b == a).mean())


def model_verdict(model, rid, conds, grid, k, cell):
    parsed = {c: sum(v is not None for key, xs in cell.items() if key[0] == c for _, v, _ in xs) / (len(grid) * k) for c in conds}
    tok = {c: slope([n for n in grid for _ in range(k)], [t for n in grid for _, _, t in sorted(cell[(c, n)])]) for c in conds}
    T = tok["long"] / tok["short"]
    out = {"parsed": parsed, "tokens_per_word": tok, "T": T}
    if min(parsed.values()) < GATE:
        return {**out, "reason": f"gate: parsed {min(parsed.values()):.3f} < {GATE}"}
    if T < MIN_T:
        return {**out, "reason": f"manipulation: T {T:.3f} < {MIN_T}"}
    est = {key: np.array([v for _, v, _ in sorted(xs)], float) for key, xs in cell.items()}
    point = iota(est, grid, T)
    if point is None:
        return {**out, "reason": "a median estimate is not positive"}
    rng = np.random.Generator(np.random.PCG64(int(hashlib.sha256(f"{rid}:bootstrap:{model}".encode()).hexdigest()[:16], 16)))
    boots = []
    for _ in range(B):
        e = {key: v[rng.integers(0, len(v), len(v))] for key, v in sorted(est.items())}
        b = iota(e, grid, T)
        boots.append(b if b is not None else float("nan"))
    boots = np.array(boots)
    if np.isnan(boots).any():
        return {**out, "iota": point, "reason": "a resample has a non-positive median"}
    lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    return {**out, "iota": point, "ci95": [lo, hi]}


def verdicts_of(r):
    """H1: the estimate follows the model's own duration (ι > 0.5). H2: it ignores word length (|ι| < EVENTS)."""
    if "ci95" not in r:
        return {h: {**r, "verdict": "inconclusive"} for h in ("H1", "H2")}
    lo, hi = r["ci95"]
    h1 = "supported" if lo > 0.5 else "contradicted" if hi < 0.5 else "inconclusive"
    h2 = "supported" if -EVENTS < lo and hi < EVENTS else "contradicted" if hi <= -EVENTS or lo >= EVENTS else "inconclusive"
    return {"H1": {**r, "verdict": h1}, "H2": {**r, "verdict": h2}}


def describe(conds, grid, cell):
    rows = {}
    for c in conds:
        for n in grid:
            v = np.array([x for _, x, _ in cell[(c, n)] if x is not None], float)
            rows[f"{c}:{n}"] = {"median": float(np.median(v)), "mean": float(v.mean()), "sd": float(v.std()),
                                "ratio": float(np.median(v) / n), "distinct": int(len(set(v))),
                                "round5": float((v % 5 == 0).mean()), "round10": float((v % 10 == 0).mean())}
        for i in range(len(grid) - 2):
            a = [x for _, x, _ in cell[(c, grid[i])] if x is not None]
            b = [x for _, x, _ in cell[(c, grid[i + 2])] if x is not None]
            rows[f"{c}:auc:{grid[i]}-{grid[i + 2]}"] = auc(a, b)
    return rows


def main(d):
    design, rid, conds, grid, k, cells = load(Path(d))
    if sorted(conds) != ["long", "short"]:
        refuse(f"conditions {conds} are not short and long")
    out, desc = {}, {}
    for m in design["models"]:
        for h, v in verdicts_of(model_verdict(m, rid, conds, grid, k, cells[m])).items():
            out[f"{h}:{m}"] = v
        desc[m] = describe(conds, grid, cells[m])
    print(json.dumps({"verdicts": out, "descriptives": desc}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
