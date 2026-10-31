"""q8-what-counts registered analysis: does a model's count estimate follow the characters of the list or its tokens?

    python -I q8-what-counts/analyze.py DIR          prints {verdicts, descriptives} as JSON; refuses incomplete data

Per model and condition: tau = tokens per word and chi = characters per word, each the slope over n of the prompt's tokens
(as ollama counts them) and of the list's characters. Over n in the grid, with medians of the parsed estimates:
    iota_chars  = mean_n log(median one / median short) / log(chi_one / chi_short)    the tokens equal, the characters not
    iota_tokens = mean_n log(median many / median one) / log(tau_many / tau_one)       the characters equal, the tokens not
    chars-minus-tokens = iota_chars - iota_tokens
0: the estimate ignores that axis; 1: it grows in proportion; below 0: it falls. A 95% interval comes from 2000 resamples of
the lists of each (condition, n) cell (inverted-cdf quantiles), the same resamples for all three, seeded from the
registration id and the model.

Inconclusive by rule: fewer than 95% of a condition's answers hold an integer; the manipulation fails (the axis held equal
differs by more than 10%, or the axis varied changes by less than 1.5 times); a median estimate is not positive. Its bytes
are the registration's `analysis` blob."""
import hashlib, json, math, re, sys
from pathlib import Path

import numpy as np

B = 2000
GATE = 0.95                 # parsed answers per (model, condition)
HELD = 0.10                 # the axis held equal may differ by at most this share
MIN_RATIO = 1.5             # the axis varied must change by at least this factor
INT = re.compile(r"\d[\d,]*")
INDEX = {"chars": ("one", "short", "chi"), "tokens": ("many", "one", "tau")}      # {index: (numerator, denominator, axis)}
DIFF = "chars-minus-tokens"                 # iota_chars - iota_tokens from the same resamples
HYPOTHESES = {              # {H: (model, index, kind, x)}: supported when the interval lies above / below / within (-x, x)
    "H1": ("gemma3:27b", "chars", "above", 0.15),
    "H2": ("gemma3:27b", DIFF, "above", 0.0),
    "H3": ("gemma4:26b-a4b-it-q4_K_M", "chars", "within", 0.1),
    "H4": ("gemma4:26b-a4b-it-q4_K_M", "tokens", "below", -0.2),
    "H5": ("qwen3.8:27b-mlx", "chars", "below", 0.0),
    "H6": ("qwen3.8:27b-mlx", "tokens", "below", 0.0),
}
CONDITIONS = ("short", "one", "many")


def parse(answer):
    m = INT.search(answer)
    return int(m.group().replace(",", "")) if m and m.group().replace(",", "") else None


def refuse(why):
    sys.exit(f"REFUSED {why}")


def finite(x):
    return x if x is not None and math.isfinite(x) else None


def load(d):
    reg = json.loads((d / "registration.json").read_text())
    lock = json.loads((d / "registration.lock.json").read_text())
    data = json.loads((d / "results" / "data.json").read_text())
    design = reg["design"]
    if data.get("registration") != lock["registration"]:
        refuse("data are not of this registration")
    conds, grid, k = [c["name"] for c in design["conditions"]], design["grid"], design["lists"]
    if sorted(conds) != sorted(CONDITIONS):
        refuse(f"conditions {conds} are not {CONDITIONS}")
    if set(data["models"]) != set(design["models"]):
        refuse(f"models {sorted(data['models'])} are not the registered {sorted(design['models'])}")
    want = {(c, n, i) for c in conds for n in grid for i in range(k)}
    cells = {}
    for m, md in data["models"].items():
        keys = [(r["condition"], r["n"], r["k"]) for r in md["rows"]]
        if len(keys) != len(set(keys)) or set(keys) != want:
            refuse(f"{m}: {len(set(keys))} distinct cells of {len(want)} registered, {len(keys)} rows")
        for r in md["rows"]:
            if not isinstance(r.get("prompt_tokens"), int) or not isinstance(r.get("chars"), int):
                refuse(f"{m}: a row without integer prompt_tokens and chars")
            cells.setdefault(m, {}).setdefault((r["condition"], r["n"]), []).append((r["k"], parse(r["answer"]), r["prompt_tokens"], r["chars"]))
    return design, lock["registration"], grid, k, cells


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())


def per_word(cell, grid, c, j):
    """The slope over n of column j (2: tokens, 3: characters) of condition c."""
    pts = [(n, row[j]) for n in grid for row in cell[(c, n)]]
    return slope([p[0] for p in pts], [p[1] for p in pts])


def iota(est, grid, num, den, ratio):
    xs = []
    for n in grid:
        if not len(est[(num, n)]) or not len(est[(den, n)]):
            return None
        a, b = np.median(est[(num, n)]), np.median(est[(den, n)])
        if a <= 0 or b <= 0:
            return None
        xs.append(math.log(a / b) / math.log(ratio))
    return float(np.mean(xs))


def model_indices(model, rid, grid, k, cell):
    """Both indices and their difference, each resample drawing the lists of every cell once for all three."""
    parsed = {c: sum(v is not None for (cc, _), xs in cell.items() if cc == c for _, v, _, _ in xs) / (len(grid) * k) for c in CONDITIONS}
    axes = {"tau": {c: per_word(cell, grid, c, 2) for c in CONDITIONS}, "chi": {c: per_word(cell, grid, c, 3) for c in CONDITIONS}}
    out = {"parsed": parsed, "tokens_per_word": axes["tau"], "chars_per_word": axes["chi"], "indices": {}}
    est = {key: np.array([v for _, v, _, _ in sorted(xs) if v is not None], float) for key, xs in cell.items()}
    ok = {}
    for name, (num, den, axis) in INDEX.items():
        other = "chi" if axis == "tau" else "tau"
        ratio, held = axes[axis][num] / axes[axis][den], axes[other][num] / axes[other][den]
        out["indices"][name] = {"ratio": ratio, "held": held}
        if min(parsed[num], parsed[den]) < GATE:
            out["indices"][name]["reason"] = f"gate: parsed {min(parsed[num], parsed[den]):.3f} < {GATE}"
        elif abs(held - 1) > HELD or ratio < MIN_RATIO:
            out["indices"][name]["reason"] = f"manipulation: varied {ratio:.3f}, held {held:.3f}"
        else:
            ok[name] = (num, den, ratio)
    out["indices"][DIFF] = {}
    if len(ok) < len(INDEX):
        out["indices"][DIFF]["reason"] = "an index is not assessed"
    both = lambda e: {name: iota(e, grid, *ok[name]) for name in ok}
    point = both(est)
    for name, v in point.items():
        if v is None:
            out["indices"][name]["reason"] = "a median estimate is not positive, or a cell has no parsed answer"
    good = [name for name, v in point.items() if v is not None]
    rng = np.random.Generator(np.random.PCG64(int(hashlib.sha256(f"{rid}:bootstrap:{model}".encode()).hexdigest()[:16], 16)))
    boots = {name: [] for name in good + ([DIFF] if len(good) == len(INDEX) else [])}
    if good:
        for _ in range(B):
            e = {key: v[rng.integers(0, len(v), len(v))] for key, v in sorted(est.items())}
            b = both(e)
            for name in good:
                boots[name].append(b[name] if b[name] is not None else math.nan)
            if DIFF in boots:
                boots[DIFF].append(math.nan if None in b.values() else b["chars"] - b["tokens"])
    if DIFF in boots:
        point[DIFF] = point["chars"] - point["tokens"]
    elif "reason" not in out["indices"][DIFF]:
        out["indices"][DIFF]["reason"] = "an index is not assessed"
    for name, v in boots.items():
        v = np.asarray(v)
        out["indices"][name]["iota"] = point[name]
        if np.isnan(v).any():
            out["indices"][name]["reason"] = "a resample has a non-positive median"
        else:
            out["indices"][name]["ci95"] = [float(np.quantile(v, q, method="inverted_cdf")) for q in (0.025, 0.975)]
    return out


def decide(kind, x, lo, hi):
    if kind == "above":
        return "supported" if lo > x else "contradicted" if hi < x else "inconclusive"
    if kind == "below":
        return "supported" if hi < x else "contradicted" if lo > x else "inconclusive"
    return "supported" if -x < lo and hi < x else "contradicted" if lo > x or hi < -x else "inconclusive"


def verdicts_of(results):
    out = {}
    for h, (model, index, kind, x) in HYPOTHESES.items():
        r = results[model]["indices"][index]
        if "ci95" not in r:
            out[h] = {"verdict": "inconclusive", "model": model, "index": index, **r}
        else:
            out[h] = {"verdict": decide(kind, x, *r["ci95"]), "model": model, "index": index, **r}
    return out


def describe(grid, cell):
    rows = {}
    for c in CONDITIONS:
        for n in grid:
            v = np.array([x for _, x, _, _ in cell[(c, n)] if x is not None], float)
            rows[f"{c}:{n}"] = {"median": float(np.median(v)) if len(v) else None, "ratio": float(np.median(v) / n) if len(v) else None,
                                "distinct": int(len(set(v)))}
    return rows


def main(d):
    design, rid, grid, k, cells = load(Path(d))
    results = {m: model_indices(m, rid, grid, k, cells[m]) for m in design["models"]}
    print(json.dumps({"verdicts": verdicts_of(results), "indices": results,
                      "descriptives": {m: describe(grid, cells[m]) for m in design["models"]}}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
