"""q5-bound registered analysis: what q4-line's line, geared, gives a crystal that already runs near the TUR bound.

    python -I q5-bound/analyze.py DIR        prints {verdicts, descriptives} as JSON; refuses incomplete data

For each cell, from the crystal's current J at the checkpoints t0 < t1 < t2, per mean phase (Phi = J / (N d)):
    D = [var(Phi(t2) - Phi(t0)) - var(Phi(t1) - Phi(t0))] / (2 (t2 - t1))     (a bounded jitter cancels)
    v = mean(Phi(t2) - Phi(t0)) / (t2 - t0)
    S = mean of the heat given to the bath over [t0, t2] by the crystal and by the line, / (sigma² (t2 - t0))
    eta = v² / (D S)       the clock's precision per entropy, all its matter counted (TUR: eta <= ~1 for an autonomous clock)
The gain of a cell is G = D(alone) / D(cell). Each hypothesis is log10 of a ratio of gains or of etas, with a 95% interval
(inverted-cdf quantiles, no interpolation) over resamples of the trajectories of every cell, seeded from the registration
id. A resample in which a D is not positive gives that side an infinite precision; one in which both are, an undefined
ratio; any undefined resample makes that hypothesis inconclusive. Its bytes are the registration's `analysis` blob."""
import hashlib, json, math, sys
from pathlib import Path

import numpy as np

B = 2000
LOG2, LOG3, LOG15, LOG125 = math.log10(2), math.log10(3), math.log10(1.5), math.log10(1.25)
NEAR = 0.6                                  # the adequacy condition of H2: the crystal alone runs at eta >= NEAR (q4-line: 0.26)
TESTS = {                                   # {test: (kind, a, b)}: log10 G(a) / G(b) = log10 D(b) / D(a), or log10 eta(a) / eta(b)
    "H1-precision-for-sale": ("gain", "line-n128", "alone"),
    "H2-efficiency-not": ("eta", "line-n128", "alone"),
    "H3-light-line-hurts": ("gain", "line-n8", "alone"),
    "H4-heavy-line-costs": ("eta", "line-n512", "alone"),
}
PARTS = (("H1", "H1-precision-for-sale"), ("H2", "H2-efficiency-not"), ("H3", "H3-light-line-hurts"), ("H4", "H4-heavy-line-costs"))
RULES = {                                   # {H: (kind, x)} on the log10 test of PARTS
    "H1": ("above", LOG3),                  # the interval lies above x: supported; wholly below: contradicted
    "H2": ("below", LOG125),                # the interval lies below x: supported; wholly above: contradicted
    "H3": ("below", 0.0),
    "H4": ("below", -LOG15),
}


def finite(x):
    """A record carries no NaN or infinity: an undefined number is null."""
    return x if math.isfinite(x) else None


def refuse(why):
    sys.exit(f"REFUSED {why}")


def load(d):
    reg = json.loads((d / "registration.json").read_text())
    lock = json.loads((d / "registration.lock.json").read_text())
    data = json.loads((d / "results" / "data.json").read_text())
    design = reg["design"]
    if data.get("registration") != lock["registration"]:
        refuse("data are not of this registration")
    want = {c["name"]: c for c in design["cells"]}
    if set(data["cells"]) != set(want):
        refuse(f"{len(set(data['cells']) & set(want))} of {len(want)} registered cells")
    cells = {}
    for name in want:
        c = data["cells"][name]
        x = np.asarray([c["J"], c["heat"], c["heat_line"]], float)
        if x.shape != (3, 3, design["trajectories"]) or not np.isfinite(x).all():
            refuse(f"{name}: J, heat and heat_line of shape {x.shape}, finite {bool(np.isfinite(x).all())}")
        cells[name] = np.concatenate([x[0] / (design["N"] * design["base"]["d"]), x[1] + x[2]])   # Phi (3 rows), heat (3 rows)
    return design, lock["registration"], data, cells


def clock(x, design):
    """(D, v, S) of one cell from its rows: Phi at t0, t1, t2, then the total heat at t0, t1, t2."""
    t0, t1, t2 = design["checkpoints"]
    D = ((x[2] - x[0]).var() - (x[1] - x[0]).var()) / (2 * (t2 - t1))
    v = (x[2] - x[0]).mean() / (t2 - t0)
    S = (x[5] - x[3]).mean() / (design["base"]["sigma"] ** 2 * (t2 - t0))
    return D, v, S


def log_ratio(kind, a, b):
    """log10 G(a) / G(b) or log10 eta(a) / eta(b) from the clocks (D, v, S); a D that is not positive is infinitely precise."""
    (Da, va, Sa), (Db, vb, Sb) = a, b
    if Da <= 0 and Db <= 0:
        return math.nan
    if Da <= 0:
        return math.inf
    if Db <= 0:
        return -math.inf
    if kind == "gain":
        return math.log10(Db / Da)
    return math.log10((va * va / (Da * Sa)) / (vb * vb / (Db * Sb)))


def tests(cells, design):
    c = {n: clock(x, design) for n, x in cells.items()}
    return {t: log_ratio(kind, c[a], c[b]) for t, (kind, a, b) in TESTS.items()}


def decide(kind, x, lo, hi):
    if kind == "above":
        return "supported" if lo > x else "contradicted" if hi < x else "inconclusive"
    if kind == "below":
        return "supported" if hi < x else "contradicted" if lo > x else "inconclusive"
    return "supported" if -x < lo and hi < x else "contradicted" if lo > x or hi < -x else "inconclusive"


def judge(ci, near):
    """near: the interval of eta(alone). H2 asks about a crystal near the bound: it is assessed only when that interval
    lies above NEAR, else inconclusive."""
    out = {}
    for h, t in PARTS:
        lo, hi = ci[t]
        if math.isnan(lo) or math.isnan(hi):
            out[h] = {"verdict": "inconclusive", "reason": f"a resample of {t} is undefined"}
        elif h == "H2" and not near[0] > NEAR:
            out[h] = {"verdict": "inconclusive", "reason": f"the crystal alone is not near the bound: eta interval {near}"}
        else:
            out[h] = {"verdict": decide(*RULES[h], lo, hi)}
    return out


def eta(x, design):
    D, v, S = clock(x, design)
    return v * v / (D * S) if D > 0 else math.inf


def main(d):
    design, rid, data, cells = load(Path(d))
    point = tests(cells, design)
    rng = np.random.Generator(np.random.PCG64(int(hashlib.sha256(f"{rid}:bootstrap".encode()).hexdigest()[:16], 16)))
    T = design["trajectories"]
    boots = {t: [] for t in TESTS}
    alone = []
    for _ in range(B):
        sample = {n: x[:, rng.integers(0, T, T)] for n, x in sorted(cells.items())}
        r = tests(sample, design)
        for t, v in r.items():
            boots[t].append(v)
        alone.append(eta(sample["alone"], design))
    ci = {}
    for t, v in boots.items():
        v = np.asarray(v)
        ci[t] = [math.nan, math.nan] if np.isnan(v).any() else [float(np.quantile(v, q, method="inverted_cdf")) for q in (0.025, 0.975)]
    near = [float(np.quantile(alone, q, method="inverted_cdf")) for q in (0.025, 0.975)]
    verdicts = judge(ci, near)
    verdicts["H2"]["eta_alone"] = {"point": finite(eta(cells["alone"], design)), "ci95": [finite(x) for x in near]}
    for h, t in PARTS:
        verdicts[h][t] = {"log10_ratio": finite(point[t]), "ci95": [finite(x) for x in ci[t]]}
    t0, t1, t2 = design["checkpoints"]
    s = design["base"]["sigma"] ** 2
    D_alone = clock(cells["alone"], design)[0]
    desc = {}
    for n, x in cells.items():
        D, v, S = clock(x, design)
        c = data["cells"][n]
        S_line = float(np.mean(np.asarray(c["heat_line"][2]) - np.asarray(c["heat_line"][0])) / (s * (t2 - t0)))
        lag = np.asarray(c["lag"], float)
        desc[n] = {"v": float(v), "D": finite(float(D)), "D_time": finite(float(D) / v ** 2) if v else None,
                   "G": finite(D_alone / D) if D > 0 else None, "entropy_rate": float(S) - S_line, "entropy_rate_line": S_line,
                   "eta_main": finite(v * v / (D * (S - S_line))) if D > 0 and S > S_line else None,
                   "eta_total": finite(v * v / (D * S)) if D > 0 and S > 0 else None, "clipped": c["clipped"]}
        if c["p"]["reference"] == "line":
            desc[n].update(lag_mean=float(lag[2].mean()), lag_sd=float(lag[2].std()),
                           slipped=float(np.mean(np.abs(lag[2] - c["p"]["Lambda"]) > np.pi)))
    print(json.dumps({"verdicts": verdicts, "descriptives": desc}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
