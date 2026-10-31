"""q6-renewal registered analysis: what a clock is worth in worlds from memoryless to periodic, and what that worth costs.

    python -I q6-renewal/analyze.py DIR        prints {verdicts, descriptives} as JSON; refuses incomplete data

For each clock, from its current J and heat at the checkpoints t0 < t1 < t2 (Phi = J / (N d)):
    D = [var(Phi(t2) - Phi(t0)) - var(Phi(t1) - Phi(t0))] / (2 (t2 - t1)),   v = mean(Phi(t2) - Phi(t0)) / (t2 - t0)
    S = mean heat given to the bath over [t0, t2] / (sigma² (t2 - t0)),       E = S T, the entropy per mean interval
For each clock and world, over all queries of all trajectories:
    MSE = the sum of squared errors of the predicted time to the next event / the number of queries
    A = MSE(none) / MSE(clock)        the clock's anticipation: how many times better than no clock it predicts
the share of the ceiling C = MSE(perfect) / MSE(clock), how much of what a perfect clock anticipates in that world the
clock gets, and in the periodic world, where the time to the next event is exactly T - a, the error of a Brownian clock
with the clock's own long-time diffusion: its reading of an elapsed time a errs with variance 2 D a / v², so
    MSE_B = 2 (D / v²) mean(a)
Each hypothesis is log10 of a ratio with a 95% interval (inverted-cdf quantiles, no interpolation) over resamples of the
trajectories, the same indices for every clock and world (the worlds are shared), seeded from the registration id. A
resample in which a ratio is undefined (an error or a D that is not positive) makes that hypothesis inconclusive. Its
bytes are the registration's `analysis` blob."""
import hashlib, json, math, sys
from pathlib import Path

import numpy as np

B = 2000
LOG11, LOG125, LOG4 = math.log10(1.1), math.log10(1.25), math.log10(4)
TESTS = {                                   # {test: (kind, args)}
    "H1-cap-k16": ("mse", ("N1", "N64", "k16")),             # log10 MSE(N1) / MSE(N64) in the world k16
    "H2-price-periodic": ("price", ("N1", "N64", "periodic")),  # log10 [MSE(N1) / MSE(N64)] - log10 [S(N64) / S(N1)]
    "H3-cap-k4096": ("mse", ("N1", "N64", "k4096")),
    "H4-inertia": ("brownian", ("N1", "periodic")),           # log10 MSE_B(N1) / MSE(N1) in the periodic world
    "H5-scaling": ("capture", ("N16", "k4096", "N1", "k256")),  # log10 C(N16 in k4096) / C(N1 in k256)
}
PARTS = (("H1", "H1-cap-k16"), ("H2", "H2-price-periodic"), ("H3", "H3-cap-k4096"), ("H4", "H4-inertia"),
         ("H5", "H5-scaling"))
RULES = {                                   # {H: (kind, x)} on the log10 test of PARTS
    "H1": ("below", LOG11),                 # the interval lies below x: supported; wholly above: contradicted
    "H2": ("within", LOG125),               # the interval lies within (-x, x): supported; wholly outside: contradicted
    "H3": ("below", LOG4),
    "H4": ("above", 0.0),                   # the interval lies above x: supported; wholly below: contradicted
    "H5": ("within", LOG11),
}
WORLD_ROWS = ("sse_perfect", "sum_perfect", "sse_none", "sum_none", "sum_elapsed", "n")


def finite(x):
    """A record carries no NaN or infinity: an undefined number is null."""
    return x if x is not None and math.isfinite(x) else None


def refuse(why):
    sys.exit(f"REFUSED {why}")


def matrix(what, rows, k, T):
    """k rows of T finite numbers, or a refusal: a ragged or short row is refused, not read."""
    if not isinstance(rows, list) or len(rows) != k or any(not isinstance(r, list) or len(r) != T for r in rows):
        refuse(f"{what}: not {k} rows of {T}")
    x = np.asarray(rows, float)
    if not np.isfinite(x).all():
        refuse(f"{what}: a value that is not finite")
    return x


def load(d):
    reg = json.loads((d / "registration.json").read_text())
    lock = json.loads((d / "registration.lock.json").read_text())
    data = json.loads((d / "results" / "data.json").read_text())
    design = reg["design"]
    T = design["trajectories"]
    if data.get("registration") != lock["registration"]:
        refuse("data are not of this registration")
    cn, wn = [c["name"] for c in design["clocks"]], [w["name"] for w in design["worlds"]]
    if set(data["clocks"]) != set(cn) or set(data["worlds"]) != set(wn):
        refuse(f"{len(set(data['clocks']) & set(cn))} of {len(cn)} registered clocks, "
               f"{len(set(data['worlds']) & set(wn))} of {len(wn)} registered worlds")
    worlds = {}
    for w in wn:
        x = matrix(f"world {w}", [data["worlds"][w].get(r) for r in WORLD_ROWS], len(WORLD_ROWS), T)
        if (x[-1] <= 0).any():
            refuse(f"world {w}: a trajectory without queries")
        worlds[w] = x
    clocks = {}
    for c in design["clocks"]:
        got = data["clocks"][c["name"]]
        if set(got["worlds"]) != set(wn):
            refuse(f"clock {c['name']}: {len(set(got['worlds']) & set(wn))} of {len(wn)} worlds")
        J, Q = matrix(f"clock {c['name']} J", got["J"], 3, T), matrix(f"clock {c['name']} heat", got["heat"], 3, T)
        sse = matrix(f"clock {c['name']} errors", [got["worlds"][w].get("sse") for w in wn], len(wn), T)
        clocks[c["name"]] = {"rows": np.concatenate([J / (c["N"] * design["base"]["d"]), Q]), "sse": dict(zip(wn, sse))}
    return design, lock["registration"], data, clocks, worlds


def clock(rows, design):
    """(D, v, S) of a clock from its rows: Phi at t0, t1, t2, then the heat at t0, t1, t2."""
    t0, t1, t2 = design["checkpoints"]
    D = ((rows[2] - rows[0]).var() - (rows[1] - rows[0]).var()) / (2 * (t2 - t1))
    v = (rows[2] - rows[0]).mean() / (t2 - t0)
    S = (rows[5] - rows[3]).mean() / (design["base"]["sigma"] ** 2 * (t2 - t0))
    return D, v, S


def mse(sse, n):
    return sse.sum() / n.sum()


def lg(a, b):
    """log10 a / b, undefined unless both are positive."""
    return math.log10(a / b) if a > 0 and b > 0 else math.nan


def tests(clocks, worlds, design, idx):
    """The log10 tests on the trajectories idx (an index array, repeats allowed)."""
    n = {w: x[-1][idx] for w, x in worlds.items()}
    m = lambda c, w: mse(clocks[c]["sse"][w][idx], n[w])
    out = {}
    for t, (kind, a) in TESTS.items():
        if kind == "mse":
            out[t] = lg(m(a[0], a[2]), m(a[1], a[2]))
        elif kind == "capture":
            P = WORLD_ROWS.index("sse_perfect")
            out[t] = lg(mse(worlds[a[1]][P][idx], n[a[1]]), m(a[0], a[1])) - lg(mse(worlds[a[3]][P][idx], n[a[3]]), m(a[2], a[3]))
        elif kind == "price":
            S0, S1 = clock(clocks[a[0]]["rows"][:, idx], design)[2], clock(clocks[a[1]]["rows"][:, idx], design)[2]
            out[t] = lg(m(a[0], a[2]), m(a[1], a[2])) - lg(S1, S0)
        else:
            D, v, _ = clock(clocks[a[0]]["rows"][:, idx], design)
            elapsed = worlds[a[1]][WORLD_ROWS.index("sum_elapsed")][idx].sum() / n[a[1]].sum()
            out[t] = lg(2 * D / v ** 2 * elapsed, m(a[0], a[1]))
    return out


def decide(kind, x, lo, hi):
    if kind == "above":
        return "supported" if lo > x else "contradicted" if hi < x else "inconclusive"
    if kind == "below":
        return "supported" if hi < x else "contradicted" if lo > x else "inconclusive"
    return "supported" if -x < lo and hi < x else "contradicted" if lo > x or hi < -x else "inconclusive"


def judge(ci):
    out = {}
    for h, t in PARTS:
        lo, hi = ci[t]
        if math.isnan(lo) or math.isnan(hi):
            out[h] = {"verdict": "inconclusive", "reason": f"a resample of {t} is undefined"}
        else:
            out[h] = {"verdict": decide(*RULES[h], lo, hi)}
    return out


def main(d):
    design, rid, data, clocks, worlds = load(Path(d))
    T = design["trajectories"]
    every = np.arange(T)
    point = tests(clocks, worlds, design, every)
    rng = np.random.Generator(np.random.PCG64(int(hashlib.sha256(f"{rid}:bootstrap".encode()).hexdigest()[:16], 16)))
    boots = {t: [] for t in TESTS}
    for _ in range(B):
        for t, v in tests(clocks, worlds, design, rng.integers(0, T, T)).items():
            boots[t].append(v)
    ci = {}
    for t, v in boots.items():
        v = np.asarray(v)
        ci[t] = [math.nan, math.nan] if np.isnan(v).any() else [float(np.quantile(v, q, method="inverted_cdf")) for q in (0.025, 0.975)]
    verdicts = judge(ci)
    for h, t in PARTS:
        verdicts[h][t] = {"log10_ratio": finite(point[t]), "ci95": [finite(x) for x in ci[t]]}
    R = WORLD_ROWS.index
    desc = {"worlds": {}, "clocks": {}}
    for w, x in worlds.items():
        n = x[R("n")].sum()
        none, perfect = x[R("sse_none")].sum() / n, x[R("sse_perfect")].sum() / n
        desc["worlds"][w] = {"mse_none": float(none), "mse_perfect": float(perfect),
                             "A_perfect": finite(float(none / perfect)) if perfect > 0 else None,
                             "bias_perfect": float(x[R("sum_perfect")].sum() / n), "mean_elapsed": float(x[R("sum_elapsed")].sum() / n)}
    for c in design["clocks"]:
        D, v, S = clock(clocks[c["name"]]["rows"], design)
        row = {"N": c["N"], "v": float(v), "D": float(D), "D_time": float(D / v ** 2), "entropy_rate": float(S),
               "entropy_per_interval": float(S * design["T"]), "eta": finite(float(v * v / (D * S))) if D > 0 else None, "worlds": {}}
        for w in worlds:
            n = worlds[w][R("n")].sum()
            e = float(mse(clocks[c["name"]]["sse"][w], worlds[w][R("n")]))
            got = data["clocks"][c["name"]]["worlds"][w]
            wd = desc["worlds"][w]
            row["worlds"][w] = {"mse": e, "A": float(wd["mse_none"] / e) if e > 0 else None, "bias": float(np.sum(got["sum"]) / n),
                                "C": float(wd["mse_perfect"] / e) if e > 0 and wd["mse_perfect"] > 0 else None}
        desc["clocks"][c["name"]] = row
    print(json.dumps({"verdicts": verdicts, "descriptives": desc}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
