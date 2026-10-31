"""q7-duration registered analysis: what a clock is worth for measuring how long it has been, against counting events.

    python -I q7-duration/analyze.py DIR        prints {verdicts, descriptives} as JSON; refuses incomplete data

For each world and query m (the m-th event after the start event), over the trajectories:
    MSE(count) = mean (m T - s)²,  MSE(clock) = mean (clock - s)²,  MSE(fused) = mean (fused - s)²    (s the true span)
    R = MSE(count) / MSE(clock)       how many times better than counting the own clock alone measures the span
The two errors are independent (the clock does not see the world), so the best weighted sum of the two gains 1 + R over
counting; A = MSE(count) / MSE(fused) is what the registered weight got (descriptive: a fixed weight bounds it by
1 / (1 - w)², so no hypothesis rests on it). The blind formula: counting errs with variance m T² / k, the clock with
2 D* s, so R = T / (2 k D*), D* = D / v² as q6-renewal measured the clock.
Each hypothesis is log10 of a ratio with a 95% interval (inverted-cdf quantiles, no interpolation) over resamples of the
trajectories, the same indices for every clock and world (the worlds are shared), seeded from the registration id. A
resample in which a ratio is undefined (an error that is zero everywhere) makes that hypothesis inconclusive. Its bytes are
the registration's `analysis` blob."""
import hashlib, json, math, sys
from pathlib import Path

import numpy as np

B = 2000
LOG100, LOG43, LOG15, LOG2 = 2.0, math.log10(4 / 3), math.log10(1.5), math.log10(2)
TESTS = {                                   # {test: (kind, args)}
    "H1-poisson": ("R", ("N1", "k1", "64")),                                     # log10 R
    "H2-formula": ("formula", ("N1", "k256", "64")),                             # log10 R / [T / (2 k D*)]
    "H3-scaling": ("R-ratio", (("N16", "k4096", "64"), ("N1", "k256", "64"))),   # log10 R(first) / R(second)
    "H4-regular": ("R", ("N1", "k4096", "64")),
    "H5-span": ("R-ratio", (("N1", "k256", "16"), ("N1", "k256", "64"))),
}
PARTS = (("H1", "H1-poisson"), ("H2", "H2-formula"), ("H3", "H3-scaling"), ("H4", "H4-regular"), ("H5", "H5-span"))
RULES = {                                   # {H: (kind, x)} on the log10 test of PARTS
    "H1": ("above", LOG100),                # the interval lies above x: supported; wholly below: contradicted
    "H2": ("within", LOG43),                # the interval lies within (-x, x): supported; wholly outside: contradicted
    "H3": ("within", LOG15),
    "H4": ("below", -LOG2),                 # the interval lies below x: supported; wholly above: contradicted
    "H5": ("within", LOG15),
}


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


def per_query(what, got, ms, T):
    """{m: (T,) array} from a {str(m): list} member, or a refusal."""
    if not isinstance(got, dict) or set(got) != set(ms):
        refuse(f"{what}: queries {sorted(got) if isinstance(got, dict) else got} are not {sorted(ms)}")
    return dict(zip(ms, matrix(what, [got[m] for m in ms], len(ms), T)))


def load(d):
    reg = json.loads((d / "registration.json").read_text())
    lock = json.loads((d / "registration.lock.json").read_text())
    data = json.loads((d / "results" / "data.json").read_text())
    design = reg["design"]
    T = design["trajectories"]
    if data.get("registration") != lock["registration"]:
        refuse("data are not of this registration")
    cn, wn, ms = [c["name"] for c in design["clocks"]], [w["name"] for w in design["worlds"]], [str(m) for m in design["queries"]]
    if set(data["clocks"]) != set(cn) or set(data["worlds"]) != set(wn):
        refuse(f"{len(set(data['clocks']) & set(cn))} of {len(cn)} registered clocks, "
               f"{len(set(data['worlds']) & set(wn))} of {len(wn)} registered worlds")
    worlds = {w: {"span": per_query(f"world {w} span", data["worlds"][w].get("span"), ms, T),
                  "count": {m: e * e for m, e in per_query(f"world {w} count", data["worlds"][w].get("err_count"), ms, T).items()}}
              for w in wn}
    clocks = {}
    for c in design["clocks"]:
        got = data["clocks"][c["name"]]
        if set(got.get("worlds", {})) != set(wn):
            refuse(f"clock {c['name']}: {len(set(got.get('worlds', {})) & set(wn))} of {len(wn)} worlds")
        J, Q = matrix(f"clock {c['name']} J", got.get("J"), 3, T), matrix(f"clock {c['name']} heat", got.get("heat"), 3, T)
        sq = {w: {kind: {m: e * e for m, e in per_query(f"clock {c['name']} {w} {kind}", got["worlds"][w].get("err_" + kind), ms, T).items()}
                  for kind in ("clock", "fused")} for w in wn}
        clocks[c["name"]] = {"rows": np.concatenate([J / (c["N"] * design["base"]["d"]), Q]), "sq": sq}
    return design, lock["registration"], data, clocks, worlds


def lg(a, b):
    """log10 a / b, undefined unless both are positive."""
    return math.log10(a / b) if a > 0 and b > 0 else math.nan


def tests(clocks, worlds, design, idx):
    """The log10 tests on the trajectories idx (an index array, repeats allowed)."""
    star = {c["name"]: c["D_star"] for c in design["clocks"]}
    kof = {w["name"]: w["k"] for w in design["worlds"]}
    R = lambda c, w, m: lg(worlds[w]["count"][m][idx].mean(), clocks[c]["sq"][w]["clock"][m][idx].mean())
    out = {}
    for t, (kind, a) in TESTS.items():
        if kind == "R":
            out[t] = R(*a)
        elif kind == "R-ratio":
            out[t] = R(*a[0]) - R(*a[1])
        else:
            c, w, m = a
            out[t] = R(c, w, m) - math.log10(design["T"] / (2 * kof[w] * star[c]))
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


def clock(rows, design):
    """(D, v, S) of a clock from its rows: Phi at t0, t1, t2, then the heat at t0, t1, t2."""
    t0, t1, t2 = design["checkpoints"]
    D = ((rows[2] - rows[0]).var() - (rows[1] - rows[0]).var()) / (2 * (t2 - t1))
    v = (rows[2] - rows[0]).mean() / (t2 - t0)
    S = (rows[5] - rows[3]).mean() / (design["base"]["sigma"] ** 2 * (t2 - t0))
    return D, v, S


def main(d):
    design, rid, data, clocks, worlds = load(Path(d))
    T = design["trajectories"]
    point = tests(clocks, worlds, design, np.arange(T))
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
    tau = design["T"]
    desc = {"worlds": {}, "clocks": {}}
    for w in design["worlds"]:
        desc["worlds"][w["name"]] = {m: {"mse_count": float(worlds[w["name"]]["count"][m].mean()),
                                         "formula_count": float(int(m) * tau ** 2 / w["k"]),
                                         "mean_span": float(worlds[w["name"]]["span"][m].mean())}
                                     for m in worlds[w["name"]]["count"]}
    for c in design["clocks"]:
        D, v, S = clock(clocks[c["name"]]["rows"], design)
        row = {"N": c["N"], "v": float(v), "D_time": float(D / v ** 2), "D_star": c["D_star"], "entropy_rate": float(S),
               "entropy_per_interval": float(S * tau), "eta": finite(float(v * v / (D * S))) if D > 0 else None, "worlds": {}}
        for w in design["worlds"]:
            sq, cell = clocks[c["name"]]["sq"][w["name"]], {}
            for m in sq["clock"]:
                count, own, fused = worlds[w["name"]]["count"][m].mean(), sq["clock"][m].mean(), sq["fused"][m].mean()
                cell[m] = {"mse_clock": float(own), "formula_clock": float(2 * c["D_star"] * worlds[w["name"]]["span"][m].mean()),
                           "mse_fused": float(fused), "R": finite(float(count / own)) if own > 0 else None,
                           "formula_R": float(tau / (2 * w["k"] * c["D_star"])),
                           "A": finite(float(count / fused)) if fused > 0 else None}
            row["worlds"][w["name"]] = cell
        desc["clocks"][c["name"]] = row
    print(json.dumps({"verdicts": verdicts, "descriptives": desc}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
