"""q3-self registered analysis: how much the crystal's reading of its own past sharpens its clock, and whose time that is.

    python -I q3-self/analyze.py DIR        prints {verdicts, descriptives} as JSON; refuses incomplete data

For each cell, from the main crystal's current J at the checkpoints t0 < t1 < t2, per mean phase (J / (N d)):
    D = [var(J(t2) - J(t0)) - var(J(t1) - J(t0))] / (2 (t2 - t1) (N d)²)
the slope of the variance, so that a bounded jitter around a held rhythm, which does not grow, cancels. The gain of a cell
is G = D(alone) / D(cell). Each hypothesis is log10 of a gain or of a ratio of two gains, with a 95% interval (inverted-cdf
quantiles, no interpolation) over resamples of the trajectories of every cell, seeded from the registration id. A
resample in which a D is not positive gives that D's side an infinite precision; one in which both are, an undefined
ratio; any undefined resample makes that hypothesis inconclusive. Its bytes are the registration's `analysis` blob."""
import hashlib, json, math, sys
from pathlib import Path

import numpy as np

B = 2000
LOG2, LOG5 = math.log10(2), math.log10(5)
TESTS = {                                   # {test: (numerator cell, denominator cell)}: log10 G(a) / G(b) = log10 D(b) / D(a)
    "H1-ideal": ("ideal-long", "alone"),
    "H2-ideal-span": ("ideal-long", "ideal-short"),
    "H3-memory-size": ("memory-N128-long", "memory-N8-long"),
    "H4-own-size": ("memory-N32-long", "alone"),
    "H5-memory-span": ("memory-N32-long", "memory-N32-short"),
    "H6-between-ticks": ("memory-N128-long", "memory-N128-offtick"),
}
PARTS = (("H1", "H1-ideal"), ("H2", "H2-ideal-span"), ("H3", "H3-memory-size"), ("H4", "H4-own-size"),
         ("H5", "H5-memory-span"), ("H6", "H6-between-ticks"))


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
        J = np.asarray(data["cells"][name]["J"], float)
        if J.shape != (3, design["trajectories"]) or not np.isfinite(J).all():
            refuse(f"{name}: J of shape {J.shape}, finite {bool(np.isfinite(J).all())}")
        cells[name] = J / (design["N"] * design["base"]["d"])
    return design, lock["registration"], data, cells


def diffusion(phi, design):
    t0, t1, t2 = design["checkpoints"]
    return ((phi[2] - phi[0]).var() - (phi[1] - phi[0]).var()) / (2 * (t2 - t1))


def log_ratio(Da, Db):
    """log10 G(a) / G(b) = log10 D(b) / D(a); a D that is not positive is infinitely precise."""
    if Da <= 0 and Db <= 0:
        return math.nan
    if Da <= 0:
        return math.inf
    if Db <= 0:
        return -math.inf
    return math.log10(Db / Da)


def tests(phis, design):
    D = {n: diffusion(x, design) for n, x in phis.items()}
    return {t: log_ratio(D[a], D[b]) for t, (a, b) in TESTS.items()}


RULES = {                                   # {H: (kind, x)} on the log10 test of PARTS
    "H1": ("above", 1.0),                   # the interval lies above x: supported; wholly below: contradicted
    "H2": ("above", LOG2),
    "H3": ("above", LOG5),
    "H4": ("below", LOG2),                  # the interval lies below x: supported; wholly above: contradicted
    "H5": ("within", LOG2),                 # the interval lies inside (-x, x): supported; wholly outside: contradicted
    "H6": ("above", LOG2),
}


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
    design, rid, data, phis = load(Path(d))
    point = tests(phis, design)
    rng = np.random.Generator(np.random.PCG64(int(hashlib.sha256(f"{rid}:bootstrap".encode()).hexdigest()[:16], 16)))
    T = design["trajectories"]
    boots = {t: [] for t in TESTS}
    for _ in range(B):
        r = tests({n: x[:, rng.integers(0, T, T)] for n, x in sorted(phis.items())}, design)
        for t, v in r.items():
            boots[t].append(v)
    ci = {}
    for t, v in boots.items():
        v = np.asarray(v)
        ci[t] = [math.nan, math.nan] if np.isnan(v).any() else [float(np.quantile(v, q, method="inverted_cdf")) for q in (0.025, 0.975)]
    verdicts = judge(ci)
    for h, t in PARTS:
        verdicts[h][t] = {"log10_ratio": finite(point[t]), "ci95": [finite(x) for x in ci[t]]}
    t0, t1, t2 = design["checkpoints"]
    D_alone = diffusion(phis["alone"], design)
    desc = {}
    for n, phi in phis.items():
        D = diffusion(phi, design)
        c = data["cells"][n]
        v = float((phi[2] - phi[0]).mean() / (t2 - t0))
        s = design["base"]["sigma"] ** 2
        heat = float(np.mean(np.asarray(c["heat"][2]) - np.asarray(c["heat"][0])) / (t2 - t0) / s)
        heat_m = float(np.mean(np.asarray(c["heat_memory"][2]) - np.asarray(c["heat_memory"][0])) / (t2 - t0) / s)
        desc[n] = {"v": v, "D": finite(float(D)), "D_time": finite(float(D) / v ** 2) if v else None,
                   "G": finite(D_alone / D) if D > 0 else None, "entropy_rate": heat, "entropy_rate_memory": heat_m,
                   "eta_main": finite(v * v / (D * heat)) if D > 0 and heat > 0 else None,
                   "eta_total": finite(v * v / (D * (heat + heat_m))) if D > 0 and heat + heat_m > 0 else None,
                   "clipped": c["clipped"]}
        if c["p"]["reference"] == "memory":
            psi = np.asarray(c["J_memory"], float) / (c["p"]["N_mem"] * design["base"]["d"])
            Dm, vm = diffusion(psi, design), float((psi[2] - psi[0]).mean() / (t2 - t0))
            desc[n].update(v_memory=vm, D_time_memory=finite(float(Dm) / vm ** 2) if vm else None)
    print(json.dumps({"verdicts": verdicts, "descriptives": desc}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
