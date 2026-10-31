"""q2-crystal registered analysis: how much a world's rhythm sharpens the crystal clock, and what that depends on.

    python -I q2-crystal/analyze.py DIR        prints {verdicts, descriptives} as JSON; refuses incomplete data

For each cell, over the window [t1, t2]: per-phase velocity v = mean(ΔJ) / (Δt N d) and diffusion D = var(ΔJ) / (2 Δt N d).
The lock gain of a (coupling, N, sigma, j) is G = D without the rhythm (F1 = 0) / D with it (F1 > 0). Each hypothesis is a
log10 ratio of gains, with a 95% percentile interval over resamples of the trajectories of every cell, seeded from the
registration id. Its bytes are the registration's `analysis` blob."""
import hashlib, json, math, sys
from pathlib import Path

import numpy as np

B = 2000
LOG2, LOG5 = math.log10(2), math.log10(5)


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
    for name, c in want.items():
        dJ = np.asarray(data["cells"][name]["dJ"], float)
        if dJ.shape != (design["trajectories"],) or not np.isfinite(dJ).all():
            refuse(f"{name}: {dJ.shape[0]} values, finite {bool(np.isfinite(dJ).all())}")
        cells[name] = (c, dJ)
    return design, lock["registration"], cells


def name(coupling, N, sigma, j, F1):
    return f"{coupling}-N{N}-s{sigma}-j{j}-F1{F1}"


def gain(dJ, design, coupling, N, sigma, j):
    """D without the rhythm over D with it, both from the (resampled) ΔJ of each cell."""
    off, on = dJ[name(coupling, N, sigma, j, 0.0)], dJ[name(coupling, N, sigma, j, design["F1"])]
    return off.var() / on.var() if on.var() > 0 else math.inf


def hypotheses(design):
    """{H: (numerator gains, denominator gains)}: each a log10 ratio of lock gains."""
    return {
        "H1-size-2-8": (("crystal", 8, 0.3, 0.0), ("crystal", 2, 0.3, 0.0)),
        "H1-size-8-32": (("crystal", 32, 0.3, 0.0), ("crystal", 8, 0.3, 0.0)),
        "H2-coupling": (("crystal", 32, 0.3, 0.0), ("loose", 32, 0.3, 0.0)),
        "H3-noise": (("crystal", 32, 0.5, 0.0), ("crystal", 8, 0.5, 0.0)),
        "H4-gyroscope": (("crystal", 32, 0.3, 0.1), ("crystal", 32, 0.3, 0.0)),
    }


def ratios(dJ, design):
    out = {}
    for h, (a, b) in hypotheses(design).items():
        ga, gb = gain(dJ, design, *a), gain(dJ, design, *b)
        out[h] = math.log10(ga / gb) if 0 < ga < math.inf and 0 < gb < math.inf else math.nan
    return out


def judge(ci):
    lo = {h: c[0] for h, c in ci.items()}
    hi = {h: c[1] for h, c in ci.items()}
    if any(math.isnan(x) for c in ci.values() for x in c):
        bad = sorted(h for h, c in ci.items() if any(math.isnan(x) for x in c))
        return {k: {"verdict": "inconclusive", "reason": f"a gain is undefined in {bad}"} for k in ("H1", "H2", "H3", "H4")}
    sizes = ("H1-size-2-8", "H1-size-8-32")
    return {
        "H1": "supported" if all(lo[h] > LOG2 for h in sizes) else "contradicted" if any(hi[h] < LOG2 for h in sizes) else "inconclusive",
        "H2": "supported" if lo["H2-coupling"] > 1 else "contradicted" if hi["H2-coupling"] < 1 else "inconclusive",
        "H3": "supported" if lo["H3-noise"] > LOG5 else "contradicted" if hi["H3-noise"] < LOG5 else "inconclusive",
        "H4": "supported" if hi["H4-gyroscope"] < -LOG2 else "contradicted" if lo["H4-gyroscope"] > -LOG2 else "inconclusive",
    }


def main(d):
    design, rid, cells = load(Path(d))
    dJ = {n: x for n, (_, x) in cells.items()}
    point = ratios(dJ, design)
    rng = np.random.Generator(np.random.PCG64(int(hashlib.sha256(f"{rid}:bootstrap".encode()).hexdigest()[:16], 16)))
    boots = {h: [] for h in point}
    for _ in range(B):
        r = ratios({n: x[rng.integers(0, len(x), len(x))] for n, x in sorted(dJ.items())}, design)
        for h, v in r.items():
            boots[h].append(v)
    ci = {h: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if not np.isnan(v).any() else [math.nan, math.nan]
          for h, v in boots.items()}
    verdict = judge(ci)
    parts = {"H1": ("H1-size-2-8", "H1-size-8-32"), "H2": ("H2-coupling",), "H3": ("H3-noise",), "H4": ("H4-gyroscope",)}
    verdicts = {}
    for h, keys in parts.items():
        v = verdict[h] if isinstance(verdict[h], dict) else {"verdict": verdict[h]}
        verdicts[h] = {**v, **{k: {"log10_ratio": finite(point[k]), "ci95": [finite(x) for x in ci[k]]} for k in keys}}
    t1, t2 = design["window"]
    desc = {}
    for n, (c, x) in cells.items():
        phases = c["N"] * design["base"]["d"]
        desc[n] = {"v_per_phase": float(x.mean() / ((t2 - t1) * phases)), "D_per_phase": float(x.var() / (2 * (t2 - t1) * phases))}
    for coupling in ("crystal", "loose"):
        for N in (2, 8, 32):
            for sigma in (0.3, 0.5):
                desc[f"G:{coupling}-N{N}-s{sigma}-j0.0"] = finite(gain(dJ, design, coupling, N, sigma, 0.0))
    desc["G:crystal-N32-s0.3-j0.1"] = finite(gain(dJ, design, "crystal", 32, 0.3, 0.1))
    print(json.dumps({"verdicts": verdicts, "descriptives": desc}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
