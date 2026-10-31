"""Calibration probes of q6-renewal (seeds from the namespace calibration:…, never a registration's).

    python -I q6-renewal/calib.py NAME     runs the probe set NAME below; prints a line per probe, writes calibration/NAME.json"""
import hashlib, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import readout, residual_life, trace, world

HERE = Path(__file__).resolve().parent
BASE = {"d": 2, "dt": 1.0, "S": 4, "gamma": 0.5, "kappa": 1.0, "g_lat": 1.0, "g_pin": 0.5, "m_pin": 2, "F": 1.3, "sigma": 0.3}


def twin(v, D, B, steps, h, seed):
    """B Brownian clocks Phi = v t + sqrt(2 D) W(t) on the sub-step grid, shape (B, steps + 1): the crystal's diffusion
    without its ticks or its inertia."""
    rng = np.random.Generator(np.random.PCG64(seed))
    phi = np.zeros((B, steps + 1))
    phi[:, 1:] = np.cumsum(rng.standard_normal((B, steps)) * np.sqrt(2 * D * h) + v * h, 1)
    return phi


def seed(*a):
    return int(hashlib.sha256(":".join(map(str, ("calibration",) + a)).encode()).hexdigest()[:12], 16)


def clock(args):
    """The crystal alone of N phases: v, the slope-of-variance D, the entropy rate, eta, and the reading error of an
    elapsed time L from any sub-step: var((Phi(t + L) - Phi(t)) / v) against the diffusive 2 D L / v²."""
    N, B, t0, t1, t2 = args
    p = {**BASE, "N": N}
    t = time.time()
    phi, J, Q = trace(p, B, t2, (t0, t1, t2), seed("clock", N))
    h = p["dt"] / p["S"]
    x = J / (N * p["d"])
    D = ((x[:, 2] - x[:, 0]).var() - (x[:, 1] - x[:, 0]).var()) / (2 * (t2 - t1))
    v = (x[:, 2] - x[:, 0]).mean() / (t2 - t0)
    S = (Q[:, 2] - Q[:, 0]).mean() / (p["sigma"] ** 2 * (t2 - t0))
    i0 = int(round(t0 / h))
    read = {}
    for L in (0.25, 0.5, 1, 2, 5, 10, 20, 50, 100):
        n = int(round(L / h))
        d = (phi[:, i0 + n:] - phi[:, i0:-n]) / v - L
        read[L] = {"bias": float(d.mean()), "var": float(d.var()), "diffusive": float(2 * D * L / v ** 2)}
    return {"N": N, "B": B, "v": float(v), "D": float(D), "D_time": float(D / v ** 2), "entropy_rate": float(S),
            "eta": float(v * v / (D * S)), "read": read, "seconds": round(time.time() - t, 1)}


def show(r):
    reads = " ".join(f"L{L}:{x['var']:.2e}/{x['diffusive']:.2e}" for L, x in r["read"].items())
    print(f"N={r['N']:<3} v={r['v']:.4f} D={r['D']:.3e} D_time={r['D_time']:.3e} S={r['entropy_rate']:.0f} eta={r['eta']:.3f} "
          f"{r['seconds']}s\n    read var / diffusive: {reads}", flush=True)


def anticipation(args):
    """One crystal of N phases (and its Brownian twin, v and D from calibration/clocks.json) read in worlds of shape k:
    the mean squared error of the predicted time to the next event, against a perfect clock and no clock."""
    N, B, T, ks, warm, t_total, q_every = args
    p = {**BASE, "N": N}
    h = p["dt"] / p["S"]
    steps = int(round(t_total / h))
    queries = np.arange(int(round(warm / h)), steps + 1, int(round(q_every / h)))
    cal = {c["N"]: c for c in json.loads((HERE / "calibration" / "clocks.json").read_text())}[N]
    t = time.time()
    phi, _, Q = trace(p, B, t_total, (warm, (warm + t_total) / 2, t_total), seed("anticipation", N, T))
    tw = twin(cal["v"], cal["D"], B, steps, h, seed("anticipation-twin", N, T))
    S = float((Q[:, 2] - Q[:, 0]).mean() / (p["sigma"] ** 2 * (t_total - warm)))
    rows = {}
    for k in ks:
        ev = world(k, T, B, steps, h, seed("anticipation-world", T, k))
        m = residual_life(k, T)
        none = T / 2 if k is None else T * (1 + 1 / k) / 2
        n = len(queries) * B
        mse = lambda r: float(r[0].sum() / n)
        rows[str(k)] = {"clock": mse(readout(phi, ev, queries, cal["v"], m, h)), "twin": mse(readout(tw, ev, queries, cal["v"], m, h)),
                        "perfect": mse(readout(None, ev, queries, 1.0, m, h)),
                        "none": mse(readout(None, ev, queries, 1.0, lambda a: np.full_like(a, none), h))}
    return {"N": N, "T": T, "B": B, "entropy_rate": S, "entropy_per_interval": S * T, "D_time": cal["D_time"], "rows": rows,
            "seconds": round(time.time() - t, 1)}


def show_anticipation(r):
    print(f"N={r['N']} T={r['T']} E/interval={r['entropy_per_interval']:.0f} {r['seconds']}s", flush=True)
    for k, x in r["rows"].items():
        A, Ap = x["none"] / x["clock"], x["none"] / x["perfect"] if x["perfect"] > 0 else float("inf")
        print(f"    k={k:<5} mse clock={x['clock']:.4g} twin={x['twin']:.4g} perfect={x['perfect']:.4g} none={x['none']:.4g} "
              f"A={A:.4g} A_twin={x['none'] / x['twin']:.4g} A_perfect={Ap:.4g}", flush=True)


SETS = {
    # the crystal alone by size: speed, diffusion, price and how well it reads an elapsed time from any moment
    "clocks": [(N, 256, 300, 700, 1300) for N in (1, 2, 4, 8, 16, 32, 64)],
    # anticipation in worlds from Poisson (k = 1) to periodic (None), mean interval T
    "anticipation": [(N, 256, T, (1, 4, 16, 64, 256, 1024, 4096, None), 400, 1800, 1.0) for T in (20,) for N in (1, 4, 16, 64)],
}


def full():
    """The registered design with calibration seeds, analysed by the registered analysis: the intervals the confirmatory
    run will be judged by, before choosing its thresholds (q4-line's lesson). Data under the ignored runs/."""
    import shutil, subprocess
    from run import one
    reg = json.loads((HERE / "registration.json").read_text())
    rid, rnd = "calibration-" + "0" * 52, "calibration-full"
    work = [(reg["design"], c, rid, rnd) for c in [None] + reg["design"]["clocks"]]
    with ProcessPoolExecutor(int(os.environ.get("JOBS", "5"))) as pool:
        res = dict(pool.map(one, work))
    d = HERE / "runs" / "calibration-full"
    (d / "results").mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / "registration.json", d)
    (d / "registration.lock.json").write_text(json.dumps({"registration": rid}))
    (d / "results" / "data.json").write_text(json.dumps({"registration": rid, "worlds": res.pop("worlds"), "clocks": res}))
    a = subprocess.run([sys.executable, "-I", str(HERE / "analyze.py"), str(d)], capture_output=True, text=True, check=True).stdout
    (HERE / "calibration" / "full-analysis.json").write_text(a)
    print(a)


def main(name):
    if name == "full":
        return full()
    out = []
    fn, sh = (anticipation, show_anticipation) if name == "anticipation" else (clock, show)
    with ProcessPoolExecutor(int(os.environ.get("JOBS", "4"))) as pool:
        for r in pool.map(fn, SETS[name]):
            sh(r)
            out.append(r)
    (HERE / "calibration" / f"{name}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
