"""Calibration probes of q5-bound (seeds from the namespace calibration:…, never a registration's).

    python -I q5-bound/calib.py NAME     runs the probe set NAME below; prints a line per probe, writes calibration/NAME.json"""
import hashlib, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import run

HERE = Path(__file__).resolve().parent
BASE = {"d": 2, "dt": 1.0, "S": 4, "gamma": 0.5, "kappa": 1.0, "g_lat": 1.0, "g_pin": 0.5, "m_pin": 2, "F": 1.3, "sigma": 0.3,
        "N": 32, "reference": "alone", "k": 0.0, "v_star": 0.0, "tau": 0.0, "L": 16, "K": 2.0, "gamma_line": 2.0, "n_s": 0,
        "Lambda": 0.0, "gear": 1.0}


def seed(*a):
    return int(hashlib.sha256(":".join(map(str, ("calibration",) + a)).encode()).hexdigest()[:12], 16)


def probe(args):
    name, kw, B, t0, lags, every = args
    p = {**BASE, **kw}
    t = time.time()
    J, Q, Ql, lag, clip = run(p, B, t0 + max(lags), every, seed(name, json.dumps(p, sort_keys=True)),
                              seed(name, json.dumps(p, sort_keys=True), "line"))
    k0 = int(round(t0 / every)) - 1
    phi = J / (p["N"] * p["d"])
    Ls = sorted(lags)
    var = {L: float((phi[:, k0 + int(round(L / every))] - phi[:, k0]).var()) for L in Ls}
    D = (var[Ls[-1]] - var[Ls[-2]]) / (2 * (Ls[-1] - Ls[-2]))
    v = float((phi[:, -1] - phi[:, k0]).mean() / Ls[-1])
    s = p["sigma"] ** 2
    S = float((Q[:, -1] - Q[:, k0]).mean() / Ls[-1] / s)
    Sl = float((Ql[:, -1] - Ql[:, k0]).mean() / Ls[-1] / s)
    tail = lag[:, k0:]
    adv = phi[:, -1] - phi[:, k0]
    dev = adv - np.median(adv)
    stalls = int((np.abs(dev) > 10 * max(np.median(np.abs(dev)), 1e-9)).sum())
    return {"name": name, "p": p, "B": B, "t0": t0, "v": v, "D": D, "D_time": D / v ** 2, "var": var, "entropy_rate": S,
            "entropy_rate_line": Sl, "eta_main": v * v / (D * S) if D > 0 else None,
            "eta_total": v * v / (D * (S + Sl)) if D > 0 else None, "lag_mean": float(tail.mean()), "lag_sd": float(tail.std()),
            "outliers": stalls, "clipped": float(clip[:, -1].mean()), "seconds": round(time.time() - t, 1)}


def show(r):
    e = lambda x: "-" if x is None else f"{x:.3f}"
    print(f"{r['name']:<30} v={r['v']:.4f} D={r['D']:.3e} D_time={r['D_time']:.3e} S={r['entropy_rate']:.0f}+{r['entropy_rate_line']:.0f} "
          f"eta_main={e(r['eta_main'])} eta_total={e(r['eta_total'])} lag={r['lag_mean']:.4f}±{r['lag_sd']:.4f} "
          f"out={r['outliers']} clip={r['clipped']:.4f} {r['seconds']}s", flush=True)


SETS = {
    # the crystal alone by friction (and by torque at gamma = 0.5): how near the TUR bound it runs
    "alone": [(f"alone g{g}", {"gamma": g}, 256, 300, (300, 1000), 50) for g in (0.25, 0.5, 1.0, 2.0)]
             + [(f"alone g0.5 F{F}", {"F": F}, 256, 300, (300, 1000), 50) for F in (1.1, 1.6)],
    # q4-line's line unchanged (gamma_line = 2, K = 2, L = 16) reading Phi / r, r = 2.5924 / 0.4885: it moves as in q4-line
    "gear": [(f"gear line n{n} k0", {"reference": "line", "n_s": n, "gear": 5.307}, 256, 300, (300, 1000), 50)
             for n in (8, 32, 128, 512)],
    # the crystal reading itself through the geared line (mu = dv/dF ≈ 2.06: k 0.06 is q4-line's loop gain), and a free
    # exact buffer of the line's delay
    "gain": [(f"gear line n{n} k{k}", {"reference": "line", "n_s": n, "gear": 5.307, "k": k, "Lambda": lam}, 256, 300, (300, 1000), 50)
             for n, lam in ((8, 8.0659), (32, 8.0030), (128, 7.9876), (512, 7.9857)) for k in (0.02, 0.04, 0.06, 0.09)]
            + [(f"ideal tau16.5 k{k}", {"reference": "ideal", "tau": 16.5, "k": k, "v_star": 2.5924}, 256, 300, (300, 1000), 50)
               for k in (0.02, 0.04, 0.06, 0.09)],
}


def full():
    """The registered design with calibration seeds, analysed by the registered analysis: the intervals the confirmatory
    run will be judged by, before choosing its thresholds (q4-line's lesson). Data under the ignored runs/."""
    import shutil, subprocess
    from run import cell_params, one
    reg = json.loads((HERE / "registration.json").read_text())
    rid, rnd = "calibration-" + "0" * 52, "calibration-full"
    work = [(reg["design"], c, rid, rnd) for c in reg["design"]["cells"]]
    with ProcessPoolExecutor(int(os.environ.get("JOBS", "6"))) as pool:
        cells = dict(pool.map(one, work))
    d = HERE / "runs" / "calibration-full"
    (d / "results").mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / "registration.json", d)
    (d / "registration.lock.json").write_text(json.dumps({"registration": rid}))
    (d / "results" / "data.json").write_text(json.dumps({"registration": rid, "cells": cells}))
    a = subprocess.run([sys.executable, "-I", str(HERE / "analyze.py"), str(d)], capture_output=True, text=True, check=True).stdout
    (HERE / "calibration" / "full-analysis.json").write_text(a)
    print(a)


def main(name):
    if name == "full":
        return full()
    out = []
    with ProcessPoolExecutor(int(os.environ.get("JOBS", "4"))) as pool:
        for r in pool.map(probe, SETS[name]):
            show(r)
            out.append(r)
    (HERE / "calibration" / f"{name}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
