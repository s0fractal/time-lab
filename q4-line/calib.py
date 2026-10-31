"""Calibration probes of q4-line (seeds from the namespace calibration:…, never a registration's).

    python -I q4-line/calib.py NAME     runs the probe set NAME below; prints a line per probe, writes calibration/NAME.json"""
import hashlib, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import run

HERE = Path(__file__).resolve().parent
BASE = {"d": 2, "dt": 1.0, "S": 4, "gamma": 2.0, "kappa": 1.0, "g_lat": 1.0, "g_pin": 0.5, "m_pin": 2, "F": 1.3, "sigma": 0.3,
        "N": 32, "reference": "alone", "k": 0.0, "v_star": 0.4885, "tau": 0.0, "L": 16, "K": 2.0, "n_s": 0, "Lambda": 0.0}


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
    return {"name": name, "p": p, "B": B, "t0": t0, "v": v, "D": D, "D_time": D / v ** 2, "var": var, "entropy_rate": S,
            "entropy_rate_line": Sl, "eta_main": v * v / (D * S) if D > 0 else None,
            "eta_total": v * v / (D * (S + Sl)) if D > 0 else None, "lag_mean": float(tail.mean()), "lag_sd": float(tail.std()),
            "clipped": float(clip[:, -1].mean()), "seconds": round(time.time() - t, 1)}


def show(r):
    e = lambda x: "-" if x is None else f"{x:.3f}"
    print(f"{r['name']:<28} v={r['v']:.4f} D={r['D']:.3e} D_time={r['D_time']:.3e} S={r['entropy_rate']:.0f}+{r['entropy_rate_line']:.0f} "
          f"eta_main={e(r['eta_main'])} eta_total={e(r['eta_total'])} lag={r['lag_mean']:.4f}±{r['lag_sd']:.4f} "
          f"clip={r['clipped']:.4f} {r['seconds']}s", flush=True)


SETS = {
    # the line hanging on the free crystal (k = 0): its mean lag, the lag's spread and the line's heat, by stage size
    "lag": [("alone", {}, 256, 300, (300, 1000), 50)]
           + [(f"line n{n} k0", {"reference": "line", "n_s": n}, 256, 300, (300, 1000), 50) for n in (1, 4, 16, 64, 256)],
    # the crystal reading itself through the line (Lambda* = the k = 0 mean lag), and a free exact buffer of the line's delay
    "gain": [(f"line n{n} k{k}", {"reference": "line", "n_s": n, "k": k, "Lambda": lam}, 256, 300, (300, 1000), 50)
             for n, lam in ((4, 8.1865), (16, 8.0557), (64, 8.0294), (256, 8.0205)) for k in (0.16, 0.32, 0.64)]
            + [(f"ideal tau16.5 k{k}", {"reference": "ideal", "tau": 16.5, "k": k}, 256, 300, (300, 1000), 50) for k in (0.16, 0.32, 0.64)],
    # the stable range of k, the stage sizes around the slipping one, and free exact buffers of 16.5 and 18.3 (dLambda/dv)
    "k": [(f"line n{n} k{k}", {"reference": "line", "n_s": n, "k": k, "Lambda": lam}, 256, 300, (300, 1000), 50)
          for n, lam in ((8, 8.10), (16, 8.0557), (64, 8.0294)) for k in (0.08, 0.12, 0.16, 0.2, 0.24)]
         + [(f"line n4 k{k}", {"reference": "line", "n_s": 4, "k": k, "Lambda": 8.1865}, 256, 300, (300, 1000), 50) for k in (0.04, 0.08)]
         + [(f"line n8 k0", {"reference": "line", "n_s": 8}, 256, 300, (300, 1000), 50)]
         + [(f"ideal tau{tau} k{k}", {"reference": "ideal", "tau": tau, "k": k}, 256, 300, (300, 1000), 50)
            for tau in (16.5, 18.25) for k in (0.08, 0.12, 0.2, 0.24)],
    # at the size of the registered run (1024 trajectories, checkpoints 400 / 800 / 1800)
    "full": [("alone", {}, 1024, 400, (400, 1400), 50), ("ideal tau16.5 k0.16", {"reference": "ideal", "tau": 16.5, "k": 0.16}, 1024, 400, (400, 1400), 50)]
            + [(f"line n{n} k{k}", {"reference": "line", "n_s": n, "k": k, "Lambda": lam}, 1024, 400, (400, 1400), 50)
               for n, lam in ((8, 8.1019), (16, 8.0557), (32, 8.042)) for k in (0.12, 0.16)]
            + [("line n4 k0.16", {"reference": "line", "n_s": 4, "k": 0.16, "Lambda": 8.1865}, 1024, 400, (400, 1400), 50)],
}


def main(name):
    out = []
    with ProcessPoolExecutor(int(os.environ.get("JOBS", "4"))) as pool:
        for r in pool.map(probe, SETS[name]):
            show(r)
            out.append(r)
    (HERE / "calibration" / f"{name}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
