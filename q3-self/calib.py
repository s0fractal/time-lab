"""Calibration probes of q3-self (seeds from the namespace calibration:…, never a registration's).

    python -I q3-self/calib.py NAME     runs the probe set NAME below; prints a line per probe, writes calibration/NAME.json"""
import hashlib, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as runner
from run import run


class Brownian:
    """A memory that is not a crystal: its mean phase is v t + sqrt(2 D) W(t) (D = 0: a perfect clock through the same
    pointer). Only for calibration: it checks the pointer against the linear theory."""

    def __init__(self, p, N, B, seed):
        self.h, self.v, self.D, self.N, self.p = p["dt"] / p["S"], p["v_mem"], p["brownian"], N, p
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.J, self.heat = np.zeros(B), np.zeros(B)

    def drift(self):
        step = self.v * self.h + np.sqrt(2 * self.D * self.h) * self.rng.standard_normal(self.J.shape)
        self.J += step * self.N * self.p["d"]

    def kick(self, F):
        pass

    def phase(self):
        return self.J / (self.N * self.p["d"])


class Memory:
    def __new__(cls, p, N, B, seed):
        return Brownian(p, N, B, seed) if "brownian" in p else runner.Crystal(p, N, B, seed)


_Crystal = runner.Crystal

HERE = Path(__file__).resolve().parent
BASE = {"d": 2, "dt": 1.0, "S": 4, "gamma": 2.0, "kappa": 1.0, "g_lat": 1.0, "g_pin": 0.5, "m_pin": 2, "F": 1.3, "sigma": 0.3,
        "N": 32, "reference": "alone", "tau": 0.0, "N_mem": 0, "k": 0.0, "v_star": 0.489, "v_mem": 0.489}


def seed(*a):
    return int(hashlib.sha256(":".join(map(str, ("calibration",) + a)).encode()).hexdigest()[:12], 16)


def probe(args):
    name, kw, B, t0, lags, every = args
    p = {**BASE, **kw}
    if "brownian" in p:                                  # the memory is built second: the main stays a crystal
        made = []
        runner.Crystal = lambda q, N, B_, s: (made.append(1), _Crystal(q, N, B_, s) if len(made) == 1 else Brownian(q, N, B_, s))[1]
    t = time.time()
    J, Q, Jm, Qm, clip = run(p, B, t0 + max(lags), every, seed(name, json.dumps(p, sort_keys=True)),
                         seed(name, json.dumps(p, sort_keys=True), "memory"))
    nd = p["N"] * p["d"]
    k0 = int(round(t0 / every)) - 1
    phi = J / nd
    var = {L: float((phi[:, k0 + int(round(L / every))] - phi[:, k0]).var()) for L in lags}
    Ls = sorted(lags)
    D = (var[Ls[-1]] - var[Ls[-2]]) / (2 * (Ls[-1] - Ls[-2]))
    v = float((phi[:, -1] - phi[:, k0]).mean() / Ls[-1])
    if p["reference"] == "memory":
        psi = Jm / (p["N_mem"] * p["d"])
        vm = {L: float((psi[:, k0 + int(round(L / every))] - psi[:, k0]).var()) for L in lags}
        D_mem = (vm[Ls[-1]] - vm[Ls[-2]]) / (2 * (Ls[-1] - Ls[-2]))
        v_m = float((psi[:, -1] - psi[:, k0]).mean() / Ls[-1])
        mem = {"v_memory": v_m, "D_memory": D_mem, "D_time_memory": D_mem / v_m ** 2,
               "lock": float(np.corrcoef(phi[:, -1] - phi[:, k0], psi[:, -1] - psi[:, k0])[0, 1])}
    else:
        mem = {}
    heat = float((Q[:, -1] - Q[:, k0]).mean() / Ls[-1] / p["sigma"] ** 2)
    heat_m = float((Qm[:, -1] - Qm[:, k0]).mean() / Ls[-1] / p["sigma"] ** 2)
    return {"name": name, "p": p, "B": B, "t0": t0, "v": v, "D_phase": D, "D_time": D / v ** 2 if v else None, "var": var,
            "entropy_rate": heat, "entropy_rate_memory": heat_m, "clipped": float(clip[:, -1].mean()),
            "seconds": round(time.time() - t, 1), **mem}


def show(r):
    print(f"{r['name']:<40} v={r['v']:.4f} D_phase={r['D_phase']:.3e} D_time={r['D_time']:.3e} "
          f"var={ {k: round(x, 4) for k, x in r['var'].items()} } S={r['entropy_rate']:.1f}+{r['entropy_rate_memory']:.1f} "
          f"clip={r['clipped']:.4f} {r['seconds']}s"
          + (f" | memory v={r['v_memory']:.4f} D_time={r['D_time_memory']:.3e} lock={r['lock']:.3f}" if "lock" in r else ""), flush=True)


SETS = {
    # the autonomous crystal by size and the main crystal's mobility dv/dF
    "alone": [(f"alone N{N}", {"N": N}, 256, 200, (200, 500, 800), 10) for N in (8, 32, 128)]
             + [(f"alone N32 F{F}", {"F": F}, 256, 200, (200, 500, 800), 10) for F in (1.2, 1.4)],
    # a perfect delay line by gain k and span tau; a memory crystal by size at k = 1.28, tau = 8 (mu = dv/dF ≈ 0.78:
    # k mu tau = 8). The short span of a memory crystal is left uncalibrated on purpose (H5 is predicted blind).
    "feedback": [("alone T1300", {}, 256, 300, (300, 1000), 50)]
                + [(f"ideal k{k} tau{tau}", {"reference": "ideal", "k": k, "tau": tau}, 256, 300, (300, 1000), 50)
                   for k in (0.64, 1.28, 2.56) for tau in (2.0, 8.0, 32.0)]
                + [(f"memory N{n} k1.28 tau8", {"reference": "memory", "k": 1.28, "tau": 8.0, "N_mem": n, "v_mem": v}, 256, 300,
                    (300, 1000), 50) for n, v in ((8, 0.4909), (32, 0.4885), (128, 0.4881))],
    # the memory's own clock in the same runs, and the large memory at longer lags and other gains
    # the pointer itself: a memory that is a perfect clock (D = 0) and Brownian clocks as precise as the crystals N8/N128
    "pointer": [(f"memory brownian D{D} k1.28 tau8", {"reference": "memory", "k": 1.28, "tau": 8.0, "N_mem": 128, "v_mem": 0.4881,
                 "brownian": D}, 256, 300, (300, 1000), 50) for D in (0.0, 4.76e-4, 7.9e-3)],
    # spans of whole memory ticks (M = n π, the pinning's period): does the misfit of 3.9 rad cause the excess?
    "ticks": [(f"memory N{n} k{k} M{m}pi", {"reference": "memory", "k": k, "tau": m * np.pi / v, "N_mem": n, "v_mem": v}, 256, 300,
               (300, 1000), 50) for n, v in ((8, 0.4909), (32, 0.4885), (128, 0.4881)) for k, m in ((1.28, 1), (0.64, 2))]
             + [(f"ideal k{k} M{m}pi", {"reference": "ideal", "k": k, "tau": m * np.pi / 0.4885}, 256, 300, (300, 1000), 50)
                for k, m in ((1.28, 1), (0.64, 2), (0.32, 4), (0.32, 1))]
             + [("memory N128 k0.32 M4pi", {"reference": "memory", "k": 0.32, "tau": 4 * np.pi / 0.4881, "N_mem": 128, "v_mem": 0.4881},
                 256, 300, (300, 1000), 50),
                ("memory N32 k0.32 M4pi", {"reference": "memory", "k": 0.32, "tau": 4 * np.pi / 0.4885, "N_mem": 32, "v_mem": 0.4885},
                 256, 300, (300, 1000), 50)],
    # the design's gain (k = 0.32): the small memory at four ticks and the large one read between ticks (4.5)
    "design": [("memory N8 k0.32 M4pi", {"reference": "memory", "k": 0.32, "tau": 4 * np.pi / 0.4909, "N_mem": 8, "v_mem": 0.4909},
                256, 300, (300, 1000), 50),
               ("memory N128 k0.32 M4.5pi", {"reference": "memory", "k": 0.32, "tau": 4.5 * np.pi / 0.4881, "N_mem": 128,
                                             "v_mem": 0.4881}, 256, 300, (300, 1000), 50)],
    "memory": [(f"memory N{n} k{k} tau8", {"reference": "memory", "k": k, "tau": 8.0, "N_mem": n, "v_mem": v}, 256, 300,
                (300, 1000), 50) for n, v in ((8, 0.4909), (32, 0.4885), (128, 0.4881)) for k in (0.64, 1.28)]
              + [("memory N128 k1.28 tau8 T2300", {"reference": "memory", "k": 1.28, "tau": 8.0, "N_mem": 128, "v_mem": 0.4881},
                  256, 300, (1000, 2000), 50)],
}


def main(name):
    out = []
    with ProcessPoolExecutor(int(os.environ.get("JOBS", "4"))) as pool:
        for r in pool.map(probe, SETS[name]):
            show(r)
            out.append(r)
    (HERE / "calibration").mkdir(exist_ok=True)
    (HERE / "calibration" / f"{name}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
