"""q7-duration: how long has it been? A clock's second use. The agent starts at a world event and, at a later event, says
how much time has passed since the start, by counting events, by its own clock, or by both.

The world: events at intervals independent and gamma-distributed with mean T and shape k (k = 1: Poisson; larger k: more
regular), the first at t0 (when the clock has run from rest for t0), on the sub-step grid (h = dt / S).

The clock: q6-renewal's crystal alone (q6-renewal/run.py, runner blob 10ed7700; q5-bound's alone cell, gamma = 0.5, near
the TUR bound) of N phases per dimension. The clock does not act on the world, and every clock reads the same worlds.

At the m-th event after the start, with s the true span:
    count   m T: the world as the clock (free, and as good as the world is regular: var m T² / k)
    clock   (Phi(t_m) - Phi(t0)) / v*: the agent's own clock (var 2 (D/v²) s, at the entropy it produces)
    fused   w clock + (1 - w) count,  w = (T / k) / (T / k + 2 D*):  each weighted by the inverse of its variance per
            unit time, D* = D/v² as q6-renewal measured this clock

    python -I q7-duration/run.py [--jobs K]      the registered cells, seeded from the registration and the beacon: results/data.json"""
import hashlib, json, platform, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def attention(theta, kappa):
    d = theta.shape[-1]
    c, s = np.cos(theta), np.sin(theta)
    C = (c @ np.swapaxes(c, -1, -2) + s @ np.swapaxes(s, -1, -2)) / d
    C = kappa * C
    A = np.exp(C - C.max(-1, keepdims=True))
    A /= A.sum(-1, keepdims=True)
    A = 0.5 * (A + np.swapaxes(A, -1, -2))
    return -(kappa / d) * (s * (A @ c) - c * (A @ s))


def force(theta, p):
    f = attention(theta, p["kappa"]) if p["kappa"] else np.zeros_like(theta)
    fwd = np.roll(theta, -1, axis=-2) - theta
    bwd = theta - np.roll(theta, 1, axis=-2)
    f = f + p["g_lat"] * (np.sin(fwd) - np.sin(bwd)) - p["m_pin"] * p["g_pin"] * np.sin(p["m_pin"] * theta)
    return f + p["F"]


class Crystal:
    """B trajectories of one crystal from rest, stepped one BAOAB sub-step at a time (q6-renewal's, which is q5-bound's alone
    cell)."""

    def __init__(self, p, N, B, seed):
        self.p, self.N, self.h = p, N, p["dt"] / p["S"]
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.c1 = np.exp(-p["gamma"] * self.h)
        self.c2 = np.sqrt(p["sigma"] ** 2 * (1 - self.c1 ** 2))
        self.theta = np.zeros((B, N, p["d"]))
        self.P = np.zeros((B, N, p["d"]))
        self.J = np.zeros(B)
        self.heat = np.zeros(B)
        self.f = force(self.theta, p)

    def step(self):
        """B(h/2) A(h/2) O(h) A(h/2) B(h/2), the force at the new positions carried to the next sub-step."""
        h = self.h
        self.P = self.P + 0.5 * h * self.f
        self.theta = self.theta + 0.5 * h * self.P
        self.J += 0.5 * h * self.P.sum((-2, -1))
        k0 = 0.5 * (self.P ** 2).sum((-2, -1))
        self.P = self.c1 * self.P + self.c2 * self.rng.standard_normal(self.P.shape)
        self.heat += k0 - 0.5 * (self.P ** 2).sum((-2, -1))
        self.theta = self.theta + 0.5 * h * self.P
        self.J += 0.5 * h * self.P.sum((-2, -1))
        self.theta = np.remainder(self.theta + np.pi, 2 * np.pi) - np.pi
        self.f = force(self.theta, self.p)
        self.P = self.P + 0.5 * h * self.f


def trace(p, B, t_total, checkpoints, seed):
    """The crystal's mean phase Phi = J / (N d) at every sub-step from 0 to t_total, shape (B, steps + 1), and its current
    and heat at the checkpoints, two (B, 3) arrays."""
    h = p["dt"] / p["S"]
    steps = int(round(t_total / h))
    if list(checkpoints) != sorted(set(checkpoints)) or checkpoints[-1] > t_total:
        raise ValueError(f"checkpoints {checkpoints} are not increasing within {t_total}")
    at = {int(round(t / h)): i for i, t in enumerate(checkpoints)}
    c = Crystal(p, p["N"], B, seed)
    phi = np.zeros((B, steps + 1))
    J, Q = np.zeros((B, len(checkpoints))), np.zeros((B, len(checkpoints)))
    for i in range(1, steps + 1):
        c.step()
        phi[:, i] = c.J / (p["N"] * p["d"])
        if i in at:
            J[:, at[i]], Q[:, at[i]] = c.J, c.heat
    return phi, J, Q


def world(k, T, B, start, steps, h, m_max, seed):
    """Each trajectory's first m_max + 1 event indices on the sub-step grid, the first at `start`: a (B, m_max + 1) int
    array. Refused if an event the queries need falls past the trajectory."""
    rng = np.random.Generator(np.random.PCG64(seed))
    tau = rng.gamma(k, T / k, (B, m_max))
    t = start + np.concatenate([np.zeros((B, 1)), np.cumsum(tau, 1)], 1)
    e = np.rint(t / h).astype(np.int64)
    if (e[:, -1] > steps).any():
        raise RuntimeError("an event the queries need falls past the trajectory")
    return e


def seed(registration, randomness, name):
    # the registered seed rule (registration.json `seeds`, METHOD §3)
    return int(hashlib.sha256(f"{registration}:{randomness}:{name}".encode()).hexdigest()[:12], 16)


def worlds(design, rid, rnd):
    h = design["base"]["dt"] / design["base"]["S"]
    steps = int(round(design["t_total"] / h))
    return {w["name"]: world(w["k"], design["T"], design["trajectories"], design["start"], steps, h, max(design["queries"]),
                             seed(rid, rnd, "world:" + w["name"])) for w in design["worlds"]}


def one(args):
    """One clock over all worlds, or (clock None) the worlds alone: per world and query m, each trajectory's error of
    each estimate of the span."""
    design, clock, rid, rnd = args
    h, T, B = design["base"]["dt"] / design["base"]["S"], design["T"], design["trajectories"]
    ev = worlds(design, rid, rnd)
    rows = lambda x: x.tolist()
    if clock is None:
        out = {}
        for w in design["worlds"]:
            e = ev[w["name"]]
            out[w["name"]] = {"span": {}, "err_count": {}}
            for m in design["queries"]:
                s = (e[:, m] - e[:, 0]) * h
                out[w["name"]]["span"][str(m)] = rows(s)
                out[w["name"]]["err_count"][str(m)] = rows(m * T - s)
        return "worlds", out
    p = {**design["base"], "N": clock["N"]}
    phi, J, Q = trace(p, B, design["t_total"], design["checkpoints"], seed(rid, rnd, clock["name"]))
    out = {"p": p, "J": J.T.tolist(), "heat": Q.T.tolist(), "worlds": {}}
    b = np.arange(B)
    for w in design["worlds"]:
        e = ev[w["name"]]
        wt = (T / w["k"]) / (T / w["k"] + 2 * clock["D_star"])
        got = {"err_clock": {}, "err_fused": {}}
        for m in design["queries"]:
            s = (e[:, m] - e[:, 0]) * h
            c = (phi[b, e[:, m]] - phi[b, e[:, 0]]) / clock["v_star"]
            got["err_clock"][str(m)] = rows(c - s)
            got["err_fused"][str(m)] = rows(wt * c + (1 - wt) * m * T - s)
        out["worlds"][w["name"]] = got
    return clock["name"], out


def main():
    here = Path(__file__).resolve().parent
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 1
    reg = json.loads((here / "registration.json").read_text())
    rid = json.loads((here / "registration.lock.json").read_text())["registration"]
    beacon = json.loads((here / "beacon.json").read_text())
    rnd, design = beacon["beacon"]["randomness"], reg["design"]
    work = [(design, c, rid, rnd) for c in [None] + design["clocks"]]
    res = {}
    with ProcessPoolExecutor(jobs) as pool:
        for name, out in pool.map(one, work):
            res[name] = out
            print(name, flush=True)
    data = {"registration": rid, "beacon": beacon["attest"], "randomness": rnd,
            "env": {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine()},
            "worlds": res["worlds"], "clocks": {c["name"]: res[c["name"]] for c in design["clocks"]}}
    (here / "results").mkdir(exist_ok=True)
    (here / "results" / "data.json").write_text(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()
