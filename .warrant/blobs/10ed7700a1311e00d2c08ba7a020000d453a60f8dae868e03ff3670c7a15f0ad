"""q6-renewal: what a clock is worth in a world whose events come at intervals from memoryless to periodic, and what it
costs to get that worth.

The world: events whose intervals are independent and gamma-distributed with mean T and shape k (k = 1: Poisson, a world
with no memory; larger k: more regular; no k: periodic), from a uniform offset in [0, T). Event times fall on the
sub-step grid (h = dt / S).

The clock: q5-bound's crystal alone (gamma = 0.5, near the TUR bound; q5-bound/run.py, runner blob bf92b355) of N phases
per dimension. The clock does not act on the world, and every clock reads the same worlds. The agent stores the clock's
phase at each event; at a query moment it reads the elapsed time

    a_hat = (Phi(now) - Phi(last event)) / v*,    v* the clock's calibrated speed

and predicts the time to the next event by the world's mean residual life at a_hat, m_k(a) = E[tau - a | tau > a]. On the
same worlds and queries:
    perfect   m_k at the true elapsed time a (what the world lets a perfect clock anticipate)
    none      no clock: the mean residual life of a random moment, T (1 + 1/k) / 2 (T / 2 when periodic)
In a Poisson world m_1 is the constant T, so a clock changes no prediction: that is memorylessness, not a finding.

    python -I q6-renewal/run.py [--jobs K]      the registered cells, seeded from the registration and the beacon: results/data.json"""
import hashlib, json, math, platform, sys
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
    """B trajectories of one crystal from rest, stepped one BAOAB sub-step at a time (q5-bound's, without its torque
    correction: the torque is the constant F)."""

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


def residual_life(k, T):
    """m(a) = E[tau - a | tau > a] of the world's intervals as a function of the elapsed time a (array in, array out). For
    gamma of shape k and scale theta = T / k, with x = a / theta: m = theta sum_{i<k} (k - i) x^i / i! / sum_{i<k} x^i / i!,
    tabulated on [0, T (1 + 10 / sqrt k)] and held at the last value beyond it; periodic: max(T - a, 0)."""
    if k is None:
        return lambda a: np.maximum(T - a, 0.0)
    theta = T / k
    a = np.linspace(0.0, T * (1 + 10 / math.sqrt(k)), 4001)
    x = np.maximum(a[1:] / theta, 1e-300)
    i = np.arange(k)
    lw = np.log(x)[:, None] * i[None, :] - np.array([math.lgamma(j + 1) for j in i])[None, :]
    w = np.exp(lw - lw.max(1, keepdims=True))
    m = np.concatenate([[T], theta * (w * (k - i)).sum(1) / w.sum(1)])
    return lambda q: np.interp(q, a, m)


def world(k, T, B, steps, h, seed):
    """Each trajectory's event indices on the sub-step grid, sorted, past the end: a list of B int arrays."""
    rng = np.random.Generator(np.random.PCG64(seed))
    t_total = steps * h
    M = int(1.5 * t_total / T) + 64
    tau = np.full((B, M), float(T)) if k is None else rng.gamma(k, T / k, (B, M))
    t = rng.uniform(0, T, (B, 1)) + np.concatenate([np.zeros((B, 1)), np.cumsum(tau[:, :-1], 1)], 1)
    if not (t[:, -1] > t_total + T).all():
        raise RuntimeError("a world ended before the trajectory")
    return [np.rint(row / h).astype(np.int64) for row in t]


def readout(phi, events, queries, v, m, h):
    """For each trajectory, the sums over queries of the squared error and of the error of the prediction m(a_hat) of the
    time to the next event, a_hat = (phi(q) - phi(last event)) / v (phi None: the true elapsed time a), and of a itself.
    Three (B,) arrays."""
    B = len(events)
    sse, s, sa = np.zeros(B), np.zeros(B), np.zeros(B)
    for b in range(B):
        e = events[b]
        j = np.searchsorted(e, queries, side="right") - 1
        last, nxt = e[j], e[j + 1]
        a = (queries - last) * h
        err = m(a if phi is None else (phi[b, queries] - phi[b, last]) / v) - (nxt - queries) * h
        sse[b], s[b], sa[b] = (err * err).sum(), err.sum(), a.sum()
    return sse, s, sa


def seed(registration, randomness, name):
    # the registered seed rule (registration.json `seeds`, METHOD §3)
    return int(hashlib.sha256(f"{registration}:{randomness}:{name}".encode()).hexdigest()[:12], 16)


def grid(design):
    h = design["base"]["dt"] / design["base"]["S"]
    steps = int(round(design["t_total"] / h))
    queries = np.arange(int(round(design["warm"] / h)), steps + 1, int(round(design["query_every"] / h)))
    return h, steps, queries


def worlds(design, rid, rnd):
    h, steps, _ = grid(design)
    return {w["name"]: world(w["k"], design["T"], design["trajectories"], steps, h, seed(rid, rnd, "world:" + w["name"]))
            for w in design["worlds"]}


def one(args):
    """One clock (a crystal of N phases per dimension, or `None` for the world-only references) over all worlds."""
    design, clock, rid, rnd = args
    h, steps, queries = grid(design)
    T, B = design["T"], design["trajectories"]
    ev = worlds(design, rid, rnd)
    out = {}
    if clock is None:
        for w in design["worlds"]:
            m = residual_life(w["k"], T)
            none = T / 2 if w["k"] is None else T * (1 + 1 / w["k"]) / 2
            sse, s, sa = readout(None, ev[w["name"]], queries, 1.0, m, h)
            sse0, s0, _ = readout(None, ev[w["name"]], queries, 1.0, lambda a: np.full_like(a, none), h)
            out[w["name"]] = {"sse_perfect": sse.tolist(), "sum_perfect": s.tolist(), "sse_none": sse0.tolist(),
                              "sum_none": s0.tolist(), "sum_elapsed": sa.tolist(), "n": [len(queries)] * B}
        return "worlds", out
    p = {**design["base"], "N": clock["N"]}
    phi, J, Q = trace(p, B, design["t_total"], design["checkpoints"], seed(rid, rnd, clock["name"]))
    out = {"p": p, "J": J.T.tolist(), "heat": Q.T.tolist(), "worlds": {}}
    for w in design["worlds"]:
        sse, s, _ = readout(phi, ev[w["name"]], queries, clock["v_star"], residual_life(w["k"], T), h)
        out["worlds"][w["name"]] = {"sse": sse.tolist(), "sum": s.tolist()}
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
