"""q3-self: can the soliton crystal hold its own rhythm by reading its own past, and where does that precision come from?

The crystal is q2-crystal's (copied from q2-crystal/run.py, runner blob eb522783, without the gyroscope), driven by a
constant torque F at temperature sigma². It reads its own mean phase Phi(t) = J(t) / (N d) and corrects its own torque:

    F(t) = F - k (Phi(t) - Phi(past) - v* tau),   clipped to [0, 2F]

"I should have advanced v* tau since tau ago." Where "tau ago" comes from is the factor of the question:
    alone    k = 0: the autonomous clock
    ideal    past = t - tau exactly: a perfect delay line, which keeps the world's time for free
    memory   past = the moment a second, independent crystal of N_mem nodes (the memory: autonomous, same physics, its own
             noise) stood M = v_mem tau radians of its own mean phase behind where it stands now: a delay line whose time
             is kept by a crystal. The main crystal's phase is stored at every sub-step and looked up by the memory's phase.
A span is counted in ticks n (a tick is a slip over the pinning's period, pi): tau = n pi / v* for the perfect delay line
(n of the crystal's own ticks at its natural rate v*), and M = n pi for a memory (n of the memory's ticks; tau = n pi / v_mem
is their expected duration).

State Theta, P of shape (B, N, d) on the cotangent torus, masses 1, temperature sigma². Forces:
    attention  -(kappa/d) sum_j A_sym_ij sin(theta_ik - theta_jk),  A = softmax_j(kappa C), C_ij = (1/d) sum_k cos(...)
    lattice    g_lat [sin(theta_{i+1} - theta_i) - sin(theta_i - theta_{i-1})] - m g_pin sin(m theta_i)
    torque     F(t) on every phase
One sub-step: B(h/2) -> A(h/2) -> O(h) -> A(h/2) -> B(h/2). The current J is the unwrapped phase advance summed over all
N·d phases; the heat is what the O steps take from the kinetic energy.

    python -I q3-self/run.py [--jobs K]      the registered cells, seeded from the registration and the beacon: results/data.json"""
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


def force(theta, p, F):
    """F: the torque on every phase of each trajectory, shape (B,)."""
    f = attention(theta, p["kappa"]) if p["kappa"] else np.zeros_like(theta)
    fwd = np.roll(theta, -1, axis=-2) - theta
    bwd = theta - np.roll(theta, 1, axis=-2)
    f = f + p["g_lat"] * (np.sin(fwd) - np.sin(bwd)) - p["m_pin"] * p["g_pin"] * np.sin(p["m_pin"] * theta)
    return f + F[:, None, None]


class Crystal:
    """B trajectories of one crystal from rest, stepped one BAOAB sub-step at a time."""

    def __init__(self, p, N, B, seed):
        self.p, self.N, self.h = p, N, p["dt"] / p["S"]
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.c1 = np.exp(-p["gamma"] * self.h)
        self.c2 = np.sqrt(p["sigma"] ** 2 * (1 - self.c1 ** 2))
        self.theta = np.zeros((B, N, p["d"]))
        self.P = np.zeros((B, N, p["d"]))
        self.J = np.zeros(B)
        self.heat = np.zeros(B)
        self.f = force(self.theta, p, np.full(B, p["F"]))

    def drift(self):
        """B(h/2) A(h/2) O(h) A(h/2) with the force of the previous sub-step."""
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

    def kick(self, F):
        """The force at the new positions under the torque F (B,), then B(h/2)."""
        self.f = force(self.theta, self.p, F)
        self.P = self.P + 0.5 * self.h * self.f

    def phase(self):
        return self.J / (self.N * self.p["d"])


def run(p, B, t_total, every, seed, mem_seed=None):
    """B trajectories for t_total time units under the cell's reference (p["reference"]: alone, ideal, memory).
    Returns, at each `every` time units, the main crystal's current, its heat, the memory's current and heat (zeros
    without a memory), and the share of sub-steps whose torque was clipped so far: five (B, K) arrays."""
    h = p["dt"] / p["S"]
    steps, per = int(round(t_total / h)), int(round(every / h))
    main = Crystal(p, p["N"], B, seed)
    mem = Crystal(p, p["N_mem"], B, mem_seed) if p["reference"] == "memory" else None
    lag = int(round(p["tau"] / h)) if p["reference"] == "ideal" else 0
    span = p["v_mem"] * p["tau"] if mem else 0.0
    hist = np.zeros((steps + 1, B))                     # Phi of the main crystal at every sub-step
    mhist = np.zeros((steps + 1, B)) if mem else None   # the memory's mean phase at every sub-step
    ptr = np.zeros(B, dtype=np.int64)
    clipped = np.zeros(B)
    cols = [[], [], [], [], []]
    F0, k = p["F"], p["k"]
    target = p["v_star"] * (lag * h if lag else p["tau"])   # the advance expected over the span the crystal actually reads
    rows = np.arange(B)
    for i in range(1, steps + 1):
        main.drift()
        hist[i] = main.phase()
        F = np.full(B, F0)
        if mem is not None:
            mem.drift()
            mhist[i] = mem.phase()
            want = mhist[i] - span
            ok = want >= 0.0
            while True:                                  # the last sub-step at which the memory stood at or behind `want`
                nxt = np.minimum(ptr + 1, i)
                adv = (nxt > ptr) & (mhist[nxt, rows] <= want)
                if not adv.any():
                    break
                ptr = np.where(adv, nxt, ptr)
            a, b = mhist[ptr, rows], mhist[np.minimum(ptr + 1, i), rows]
            w = np.where(b > a, np.clip((want - a) / np.where(b > a, b - a, 1.0), 0.0, 1.0), 0.0)
            past = (1 - w) * hist[ptr, rows] + w * hist[np.minimum(ptr + 1, i), rows]
            u = np.where(ok, -k * (hist[i] - past - target), 0.0)
            F = F0 + u
            mem.kick(np.full(B, F0))
        elif lag and i >= lag:
            F = F0 - k * (hist[i] - hist[i - lag] - target)
        lo, hi = 0.0, 2.0 * F0
        clipped += (F < lo) | (F > hi)
        main.kick(np.clip(F, lo, hi))
        if i % per == 0:
            for col, x in zip(cols, (main.J, main.heat, mem.J if mem else np.zeros(B), mem.heat if mem else np.zeros(B),
                                     clipped / i)):
                col.append(x.copy())
    return tuple(np.stack(c, 1) for c in cols)


def seed(registration, randomness, name):
    # the registered seed rule (registration.json `seeds`, METHOD §3)
    return int(hashlib.sha256(f"{registration}:{randomness}:{name}".encode()).hexdigest()[:12], 16)


def cell_params(design, cell):
    """A cell names its reference, its span (a key of design["span_ticks"]) and the memory's size."""
    ref = cell["reference"]
    v_mem = design["v_mem"][str(cell["N_mem"])] if ref == "memory" else 0.0
    n = design["span_ticks"][cell["span"]] if ref != "alone" else 0
    return {**design["base"], "reference": ref, "N": design["N"], "N_mem": cell.get("N_mem", 0),
            "tau": n * np.pi / (v_mem if ref == "memory" else design["v_star"]), "k": design["k"] if ref != "alone" else 0.0,
            "v_star": design["v_star"], "v_mem": v_mem}


def one(args):
    design, cell, rid, rnd = args
    p = cell_params(design, cell)
    t0, t1, t2 = design["checkpoints"]
    J, Q, Jm, Qm, clip = run(p, design["trajectories"], t2, design["every"], seed(rid, rnd, cell["name"]),
                             seed(rid, rnd, cell["name"] + ":memory"))
    at = [int(round(t / design["every"])) - 1 for t in (t0, t1, t2)]
    return cell["name"], {"p": p, **{k: [x[:, a].tolist() for a in at] for k, x in
                                     (("J", J), ("heat", Q), ("J_memory", Jm), ("heat_memory", Qm))},
                          "clipped": float(clip[:, -1].mean())}


def main():
    here = Path(__file__).resolve().parent
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 1
    reg = json.loads((here / "registration.json").read_text())
    rid = json.loads((here / "registration.lock.json").read_text())["registration"]
    beacon = json.loads((here / "beacon.json").read_text())
    rnd, design = beacon["beacon"]["randomness"], reg["design"]
    work = [(design, c, rid, rnd) for c in design["cells"]]
    cells = {}
    with ProcessPoolExecutor(jobs) as pool:
        for name, out in pool.map(one, work):
            cells[name] = out
            print(name, flush=True)
    data = {"registration": rid, "beacon": beacon["attest"], "randomness": rnd,
            "env": {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine()},
            "cells": {c["name"]: cells[c["name"]] for c in design["cells"]}}
    (here / "results").mkdir(exist_ok=True)
    (here / "results" / "data.json").write_text(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()
