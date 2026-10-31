"""q4-line: can the soliton crystal read its own past through a physical delay line, where both the delay and the stored
value are carried by matter that costs, and is the whole thing a cheaper clock than the crystal alone?

The crystal is q3-self's (copied from q3-self/run.py, runner blob 36e7fc16): N = 32 phases pairs, driven by a constant torque
F at temperature sigma², no gyroscope. In q3-self the crystal's past phase lay in a free, exact buffer; here it travels
through a line of L stages, the same kind of matter as the crystal without the pinning: stage i is a rotor of n_s phases
bound together (mass n_s, friction gamma, temperature sigma²) pulled toward the stage before it,

    n_s psi_i'' = n_s K sin(psi_{i-1} - psi_i) - gamma n_s psi_i' + noise,     psi_0 = Phi (the crystal's mean phase)

A stage moving at the crystal's velocity v lags it by arcsin(gamma v / K): the line's end psi_L is the crystal's phase of
about L arcsin(gamma v / K) / v ago, blurred by the line's own heat. The pull is one way (a reader: the crystal feels
nothing), and its work ends as the line's heat, which is counted. The crystal corrects its torque by the line's lag:

    F(t) = F - k (Phi(t) - psi_L(t) - Lambda*),   clipped to [0, 2F]

with Lambda* the line's mean lag at the crystal's natural velocity. References:
    alone    k = 0, no line
    ideal    the crystal's exact phase tau ago from a free buffer (q3-self's perfect delay line), tau the line's delay
    line     the line of stages of n_s phases

One sub-step: B(h/2) -> A(h/2) -> O(h) -> A(h/2) -> B(h/2) for the crystal and for the line. The heat is what the O steps
take from the kinetic energy.

    python -I q4-line/run.py [--jobs K]      the registered cells, seeded from the registration and the beacon: results/data.json"""
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


class Line:
    """B trajectories of a line of L stages from rest; stage i is n_s phases bound into one rotor (mass n_s): per unit
    mass its velocity u_i takes K sin(psi_{i-1} - psi_i) and the bath's friction gamma at temperature sigma² / n_s.
    Phases are unwrapped."""

    def __init__(self, p, B, seed):
        self.p, self.h, self.K, self.ns = p, p["dt"] / p["S"], p["K"], p["n_s"]
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.c1 = np.exp(-p["gamma"] * self.h)
        self.c2 = np.sqrt(p["sigma"] ** 2 / self.ns * (1 - self.c1 ** 2))
        self.psi = np.zeros((B, p["L"]))
        self.u = np.zeros((B, p["L"]))
        self.a = np.zeros((B, p["L"]))
        self.heat = np.zeros(B)

    def drift(self):
        h = self.h
        self.u = self.u + 0.5 * h * self.a
        self.psi = self.psi + 0.5 * h * self.u
        k0 = 0.5 * self.ns * (self.u ** 2).sum(-1)
        self.u = self.c1 * self.u + self.c2 * self.rng.standard_normal(self.u.shape)
        self.heat += k0 - 0.5 * self.ns * (self.u ** 2).sum(-1)
        self.psi = self.psi + 0.5 * h * self.u

    def kick(self, leader):
        """The pull of each stage toward the one before it, the first toward `leader` (B,), then B(h/2)."""
        before = np.concatenate([leader[:, None], self.psi[:, :-1]], 1)
        self.a = self.K * np.sin(before - self.psi)
        self.u = self.u + 0.5 * self.h * self.a


def run(p, B, t_total, every, seed, line_seed=None):
    """B trajectories for t_total time units under the cell's reference (p["reference"]: alone, ideal, line).
    Returns, at each `every` time units, the crystal's current, its heat, the line's heat, the lag Phi - psi_L (zeros
    without a line) and the share of sub-steps whose torque was clipped so far: five (B, K) arrays."""
    h = p["dt"] / p["S"]
    steps, per = int(round(t_total / h)), int(round(every / h))
    main = Crystal(p, p["N"], B, seed)
    line = Line(p, B, line_seed) if p["reference"] == "line" else None
    lag = int(round(p["tau"] / h)) if p["reference"] == "ideal" else 0
    hist = np.zeros((lag + 1, B)) if lag else None      # a ring of the crystal's phase over the last tau
    clipped = np.zeros(B)
    cols = [[], [], [], [], []]
    F0, k = p["F"], p["k"]
    for i in range(1, steps + 1):
        main.drift()
        phi = main.phase()
        F = np.full(B, F0)
        if line is not None:
            line.drift()
            line.kick(phi)
            F = F0 - k * (phi - line.psi[:, -1] - p["Lambda"])
        elif lag:
            if i >= lag:                                 # the slot after i's holds step i - lag (the phase 0 at rest)
                F = F0 - k * (phi - hist[(i + 1) % (lag + 1)] - p["v_star"] * lag * h)
            hist[i % (lag + 1)] = phi
        lo, hi = 0.0, 2.0 * F0
        clipped += (F < lo) | (F > hi)
        main.kick(np.clip(F, lo, hi))
        if i % per == 0:
            for col, x in zip(cols, (main.J, main.heat, line.heat if line else np.zeros(B),
                                     phi - line.psi[:, -1] if line else np.zeros(B), clipped / i)):
                col.append(x.copy())
    return tuple(np.stack(c, 1) for c in cols)


def seed(registration, randomness, name):
    # the registered seed rule (registration.json `seeds`, METHOD §3)
    return int(hashlib.sha256(f"{registration}:{randomness}:{name}".encode()).hexdigest()[:12], 16)


def cell_params(design, cell):
    """A cell names its reference and, for a line, the size n_s of its stages (design["Lambda"] keys its mean lag)."""
    ref = cell["reference"]
    line = design["line"]
    return {**design["base"], "reference": ref, "N": design["N"], "k": design["k"] if ref != "alone" else 0.0,
            "v_star": design["v_star"], "tau": design["tau"] if ref == "ideal" else 0.0,
            "L": line["L"], "K": line["K"], "n_s": cell.get("n_s", 0),
            "Lambda": design["Lambda"][str(cell["n_s"])] if ref == "line" else 0.0}


def one(args):
    design, cell, rid, rnd = args
    p = cell_params(design, cell)
    t0, t1, t2 = design["checkpoints"]
    J, Q, Ql, lag, clip = run(p, design["trajectories"], t2, design["every"], seed(rid, rnd, cell["name"]),
                              seed(rid, rnd, cell["name"] + ":line"))
    at = [int(round(t / design["every"])) - 1 for t in (t0, t1, t2)]
    return cell["name"], {"p": p, **{k: [x[:, a].tolist() for a in at] for k, x in
                                     (("J", J), ("heat", Q), ("heat_line", Ql), ("lag", lag))},
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
