"""The soliton crystal of the archived lab (ARCHIVED / NOT CURRENT PRECEDENT: phs-soliton/phs/core.py, rewritten here in
numpy, no import), untrained and driven: a clock is a crystal whose phases a constant torque F pushes over the pinning.

State Theta, P of shape (B, N, d) on the cotangent torus, masses 1, temperature tau = sigma². Forces:
    attention  -(kappa/d) sum_j A_sym_ij sin(theta_ik - theta_jk),  A = softmax_j(kappa C), C_ij = (1/d) sum_k cos(...)
    lattice    g_lat [sin(theta_{i+1} - theta_i) - sin(theta_i - theta_{i-1})] - m g_pin sin(m theta_i)
    torque     F + F1 sin(omega t) on every phase (F1 = 0: an autonomous clock; F1 > 0: a world with a rhythm)
Gyroscopic J = j_scale (W - W^T), W ~ N(0, 1) of the crystal's seed, integrated as an exact Cayley rotation. One sub-step:
    B(h/2) -> G(h/2) -> A(h/2) -> O(h) -> A(h/2) -> G(h/2) -> B(h/2)
The current is the unwrapped phase advance summed over all N·d phases; the heat is what the O steps take from the kinetic
energy. In a steady state the torque's work equals the heat, and the entropy production rate is F·v / tau.

    python -I q2-crystal/run.py        the registered cells, seeded from the registration and the beacon: results/data.json"""
import hashlib, json, platform, sys
from pathlib import Path

import numpy as np


def crystal(N, d, j_scale, seed):
    """The gyroscopic matrix of one crystal: the same seed gives the same crystal."""
    W = np.random.Generator(np.random.PCG64(seed)).standard_normal((N, N))
    return j_scale * (W - W.T)


def attention(theta, kappa):
    d = theta.shape[-1]
    c, s = np.cos(theta), np.sin(theta)
    C = (c @ np.swapaxes(c, -1, -2) + s @ np.swapaxes(s, -1, -2)) / d
    C = kappa * C
    A = np.exp(C - C.max(-1, keepdims=True))
    A /= A.sum(-1, keepdims=True)
    A = 0.5 * (A + np.swapaxes(A, -1, -2))
    return -(kappa / d) * (s * (A @ c) - c * (A @ s))


def force(theta, p, t):
    f = attention(theta, p["kappa"]) if p["kappa"] else np.zeros_like(theta)
    if p["g_lat"] or p["g_pin"]:
        fwd = np.roll(theta, -1, axis=-2) - theta
        bwd = theta - np.roll(theta, 1, axis=-2)
        f = f + p["g_lat"] * (np.sin(fwd) - np.sin(bwd)) - p["m_pin"] * p["g_pin"] * np.sin(p["m_pin"] * theta)
    return f + p["F"] + p.get("F1", 0.0) * np.sin(p.get("omega", 0.0) * t)


def cayley(J, h):
    eye = np.eye(J.shape[0])
    return np.linalg.solve(eye - 0.5 * h * J, eye + 0.5 * h * J)


def run(p, B, t_total, every, seed):
    """B independent trajectories from rest for t_total time units. Returns the current and the heat given to the bath,
    per trajectory, at each `every` time units: two (B, K) arrays."""
    N, d, h = p["N"], p["d"], p["dt"] / p["S"]
    rng = np.random.Generator(np.random.PCG64(seed))
    G = cayley(crystal(N, d, p["j_scale"], p["crystal"]), 0.5 * h) if p["j_scale"] else None
    c1 = np.exp(-p["gamma"] * h)
    c2 = np.sqrt(p["sigma"] ** 2 * (1 - c1 ** 2))
    theta = np.zeros((B, N, d))
    P = np.zeros((B, N, d))
    J = np.zeros(B)
    heat = np.zeros(B)
    f = force(theta, p, 0.0)
    steps, per = int(round(t_total / h)), int(round(every / h))
    out, heats = [], []
    for i in range(1, steps + 1):
        P = P + 0.5 * h * f
        if G is not None:
            P = G @ P
        theta = theta + 0.5 * h * P
        J += 0.5 * h * P.sum((-2, -1))
        k0 = 0.5 * (P ** 2).sum((-2, -1))
        P = c1 * P + c2 * rng.standard_normal(P.shape)
        heat += k0 - 0.5 * (P ** 2).sum((-2, -1))
        theta = theta + 0.5 * h * P
        J += 0.5 * h * P.sum((-2, -1))
        theta = np.remainder(theta + np.pi, 2 * np.pi) - np.pi
        if G is not None:
            P = G @ P
        f = force(theta, p, i * h)
        P = P + 0.5 * h * f
        if i % per == 0:
            out.append(J.copy())
            heats.append(heat.copy())
    return np.stack(out, 1), np.stack(heats, 1)


def clock(p, B, t_total, every, seed, burn=0.25):
    """v, D, the entropy production rate and eta = v² / (D Σ), from the window after the first `burn` of the run.
    v and D are slopes of the mean and the variance of the current; Σ is the measured heat rate over tau."""
    Js, Qs = run(p, B, t_total, every, seed)
    t = every * np.arange(1, Js.shape[1] + 1)
    w = t >= burn * t_total
    v = np.polyfit(t[w], Js[:, w].mean(0), 1)[0]
    D = np.polyfit(t[w], Js[:, w].var(0), 1)[0] / 2
    sigma_rate = np.polyfit(t[w], Qs[:, w].mean(0), 1)[0] / p["sigma"] ** 2
    work_rate = p["F"] * v / p["sigma"] ** 2 if not p.get("F1") else None
    return {"v": float(v), "D": float(D), "entropy_rate": float(sigma_rate), "work_rate": None if work_rate is None else float(work_rate),
            "eta": float(v * v / (D * sigma_rate)) if D > 0 and sigma_rate > 0 else None}


def seed(registration, randomness, name):
    # the registered seed rule (registration.json `seeds`, METHOD §3)
    return int(hashlib.sha256(f"{registration}:{randomness}:{name}".encode()).hexdigest()[:12], 16)


def cell_params(design, cell, registration, randomness):
    p = {**design["base"], **{k: cell[k] for k in ("N", "sigma", "j_scale", "F1")}}
    if cell["coupling"] == "loose":
        p.update(kappa=0.0, g_lat=0.0)
    p["crystal"] = seed(registration, randomness, "crystal")
    return p


def main():
    here = Path(__file__).resolve().parent
    reg = json.loads((here / "registration.json").read_text())
    rid = json.loads((here / "registration.lock.json").read_text())["registration"]
    beacon = json.loads((here / "beacon.json").read_text())
    rnd, design = beacon["beacon"]["randomness"], reg["design"]
    t1, t2 = design["window"]
    cells = {}
    for c in design["cells"]:
        p = cell_params(design, c, rid, rnd)
        Js, Qs = run(p, design["trajectories"], t2, t1, seed(rid, rnd, c["name"]))
        k1, k2 = 0, int(round(t2 / t1)) - 1
        cells[c["name"]] = {"p": p, "dJ": (Js[:, k2] - Js[:, k1]).tolist(), "dQ": (Qs[:, k2] - Qs[:, k1]).tolist()}
        print(c["name"], flush=True)
    data = {"registration": rid, "beacon": beacon["attest"], "randomness": rnd,
            "env": {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine()},
            "cells": cells}
    (here / "results").mkdir(exist_ok=True)
    (here / "results" / "data.json").write_text(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()
