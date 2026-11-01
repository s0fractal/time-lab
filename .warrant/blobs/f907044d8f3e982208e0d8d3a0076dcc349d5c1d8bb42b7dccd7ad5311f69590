"""q9-field: the field of a pacemaker in a crystal of clocks. Does its pull fall as 1/r^(D-1), like gravity and
electrostatics in three dimensions, and how precisely does a distant site borrow its time?

The lattice: the crystal's rotors on a periodic L^D lattice (q5-bound's local physics: friction gamma, bath sigma, torque
F, pin -m g_pin sin(m theta); no attention), each coupled to its 2D nearest neighbours by g sin(theta_j - theta_i), stepped
by BAOAB (h = dt / S). The site at the origin is a pacemaker: its torque is F + delta. Running (F = 1.3) the rotors tick;
resting (F = 0) they sit in the pin's wells.

Per trajectory, over [warm, t_total], sampled every time unit, with X the unwrapped phase:
    lead      X_0 - mean X at warm and at t_total
    range     the largest range (max - min) over time of any bond's X_x - X_(x+e): a phase slip moves it by 2 pi
    phi       <X_i - mean X>, each site's phase lead
    var       the variance in time of X_i - X_0: how precisely site i holds the pacemaker's time
    c         <g sum_j sin(theta_j - theta_i)>, the torque site i takes from its neighbours
    E         <g sin(theta_x - theta_(x+e_a))>, the torque bond (x, a) passes outward along +e_a: the pull
    spectrum  g lam_k Var(X_k) / L^D for each mode k != 0 (equipartition: sigma^2), lam_k = sum_d (2 - 2 cos k_d), binned
              in lam by width 1/4
Site quantities are averaged over the orbits of the lattice's symmetries that fix the origin (reflections and axis
permutations): the key is the sorted |x_d| (minimum image). Bonds are keyed by their midpoint m: (|m_a|, sorted |m_d| of
the other axes), with E signed outward (times the sign of m_a). Each key keeps its multiplicity.

    python -I q9-field/run.py              the registered cells, seeded from the registration and the beacon: results/data.json"""
import hashlib, json, platform, sys
from pathlib import Path

import numpy as np

D_ = Path(__file__).resolve().parent


def coupling(theta, g):
    f = np.zeros_like(theta)
    for ax in range(1, theta.ndim):
        f += np.sin(np.roll(theta, -1, ax) - theta) + np.sin(np.roll(theta, 1, ax) - theta)
    return g * f


class Lattice:
    """B trajectories of one L^D lattice from rest, one BAOAB sub-step at a time."""

    def __init__(self, p, L, D, B, delta, seed):
        self.p, self.h, self.g = p, p["dt"] / p["S"], p["g_lat"]
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.c1 = np.exp(-p["gamma"] * self.h)
        self.c2 = np.sqrt(p["sigma"] ** 2 * (1 - self.c1 ** 2))
        shape = (B,) + (L,) * D
        self.theta, self.P, self.X = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        self.Fi = np.full((1,) + (L,) * D, float(p["F"]))
        self.Fi[(0,) * (D + 1)] += delta
        self.c = coupling(self.theta, self.g)
        self.f = self.force()

    def force(self):
        p = self.p
        return self.c - p["m_pin"] * p["g_pin"] * np.sin(p["m_pin"] * self.theta) + self.Fi

    def step(self):
        """B(h/2) A(h/2) O(h) A(h/2) B(h/2), the force at the new positions carried to the next sub-step."""
        h = self.h
        self.P = self.P + 0.5 * h * self.f
        self.theta, self.X = self.theta + 0.5 * h * self.P, self.X + 0.5 * h * self.P
        self.P = self.c1 * self.P + self.c2 * self.rng.standard_normal(self.P.shape)
        self.theta, self.X = self.theta + 0.5 * h * self.P, self.X + 0.5 * h * self.P
        self.theta = np.remainder(self.theta + np.pi, 2 * np.pi) - np.pi
        self.c = coupling(self.theta, self.g)
        self.f = self.force()
        self.P = self.P + 0.5 * h * self.f


def minimum_image(L, D):
    """|x_d| of every site, shape (D,) + (L,) * D."""
    i = np.arange(L)
    return np.stack(np.meshgrid(*([np.minimum(i, L - i)] * D), indexing="ij"))


def site_orbits(L, D):
    """For every site the index of its orbit, and the orbits' keys (sorted |x_d|)."""
    m = minimum_image(L, D).reshape(D, -1).T
    keys = [tuple(sorted(int(v) for v in row)) for row in m]
    uniq = sorted(set(keys))
    at = {k: j for j, k in enumerate(uniq)}
    return np.array([at[k] for k in keys]), uniq


def bond_orbits(L, D):
    """For every bond (axis a, site x -> x + e_a) its orbit index and outward sign; the orbits' keys (|m_a|, sorted |m_d|),
    in half units (2 |m|) so that every key is integer. Index arrays of shape (D, L^D)."""
    i = np.arange(L)
    mid = (2 * i + 1) % (2 * L)
    mid = np.where(mid > L, mid - 2 * L, mid)               # 2 m_a in (-L, L], minimum image of the midpoint
    full = minimum_image(L, D).reshape(D, -1)
    coords = np.stack(np.meshgrid(*([i] * D), indexing="ij")).reshape(D, -1)
    keys, sign = [], np.zeros((D, L ** D))
    for a in range(D):
        ma = mid[coords[a]]
        sign[a] = np.sign(ma)
        others = [full[d] for d in range(D) if d != a]
        for s, row in enumerate(zip(np.abs(ma), *others)):
            keys.append((a, s, (int(row[0]),) + tuple(sorted(2 * int(v) for v in row[1:]))))
    uniq = sorted({k for _, _, k in keys})
    at = {k: j for j, k in enumerate(uniq)}
    idx = np.zeros((D, L ** D), dtype=np.int64)
    for a, s, k in keys:
        idx[a, s] = at[k]
    return idx, sign, uniq


def fold(values, idx, n):
    """Mean of values (B, sites) over each orbit: (B, n)."""
    out = np.zeros((values.shape[0], n))
    np.add.at(out.T, idx, values.T)
    return out / np.bincount(idx, minlength=n)


def observe(p, L, D, B, delta, warm, t_total, seed):
    lat = Lattice(p, L, D, B, delta, seed)
    h = lat.h
    n_warm, n_total, every = int(round(warm / h)), int(round(t_total / h)), int(round(1.0 / h))
    ax = tuple(range(1, D + 1))
    o = (slice(None),) + (0,) * D
    k = 2 * np.pi * np.fft.fftfreq(L)
    lam = sum(np.meshgrid(*([2 - 2 * np.cos(k)] * D), indexing="ij")).ravel()
    s_phi = s_rel = s_rel2 = s_c = s_k = s_k2 = 0.0
    s_E = np.zeros((D, B) + (L,) * D)
    lo, hi = np.full((D, B) + (L,) * D, np.inf), np.full((D, B) + (L,) * D, -np.inf)
    lead, m = np.zeros((B, 2)), 0
    for i in range(1, n_total + 1):
        lat.step()
        if i == n_warm:
            lead[:, 0] = lat.X[o] - lat.X.mean(ax)
        if i > n_warm and i % every == 0:
            xb = lat.X.mean(ax, keepdims=True)
            rel = lat.X - lat.X[o].reshape((B,) + (1,) * D)
            s_phi, s_rel, s_rel2, s_c = s_phi + (lat.X - xb), s_rel + rel, s_rel2 + rel ** 2, s_c + lat.c
            for a in range(D):
                s_E[a] += lat.g * np.sin(lat.theta - np.roll(lat.theta, -1, a + 1))
                dx = lat.X - np.roll(lat.X, -1, a + 1)
                lo[a], hi[a] = np.minimum(lo[a], dx), np.maximum(hi[a], dx)
            Xk = np.fft.fftn(lat.X - xb, axes=ax).reshape(B, -1)
            s_k, s_k2 = s_k + Xk, s_k2 + np.abs(Xk) ** 2
            m += 1
    lead[:, 1] = lat.X[o] - lat.X.mean(ax)
    flat = lambda a: a.reshape(B, -1) / m
    var_k = flat(s_k2) - np.abs(flat(s_k)) ** 2
    energy = p["g_lat"] * lam[None, :] * var_k / L ** D
    bins = np.floor(lam * 4).astype(np.int64)
    keep = lam > 1e-12
    nb = int(bins.max()) + 1
    spec = np.zeros((B, nb))
    np.add.at(spec.T, bins[keep], energy[:, keep].T)
    count = np.bincount(bins[keep], minlength=nb)
    sidx, skeys = site_orbits(L, D)
    bidx, bsign, bkeys = bond_orbits(L, D)
    mu = flat(s_rel)
    E = np.concatenate([s_E[a].reshape(B, -1) * bsign[a] for a in range(D)], 1) / m
    rng = (hi - lo).reshape(D, B, -1).max(axis=(0, 2))
    return {"lead": lead.tolist(), "range": rng.tolist(), "speed": (lat.X.mean(ax) / t_total).tolist(),
            "site_keys": [list(k) for k in skeys], "site_count": np.bincount(sidx).tolist(),
            "phi": fold(flat(s_phi), sidx, len(skeys)).tolist(), "var": fold(flat(s_rel2) - mu ** 2, sidx, len(skeys)).tolist(),
            "c": fold(flat(s_c), sidx, len(skeys)).tolist(),
            "bond_keys": [list(k) for k in bkeys], "bond_count": np.bincount(bidx.ravel()).tolist(),
            "E": fold(E, bidx.ravel(), len(bkeys)).tolist(),
            "spectrum_bin": 0.25, "spectrum_count": count.tolist(),
            "spectrum": np.divide(spec, np.maximum(count, 1)).tolist()}


def seed(registration, randomness, name):
    # the registered seed rule (registration.json `seeds`, METHOD §3)
    return int(hashlib.sha256(f"{registration}:{randomness}:{name}".encode()).hexdigest()[:12], 16)


def main():
    reg = json.loads((D_ / "registration.json").read_text())
    lock = json.loads((D_ / "registration.lock.json").read_text())
    beacon = json.loads((D_ / "beacon.json").read_text())
    design, rid, rnd = reg["design"], lock["registration"], beacon["beacon"]["randomness"]
    cells = {}
    for c in design["cells"]:
        p = {**design["base"], **c["params"]}
        cells[c["name"]] = observe(p, c["L"], c["D"], design["trajectories"], design["delta"], design["warm"],
                                   design["t_total"], seed(rid, rnd, c["name"]))
        print(c["name"], file=sys.stderr, flush=True)
    import numpy
    data = {"registration": rid, "beacon": beacon["attest"], "randomness": rnd,
            "env": {"python": platform.python_version(), "numpy": numpy.__version__, "machine": platform.machine()}, "cells": cells}
    (D_ / "results").mkdir(exist_ok=True)
    (D_ / "results" / "data.json").write_text(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()
