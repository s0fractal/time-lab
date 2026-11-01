"""q9-field registered analysis: the pull of a pacemaker against the lattice Coulomb field, its screening at rest, how
precisely a distant site holds the pacemaker's time, and the heat the ticking pumps into the lattice's modes.

    python -I q9-field/analyze.py DIR          prints {verdicts, descriptives} as JSON; refuses incomplete data

G is the lattice Green's function on the L^D torus, (-lap + mu2) G = delta_0 (mu2 = 0: minus the uniform 1/L^D), by FFT.
Each statistic is computed from the trajectory means of the cell's orbit values; a 95% interval comes from 2000 resamples
of each cell's trajectories (inverted-cdf quantiles), seeded from the registration id and the cell's name.

    field slope    sum w E p / sum w p^2 over bond orbits whose endpoints both lie at 2 < r <= L/4, where E is the measured
                   outward torque, p = delta (G(x) - G(x + e)) the outward field of a point charge delta, w the multiplicity.
                   1: the pull is the lattice Coulomb field (1/r^2 in three dimensions, 1/r in two), with no free constant
    screening      flux(1) / flux(0) over its linear prediction, flux(R) = -sum of <c> over the cube of half-width R (the
                   torque that leaves it), predicted delta (1 - mu2 sum_cube G^mu), mu2 = m^2 g_pin / g (the wells' stiffness)
    time ratio     var at (L/4, 0, ...) / var at (1, 0, ...): how much worse the pacemaker's time is held at a quarter of the
                   lattice than next to it; G predicts 2 (G(0) - G(x)) for each
    heating        mean mode energy of a band of lam over that of the slow modes (0 <= lam < 2; equipartition: 1); the
                   resonant band is g lam within 15% of (m v / 2)^2, v the cell's mean speed (the pin turns at m v)

Inconclusive by rule: a slip (in any trajectory of a cell used, some bond's X_x - X_(x+e) ranged over 1.5 pi or more in
time: a slip moves it by 2 pi, the bath alone moved the calibration's by at most 3.6),
or a resample in which a statistic is undefined. Its bytes are the registration's `analysis` blob."""
import hashlib, json, math, sys
from itertools import product
from pathlib import Path

import numpy as np

B_RESAMPLES = 2000
HYPOTHESES = {          # {H: (statistic, cells, kind, x)}: supported when the interval of log(statistic) lies above / below
                        # log x, or within (-log x, log x)
    "H1": ("field", ("run-D3",), "within", 1.15),
    "H2": ("field", ("run-D2",), "within", 1.15),
    "H3": ("screening", ("rest-D3",), "within", 1.25),
    "H4": ("time", ("run-D3",), "below", 1.6),
    "H5": ("time_vs_G", ("run-D1",), "within", 1.25),
    "H6": ("heating", ("run-D3",), "above", 1.5),
    "H7": ("heating_gain", ("run-D1-g2", "run-D1"), "above", 1.3),
}
SLOW = (0.0, 2.0)
SLIP = 1.5 * math.pi
TOP_1D = (3.0, 4.0)


def refuse(why):
    sys.exit(f"REFUSED {why}")


def green(L, D, mu2=0.0):
    k = 2 * np.pi * np.fft.fftfreq(L)
    lam = sum(np.meshgrid(*([2 - 2 * np.cos(k)] * D), indexing="ij"))
    with np.errstate(divide="ignore"):
        inv = 1.0 / (lam + mu2)
    if mu2 == 0:
        inv[(0,) * D] = 0.0
    return np.real(np.fft.ifftn(inv))


def expected(L, D):
    """Site orbits {sorted |x|: count} and bond orbits {(2|m_a|, sorted 2|m_d|): count}, enumerated afresh."""
    mi = lambda v: min(v % L, (-v) % L)
    sites, bonds = {}, {}
    for x in product(range(L), repeat=D):
        k = tuple(sorted(mi(v) for v in x))
        sites[k] = sites.get(k, 0) + 1
        for a in range(D):
            two_m = (2 * x[a] + 1) % (2 * L)
            two_m = min(two_m, 2 * L - two_m)
            k = (two_m,) + tuple(sorted(2 * mi(x[d]) for d in range(D) if d != a))
            bonds[k] = bonds.get(k, 0) + 1
    return sites, bonds


def load(d):
    reg = json.loads((d / "registration.json").read_text())
    lock = json.loads((d / "registration.lock.json").read_text())
    data = json.loads((d / "results" / "data.json").read_text())
    design = reg["design"]
    if data.get("registration") != lock["registration"]:
        refuse("data are not of this registration")
    names = [c["name"] for c in design["cells"]]
    if sorted(data["cells"]) != sorted(names):
        refuse(f"cells {sorted(data['cells'])} are not the registered {sorted(names)}")
    B = design["trajectories"]
    for c in design["cells"]:
        x, L, D = data["cells"][c["name"]], c["L"], c["D"]
        sites, bonds = expected(L, D)
        if [tuple(k) for k in x["site_keys"]] != sorted(sites) or x["site_count"] != [sites[k] for k in sorted(sites)]:
            refuse(f"{c['name']}: site orbits are not those of the {L}^{D} lattice")
        if [tuple(k) for k in x["bond_keys"]] != sorted(bonds) or x["bond_count"] != [bonds[k] for k in sorted(bonds)]:
            refuse(f"{c['name']}: bond orbits are not those of the {L}^{D} lattice")
        for f, n in (("phi", len(sites)), ("var", len(sites)), ("c", len(sites)), ("E", len(bonds)),
                     ("spectrum", len(x["spectrum_count"])), ("lead", 2)):
            if len(x[f]) != B or any(len(row) != n for row in x[f]):
                refuse(f"{c['name']}: {f} is not {B} trajectories of {n}")
        if len(x["speed"]) != B or len(x["range"]) != B:
            refuse(f"{c['name']}: speed or range is not {B} trajectories")
    return design, lock["registration"], data["cells"]


class Cell:
    """One cell's arrays and the fixed quantities its statistics weigh them by."""

    def __init__(self, spec, x, base, delta):
        self.name, self.L, self.D = spec["name"], spec["L"], spec["D"]
        self.prm = {**base, **spec["params"]}
        self.delta = delta
        self.arr = {f: np.asarray(x[f], float) for f in ("phi", "var", "c", "E", "spectrum", "lead", "speed", "range")}
        self.skeys = [tuple(k) for k in x["site_keys"]]
        self.sw = np.asarray(x["site_count"], float)
        bkeys = [tuple(k) for k in x["bond_keys"]]
        self.bw = np.asarray(x["bond_count"], float)
        G = green(self.L, self.D)
        at = lambda c: G[tuple(int(round(v)) % self.L for v in c)]
        p, inside = [], []
        for k in bkeys:
            x0 = ((k[0] - 1) / 2,) + tuple(v / 2 for v in k[1:])
            x1 = ((k[0] + 1) / 2,) + x0[1:]
            p.append(delta * (at(x0) - at(x1)))
            r0, r1 = math.sqrt(sum(v * v for v in x0)), math.sqrt(sum(v * v for v in x1))
            inside.append(2 < min(r0, r1) and max(r0, r1) <= self.L / 4)
        self.pred, self.inside = np.asarray(p), np.asarray(inside)
        mu2 = self.prm["m_pin"] ** 2 * self.prm["g_pin"] / self.prm["g_lat"]
        Gm = green(self.L, self.D, mu2)
        self.flux_lin = {R: delta * (1 - mu2 * sum(self.sw[i] * Gm[k] for i, k in enumerate(self.skeys) if max(k) <= R))
                         for R in (0, 1, 2, 3)}
        self.lam_mid = (np.arange(len(x["spectrum_count"])) + 0.5) * x["spectrum_bin"]
        self.mcount = np.asarray(x["spectrum_count"], float)
        axis = lambda r: self.skeys.index(tuple(sorted((r,) + (0,) * (self.D - 1))))
        self.near, self.far = axis(1), axis(self.L // 4)
        Gx = lambda r: G[(r,) + (0,) * (self.D - 1)]
        self.time_G = (G[(0,) * self.D] - Gx(self.L // 4)) / (G[(0,) * self.D] - Gx(1))
        self.slip = float(self.arr["range"].max())
        self.drift = float(np.abs(self.arr["lead"][:, 1] - self.arr["lead"][:, 0]).max())

    def mean(self, idx):
        return {f: a[idx].mean(0) for f, a in self.arr.items()}

    def field(self, m):
        i = self.inside
        return float((self.bw[i] * m["E"][i] * self.pred[i]).sum() / (self.bw[i] * self.pred[i] ** 2).sum())

    def flux(self, m, R):
        return float(-sum(self.sw[i] * m["c"][i] for i, k in enumerate(self.skeys) if max(k) <= R))

    def band(self, m, lo, hi):
        sel = (self.lam_mid >= lo) & (self.lam_mid < hi) & (self.mcount > 0)
        if not sel.any():
            return math.nan
        return float((m["spectrum"][sel] * self.mcount[sel]).sum() / self.mcount[sel].sum())

    def resonant(self, m):
        v = float(m["speed"].mean())
        target = (self.prm["m_pin"] * v / 2) ** 2 / self.prm["g_lat"]
        return self.band(m, 0.85 * target, 1.15 * target)


def statistic(name, cells, ms):
    c, m = cells[0], ms[0]
    if name == "field":
        return c.field(m)
    if name == "screening":
        f0 = c.flux(m, 0)
        return c.flux(m, 1) / f0 / (c.flux_lin[1] / c.flux_lin[0]) if f0 > 0 else math.nan
    if name == "time":
        return float(m["var"][c.far] / m["var"][c.near])
    if name == "time_vs_G":
        return float(m["var"][c.far] / m["var"][c.near]) / c.time_G
    if name == "heating":
        return c.resonant(m) / c.band(m, *SLOW)
    if name == "heating_gain":
        top = lambda cc, mm: cc.band(mm, *TOP_1D) / cc.band(mm, *SLOW)
        return top(cells[0], ms[0]) / top(cells[1], ms[1])
    raise ValueError(name)


def decide(kind, x, lo, hi):
    t = math.log(x)
    if kind == "above":
        return "supported" if lo > t else "contradicted" if hi < t else "inconclusive"
    if kind == "below":
        return "supported" if hi < t else "contradicted" if lo > t else "inconclusive"
    return "supported" if -t < lo and hi < t else "contradicted" if lo > t or hi < -t else "inconclusive"


def verdicts_of(rid, cells, B):
    out = {}
    for h, (name, used, kind, x) in HYPOTHESES.items():
        cs = [cells[u] for u in used]
        r = {"statistic": name, "cells": list(used), "kind": kind, "threshold": x}
        slips = {cc.name: cc.slip for cc in cs if not cc.slip < SLIP}
        point = statistic(name, cs, [cc.mean(np.arange(B)) for cc in cs])
        r["value"] = point
        if slips:
            out[h] = {**r, "verdict": "inconclusive", "reason": f"slip: bond range {slips}"}
            continue
        rngs = [np.random.Generator(np.random.PCG64(int(hashlib.sha256(f"{rid}:bootstrap:{cc.name}".encode()).hexdigest()[:16], 16)))
                for cc in cs]
        logs = []
        for _ in range(B_RESAMPLES):
            v = statistic(name, cs, [cc.mean(g.integers(0, B, B)) for cc, g in zip(cs, rngs)])
            logs.append(math.log(v) if v > 0 and math.isfinite(v) else math.nan)
        logs = np.asarray(logs)
        if not (point > 0 and math.isfinite(point)) or np.isnan(logs).any():
            out[h] = {**r, "verdict": "inconclusive", "reason": "a statistic is undefined in a resample"}
            continue
        lo, hi = (float(np.quantile(logs, q, method="inverted_cdf")) for q in (0.025, 0.975))
        out[h] = {**r, "log": math.log(point), "ci95_log": [lo, hi], "ci95": [math.exp(lo), math.exp(hi)],
                  "verdict": decide(kind, x, lo, hi)}
    return out


def describe(cells, B):
    rows = {}
    for name, c in cells.items():
        m = c.mean(np.arange(B))
        G = green(c.L, c.D)
        on_axis = {r: c.skeys.index(tuple(sorted((r,) + (0,) * (c.D - 1)))) for r in (1, 2, 4, c.L // 4)}
        g0 = G[(0,) * c.D]
        rows[name] = {
            "speed": float(m["speed"].mean()), "bond_range": c.slip, "lead_drift": c.drift, "field_slope": c.field(m),
            "flux": {R: [c.flux(m, R), c.flux_lin[R]] for R in (0, 1, 2, 3)},
            "flux_unscreened": {R: c.delta * (1 - (2 * R + 1) ** c.D / c.L ** c.D) for R in (0, 1, 2, 3)},
            "var_on_axis": {r: [float(m["var"][i]), float(2 * (g0 - G[(r,) + (0,) * (c.D - 1)]))] for r, i in on_axis.items()},
            "time_ratio": [float(m["var"][c.far] / m["var"][c.near]), c.time_G],
            "spectrum": {f"{c.lam_mid[j]:.3f}": float(m["spectrum"][j]) for j in range(len(c.lam_mid)) if c.mcount[j] > 0},
            "resonant_band_over_slow": c.resonant(m) / c.band(m, *SLOW),
        }
    return rows


def main(d):
    design, rid, data = load(Path(d))
    B = design["trajectories"]
    cells = {c["name"]: Cell(c, data[c["name"]], design["base"], design["delta"]) for c in design["cells"]}
    print(json.dumps({"verdicts": verdicts_of(rid, cells, B), "descriptives": describe(cells, B)}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
