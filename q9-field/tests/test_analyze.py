"""Controls for q9-field/analyze.py on synthetic cells whose statistics are set in advance (orbits from the runner's own
enumeration, fields from a separate Green's function), reference values computed here, and an end-to-end smoke run of the
real runner on a small design.

    python -m unittest q9-field/tests/test_analyze.py      (ANALYZE=path runs another copy, for mutants)"""
import importlib.util, json, math, os, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ANALYZE = os.environ.get("ANALYZE", str(HERE / "analyze.py"))
SCRATCH = HERE / "runs"
REG = json.loads((HERE / "registration.json").read_text())
DES = REG["design"]
B = DES["trajectories"]
DELTA = DES["delta"]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = module("q9run", HERE / "run.py")
A = module("q9analyze", HERE / "analyze.py")
HYP = A.HYPOTHESES
CELLS = {c["name"]: c for c in DES["cells"]}


def green(L, D, mu2=0.0):
    """A separate (-lap + mu2) G = delta_0 on the torus: the mode sum with cos(k.x) as a product of cosines (the sines
    cancel between k and -k), contracted axis by axis instead of an FFT."""
    k = 2 * np.pi * np.arange(L) / L
    lam = sum(np.meshgrid(*([2 - 2 * np.cos(k)] * D), indexing="ij"))
    w = np.where(lam + mu2 > 1e-12, 1.0 / np.maximum(lam + mu2, 1e-12), 0.0)
    C = np.cos(np.outer(k, np.arange(L)))                    # C[k, x]
    G = w
    for _ in range(D):
        G = np.tensordot(G, C, axes=([0], [0]))              # contracts the leading k axis, appends an x axis
    return G / L ** D


_G = {}


def G_of(L, D, mu2=0.0):
    if (L, D, mu2) not in _G:
        _G[(L, D, mu2)] = green(L, D, mu2)
    return _G[(L, D, mu2)]


def params(name):
    return {**DES["base"], **CELLS[name]["params"]}


def bond_geometry(name):
    c = CELLS[name]
    L, D = c["L"], c["D"]
    _, _, keys = R.bond_orbits(L, D)
    G = G_of(L, D)
    pred, inside = [], []
    for k in keys:
        x0 = ((k[0] - 1) // 2,) + tuple(v // 2 for v in k[1:])
        x1 = ((k[0] + 1) // 2,) + x0[1:]
        pred.append(DELTA * (G[tuple(v % L for v in x0)] - G[tuple(v % L for v in x1)]))
        r0, r1 = math.dist(x0, (0,) * D), math.dist(x1, (0,) * D)
        inside.append(2 < min(r0, r1) and max(r0, r1) <= L / 4)
    idx, _, _ = R.bond_orbits(L, D)
    return keys, np.bincount(idx.ravel()), np.asarray(pred), np.asarray(inside)


def lam_bins(L, D):
    k = 2 * np.pi * np.fft.fftfreq(L)
    lam = sum(np.meshgrid(*([2 - 2 * np.cos(k)] * D), indexing="ij")).ravel()
    lam = lam[lam > 1e-12]
    count = np.bincount(np.floor(lam * 4).astype(int))
    return (np.arange(len(count)) + 0.5) / 4, count


def resonant_band(name, v):
    p = params(name)
    t = (p["m_pin"] * v / 2) ** 2 / p["g_lat"]
    return 0.85 * t, 1.15 * t


def flux_linear(name, R_):
    c, p = CELLS[name], params(name)
    L, D = c["L"], c["D"]
    mu2 = p["m_pin"] ** 2 * p["g_pin"] / p["g_lat"]
    Gm = G_of(L, D, mu2)
    idx = [x for x in np.ndindex(*(L,) * D) if max(min(v, L - v) for v in x) <= R_]
    return DELTA * (1 - mu2 * sum(Gm[x] for x in idx))


# calibration-like settings (calibration/full-analysis.json)
DEFAULT = {"field": {"run-D3": 0.97, "run-D2": 0.995}, "screen": {"rest-D3": 1.01}, "time": {"run-D3": 1.29},
           "time_vs_G": {"run-D1": 0.93}, "heat": {"run-D3": 2.07}, "top": {"run-D1": 1.13, "run-D1-g2": 1.13 * 1.66}}


def data(set_=None, noise=0.03, slip=None, speed=None, outside=None, registration="r" * 64):
    """Synthetic cells: every statistic of DEFAULT (or set_) holds in the trajectory mean, each trajectory scattered by a
    lognormal of sd `noise`. slip: {cell: range of one trajectory}; speed: {cell: v}; outside: {cell: factor} on the field
    of bonds outside 2 < r <= L/4."""
    st = {k: dict(v) for k, v in DEFAULT.items()}
    for k, v in (set_ or {}).items():
        st[k].update(v)
    rng = np.random.Generator(np.random.PCG64(9))
    jitter = lambda shape: np.exp(0.2 * noise * rng.standard_normal(shape))

    def per_trajectory(extra=1):
        j = np.exp(noise * rng.standard_normal(B))
        return (j / j.mean()).reshape((B,) + (1,) * extra)
    cells = {}
    for name, c in CELLS.items():
        L, D, p = c["L"], c["D"], params(name)
        running = p["F"] > 0
        sidx, skeys = R.site_orbits(L, D)
        scount = np.bincount(sidx)
        G = G_of(L, D)
        g0 = G[(0,) * D]
        gk = np.array([G[k] for k in skeys])
        var = np.tile(0.1 * 2 * (g0 - gk) + (gk == g0) * 1e-3, (B, 1)) * jitter((B, len(skeys)))
        near, far = skeys.index(tuple(sorted((1,) + (0,) * (D - 1)))), skeys.index(tuple(sorted((L // 4,) + (0,) * (D - 1))))
        G_ratio = (g0 - G[(L // 4,) + (0,) * (D - 1)]) / (g0 - G[(1,) + (0,) * (D - 1)])
        target = st["time"].get(name) or (st["time_vs_G"].get(name, 1.0) * G_ratio)
        var[:, far] = var[:, near] * target * per_trajectory(0)
        # c: the flux through cubes; running unscreened, resting the linear screening with the ratio scaled by `screen`
        f = {r: DELTA * (1 - (2 * r + 1) ** D / L ** D) if running else flux_linear(name, r) for r in (0, 1, 2, 3)}
        if not running:
            f[1] = f[0] * (f[1] / f[0]) * st["screen"].get(name, 1.0)
        shell = np.array([max(k) for k in skeys])
        cmean = np.zeros(len(skeys))
        for r in (0, 1, 2, 3):
            sel = shell == r
            cmean[sel] = -(f[r] - (f[r - 1] if r else 0.0)) / scount[sel].sum()
        cvals = np.tile(cmean, (B, 1)) * jitter((B, len(skeys)))
        cvals[:, shell == 1] *= per_trajectory()
        keys, bcount, pred, inside = bond_geometry(name)
        s = st["field"].get(name, 1.0 if running else 0.05)
        E = np.tile(pred * s, (B, 1)) * jitter((B, len(keys))) * per_trajectory()
        if outside and name in outside:
            E[:, ~inside] *= outside[name]
        mids, mcount = lam_bins(L, D)
        v = (speed or {}).get(name, 2.59 if running else 0.0)
        spec = np.tile(np.full(len(mids), 0.09), (B, 1)) * jitter((B, len(mids)))
        if running:
            lo, hi = resonant_band(name, v)
            band = (mids >= lo) & (mids < hi)
            spec[:, band] *= st["heat"].get(name, 1.0) * per_trajectory()
            if D == 1:
                top = (mids >= 3.0) & (mids < 4.0)
                spec[:, top] = 0.09 * st["top"].get(name, 1.0) * per_trajectory()
        rngs = np.full(B, 2.5)
        if slip and name in slip:
            rngs[0] = slip[name]
        cells[name] = {"lead": [[0.0, 0.1]] * B, "range": rngs.tolist(), "speed": [v] * B,
                       "site_keys": [list(k) for k in skeys], "site_count": scount.tolist(), "phi": np.zeros((B, len(skeys))).tolist(),
                       "var": var.tolist(), "c": cvals.tolist(), "bond_keys": [list(k) for k in keys],
                       "bond_count": bcount.tolist(), "E": E.tolist(), "spectrum_bin": 0.25,
                       "spectrum_count": mcount.tolist(), "spectrum": spec.tolist()}
    return {"registration": registration, "cells": cells}


def run(payload, design=None):
    SCRATCH.mkdir(exist_ok=True)
    d = Path(tempfile.mkdtemp(dir=SCRATCH))
    try:
        (d / "results").mkdir()
        (d / "registration.json").write_text(json.dumps({"design": design or DES}))
        (d / "registration.lock.json").write_text(json.dumps({"registration": "r" * 64}))
        (d / "results" / "data.json").write_text(json.dumps(payload))
        p = subprocess.run([sys.executable, "-I", ANALYZE, str(d)], capture_output=True, text=True)
        return p.returncode, (json.loads(p.stdout) if p.returncode == 0 else p.stdout + p.stderr)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def verdicts(payload):
    code, out = run(payload)
    assert code == 0, out
    return {h: v["verdict"] for h, v in out["verdicts"].items()}


KEY = {"field": "field", "screening": "screen", "time": "time", "time_vs_G": "time_vs_G", "heating": "heat"}


def put(h, value):
    """Settings that put hypothesis h's statistic at `value`."""
    name, used, _, _ = HYP[h]
    if name == "heating_gain":
        return {"top": {used[0]: DEFAULT["top"][used[1]] * value}}
    return {KEY[name]: {used[0]: value}}


def flip(h):
    _, _, kind, x = HYP[h]
    return put(h, {"above": x / 1.6, "below": x * 1.6, "within": x * 1.6}[kind])


class Controls(unittest.TestCase):
    def test_calibration_like_cells(self):
        self.assertEqual(verdicts(data()), dict.fromkeys(HYP, "supported"))

    def test_each_hypothesis_can_be_contradicted(self):
        for h in HYP:
            with self.subTest(h=h):
                self.assertEqual(verdicts(data(flip(h)))[h], "contradicted")

    def test_a_band_is_left_on_either_side(self):
        for h, (_, _, kind, x) in HYP.items():
            if kind == "within":
                with self.subTest(h=h):
                    self.assertEqual(verdicts(data(put(h, 1 / (x * 1.6))))[h], "contradicted")

    def test_a_statistic_near_its_threshold_is_inconclusive(self):
        for h, (_, _, kind, x) in HYP.items():
            with self.subTest(h=h):
                self.assertEqual(verdicts(data(put(h, x * 1.01), noise=0.15))[h], "inconclusive")

    def test_a_slip_makes_the_cells_hypotheses_inconclusive(self):
        out = verdicts(data(slip={"run-D3": 6.0}))
        for h, (_, used, _, _) in HYP.items():
            with self.subTest(h=h):
                self.assertEqual(out[h], "inconclusive" if "run-D3" in used else "supported")

    def test_the_bath_alone_is_no_slip(self):
        self.assertEqual(verdicts(data(slip={c: 4.0 for c in CELLS})), dict.fromkeys(HYP, "supported"))

    def test_bonds_outside_the_shell_do_not_count(self):
        self.assertEqual(verdicts(data(outside={"run-D3": 3.0, "run-D2": 0.2})), dict.fromkeys(HYP, "supported"))

    def test_the_resonant_band_follows_the_measured_speed(self):
        # the cell turns at 2.2 and its spectrum is heated at 2.2's band, not at 2.59's
        self.assertEqual(verdicts(data(speed={"run-D3": 2.2}))["H6"], "supported")


class Values(unittest.TestCase):
    """Each statistic's point value against a reference computed here from the same synthetic arrays, made uneven on purpose
    (multiplicities, shells, bands) so that a weight or a region left out shows."""

    @classmethod
    def setUpClass(cls):
        cls.payload = data(noise=0.3)
        rng = np.random.Generator(np.random.PCG64(11))
        for name, x in cls.payload["cells"].items():
            for f in ("E", "c", "var", "spectrum"):
                a = np.asarray(x[f])
                x[f] = (a * np.exp(0.5 * rng.standard_normal(a.shape[1]))).tolist()
        code, cls.out = run(cls.payload)
        assert code == 0, cls.out

    def mean(self, name, f):
        return np.asarray(self.payload["cells"][name][f]).mean(0)

    def value(self, h):
        return self.out["verdicts"][h]["value"]

    def test_field(self):
        for h in ("H1", "H2"):
            name = HYP[h][1][0]
            keys, bcount, pred, inside = bond_geometry(name)
            E = self.mean(name, "E")
            ref = (bcount[inside] * E[inside] * pred[inside]).sum() / (bcount[inside] * pred[inside] ** 2).sum()
            self.assertAlmostEqual(self.value(h), ref, places=9)

    def test_screening(self):
        name = HYP["H3"][1][0]
        x = self.payload["cells"][name]
        c, cnt = self.mean(name, "c"), np.asarray(x["site_count"])
        shell = np.array([max(k) for k in x["site_keys"]])
        flux = lambda r: -(cnt[shell <= r] * c[shell <= r]).sum()
        ref = flux(1) / flux(0) / (flux_linear(name, 1) / flux_linear(name, 0))
        self.assertAlmostEqual(self.value("H3"), ref, places=6)

    def test_time(self):
        for h in ("H4", "H5"):
            name = HYP[h][1][0]
            c, keys = CELLS[name], [tuple(k) for k in self.payload["cells"][name]["site_keys"]]
            L, D = c["L"], c["D"]
            var = self.mean(name, "var")
            ratio = var[keys.index(tuple(sorted((L // 4,) + (0,) * (D - 1))))] / var[keys.index(tuple(sorted((1,) + (0,) * (D - 1))))]
            G = G_of(L, D)
            g = lambda r: G[(r,) + (0,) * (D - 1)]
            ref = ratio if HYP[h][0] == "time" else ratio / ((g(0) - g(L // 4)) / (g(0) - g(1)))
            self.assertAlmostEqual(self.value(h), ref, places=9)

    def band(self, name, lo, hi):
        mids, mcount = lam_bins(CELLS[name]["L"], CELLS[name]["D"])
        s = self.mean(name, "spectrum")
        sel = (mids >= lo) & (mids < hi)
        return (s[sel] * mcount[sel]).sum() / mcount[sel].sum()

    def test_heating(self):
        name = HYP["H6"][1][0]
        v = self.mean(name, "speed").mean()
        ref = self.band(name, *resonant_band(name, v)) / self.band(name, 0.0, 2.0)
        self.assertAlmostEqual(self.value("H6"), ref, places=9)
        a, b = HYP["H7"][1]
        top = lambda n: self.band(n, 3.0, 4.0) / self.band(n, 0.0, 2.0)
        self.assertAlmostEqual(self.value("H7"), top(a) / top(b), places=9)


class Refusals(unittest.TestCase):
    def refused(self, payload, why):
        code, out = run(payload)
        self.assertNotEqual(code, 0, out)
        self.assertIn(why, out)

    def test_other_registration(self):
        self.refused(data(registration="x" * 64), "not of this registration")

    def test_a_missing_cell(self):
        p = data()
        del p["cells"]["rest-D2"]
        self.refused(p, "not the registered")

    def test_a_missing_trajectory(self):
        p = data()
        p["cells"]["run-D2"]["E"] = p["cells"]["run-D2"]["E"][:-1]
        self.refused(p, "is not")

    def test_orbits_of_another_lattice(self):
        p = data()
        x = p["cells"]["run-D2"]
        x["site_count"] = [n + (i == 3) for i, n in enumerate(x["site_count"])]
        self.refused(p, "site orbits")
        p = data()
        x = p["cells"]["run-D3"]
        x["bond_keys"] = x["bond_keys"][:-1] + [[99, 0, 0]]
        self.refused(p, "bond orbits")

    def test_a_short_range_list(self):
        p = data()
        p["cells"]["rest-D1"]["range"] = p["cells"]["rest-D1"]["range"][:-1]
        self.refused(p, "trajectories")


class Smoke(unittest.TestCase):
    def test_the_real_runner_end_to_end(self):
        if os.environ.get("ANALYZE"):
            self.skipTest("mutant runs exercise the analysis only")
        design = {**DES, "trajectories": 2, "warm": 4, "t_total": 12,
                  "cells": [{**c, "L": 8 if c["D"] == 3 else c["L"] // 4} for c in DES["cells"]]}
        cells = {c["name"]: R.observe({**design["base"], **c["params"]}, c["L"], c["D"], 2, design["delta"], 4, 12, 7 + i)
                 for i, c in enumerate(design["cells"])}
        code, out = run({"registration": "r" * 64, "cells": json.loads(json.dumps(cells))}, design)
        self.assertEqual(code, 0, out)
        self.assertEqual(set(out["verdicts"]), set(HYP))


if __name__ == "__main__":
    unittest.main()
