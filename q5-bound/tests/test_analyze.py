"""Controls for q5-bound/analyze.py on synthetic cells whose diffusions and heats are set in advance, the runner against
q4-line's physics, and an end-to-end smoke run of the real runner on a small copy of the design.

    python -m unittest q5-bound/tests/test_analyze.py      (ANALYZE=path runs another copy, for mutants)"""
import importlib.util, json, math, os, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ANALYZE = os.environ.get("ANALYZE", str(HERE / "analyze.py"))
SCRATCH = HERE / "runs"
REG = json.loads((HERE / "registration.json").read_text())
DES = REG["design"]
T = DES["trajectories"]
ND = DES["N"] * DES["base"]["d"]
S_MAIN, V, D0 = 2398.0, 2.59, 0.0028                 # eta(alone) = V² / (D0 S_MAIN) ≈ 1: near the bound
# (gain, the line's entropy rate) as the registered analysis found them on calibration seeds (calibration/full-analysis.json)
CALIB = {"alone": (1.0, 0.0), "ideal": (8.67, 0.0), "line-n8": (0.59, 686.0), "line-n32": (1.99, 2681.0),
         "line-n128": (4.66, 10660.0), "line-n512": (8.70, 42584.0)}
ALL = {"H1", "H2", "H3", "H4"}


def exact(rng, k):
    """k columns of T values, each of mean 0 and variance 1, mutually uncorrelated in the sample."""
    z = rng.standard_normal((T, k))
    q, _ = np.linalg.qr(z - z.mean(0))
    return q * np.sqrt(T)


def data(cells_in, jitter=None, drop=None, nan=None, n=T, speed=None, scale=1.0):
    """Cells whose slope-of-variance diffusion is exactly D = scale D0 / gain (a gain of inf: D = 0), with velocity V (or
    `speed[name]`), the crystal's heat rate S_MAIN and the line's given rate, plus a bounded jitter (variance
    `jitter[name]`) that does not grow. scale moves every cell, and so eta(alone), without moving any ratio."""
    rng = np.random.Generator(np.random.PCG64(7))
    t0, t1, t2 = DES["checkpoints"]
    s2 = DES["base"]["sigma"] ** 2
    cells = {}
    for c in DES["cells"]:
        g, s_line = cells_in[c["name"]]
        D = 0.0 if g == float("inf") else scale * D0 / g
        s = (jitter or {}).get(c["name"], 0.0) ** 0.5
        u = (speed or {}).get(c["name"], V)
        w1, w2, e0, e1, e2 = exact(rng, 5).T
        phi0 = u * t0 + s * e0
        phi1 = u * t1 + np.sqrt(2 * D * (t1 - t0)) * w1 + s * e1
        phi2 = u * t2 + np.sqrt(2 * D * (t1 - t0)) * w1 + np.sqrt(2 * D * (t2 - t1)) * w2 + s * e2
        J = [(x * ND)[:n].tolist() for x in (phi0, phi1, phi2)]
        heat = [[S_MAIN * s2 * t] * n for t in (t0, t1, t2)]
        heat_line = [[s_line * s2 * t] * n for t in (t0, t1, t2)]
        p = {"reference": c["reference"], "n_s": c.get("n_s", 0),
             "Lambda": DES["Lambda"][str(c["n_s"])] if c["reference"] == "line" else 0.0}
        cells[c["name"]] = {"p": p, "J": J, "heat": heat, "heat_line": heat_line, "lag": [[p["Lambda"]] * n] * 3,
                            "clipped": 0.0}
    if drop:
        del cells[drop]
    if nan:
        cells[nan]["J"][1][0] = float("nan")
    return cells


def run(cells, registration="r" * 64):
    SCRATCH.mkdir(exist_ok=True)
    d = Path(tempfile.mkdtemp(dir=SCRATCH))
    (d / "results").mkdir()
    shutil.copy(HERE / "registration.json", d)
    (d / "registration.lock.json").write_text(json.dumps({"registration": "r" * 64}))
    (d / "results" / "data.json").write_text(json.dumps({"registration": registration, "cells": cells}))
    p = subprocess.run([sys.executable, "-I", ANALYZE, str(d)], capture_output=True, text=True)
    return p.returncode, (json.loads(p.stdout) if p.returncode == 0 else p.stdout + p.stderr)


def gain_for_eta(target, s_line):
    """The gain a line cell needs for eta(cell) / eta(alone) = target, with the line's heat s_line."""
    return target * (S_MAIN + s_line) / S_MAIN


class Controls(unittest.TestCase):
    def verdicts(self, cells):
        code, out = run(cells)
        self.assertEqual(code, 0, out)
        return {h: v["verdict"] for h, v in out["verdicts"].items()}

    def test_calibration_like_cells(self):
        self.assertEqual(self.verdicts(data(CALIB)), dict.fromkeys(ALL, "supported"))

    def test_a_crystal_far_from_the_bound_is_not_asked(self):
        code, out = run(data(CALIB, scale=4.0))                          # eta(alone) ≈ 0.25, every ratio the same
        self.assertEqual(code, 0, out)
        self.assertEqual(out["verdicts"]["H2"]["verdict"], "inconclusive")
        self.assertIn("not near the bound", out["verdicts"]["H2"]["reason"])
        self.assertEqual(out["verdicts"]["H1"]["verdict"], "supported")

    def test_a_crystal_barely_near_the_bound_by_its_point_only(self):
        code, out = run(data(CALIB, scale=1.0 / 0.63))                   # eta(alone) ≈ 0.63, its interval reaches below 0.6
        self.assertEqual(code, 0, out)
        self.assertLess(out["verdicts"]["H2"]["eta_alone"]["ci95"][0], 0.6)
        self.assertEqual(out["verdicts"]["H2"]["verdict"], "inconclusive")

    def test_efficiency_that_is_for_sale(self):
        self.assertEqual(self.verdicts(data({**CALIB, "line-n128": (gain_for_eta(2.0, 10660.0), 10660.0)}))["H2"], "contradicted")

    def test_precision_that_is_not_for_sale(self):
        self.assertEqual(self.verdicts(data({**CALIB, "line-n128": (1.5, 10660.0)}))["H1"], "contradicted")

    def test_a_light_line_that_helps(self):
        self.assertEqual(self.verdicts(data({**CALIB, "line-n8": (2.0, 686.0)}))["H3"], "contradicted")

    def test_a_heavy_line_that_pays(self):
        self.assertEqual(self.verdicts(data({**CALIB, "line-n512": (gain_for_eta(1.0, 42584.0), 42584.0)}))["H4"], "contradicted")

    def test_a_heavy_line_whose_clock_runs_faster(self):
        # the same diffusion at 1.5 times the velocity is 2.25 times the precision per entropy
        self.assertEqual(self.verdicts(data(CALIB, speed={"line-n512": 1.5 * V}))["H4"], "contradicted")

    def test_ratios_at_the_thresholds_are_inconclusive(self):
        v = self.verdicts(data({**CALIB, "line-n128": (3.0, 10660.0), "line-n8": (1.0, 686.0),
                                "line-n512": (gain_for_eta(1 / 1.5, 42584.0), 42584.0)}))
        self.assertEqual((v["H1"], v["H3"], v["H4"]), ("inconclusive",) * 3)
        self.assertEqual(self.verdicts(data({**CALIB, "line-n128": (gain_for_eta(1.25, 10660.0), 10660.0)}))["H2"], "inconclusive")

    def test_a_jitter_that_does_not_grow_cancels(self):
        # var(Phi(t2) - Phi(t0)) / (2 (t2 - t0)) would put line-n128's gain below 3 here
        self.assertEqual(self.verdicts(data(CALIB, jitter={"line-n128": 1.0}))["H1"], "supported")

    def test_a_line_without_diffusion_is_infinitely_precise(self):
        code, out = run(data({**CALIB, "line-n128": (float("inf"), 10660.0)}, jitter={"line-n128": 1e-3}))
        self.assertEqual(code, 0, out)
        self.assertEqual(out["verdicts"]["H1"]["verdict"], "supported")

    def test_two_cells_without_any_spread_have_no_ratio(self):
        code, out = run(data({**CALIB, "line-n128": (float("inf"), 10660.0), "alone": (float("inf"), 0.0)}))
        self.assertEqual(code, 0, out)
        self.assertEqual(out["verdicts"]["H1"]["verdict"], "inconclusive")
        self.assertIn("undefined", out["verdicts"]["H1"]["reason"])

    def test_a_missing_cell_is_refused(self):
        code, out = run(data(CALIB, drop=DES["cells"][0]["name"]))
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)

    def test_short_cells_are_refused(self):
        code, out = run(data(CALIB, n=T - 1))
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)

    def test_a_non_finite_value_is_refused(self):
        code, out = run(data(CALIB, nan=DES["cells"][3]["name"]))
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)

    def test_data_of_another_registration_are_refused(self):
        code, out = run(data(CALIB), registration="x" * 64)
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Runner(unittest.TestCase):
    def test_q4_line_with_gear_one(self):
        """The copied physics: with gear 1 and the line in the crystal's bath, q5's runner is q4-line's bit for bit, for
        every reference."""
        if os.environ.get("ANALYZE"):
            self.skipTest("mutant runs exercise the analysis only")
        q4 = module(HERE.parent / "q4-line" / "run.py", "q4run")
        q5 = module(HERE / "run.py", "q5run")
        for cell in ({"name": "alone", "reference": "alone"}, {"name": "ideal", "reference": "ideal"},
                     {"name": "line-n8", "reference": "line", "n_s": 8}):
            p5 = {**q5.cell_params(DES, cell), "N": 4, "gear": 1.0, "gamma_line": DES["base"]["gamma"]}
            p4 = {k: x for k, x in p5.items() if k not in ("gear", "gamma_line")}
            a, b = q4.run(p4, 8, 40.0, 5.0, 11, 12), q5.run(p5, 8, 40.0, 5.0, 11, 12)
            for x, y in zip(a, b):
                np.testing.assert_array_equal(y, x)

    def test_a_geared_line_on_a_free_crystal_lags_by_its_stages(self):
        """With k = 0 the line does not touch the crystal, and its end lags Phi / r by about L arcsin(gamma_line v / (r K))."""
        if os.environ.get("ANALYZE"):
            self.skipTest("mutant runs exercise the analysis only")
        q5 = module(HERE / "run.py", "q5run")
        p = {**q5.cell_params(DES, {"name": "line-n32", "reference": "line", "n_s": 32}), "k": 0.0}
        J, Q, Ql, lag, _ = q5.run(p, 32, 200.0, 50.0, 5, 6)
        J0, *_ = q5.run({**p, "reference": "alone"}, 32, 200.0, 50.0, 5, 6)
        np.testing.assert_array_equal(J, J0)
        line = DES["line"]
        v = J[:, -1].mean() / (200.0 * ND)
        want = line["L"] * math.asin(line["gamma_line"] * v / (line["gear"] * line["K"]))
        self.assertAlmostEqual(float(lag[:, -1].mean()), want, delta=0.05 * want)
        self.assertTrue((Ql[:, -1] > 0).all())


class Smoke(unittest.TestCase):
    """The real runner end to end on the registered cells with 16 trajectories over short checkpoints, then the analysis."""

    def test_runner_then_analysis(self):
        if os.environ.get("ANALYZE"):
            self.skipTest("mutant runs exercise the analysis only")
        SCRATCH.mkdir(exist_ok=True)
        d = Path(tempfile.mkdtemp(dir=SCRATCH))
        shutil.copy(HERE / "run.py", d)
        reg = json.loads((HERE / "registration.json").read_text())
        reg["design"].update(trajectories=16, checkpoints=[40, 60, 80], every=10)
        (d / "registration.json").write_text(json.dumps(reg))
        (d / "registration.lock.json").write_text(json.dumps({"registration": "s" * 64}))
        (d / "beacon.json").write_text(json.dumps({"attest": "b" * 64, "beacon": {"randomness": "c" * 64}}))
        p = subprocess.run([sys.executable, "-I", str(d / "run.py"), "--jobs", "2"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        out = json.loads((d / "results" / "data.json").read_text())
        self.assertEqual(list(out["cells"]), [c["name"] for c in reg["design"]["cells"]])
        self.assertTrue(all(len(c["J"]) == 3 and all(len(x) == 16 for x in c["J"]) for c in out["cells"].values()))
        lines = [c for c in out["cells"].values() if c["p"]["reference"] == "line"]
        self.assertTrue(lines and all(np.mean(c["heat_line"][2]) > 0 for c in lines))
        a = subprocess.run([sys.executable, "-I", str(HERE / "analyze.py"), str(d)], capture_output=True, text=True)
        self.assertEqual(a.returncode, 0, a.stdout + a.stderr)
        self.assertEqual(set(json.loads(a.stdout)["verdicts"]), ALL)


if __name__ == "__main__":
    unittest.main()
