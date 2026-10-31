"""Controls for q4-line/analyze.py on synthetic cells whose diffusions and heats are set in advance, the runner against
q3-self's physics, and an end-to-end smoke run of the real runner on a small copy of the design.

    python -m unittest q4-line/tests/test_analyze.py      (ANALYZE=path runs another copy, for mutants)"""
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
S_MAIN = 450.0
# (gain, the line's entropy rate) as calibration found them at k = 0.16 (calibration/full.txt)
CALIB = {"alone": (1.0, 0.0), "ideal": (5.25, 0.0), "line-n4": (0.034, 359.0), "line-n8": (3.96, 698.0),
         "line-n16": (5.25, 1374.0), "line-n32": (5.45, 2728.0)}
ALL = {"H1", "H2", "H3", "H4"}


def exact(rng, k):
    """k columns of T values, each of mean 0 and variance 1, mutually uncorrelated in the sample."""
    z = rng.standard_normal((T, k))
    q, _ = np.linalg.qr(z - z.mean(0))
    return q * np.sqrt(T)


def data(cells_in, jitter=None, drop=None, nan=None, n=T, speed=None):
    """Cells whose slope-of-variance diffusion is exactly D = 0.002 / gain (a gain of inf: D = 0), with velocity 0.5 (or
    `speed[name]`), the crystal's heat rate S_MAIN and the line's given rate, plus a bounded jitter (variance
    `jitter[name]`) that does not grow."""
    rng = np.random.Generator(np.random.PCG64(7))
    t0, t1, t2 = DES["checkpoints"]
    s2 = DES["base"]["sigma"] ** 2
    cells = {}
    for c in DES["cells"]:
        g, s_line = cells_in[c["name"]]
        D = 0.0 if g == float("inf") else 0.002 / g
        s = (jitter or {}).get(c["name"], 0.0) ** 0.5
        u = (speed or {}).get(c["name"], 0.5)
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


def eta_ratio_gain(target, a_line, b):
    """The gain that line-n8 needs for eta(line-n8) / eta(b) = target, b given as (gain, line heat)."""
    gb, sb = b
    return target * gb * (S_MAIN + a_line) / (S_MAIN + sb)


class Controls(unittest.TestCase):
    def verdicts(self, cells):
        code, out = run(cells)
        self.assertEqual(code, 0, out)
        return {h: v["verdict"] for h, v in out["verdicts"].items()}

    def test_calibration_like_cells(self):
        self.assertEqual(self.verdicts(data(CALIB)), dict.fromkeys(ALL, "supported"))

    def test_a_line_that_is_a_poor_memory(self):
        self.assertEqual(self.verdicts(data({**CALIB, "line-n16": (1.5, 1374.0)}))["H1"], "contradicted")

    def test_a_line_better_than_a_perfect_buffer_by_far(self):
        self.assertEqual(self.verdicts(data({**CALIB, "line-n16": (5.25 * 3, 1374.0)}))["H1"], "contradicted")

    def test_a_line_that_costs_more_than_it_gives(self):
        v = self.verdicts(data({**CALIB, "line-n8": (1.5, 698.0)}))
        self.assertEqual((v["H2"], v["H4"]), ("contradicted", "contradicted"))

    def test_a_line_that_slows_the_clock_it_steadies(self):
        # the same diffusion at half the velocity is a quarter of the precision per entropy
        self.assertEqual(self.verdicts(data(CALIB, speed={"line-n8": 0.25}))["H2"], "contradicted")

    def test_a_light_line_that_merely_adds_noise(self):
        self.assertEqual(self.verdicts(data({**CALIB, "line-n4": (0.8, 359.0)}))["H3"], "contradicted")

    def test_ratios_at_the_thresholds_are_inconclusive(self):
        g8 = eta_ratio_gain(1.0, 698.0, CALIB["alone"])                      # eta(n8) = eta(alone)
        v = self.verdicts(data({**CALIB, "line-n8": (g8, 698.0), "line-n16": (5.25 * 2, 1374.0), "line-n4": (0.5, 359.0)}))
        self.assertEqual((v["H1"], v["H2"], v["H3"]), ("inconclusive",) * 3)
        g8 = eta_ratio_gain(1.5, 698.0, CALIB["line-n32"])                   # eta(n8) = 1.5 eta(n32)
        self.assertEqual(self.verdicts(data({**CALIB, "line-n8": (g8, 698.0)}))["H4"], "inconclusive")

    def test_a_jitter_that_does_not_grow_cancels(self):
        # var(Phi(t2) - Phi(t0)) / (2 (t2 - t0)) would put line-n16 below half of the perfect buffer here
        self.assertEqual(self.verdicts(data(CALIB, jitter={"line-n16": 0.6}))["H1"], "supported")

    def test_a_line_without_diffusion_is_infinitely_precise(self):
        code, out = run(data({**CALIB, "line-n16": (float("inf"), 1374.0)}, jitter={"line-n16": 1e-4}))
        self.assertEqual(code, 0, out)
        self.assertEqual(out["verdicts"]["H1"]["verdict"], "contradicted")

    def test_two_cells_without_any_spread_have_no_ratio(self):
        code, out = run(data({**CALIB, "line-n16": (float("inf"), 1374.0), "ideal": (float("inf"), 0.0)}))
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
    def test_alone_and_ideal_are_q3_self(self):
        """The copied physics: without a line, q4's runner reproduces q3-self's runner bit for bit, alone and with the
        perfect delay line (q4 keeps it in a ring of tau, q3 in the whole history)."""
        if os.environ.get("ANALYZE"):
            self.skipTest("mutant runs exercise the analysis only")
        q3 = module(HERE.parent / "q3-self" / "run.py", "q3run")
        q4 = module(HERE / "run.py", "q4run")
        for ref in ("alone", "ideal"):
            p4 = {**q4.cell_params(DES, {"name": ref, "reference": ref}), "N": 4}
            p3 = {**DES["base"], "N": 4, "reference": ref, "k": p4["k"], "tau": p4["tau"], "N_mem": 0, "v_star": DES["v_star"],
                  "v_mem": 0.0}
            a, b = q3.run(p3, 8, 40.0, 5.0, 11), q4.run(p4, 8, 40.0, 5.0, 11)
            np.testing.assert_array_equal(b[0], a[0])
            np.testing.assert_array_equal(b[1], a[1])

    def test_a_line_on_a_free_crystal_lags_by_its_stages(self):
        """With k = 0 the line does not touch the crystal, and its end lags by about L arcsin(gamma v / K)."""
        if os.environ.get("ANALYZE"):
            self.skipTest("mutant runs exercise the analysis only")
        q4 = module(HERE / "run.py", "q4run")
        p = {**q4.cell_params(DES, {"name": "line-n32", "reference": "line", "n_s": 32}), "k": 0.0}
        J, Q, Ql, lag, _ = q4.run(p, 32, 200.0, 50.0, 5, 6)
        J0, Q0, *_ = q4.run({**p, "reference": "alone"}, 32, 200.0, 50.0, 5, 6)
        np.testing.assert_array_equal(J, J0)
        v = J[:, -1].mean() / (200.0 * ND)
        want = DES["line"]["L"] * math.asin(DES["base"]["gamma"] * v / DES["line"]["K"])
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
