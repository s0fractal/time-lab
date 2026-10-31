"""Controls for q3-self/analyze.py on synthetic cells whose diffusions are set in advance, the runner against q2-crystal's
physics, and an end-to-end smoke run of the real runner on a small copy of the design.

    python -m unittest q3-self/tests/test_analyze.py      (ANALYZE=path runs another copy, for mutants)"""
import importlib.util, json, os, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ANALYZE = os.environ.get("ANALYZE", str(HERE / "analyze.py"))
SCRATCH = HERE / "runs"
REG = json.loads((HERE / "registration.json").read_text())
DES = REG["design"]
T = DES["trajectories"]
ND = DES["N"] * DES["base"]["d"]
# gains G = D(alone) / D(cell) as calibration found them at k = 0.32 (calibration/ticks.txt, design.txt); the short span of
# the N32 memory from the linear theory (it is not calibrated)
THEORY = {"alone": 1.0, "ideal-long": 75.0, "ideal-short": 5.7, "memory-N8-long": 0.44, "memory-N32-long": 1.29,
          "memory-N128-long": 5.3, "memory-N32-short": 1.95, "memory-N128-offtick": 1.9}
ALL = {"H1", "H2", "H3", "H4", "H5", "H6"}


def exact(rng, k):
    """k columns of T values, each of mean 0 and variance 1, mutually uncorrelated in the sample."""
    z = rng.standard_normal((T, k))
    q, _ = np.linalg.qr(z - z.mean(0))
    return q * np.sqrt(T)


def data(gains, jitter=None, drop=None, nan=None, n=T):
    """Cells whose slope-of-variance diffusion is exactly D = 1 / gain (a gain of inf: D = 0), plus a bounded jitter
    (variance `jitter[name]` at every checkpoint) that does not grow with time."""
    rng = np.random.Generator(np.random.PCG64(7))
    t0, t1, t2 = DES["checkpoints"]
    cells = {}
    for c in DES["cells"]:
        g = gains[c["name"]]
        D = 0.0 if g == float("inf") else 1.0 / g
        s = (jitter or {}).get(c["name"], 0.0) ** 0.5
        w1, w2, e0, e1, e2 = exact(rng, 5).T
        phi0 = s * e0
        phi1 = 0.5 * t1 + np.sqrt(2 * D * (t1 - t0)) * w1 + s * e1
        phi2 = 0.5 * t2 + np.sqrt(2 * D * (t1 - t0)) * w1 + np.sqrt(2 * D * (t2 - t1)) * w2 + s * e2
        J = [(x * ND)[:n].tolist() for x in (phi0, phi1, phi2)]
        heat = [[0.0] * n, [100.0 * t1] * n, [100.0 * t2] * n]
        cells[c["name"]] = {"p": {"reference": c["reference"], "N_mem": c.get("N_mem", 0)}, "J": J, "heat": heat, "J_memory": [[0.0] * n] * 3, "heat_memory": [[0.0] * n] * 3, "clipped": 0.0}
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


class Controls(unittest.TestCase):
    def verdicts(self, cells):
        code, out = run(cells)
        self.assertEqual(code, 0, out)
        return {h: v["verdict"] for h, v in out["verdicts"].items()}

    def test_the_linear_theory(self):
        self.assertEqual(self.verdicts(data(THEORY)), dict.fromkeys(ALL, "supported"))

    def test_reading_itself_gives_nothing(self):
        v = self.verdicts(data({k: 1.0 for k in THEORY}))
        self.assertEqual(v, {"H1": "contradicted", "H2": "contradicted", "H3": "contradicted", "H4": "supported",
                             "H5": "supported", "H6": "contradicted"})

    def test_reading_between_ticks_that_costs_a_third(self):
        self.assertEqual(self.verdicts(data({**THEORY, "memory-N128-offtick": 5.3 / 1.5}))["H6"], "contradicted")

    def test_a_perfect_memory_that_helps_fourfold(self):
        self.assertEqual(self.verdicts(data({**THEORY, "ideal-long": 4.0}))["H1"], "contradicted")

    def test_a_memory_that_gives_free_precision(self):
        v = self.verdicts(data({**THEORY, "memory-N32-long": 20.0, "memory-N32-short": 5.0}))
        self.assertEqual((v["H4"], v["H5"]), ("contradicted", "contradicted"))

    def test_a_longer_memory_that_hurts_fourfold(self):
        self.assertEqual(self.verdicts(data({**THEORY, "memory-N32-long": 0.45}))["H5"], "contradicted")

    def test_memory_size_that_helps_threefold(self):
        self.assertEqual(self.verdicts(data({**THEORY, "memory-N128-long": 0.37 * 3}))["H3"], "contradicted")

    def test_gains_at_the_thresholds_are_inconclusive(self):
        v = self.verdicts(data({**THEORY, "memory-N32-long": 2.0, "memory-N32-short": 1.0, "ideal-short": 75.0 / 2,
                                "memory-N128-offtick": 5.3 / 2}))
        self.assertEqual((v["H2"], v["H4"], v["H5"], v["H6"]), ("inconclusive",) * 4)

    def test_a_gain_just_above_a_threshold_within_the_noise(self):
        # the point lies 0.04 above log10 2, the interval (±0.07) straddles it
        self.assertEqual(self.verdicts(data({**THEORY, "memory-N128-offtick": 5.3 / (2 * 10 ** 0.04)}))["H6"], "inconclusive")

    def test_a_jitter_that_does_not_grow_cancels(self):
        # var(J(t2) - J(t0)) / (2 (t2 - t0)) would give the perfect long memory a gain below 10 here
        v = self.verdicts(data(THEORY, jitter={"ideal-long": 150.0, "alone": 150.0}))
        self.assertEqual(v["H1"], "supported")

    def test_a_memory_without_diffusion_is_infinitely_precise(self):
        code, out = run(data({**THEORY, "ideal-long": float("inf"), "ideal-short": float("inf")},
                             jitter={"ideal-long": 1.0, "ideal-short": 1.0}))
        self.assertEqual(code, 0, out)
        self.assertEqual(out["verdicts"]["H1"]["verdict"], "supported")
        self.assertEqual(out["verdicts"]["H2"]["verdict"], "inconclusive")

    def test_two_memories_without_any_spread_have_no_ratio(self):
        code, out = run(data({**THEORY, "ideal-long": float("inf"), "ideal-short": float("inf")}))
        self.assertEqual(code, 0, out)
        self.assertEqual(out["verdicts"]["H1"]["verdict"], "supported")
        self.assertEqual(out["verdicts"]["H2"]["verdict"], "inconclusive")
        self.assertIn("undefined", out["verdicts"]["H2"]["reason"])

    def test_a_missing_cell_is_refused(self):
        code, out = run(data(THEORY, drop=DES["cells"][0]["name"]))
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)

    def test_short_cells_are_refused(self):
        code, out = run(data(THEORY, n=T - 1))
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)

    def test_a_non_finite_value_is_refused(self):
        code, out = run(data(THEORY, nan=DES["cells"][3]["name"]))
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)

    def test_data_of_another_registration_are_refused(self):
        code, out = run(data(THEORY), registration="x" * 64)
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Runner(unittest.TestCase):
    def test_alone_is_q2_crystal_autonomous_clock(self):
        """The copied physics: without feedback, q3's runner reproduces q2-crystal's runner (F1 = 0, j_scale = 0)."""
        if os.environ.get("ANALYZE"):
            self.skipTest("mutant runs exercise the analysis only")
        q2 = module(HERE.parent / "q2-crystal" / "run.py", "q2run")
        q3 = module(HERE / "run.py", "q3run")
        p3 = {**q3.cell_params(DES, {"name": "alone", "reference": "alone"}), "N": 4}
        p2 = {**DES["base"], "N": 4, "sigma": DES["base"]["sigma"], "j_scale": 0.0, "F1": 0.0, "omega": 0.0, "crystal": 1}
        J2, Q2 = q2.run(p2, 8, 20.0, 5.0, 11)
        J3, Q3, Jm, Qm, clip = q3.run(p3, 8, 20.0, 5.0, 11)
        np.testing.assert_allclose(J3, J2, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(Q3, Q2, rtol=1e-12, atol=1e-12)
        self.assertEqual((float(np.abs(Jm).max()), float(np.abs(Qm).max()), float(clip.max())), (0.0, 0.0, 0.0))


class Smoke(unittest.TestCase):
    """The real runner end to end on the registered cells with 16 trajectories over short checkpoints, then the analysis."""

    def test_runner_then_analysis(self):
        if os.environ.get("ANALYZE"):
            self.skipTest("mutant runs exercise the analysis only")
        SCRATCH.mkdir(exist_ok=True)
        d = Path(tempfile.mkdtemp(dir=SCRATCH))
        shutil.copy(HERE / "run.py", d)
        reg = json.loads((HERE / "registration.json").read_text())
        reg["design"].update(trajectories=16, checkpoints=[20, 40, 60], every=10)
        (d / "registration.json").write_text(json.dumps(reg))
        (d / "registration.lock.json").write_text(json.dumps({"registration": "s" * 64}))
        (d / "beacon.json").write_text(json.dumps({"attest": "b" * 64, "beacon": {"randomness": "c" * 64}}))
        p = subprocess.run([sys.executable, "-I", str(d / "run.py"), "--jobs", "2"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        out = json.loads((d / "results" / "data.json").read_text())
        self.assertEqual(list(out["cells"]), [c["name"] for c in reg["design"]["cells"]])
        self.assertTrue(all(len(c["J"]) == 3 and all(len(x) == 16 for x in c["J"]) for c in out["cells"].values()))
        mem = [c for c in out["cells"].values() if c["p"]["reference"] == "memory"]
        self.assertTrue(mem and all(np.mean(c["heat_memory"][2]) > 0 for c in mem))
        a = subprocess.run([sys.executable, "-I", str(HERE / "analyze.py"), str(d)], capture_output=True, text=True)
        self.assertEqual(a.returncode, 0, a.stdout + a.stderr)
        self.assertEqual(set(json.loads(a.stdout)["verdicts"]), ALL)


if __name__ == "__main__":
    unittest.main()
