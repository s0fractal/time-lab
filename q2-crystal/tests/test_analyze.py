"""Controls for q2-crystal/analyze.py on synthetic cells whose lock gains are set in advance, and an end-to-end smoke run of
the real runner on a small copy of the design.

    python -m unittest q2-crystal/tests/test_analyze.py      (ANALYZE=path runs another copy, for mutants)"""
import json, os, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ANALYZE = os.environ.get("ANALYZE", str(HERE / "analyze.py"))
SCRATCH = HERE / "runs"
REG = json.loads((HERE / "registration.json").read_text())
T = REG["design"]["trajectories"]
CALIB = {("crystal", 2, 0.3): 2.0, ("crystal", 8, 0.3): 100.0, ("crystal", 32, 0.3): 2600.0,
         ("loose", 2, 0.3): 1.5, ("loose", 8, 0.3): 1.5, ("loose", 32, 0.3): 1.6,
         ("crystal", 2, 0.5): 1.4, ("crystal", 8, 0.5): 1.8, ("crystal", 32, 0.5): 20.0,
         ("loose", 2, 0.5): 1.0, ("loose", 8, 0.5): 1.2, ("loose", 32, 0.5): 1.3}


def data(gains, gyro=1.0, drop=None, nan=None, n=T):
    """Cells drawn so that var(off) / var(on) is the target gain; gyro scales the j = 0.1 cell's gain."""
    rng = np.random.Generator(np.random.PCG64(7))
    cells = {}
    for c in REG["design"]["cells"]:
        g = gains[(c["coupling"], c["N"], c["sigma"])] * (gyro if c["j_scale"] else 1.0)
        sd = 10.0 if c["F1"] == 0 else 10.0 / g ** 0.5
        x = rng.standard_normal(n)
        x = (x - x.mean()) / x.std() * sd + 100.0
        cells[c["name"]] = {"dJ": x.tolist(), "dQ": [0.0] * n}
    if drop:
        del cells[drop]
    if nan:
        cells[nan]["dJ"][0] = float("nan")
    return cells


def run(cells):
    SCRATCH.mkdir(exist_ok=True)
    d = Path(tempfile.mkdtemp(dir=SCRATCH))
    (d / "results").mkdir()
    shutil.copy(HERE / "registration.json", d)
    (d / "registration.lock.json").write_text(json.dumps({"registration": "r" * 64}))
    (d / "results" / "data.json").write_text(json.dumps({"registration": "r" * 64, "cells": cells}))
    p = subprocess.run([sys.executable, "-I", ANALYZE, str(d)], capture_output=True, text=True)
    return p.returncode, (json.loads(p.stdout) if p.returncode == 0 else p.stdout + p.stderr)


class Controls(unittest.TestCase):
    def verdicts(self, cells):
        code, out = run(cells)
        self.assertEqual(code, 0, out)
        return {h: v["verdict"] for h, v in out["verdicts"].items()}

    def test_calibration_like_gains(self):
        v = self.verdicts(data(CALIB, gyro=1.0))
        self.assertEqual((v["H1"], v["H2"], v["H3"]), ("supported", "supported", "supported"))
        self.assertEqual(v["H4"], "contradicted")

    def test_a_gyroscope_that_quarters_the_gain(self):
        self.assertEqual(self.verdicts(data(CALIB, gyro=0.25))["H4"], "supported")

    def test_no_collective_effect(self):
        flat = {k: 2.0 for k in CALIB}
        v = self.verdicts(data(flat))
        self.assertEqual((v["H1"], v["H2"], v["H3"]), ("contradicted", "contradicted", "contradicted"))

    def test_one_size_step_short_of_doubling(self):
        g = {**CALIB, ("crystal", 8, 0.3): 2.4}          # 2 -> 2.4 is ×1.2: one step fails
        self.assertEqual(self.verdicts(data(g))["H1"], "contradicted")

    def test_coupling_that_helps_less_than_tenfold(self):
        g = {**CALIB, ("crystal", 32, 0.3): 1.6 * 3}           # coupled ×3 over loose: helps, but not ×10
        self.assertEqual(self.verdicts(data(g))["H2"], "contradicted")

    def test_gains_at_the_thresholds_are_inconclusive(self):
        g = {**CALIB, ("crystal", 32, 0.5): 1.8 * 5}     # exactly ×5 at σ = 0.5
        self.assertEqual(self.verdicts(data(g))["H3"], "inconclusive")

    def test_a_missing_cell_is_refused(self):
        code, out = run(data(CALIB, drop=REG["design"]["cells"][0]["name"]))
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)

    def test_short_cells_are_refused(self):
        code, out = run(data(CALIB, n=T - 1))
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)

    def test_a_non_finite_value_is_refused(self):
        code, out = run(data(CALIB, nan=REG["design"]["cells"][3]["name"]))
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)


class Smoke(unittest.TestCase):
    """The real runner end to end on the registered cells with 16 trajectories over a short window, then the analysis."""

    def test_runner_then_analysis(self):
        if os.environ.get("ANALYZE"):
            self.skipTest("mutant runs exercise the analysis only")
        SCRATCH.mkdir(exist_ok=True)
        d = Path(tempfile.mkdtemp(dir=SCRATCH))
        shutil.copy(HERE / "run.py", d)
        reg = json.loads((HERE / "registration.json").read_text())
        reg["design"].update(trajectories=16, window=[10, 40])
        (d / "registration.json").write_text(json.dumps(reg))
        (d / "registration.lock.json").write_text(json.dumps({"registration": "s" * 64}))
        (d / "beacon.json").write_text(json.dumps({"attest": "b" * 64, "beacon": {"randomness": "c" * 64}}))
        p = subprocess.run([sys.executable, "-I", str(d / "run.py")], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        out = json.loads((d / "results" / "data.json").read_text())
        self.assertEqual(len(out["cells"]), len(reg["design"]["cells"]))
        self.assertTrue(all(len(c["dJ"]) == 16 for c in out["cells"].values()))
        a = subprocess.run([sys.executable, "-I", str(HERE / "analyze.py"), str(d)], capture_output=True, text=True)
        self.assertEqual(a.returncode, 0, a.stdout + a.stderr)
        self.assertEqual(set(json.loads(a.stdout)["verdicts"]), {"H1", "H2", "H3", "H4"})


if __name__ == "__main__":
    unittest.main()
