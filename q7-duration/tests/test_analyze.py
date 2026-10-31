"""Controls for q7-duration/analyze.py on synthetic errors whose variances are set in advance, the runner's parts against
q6-renewal's physics and the world's own arithmetic, and an end-to-end smoke run of the real runner on a small copy of the
design.

    python -m unittest q7-duration/tests/test_analyze.py      (ANALYZE=path runs another copy, for mutants)"""
import importlib.util, json, math, os, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ANALYZE = os.environ.get("ANALYZE", str(HERE / "analyze.py"))
SCRATCH = HERE / "runs"
REG = json.loads((HERE / "registration.json").read_text())
DES = REG["design"]
T = DES["trajectories"]
TAU = DES["T"]
STAR = {c["name"]: c["D_star"] for c in DES["clocks"]}
KOF = {w["name"]: w["k"] for w in DES["worlds"]}
ALL = {"H1", "H2", "H3", "H4", "H5"}


def exact(rng, k):
    """k columns of T values, each of mean 0 and variance 1, mutually uncorrelated in the sample."""
    z = rng.standard_normal((T, k))
    q, _ = np.linalg.qr(z - z.mean(0))
    return q * np.sqrt(T)


def data(count=None, own=None, drop=None, nan=None, n=T, ragged=None, queries=None, zero=False):
    """Errors as the blind formula has them: counting with variance m T² / k (times count[(world, m)]), each clock with
    2 D* m T (times own[(clock, world, m)]), the fused error w clock + (1 - w) count with the registered weight; heats
    and currents of a steady clock. zero sets every error to 0."""
    rng = np.random.Generator(np.random.PCG64(7))
    ms = [str(m) for m in (queries or DES["queries"])]
    t0, t1, t2 = DES["checkpoints"]
    s2 = DES["base"]["sigma"] ** 2
    out = {"registration": "r" * 64, "worlds": {}, "clocks": {}}
    cerr = {}
    for w in DES["worlds"]:
        cerr[w["name"]] = {}
        out["worlds"][w["name"]] = {"span": {}, "err_count": {}}
        for m in ms:
            var = int(m) * TAU ** 2 / w["k"] * (count or {}).get((w["name"], m), 1.0)
            e = 0 * exact(rng, 1)[:, 0] if zero else math.sqrt(var) * exact(rng, 1)[:, 0]
            cerr[w["name"]][m] = e
            out["worlds"][w["name"]]["span"][m] = [int(m) * TAU] * n
            out["worlds"][w["name"]]["err_count"][m] = e[:n].tolist()
    for c in DES["clocks"]:
        v, D = c["v_star"], c["D_star"] * c["v_star"] ** 2
        z = exact(rng, 2)
        phi = [v * t0 + 0 * z[:, 0], v * t1 + math.sqrt(2 * D * (t1 - t0)) * z[:, 0],
               v * t2 + math.sqrt(2 * D * (t1 - t0)) * z[:, 0] + math.sqrt(2 * D * (t2 - t1)) * z[:, 1]]
        nd = c["N"] * DES["base"]["d"]
        got = {"p": {"N": c["N"]}, "J": [(x * nd)[:n].tolist() for x in phi],
               "heat": [[75.0 * c["N"] * s2 * t] * n for t in (t0, t1, t2)], "worlds": {}}
        for w in DES["worlds"]:
            wt = (TAU / w["k"]) / (TAU / w["k"] + 2 * c["D_star"])
            got["worlds"][w["name"]] = {"err_clock": {}, "err_fused": {}}
            for m in ms:
                var = 2 * c["D_star"] * int(m) * TAU * (own or {}).get((c["name"], w["name"], m), 1.0)
                e = 0 * exact(rng, 1)[:, 0] if zero else math.sqrt(var) * exact(rng, 1)[:, 0]
                got["worlds"][w["name"]]["err_clock"][m] = e[:n].tolist()
                got["worlds"][w["name"]]["err_fused"][m] = (wt * e + (1 - wt) * cerr[w["name"]][m])[:n].tolist()
        out["clocks"][c["name"]] = got
    if drop:
        kind, name = drop
        del out[kind][name]
    if nan:
        out["clocks"][nan]["J"][1][0] = float("nan")
    if ragged:
        out["clocks"][ragged]["worlds"]["k64"]["err_fused"]["16"].pop()
    return out


def run(payload, registration=None):
    SCRATCH.mkdir(exist_ok=True)
    d = Path(tempfile.mkdtemp(dir=SCRATCH))
    (d / "results").mkdir()
    shutil.copy(HERE / "registration.json", d)
    (d / "registration.lock.json").write_text(json.dumps({"registration": "r" * 64}))
    if registration:
        payload = {**payload, "registration": registration}
    (d / "results" / "data.json").write_text(json.dumps(payload))
    p = subprocess.run([sys.executable, "-I", ANALYZE, str(d)], capture_output=True, text=True)
    return p.returncode, (json.loads(p.stdout) if p.returncode == 0 else p.stdout + p.stderr)


def R(clock, world, c_scale=1.0, k_scale=1.0):
    """The formula's MSE(count) / MSE(clock) when counting's variance is k_scale and the clock's c_scale times the
    formula's."""
    return (TAU / KOF[world] * k_scale) / (2 * STAR[clock] * c_scale)


class Controls(unittest.TestCase):
    def verdicts(self, payload):
        code, out = run(payload)
        self.assertEqual(code, 0, out)
        return {h: v["verdict"] for h, v in out["verdicts"].items()}

    def test_errors_as_the_formula_has_them(self):
        self.assertEqual(self.verdicts(data()), dict.fromkeys(ALL, "supported"))

    def test_a_poisson_world_where_counting_rivals_the_clock(self):
        self.assertEqual(self.verdicts(data(count={("k1", "64"): 1 / 30}))["H1"], "contradicted")

    def test_a_clock_twice_as_diffusive_as_measured(self):
        self.assertEqual(self.verdicts(data(own={("N1", "k256", "64"): 2.5}))["H2"], "contradicted")

    def test_counting_twice_as_good_as_the_world_allows(self):
        self.assertEqual(self.verdicts(data(count={("k256", "64"): 0.4}))["H2"], "contradicted")

    def test_the_formula_is_the_world_s_shape(self):
        # counting and the clock both four times as precise in k256 keep R: H2 reads the world's own k, not the errors' size
        self.assertEqual(self.verdicts(data(count={("k256", "64"): 0.25}, own={("N1", "k256", "64"): 0.25}))["H2"], "supported")
        self.assertEqual(self.verdicts(data(count={("k256", "64"): 0.25}))["H2"], "contradicted")

    def test_the_formula_is_the_clock_s_diffusion(self):
        # the clock of 4 has a quarter of N1's D*: its errors in k256, put in N1's place, are the formula's only for N4
        self.assertEqual(self.verdicts(data(own={("N1", "k256", "64"): STAR["N4"] / STAR["N1"]}))["H2"], "contradicted")

    def test_a_scaling_that_breaks(self):
        self.assertEqual(self.verdicts(data(own={("N16", "k4096", "64"): 3.0}))["H3"], "contradicted")

    def test_a_regular_world_where_the_clock_still_pays(self):
        self.assertEqual(self.verdicts(data(count={("k4096", "64"): 10.0}))["H4"], "contradicted")

    def test_a_worth_that_depends_on_the_span(self):
        self.assertEqual(self.verdicts(data(own={("N1", "k256", "16"): 3.0}))["H5"], "contradicted")

    def test_the_fused_estimate_does_not_decide(self):
        # the registered weight's fused errors made useless: no verdict moves
        payload = data()
        for c in payload["clocks"].values():
            for w in c["worlds"].values():
                for m in w["err_fused"]:
                    w["err_fused"][m] = [x * 50 for x in w["err_fused"][m]]
        self.assertEqual(self.verdicts(payload), dict.fromkeys(ALL, "supported"))

    def test_ratios_at_the_thresholds_are_inconclusive(self):
        """Each test's point exactly at its threshold, one test at a time (a scale moves every test that reads that cell)."""
        cases = {"H1": data(count={("k1", "64"): 100 / R("N1", "k1")}), "H2": data(own={("N1", "k256", "64"): 4 / 3}),
                 "H3": data(own={("N16", "k4096", "64"): R("N16", "k4096") / (1.5 * R("N1", "k256"))}),
                 "H4": data(count={("k4096", "64"): 0.5 / R("N1", "k4096")}), "H5": data(own={("N1", "k256", "16"): 1.5})}
        self.assertEqual({h: self.verdicts(d)[h] for h, d in cases.items()}, dict.fromkeys(ALL, "inconclusive"))

    def test_errors_that_are_zero_have_no_ratio(self):
        code, out = run(data(zero=True))
        self.assertEqual(code, 0, out)
        self.assertEqual({h: v["verdict"] for h, v in out["verdicts"].items()}, dict.fromkeys(ALL, "inconclusive"))
        self.assertIn("undefined", out["verdicts"]["H1"]["reason"])

    def test_a_missing_clock_is_refused(self):
        code, out = run(data(drop=("clocks", "N4")))
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)

    def test_a_missing_world_is_refused(self):
        code, out = run(data(drop=("worlds", "k64")))
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)

    def test_a_missing_query_is_refused(self):
        code, out = run(data(queries=[64]))
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)

    def test_short_rows_are_refused(self):
        code, out = run(data(n=T - 1))
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)

    def test_a_ragged_row_is_refused(self):
        code, out = run(data(ragged="N16"))
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)

    def test_a_non_finite_value_is_refused(self):
        code, out = run(data(nan="N16"))
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)

    def test_data_of_another_registration_are_refused(self):
        code, out = run(data(), registration="x" * 64)
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Runner(unittest.TestCase):
    def setUp(self):
        if os.environ.get("ANALYZE"):
            self.skipTest("mutant runs exercise the analysis only")
        self.q7 = module(HERE / "run.py", "q7run")

    def test_the_crystal_is_q6_renewal_s(self):
        q6 = module(HERE.parent / "q6-renewal" / "run.py", "q6run")
        p = {**DES["base"], "N": 4}
        for x, y in zip(q6.trace(p, 8, 40.0, (10.0, 20.0, 40.0), 11), self.q7.trace(p, 8, 40.0, (10.0, 20.0, 40.0), 11)):
            np.testing.assert_array_equal(y, x)

    def test_worlds(self):
        """The first event at the start; gamma gaps of mean T and variance T² / k plus h² / 6 from the grid; refused when
        an event the queries need falls past the trajectory."""
        h = 0.25
        for k in (1, 16, 4096):
            e = self.q7.world(k, TAU, 2048, DES["start"], 12000, h, 64, 3)
            self.assertTrue((e[:, 0] == int(DES["start"] / h)).all())
            gaps = np.diff(e, axis=1).ravel() * h
            self.assertAlmostEqual(gaps.mean() / TAU, 1.0, delta=0.01)
            self.assertAlmostEqual(gaps.var() / (TAU ** 2 / k + h * h / 6), 1.0, delta=0.05)
        with self.assertRaises(RuntimeError):
            self.q7.world(1, TAU, 64, DES["start"], int(1200 / h), h, 64, 3)


class Smoke(unittest.TestCase):
    """The real runner end to end on the registered clocks and worlds with 16 trajectories over a short span, the errors'
    arithmetic checked row by row, then the analysis. Only shapes and identities are read, never the errors' sizes: the
    predictions are blind."""

    def test_runner_then_analysis(self):
        if os.environ.get("ANALYZE"):
            self.skipTest("mutant runs exercise the analysis only")
        SCRATCH.mkdir(exist_ok=True)
        d = Path(tempfile.mkdtemp(dir=SCRATCH))
        shutil.copy(HERE / "run.py", d)
        reg = json.loads((HERE / "registration.json").read_text())
        reg["design"].update(trajectories=16, start=40.0, checkpoints=[40, 800, 2600], t_total=2600)
        (d / "registration.json").write_text(json.dumps(reg))
        (d / "registration.lock.json").write_text(json.dumps({"registration": "s" * 64}))
        (d / "beacon.json").write_text(json.dumps({"attest": "b" * 64, "beacon": {"randomness": "c" * 64}}))
        p = subprocess.run([sys.executable, "-I", str(d / "run.py"), "--jobs", "2"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        out = json.loads((d / "results" / "data.json").read_text())
        self.assertEqual(list(out["clocks"]), [c["name"] for c in reg["design"]["clocks"]])
        for c in reg["design"]["clocks"]:
            for w in reg["design"]["worlds"]:
                wt = (TAU / w["k"]) / (TAU / w["k"] + 2 * c["D_star"])
                for m in ("16", "64"):
                    cnt = np.asarray(out["worlds"][w["name"]]["err_count"][m])
                    np.testing.assert_allclose(np.asarray(out["worlds"][w["name"]]["span"][m]) + cnt, int(m) * TAU)
                    got = out["clocks"][c["name"]]["worlds"][w["name"]]
                    np.testing.assert_allclose(got["err_fused"][m], wt * np.asarray(got["err_clock"][m]) + (1 - wt) * cnt, atol=1e-9)
        a = subprocess.run([sys.executable, "-I", str(HERE / "analyze.py"), str(d)], capture_output=True, text=True)
        self.assertEqual(a.returncode, 0, a.stdout[-500:] + a.stderr[-2000:])
        self.assertEqual(set(json.loads(a.stdout)["verdicts"]), ALL)


if __name__ == "__main__":
    unittest.main()
