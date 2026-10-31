"""Controls for q6-renewal/analyze.py on synthetic clocks and worlds whose errors, diffusions and heats are set in advance,
the runner's parts against q5-bound's physics and against the world's own arithmetic, and an end-to-end smoke run of the
real runner on a small copy of the design.

    python -m unittest q6-renewal/tests/test_analyze.py      (ANALYZE=path runs another copy, for mutants)"""
import importlib.util, json, math, os, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ANALYZE = os.environ.get("ANALYZE", str(HERE / "analyze.py"))
SCRATCH = HERE / "runs"
REG = json.loads((HERE / "registration.json").read_text())
DES = REG["design"]
T = DES["trajectories"]
NQ = 1401                                            # queries per trajectory: every 1 from 400 to 1800
# as the registered analysis found them on calibration seeds (calibration/full-analysis.json)
WORLDS = {"k1": (402.93, 402.93, 19.95), "k4": (93.67, 80.58, 12.38), "k16": (46.58, 21.31, 10.49), "k64": (36.55, 5.649, 10.04),
          "k256": (34.16, 1.495, 9.915), "k1024": (33.54, 0.3897, 9.885), "k4096": (33.41, 0.1072, 9.879),
          "periodic": (33.34, 0.0, 9.875)}            # {world: (MSE none, MSE perfect, mean elapsed)}
CLOCKS = {"N1": (0.0924, 2.5953, 75.0), "N4": (0.02276, 2.5926, 299.7), "N16": (0.005842, 2.5923, 1198.8),
          "N64": (0.001353, 2.5923, 4795.3)}          # {clock: (D, v, S)}
MSE = {"N1": {"k1": 402.9, "k4": 80.61, "k16": 21.4, "k64": 5.802, "k256": 1.673, "k1024": 0.591, "k4096": 0.3142, "periodic": 0.2131},
       "N4": {"k1": 402.9, "k4": 80.59, "k16": 21.33, "k64": 5.689, "k256": 1.539, "k1024": 0.4431, "k4096": 0.1609, "periodic": 0.05559},
       "N16": {"k1": 402.9, "k4": 80.59, "k16": 21.31, "k64": 5.658, "k256": 1.506, "k1024": 0.4026, "k4096": 0.1201, "periodic": 0.0141},
       "N64": {"k1": 402.9, "k4": 80.59, "k16": 21.31, "k64": 5.65, "k256": 1.498, "k1024": 0.3928, "k4096": 0.1106, "periodic": 0.003632}}
ALL = {"H1", "H2", "H3", "H4", "H5"}


def exact(rng, k):
    """k columns of T values, each of mean 0 and variance 1, mutually uncorrelated in the sample."""
    z = rng.standard_normal((T, k))
    q, _ = np.linalg.qr(z - z.mean(0))
    return q * np.sqrt(T)


def data(mse=None, clocks=None, worlds=None, jitter=None, spread=0.05, drop=None, nan=None, n=T, short=None):
    """Clocks whose slope-of-variance diffusion is exactly D, at velocity v and entropy rate S, and whose squared errors
    in each world sum, over the trajectories, to exactly MSE times the queries (each trajectory off by a factor of
    1 + spread z); worlds whose none and perfect errors and elapsed times do the same. jitter[clock] adds a bounded
    jitter of that variance to the clock's phase; short names a clock whose current lacks a trajectory."""
    mse = {c: {**MSE[c], **(mse or {}).get(c, {})} for c in MSE}
    clocks, worlds = {**CLOCKS, **(clocks or {})}, {**WORLDS, **(worlds or {})}
    rng = np.random.Generator(np.random.PCG64(7))
    t0, t1, t2 = DES["checkpoints"]
    s2 = DES["base"]["sigma"] ** 2
    noisy = lambda x: (x * NQ * (1 + spread * exact(rng, 1)[:, 0]))[:n].tolist()
    out = {"registration": "r" * 64, "worlds": {}, "clocks": {}}
    for w in DES["worlds"]:
        none, perfect, elapsed = worlds[w["name"]]
        out["worlds"][w["name"]] = {"sse_none": noisy(none), "sum_none": [0.0] * n, "sse_perfect": noisy(perfect),
                                    "sum_perfect": [0.0] * n, "sum_elapsed": noisy(elapsed), "n": [NQ] * n}
    for c in DES["clocks"]:
        D, v, S = clocks[c["name"]]
        s = (jitter or {}).get(c["name"], 0.0) ** 0.5
        w1, w2, e0, e1, e2 = exact(rng, 5).T
        phi0 = v * t0 + s * e0
        phi1 = v * t1 + np.sqrt(2 * D * (t1 - t0)) * w1 + s * e1
        phi2 = v * t2 + np.sqrt(2 * D * (t1 - t0)) * w1 + np.sqrt(2 * D * (t2 - t1)) * w2 + s * e2
        nd = c["N"] * DES["base"]["d"]
        out["clocks"][c["name"]] = {
            "p": {"N": c["N"]}, "J": [(x * nd)[:n].tolist() for x in (phi0, phi1, phi2)],
            "heat": [[S * s2 * t] * n for t in (t0, t1, t2)],
            "worlds": {w["name"]: {"sse": noisy(mse[c["name"]][w["name"]]), "sum": [0.0] * n} for w in DES["worlds"]}}
    if drop:
        kind, name = drop
        del out[kind][name]
    if nan:
        out["clocks"][nan]["J"][1][0] = float("nan")
    if short:
        out["clocks"][short]["J"] = [x[:-1] for x in out["clocks"][short]["J"]]
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


def brownian(clock="N1", elapsed=None):
    """The periodic-world error a Brownian clock of this clock's D and v would make: 2 (D / v²) mean(a)."""
    D, v, _ = CLOCKS[clock]
    return 2 * D / v ** 2 * (elapsed or WORLDS["periodic"][2])


class Controls(unittest.TestCase):
    def verdicts(self, payload):
        code, out = run(payload)
        self.assertEqual(code, 0, out)
        return {h: v["verdict"] for h, v in out["verdicts"].items()}

    def test_calibration_like_clocks(self):
        self.assertEqual(self.verdicts(data()), dict.fromkeys(ALL, "supported"))

    def test_precision_that_an_irregular_world_still_rewards(self):
        self.assertEqual(self.verdicts(data({"N64": {"k16": 21.4 / 1.5}}))["H1"], "contradicted")

    def test_anticipation_cheaper_than_entropy(self):
        # 64 times the entropy anticipating 160 times better: contradicted on the cheap side
        self.assertEqual(self.verdicts(data({"N64": {"periodic": 0.2131 / (64 * 2.5)}}))["H2"], "contradicted")

    def test_anticipation_dearer_than_entropy(self):
        self.assertEqual(self.verdicts(data({"N64": {"periodic": 0.2131 / (64 / 2.5)}}))["H2"], "contradicted")

    def test_the_price_is_the_entropy_rates(self):
        # the same errors with the dear clock's heat 2.5 times higher: anticipation now costs more than entropy buys
        self.assertEqual(self.verdicts(data({"N64": {"periodic": 0.2131 / 64}}, clocks={"N64": (0.001353, 2.5923, 4795.3 * 2.5)}))["H2"],
                         "contradicted")

    def test_a_regular_world_that_sells_precision(self):
        self.assertEqual(self.verdicts(data({"N64": {"k4096": 0.3142 / 8}}))["H3"], "contradicted")

    def test_a_clock_worse_than_brownian(self):
        self.assertEqual(self.verdicts(data({"N1": {"periodic": 1.5 * brownian()}}))["H4"], "contradicted")

    def test_the_brownian_error_uses_the_measured_elapsed_times(self):
        # half the elapsed time halves the Brownian error: the clock that beat it by 1.2 now loses by 1.6
        self.assertEqual(self.verdicts(data({"N1": {"periodic": brownian() / 1.2}}, worlds={"periodic": (33.34, 0.0, 9.875 / 2)}))["H4"],
                         "contradicted")

    def test_the_brownian_error_reads_time_through_the_velocity(self):
        # the same D at twice the velocity reads time four times more precisely
        self.assertEqual(self.verdicts(data({"N1": {"periodic": brownian() / 1.2}}, clocks={"N1": (0.0924, 2 * 2.5953, 75.0)}))["H4"],
                         "contradicted")

    def test_a_jitter_that_does_not_grow_cancels(self):
        # a clock 1.2 times worse than Brownian; var(Phi(t2) - Phi(t0)) / (2 (t2 - t0)) would call it better
        D, t0, t2 = CLOCKS["N1"][0], DES["checkpoints"][0], DES["checkpoints"][2]
        self.assertEqual(self.verdicts(data({"N1": {"periodic": 1.2 * brownian()}}, jitter={"N1": 0.5 * D * (t2 - t0)}))["H4"],
                         "contradicted")

    def test_a_scaling_that_breaks(self):
        self.assertEqual(self.verdicts(data({"N16": {"k4096": 0.1201 * 1.3}}))["H5"], "contradicted")

    def test_the_share_is_of_the_world_s_ceiling(self):
        # the same clock errors in a k4096 world whose perfect clock does twice as well: the share halves
        self.assertEqual(self.verdicts(data(worlds={"k4096": (33.41, 0.1072 / 2, 9.879)}))["H5"], "contradicted")

    def test_ratios_at_the_thresholds_are_inconclusive(self):
        v = self.verdicts(data({"N64": {"k16": 21.4 / 1.1, "k4096": 0.3142 / 4}, "N1": {"periodic": brownian()}}, spread=0.2))
        self.assertEqual((v["H1"], v["H3"], v["H4"]), ("inconclusive",) * 3)
        self.assertEqual(self.verdicts(data({"N64": {"periodic": 0.2131 / (4795.3 / 75.0 * 1.25)}}, spread=0.2))["H2"], "inconclusive")

    def test_a_scaling_at_its_threshold_is_inconclusive(self):
        # C(N16 in k4096) / C(N1 in k256) = 1.1 exactly at the point
        c_n1 = WORLDS["k256"][1] / MSE["N1"]["k256"]
        self.assertEqual(self.verdicts(data({"N16": {"k4096": WORLDS["k4096"][1] / (1.1 * c_n1)}}, spread=0.2))["H5"], "inconclusive")

    def test_a_clock_without_diffusion_has_no_brownian_twin(self):
        code, out = run(data(clocks={"N1": (0.0, 2.5953, 75.0)}, jitter={"N1": 1e-3}))
        self.assertEqual(code, 0, out)
        self.assertEqual(out["verdicts"]["H4"]["verdict"], "inconclusive")
        self.assertIn("undefined", out["verdicts"]["H4"]["reason"])

    def test_a_clock_without_error_has_no_ratio(self):
        code, out = run(data({"N64": {"k16": 0.0}}, spread=0.0))
        self.assertEqual(code, 0, out)
        self.assertEqual(out["verdicts"]["H1"]["verdict"], "inconclusive")

    def test_a_missing_clock_is_refused(self):
        code, out = run(data(drop=("clocks", "N4")))
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)

    def test_a_missing_world_is_refused(self):
        code, out = run(data(drop=("worlds", "k64")))
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)

    def test_short_rows_are_refused(self):
        code, out = run(data(n=T - 1))
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)

    def test_a_clock_short_of_a_trajectory_is_refused(self):
        code, out = run(data(short="N4"))
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
        self.q6 = module(HERE / "run.py", "q6run")

    def test_the_crystal_is_q5_bound_s_alone(self):
        """The copied physics: q6's crystal is q5-bound's alone cell, bit for bit."""
        q5 = module(HERE.parent / "q5-bound" / "run.py", "q5run")
        p = {**DES["base"], "N": 4}
        p5 = {**p, "reference": "alone", "k": 0.0, "v_star": 0.0, "tau": 0.0, "L": 16, "K": 2.0, "gamma_line": 2.0, "n_s": 0,
              "Lambda": 0.0, "gear": 1.0}
        J5, Q5, *_ = q5.run(p5, 8, 40.0, 10.0, 11)
        phi, J, Q = self.q6.trace(p, 8, 40.0, (10.0, 20.0, 40.0), 11)
        np.testing.assert_array_equal(J, J5[:, [0, 1, 3]])
        np.testing.assert_array_equal(Q, Q5[:, [0, 1, 3]])
        np.testing.assert_array_equal(phi[:, [40, 80, 160]] * 4 * p["d"], J)

    def test_residual_life(self):
        """m_k against a direct integral of the survival function, Poisson's constant, the periodic line, and the
        Gaussian limit at the mean (sd sqrt(2 / pi))."""
        Tm = DES["T"]
        a = np.array([0.0, 3.0, 10.0, 20.0, 35.0])
        np.testing.assert_allclose(self.q6.residual_life(1, Tm)(a), Tm)
        np.testing.assert_allclose(self.q6.residual_life(None, Tm)(a), np.maximum(Tm - a, 0))
        theta = Tm / 4
        u = np.linspace(0, 400, 400001)
        surv = np.exp(-u / theta) * sum((u / theta) ** i / math.factorial(i) for i in range(4))
        tail = np.cumsum(surv[::-1])[::-1] * (u[1] - u[0])
        want = np.interp(a, u, tail) / np.interp(a, u, surv)
        np.testing.assert_allclose(self.q6.residual_life(4, Tm)(a), want, rtol=1e-3)
        sd = Tm / 64
        self.assertAlmostEqual(float(self.q6.residual_life(4096, Tm)(np.array([Tm]))[0]), sd * math.sqrt(2 / math.pi), delta=0.02 * sd)

    def test_worlds(self):
        """Gamma intervals of mean T and variance T² / k, plus h² / 6 from placing both ends on the sub-step grid; periodic
        ones exactly T; every world outlasts the trajectory."""
        h, steps = 0.25, 7200
        for k in (1, 16, 4096):
            ev = self.q6.world(k, DES["T"], 256, steps, h, 3)
            gaps = np.concatenate([np.diff(e) * h for e in ev])
            self.assertAlmostEqual(gaps.mean() / DES["T"], 1.0, delta=0.01)
            self.assertAlmostEqual(gaps.var() / (DES["T"] ** 2 / k + h * h / 6), 1.0, delta=0.05)
            self.assertTrue(all(e[-1] > steps for e in ev))
        ev = self.q6.world(None, DES["T"], 16, steps, h, 3)
        self.assertEqual({int(x) for e in ev for x in np.diff(e)}, {int(DES["T"] / h)})

    def test_readout(self):
        """A clock that is time itself reads as the perfect one; in a Poisson world every clock predicts as no clock."""
        h, steps = 0.25, 800
        queries = np.arange(160, steps + 1, 4)
        exact_clock = np.tile(np.arange(steps + 1) * h * 2.5, (8, 1))
        noisy_clock = exact_clock + np.random.Generator(np.random.PCG64(1)).standard_normal(exact_clock.shape)
        for k in (4, None):
            ev, m = self.q6.world(k, DES["T"], 8, steps, h, 5), self.q6.residual_life(k, DES["T"])
            np.testing.assert_allclose(self.q6.readout(exact_clock, ev, queries, 2.5, m, h)[:2],
                                       self.q6.readout(None, ev, queries, 1.0, m, h)[:2], rtol=1e-9, atol=1e-9)
        ev, m = self.q6.world(1, DES["T"], 8, steps, h, 5), self.q6.residual_life(1, DES["T"])
        flat = lambda a: np.full_like(a, DES["T"])
        np.testing.assert_allclose(self.q6.readout(noisy_clock, ev, queries, 2.5, m, h)[0],
                                   self.q6.readout(None, ev, queries, 1.0, flat, h)[0])


class Smoke(unittest.TestCase):
    """The real runner end to end on the registered clocks and worlds with 16 trajectories over a short span, then the
    analysis."""

    def test_runner_then_analysis(self):
        if os.environ.get("ANALYZE"):
            self.skipTest("mutant runs exercise the analysis only")
        SCRATCH.mkdir(exist_ok=True)
        d = Path(tempfile.mkdtemp(dir=SCRATCH))
        shutil.copy(HERE / "run.py", d)
        reg = json.loads((HERE / "registration.json").read_text())
        reg["design"].update(trajectories=16, checkpoints=[40, 80, 120], t_total=120, warm=40)
        (d / "registration.json").write_text(json.dumps(reg))
        (d / "registration.lock.json").write_text(json.dumps({"registration": "s" * 64}))
        (d / "beacon.json").write_text(json.dumps({"attest": "b" * 64, "beacon": {"randomness": "c" * 64}}))
        p = subprocess.run([sys.executable, "-I", str(d / "run.py"), "--jobs", "2"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        out = json.loads((d / "results" / "data.json").read_text())
        self.assertEqual(list(out["clocks"]), [c["name"] for c in reg["design"]["clocks"]])
        self.assertEqual(set(out["worlds"]), {w["name"] for w in reg["design"]["worlds"]})
        self.assertTrue(all(len(c["J"]) == 3 and all(len(x) == 16 for x in c["J"]) for c in out["clocks"].values()))
        self.assertEqual(out["worlds"]["k1"]["n"], [81] * 16)
        for c in out["clocks"].values():                                  # memorylessness: no clock changes a Poisson prediction
            self.assertEqual(c["worlds"]["k1"]["sse"], out["worlds"]["k1"]["sse_perfect"])
        a = subprocess.run([sys.executable, "-I", str(HERE / "analyze.py"), str(d)], capture_output=True, text=True)
        self.assertEqual(a.returncode, 0, a.stdout + a.stderr)
        self.assertEqual(set(json.loads(a.stdout)["verdicts"]), ALL)


if __name__ == "__main__":
    unittest.main()
