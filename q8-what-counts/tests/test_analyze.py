"""Controls for q8-what-counts/analyze.py on synthetic models whose elasticities to characters and to tokens are set in
advance, and an end-to-end smoke run of the real runner against a stub ollama.

    python -m unittest q8-what-counts/tests/test_analyze.py      (ANALYZE=path runs another copy, for mutants)"""
import importlib.util, json, math, os, shutil, subprocess, sys, tempfile, threading, unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ANALYZE = os.environ.get("ANALYZE", str(HERE / "analyze.py"))
SCRATCH = HERE / "runs"
REG = json.loads((HERE / "registration.json").read_text())
DES = REG["design"]
MODELS = DES["models"]
CHI = {"short": 5.5, "one": 13.4, "many": 13.3}
TAU = {"short": 1.0, "one": 1.0, "many": 3.6}
spec = importlib.util.spec_from_file_location("q8analyze", HERE / "analyze.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
HYP = A.HYPOTHESES
# the elasticities each model has in the calibration (calibration/full-analysis.json)
CALIB = {"gemma3:27b": {"chars": 0.37, "tokens": 0.17}, "gemma4:26b-a4b-it-q4_K_M": {"chars": -0.01, "tokens": -0.36},
         "qwen3.8:27b-mlx": {"chars": -0.20, "tokens": -0.15}}


def data(iotas=None, noise=0.08, unparsed=None, drop=None, chi=None, tau=None, nonpositive=None, registration="r" * 64):
    """A model answers n (chi_c / chi_short)^iota_chars (tau_c / tau_short)^iota_tokens, times a lognormal noise of sd
    `noise` per list, rounded; its prompt takes 40 + tau_c n tokens and its list chi_c n characters."""
    rng = np.random.Generator(np.random.PCG64(5))
    chi, tau = {**CHI, **(chi or {})}, {**TAU, **(tau or {})}
    models = {}
    for m in MODELS:
        io = {**CALIB[m], **(iotas or {}).get(m, {})}
        rows = []
        for c in ("short", "one", "many"):
            f = (chi[c] / chi["short"]) ** io["chars"] * (tau[c] / tau["short"]) ** io["tokens"]
            for n in DES["grid"]:
                for k in range(DES["lists"]):
                    v = max(1, round(n * f * math.exp(noise * rng.standard_normal())))
                    answer = str(v)
                    if unparsed and unparsed[0] == m and unparsed[1] == c and k < unparsed[2]:
                        answer = "I cannot tell."
                    if nonpositive and nonpositive == (m, c, n):
                        answer = "0"
                    rows.append({"condition": c, "n": n, "k": k, "seed": 1, "answer": answer,
                                 "prompt_tokens": int(round(40 + tau[c] * n)), "chars": int(round(chi[c] * n))})
        models[m] = {"digest": "d", "rows": rows}
    if drop:
        models[drop[0]]["rows"] = [r for r in models[drop[0]]["rows"] if (r["condition"], r["n"], r["k"]) != drop[1]]
    return {"registration": registration, "models": models}


def run(payload):
    SCRATCH.mkdir(exist_ok=True)
    d = Path(tempfile.mkdtemp(dir=SCRATCH))
    (d / "results").mkdir()
    shutil.copy(HERE / "registration.json", d)
    (d / "registration.lock.json").write_text(json.dumps({"registration": "r" * 64}))
    (d / "results" / "data.json").write_text(json.dumps(payload))
    p = subprocess.run([sys.executable, "-I", ANALYZE, str(d)], capture_output=True, text=True)
    return p.returncode, (json.loads(p.stdout) if p.returncode == 0 else p.stdout + p.stderr)


def verdicts(payload):
    code, out = run(payload)
    assert code == 0, out
    return {h: v["verdict"] for h, v in out["verdicts"].items()}


def at(model, index, value):
    """Elasticities of `model` that put `index` at `value` (the difference by moving its token elasticity)."""
    if index == A.DIFF:
        return {model: {"tokens": CALIB[model]["chars"] - value}}
    return {model: {index: value}}


def flip(h):
    """Elasticities that put hypothesis h wholly on the other side of its threshold."""
    model, index, kind, x = HYP[h]
    return at(model, index, {"above": x - 0.4, "below": x + 0.4, "within": x + 0.4}[kind])


class Controls(unittest.TestCase):
    def test_calibration_like_models(self):
        self.assertEqual(verdicts(data()), dict.fromkeys(HYP, "supported"))

    def test_each_hypothesis_can_be_contradicted(self):
        for h in HYP:
            with self.subTest(h=h):
                self.assertEqual(verdicts(data(flip(h)))[h], "contradicted")

    def test_a_band_is_left_on_either_side(self):
        for h, (model, index, kind, x) in HYP.items():
            if kind == "within":
                with self.subTest(h=h):
                    self.assertEqual(verdicts(data(at(model, index, -x - 0.4)))[h], "contradicted")

    def test_a_few_wild_answers_do_not_move_a_median(self):
        # 5 of 24 lists of every `many` cell answer twenty times n: the medians, and so the verdicts, do not move
        payload = data()
        for m in MODELS:
            for r in payload["models"][m]["rows"]:
                if r["condition"] == "many" and r["k"] < 5:
                    r["answer"] = str(20 * r["n"])
        self.assertEqual(verdicts(payload), dict.fromkeys(HYP, "supported"))

    def test_an_elasticity_at_the_threshold_is_inconclusive(self):
        for h, (model, index, kind, x) in HYP.items():
            with self.subTest(h=h):
                self.assertEqual(verdicts(data(at(model, index, x), noise=0.15))[h], "inconclusive")

    def test_characters_do_not_move_the_token_index(self):
        # a model that follows characters only: iota_tokens stays near 0 because one and many have the same characters
        m = HYP[next(h for h in HYP if HYP[h][1] == "tokens")][0]
        code, out = run(data({m: {"chars": 1.0, "tokens": 0.0}}, noise=0.02))
        self.assertEqual(code, 0, out)
        self.assertAlmostEqual(out["indices"][m]["indices"]["tokens"]["iota"], 0.0, delta=0.05)
        self.assertAlmostEqual(out["indices"][m]["indices"]["chars"]["iota"], 1.0, delta=0.05)

    def test_the_ratio_is_the_measured_one(self):
        # tokens per word of `many` 7.2, not 3.6: the same answers then mean half the elasticity
        m = HYP[next(h for h in HYP if HYP[h][1] == "tokens")][0]
        a = run(data({m: {"tokens": 0.6}}, noise=0.02))[1]["indices"][m]["indices"]["tokens"]["iota"]
        b = run(data({m: {"tokens": 0.6}}, noise=0.02, tau={"many": 7.2}))[1]["indices"][m]["indices"]["tokens"]["iota"]
        self.assertAlmostEqual(a, 0.6, delta=0.05)
        self.assertAlmostEqual(b, 0.6, delta=0.05)                  # the synthetic answers follow the new tau too

    def test_tokens_that_were_not_held_equal(self):
        # `one` taking 1.3 tokens per word: the characters index compares two token counts and is refused
        code, out = run(data(tau={"one": 1.3}))
        self.assertEqual(code, 0, out)
        for m in MODELS:
            self.assertIn("manipulation", out["indices"][m]["indices"]["chars"].get("reason", ""))
        for h, (model, index, _, _) in HYP.items():
            if index == "chars":
                self.assertEqual(out["verdicts"][h]["verdict"], "inconclusive")

    def test_characters_that_were_not_held_equal(self):
        code, out = run(data(chi={"many": 16.0}))
        self.assertEqual(code, 0, out)
        for m in MODELS:
            self.assertIn("manipulation", out["indices"][m]["indices"]["tokens"].get("reason", ""))

    def test_a_manipulation_too_small(self):
        code, out = run(data(tau={"many": 1.4}))
        self.assertEqual(code, 0, out)
        for m in MODELS:
            self.assertIn("manipulation", out["indices"][m]["indices"]["tokens"].get("reason", ""))

    def test_unparsed_answers_close_the_gate(self):
        m = MODELS[0]
        code, out = run(data(unparsed=(m, "many", 3)))                  # 3 of 24 lists per n: 87.5% parsed
        self.assertEqual(code, 0, out)
        self.assertIn("gate", out["indices"][m]["indices"]["tokens"].get("reason", ""))
        code, out = run(data(unparsed=(m, "many", 1)))                  # 1 of 24: 95.8% parsed, and medians skip it
        self.assertEqual(code, 0, out)
        self.assertIn("ci95", out["indices"][m]["indices"]["tokens"])

    def test_a_zero_median_is_inconclusive(self):
        m = MODELS[1]
        payload = data()
        for r in payload["models"][m]["rows"]:
            if r["condition"] == "short" and r["n"] == DES["grid"][0]:
                r["answer"] = "0"
        code, out = run(payload)
        self.assertEqual(code, 0, out)
        self.assertIn("not positive", out["indices"][m]["indices"]["chars"].get("reason", ""))

    def test_a_missing_list_is_refused(self):
        code, out = run(data(drop=(MODELS[2], ("one", 181, 3))))
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)

    def test_a_missing_model_is_refused(self):
        payload = data()
        del payload["models"][MODELS[1]]
        code, out = run(payload)
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)

    def test_a_row_without_counts_is_refused(self):
        payload = data()
        payload["models"][MODELS[0]]["rows"][5]["prompt_tokens"] = None
        code, out = run(payload)
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)

    def test_data_of_another_registration_are_refused(self):
        code, out = run(data(registration="x" * 64))
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)


class Stub(BaseHTTPRequestHandler):
    """A stub ollama: tags, version, and generate that answers the number of words it was shown."""

    def log_message(self, *a):
        pass

    def reply(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self.reply({"models": [{"name": m, "digest": "d"} for m in MODELS]} if self.path == "/api/tags" else {"version": "stub"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        words = body["prompt"].split("\n\n")[1].split()
        self.reply({"response": str(len(words)), "prompt_eval_count": 40 + len(words)})


class Smoke(unittest.TestCase):
    """The real runner end to end against a stub ollama on a small grid, then the analysis."""

    def test_runner_then_analysis(self):
        if os.environ.get("ANALYZE"):
            self.skipTest("mutant runs exercise the analysis only")
        srv = HTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            SCRATCH.mkdir(exist_ok=True)
            d = Path(tempfile.mkdtemp(dir=SCRATCH))
            src = (HERE / "run.py").read_text().replace('OLLAMA = "http://localhost:11434"', f'OLLAMA = "http://127.0.0.1:{srv.server_port}"')
            self.assertIn(str(srv.server_port), src)
            (d / "run.py").write_text(src)
            reg = json.loads((HERE / "registration.json").read_text())
            reg["design"].update(grid=[91, 128], lists=3)
            (d / "registration.json").write_text(json.dumps(reg))
            (d / "registration.lock.json").write_text(json.dumps({"registration": "s" * 64}))
            (d / "beacon.json").write_text(json.dumps({"attest": "b" * 64, "beacon": {"randomness": "c" * 64}}))
            p = subprocess.run([sys.executable, "-I", str(d / "run.py")], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            out = json.loads((d / "results" / "data.json").read_text())
            self.assertEqual(list(out["models"]), MODELS)
            for m in MODELS:
                rows = out["models"][m]["rows"]
                self.assertEqual(len(rows), 3 * 2 * 3)
                self.assertTrue(all(r["answer"] == str(r["n"]) for r in rows))
                vocab = {c["name"]: c["vocabulary"] for c in reg["design"]["conditions"]}
                self.assertTrue(all(r["chars"] >= r["n"] * min(map(len, vocab[r["condition"]])) for r in rows))
            a = subprocess.run([sys.executable, "-I", str(HERE / "analyze.py"), str(d)], capture_output=True, text=True)
            self.assertEqual(a.returncode, 0, a.stdout[-500:] + a.stderr[-2000:])
            self.assertEqual(set(json.loads(a.stdout)["verdicts"]), set(HYP))
        finally:
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()
