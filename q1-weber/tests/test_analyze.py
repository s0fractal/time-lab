"""Controls for q1-weber/analyze.py on synthetic models whose answer to each list is known in advance.

    python -m unittest q1-weber/tests/test_analyze.py      (ANALYZE=path runs another copy, for mutants)"""
import json, os, subprocess, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ANALYZE = os.environ.get("ANALYZE", str(HERE / "analyze.py"))
SCRATCH = HERE / "runs"
GRID = [8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512]
K = 6
TOKENS = {"short": 1.0, "long": 2.0}


def run(answer, models=("m",), drop=0, tokens=TOKENS):
    """answer(condition, n, k) -> text; drop removes rows from the end (truncation)."""
    SCRATCH.mkdir(exist_ok=True)
    d = Path(tempfile.mkdtemp(dir=SCRATCH))
    (d / "results").mkdir()
    design = {"models": list(models), "grid": GRID, "lists": K,
              "conditions": [{"name": "short", "vocabulary": ["a"]}, {"name": "long", "vocabulary": ["b"]}]}
    (d / "registration.json").write_text(json.dumps({"design": design}))
    (d / "registration.lock.json").write_text(json.dumps({"registration": "r" * 64}))
    rows = [{"condition": c, "n": n, "k": k, "seed": 0, "answer": answer(c, n, k), "prompt_tokens": int(40 + tokens[c] * n)}
            for c in ("short", "long") for n in GRID for k in range(K)]
    rows = rows[:len(rows) - drop]
    data = {"registration": "r" * 64, "models": {m: {"rows": rows} for m in models}}
    (d / "results" / "data.json").write_text(json.dumps(data))
    p = subprocess.run([sys.executable, "-I", ANALYZE, str(d)], capture_output=True, text=True)
    return p.returncode, (json.loads(p.stdout) if p.returncode == 0 else p.stdout + p.stderr)


def jitter(k):
    return 1 + 0.03 * ((k % 5) - 2)


class Controls(unittest.TestCase):
    def verdicts(self, answer, **kw):
        code, out = run(answer, **kw)
        self.assertEqual(code, 0, out)
        return out["verdicts"]["H1:m"], out["verdicts"]["H2:m"]

    def test_token_time_model_supports_h1(self):
        h1, h2 = self.verdicts(lambda c, n, k: str(round(TOKENS[c] * n * jitter(k))))
        self.assertEqual((h1["verdict"], h2["verdict"]), ("supported", "contradicted"))
        self.assertAlmostEqual(h1["iota"], 1.0, delta=0.05)

    def test_event_counting_model_supports_h2(self):
        h1, h2 = self.verdicts(lambda c, n, k: str(round(1.6 * n * jitter(k))))
        self.assertEqual((h1["verdict"], h2["verdict"]), ("contradicted", "supported"))

    def test_anti_duration_model_contradicts_both(self):
        h1, h2 = self.verdicts(lambda c, n, k: str(round((0.6 if c == "long" else 1.0) * n * jitter(k))))
        self.assertEqual((h1["verdict"], h2["verdict"]), ("contradicted", "contradicted"))
        self.assertLess(h1["iota"], -0.25)

    def test_half_way_model_is_inconclusive_for_h1(self):
        h1, _ = self.verdicts(lambda c, n, k: str(round((2 ** 0.5 if c == "long" else 1.0) * n * jitter(k))))
        self.assertEqual(h1["verdict"], "inconclusive")

    def test_unparseable_answers_close_the_gate(self):
        h1, h2 = self.verdicts(lambda c, n, k: "many" if k == 0 else str(TOKENS[c] * n))
        self.assertEqual((h1["verdict"], h2["verdict"]), ("inconclusive", "inconclusive"))
        self.assertIn("gate", h1["reason"])

    def test_no_token_difference_is_inconclusive(self):
        h1, _ = self.verdicts(lambda c, n, k: str(n), tokens={"short": 1.0, "long": 1.2})
        self.assertEqual(h1["verdict"], "inconclusive")
        self.assertIn("manipulation", h1["reason"])

    def test_truncated_data_are_refused(self):
        code, out = run(lambda c, n, k: str(n), drop=1)
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)

    def test_empty_data_are_refused(self):
        code, out = run(lambda c, n, k: str(n), drop=2 * len(GRID) * K)
        self.assertNotEqual(code, 0)
        self.assertIn("REFUSED", out)


if __name__ == "__main__":
    unittest.main()
