"""Each mutant of q10-drift/analyze.py must make a control fail on an assertion (not on an import or a crash).

    python -I q10-drift/tests/mutants.py"""
import os, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
MUTANTS = {
    "alpha-wider": ("ALPHA = 0.05", "ALPHA = 0.10"),
    "gate-lower": ("GATE = 0.95", "GATE = 0.90"),
    "gate-strict": ("if parsed(rows) < GATE:", "if parsed(rows) <= GATE:"),
    "no-gate": ("if parsed(rows) < GATE:", "if False:"),
    "stale-one-kind": ('STALE = ("semantic", "move+sem")', 'STALE = ("semantic",)'),
    "fresh-two-kinds": ('"rely_fresh": (("none", "cosmetic", "move"), "rely"),', '"rely_fresh": (("none", "cosmetic"), "rely"),'),
    "rely-none-reads": ('"rely_none": (("none",), "rely"),', '"rely_none": (("none",), "read"),'),
    "cosmetic-relies": ('"read_cosmetic": (("cosmetic",), "read"),', '"read_cosmetic": (("cosmetic",), "rely"),'),
    "rely-prefix": ('return "rely" if RELY.fullmatch(turn1)', 'return "rely" if RELY.match(turn1)'),
    "read-search": ('else "read" if READ.fullmatch(turn1)', 'else "read" if READ.search(turn1)'),
    "rely-case": ('RELY = re.compile(r"`?RELY\\s+(-?\\d+)`?\\.?", re.I)', 'RELY = re.compile(r"`?RELY\\s+(-?\\d+)`?\\.?")'),
    "read-no-period": ('READ = re.compile(r"`?READ`?\\.?", re.I)', 'READ = re.compile(r"`?READ`?", re.I)'),
    "unparsed-is-rely": ('return "rely" if RELY.fullmatch(turn1) else "read" if READ.fullmatch(turn1) else "unparsed"',
                         'return "rely" if RELY.fullmatch(turn1) else "read" if READ.fullmatch(turn1) else "rely"'),
    "count-all-rows": ("return sum(act(r[\"turn1\"]) == a for r in sel), len(sel)", "return sum(act(r[\"turn1\"]) == a for r in sel), len(rows)"),
    "tail-from-next": ("for i in range(x, n + 1))", "for i in range(x + 1, n + 1))"),
    "hi-tail-at-x": ("hi = 1.0 if x == n else root(lambda p: 1 - tail(x + 1, n, p) <= ALPHA / 2)",
                     "hi = 1.0 if x == n else root(lambda p: 1 - tail(x, n, p) <= ALPHA / 2)"),
    "lo-one-sided": ("lo = 0.0 if x == 0 else root(lambda p: tail(x, n, p) >= ALPHA / 2)",
                     "lo = 0.0 if x == 0 else root(lambda p: tail(x, n, p) >= ALPHA)"),
    "lo-zero-at-one": ("lo = 0.0 if x == 0 else", "lo = 0.0 if x <= 1 else"),
    "hi-one-near-n": ("hi = 1.0 if x == n else", "hi = 1.0 if x >= n - 1 else"),
    "few-bisections": ("for _ in range(100):", "for _ in range(6):"),
    "above-uses-hi": ('return "supported" if lo > t else "contradicted" if hi < t else "inconclusive"',
                      'return "supported" if hi > t else "contradicted" if hi < t else "inconclusive"'),
    "below-uses-lo": ('return "supported" if hi < t else "contradicted" if lo > t else "inconclusive"',
                      'return "supported" if lo < t else "contradicted" if lo > t else "inconclusive"'),
    "above-never-contradicts": ('return "supported" if lo > t else "contradicted" if hi < t else "inconclusive"',
                                'return "supported" if lo > t else "inconclusive"'),
    "contradicted-needs-all": ('"contradicted" if "contradicted" in vs', '"contradicted" if set(vs) == {"contradicted"}'),
    "supported-any": ('"supported" if set(vs) == {"supported"}', '"supported" if "supported" in vs'),
    "h1-name-threshold": ('(G4, "name-pressure", "stale_act", "above", 0.10), (QW', '(G4, "name-pressure", "stale_act", "above", 0.05), (QW'),
    "h1-name-mirror": ('(QW, "name-pressure", "stale_act", "above", 0.10)', '(QW, "name-pressure", "stale_act", "above", 0.05)'),
    "h1-pin-threshold": ('(G4, "pin-pressure", "stale_act", "below", 0.10)', '(G4, "pin-pressure", "stale_act", "below", 0.15)'),
    "h1-pin-mirror": ('(QW, "pin-pressure", "stale_act", "below", 0.10)', '(QW, "pin-pressure", "stale_act", "below", 0.15)'),
    "h1-drops-qwen": (', (QW, "name-pressure", "stale_act", "above", 0.10),', ','),
    "h2-foreign": ('(G4, "foreign", "rely_fresh", "below", 0.10)', '(G4, "foreign", "rely_fresh", "below", 0.15)'),
    "h2-foreign-mirror": ('(QW, "foreign", "rely_fresh", "below", 0.10)', '(QW, "foreign", "rely_fresh", "below", 0.15)'),
    "h2-none": ('(G4, "pin-foreign", "rely_none", "above", 0.85)', '(G4, "pin-foreign", "rely_none", "above", 0.80)'),
    "h2-none-mirror": ('(QW, "pin-foreign", "rely_none", "above", 0.85)', '(QW, "pin-foreign", "rely_none", "above", 0.80)'),
    "h2-stale": ('(G4, "pin-foreign", "stale_act", "below", 0.10)', '(G4, "pin-foreign", "stale_act", "below", 0.15)'),
    "h2-stale-mirror": ('(QW, "pin-foreign", "stale_act", "below", 0.10)', '(QW, "pin-foreign", "stale_act", "below", 0.15)'),
    "h3-threshold": ('(G3, "pin-pressure", "stale_act", "above", 0.50)', '(G3, "pin-pressure", "stale_act", "above", 0.30)'),
    "h3-model": ('(G3, "pin-pressure", "stale_act", "above", 0.50)', '(G4, "pin-pressure", "stale_act", "above", 0.50)'),
    "h4-threshold": ('(G4, "pin", "read_cosmetic", "above", 0.85)', '(G4, "pin", "read_cosmetic", "above", 0.80)'),
    "h4-mirror": ('(QW, "pin", "read_cosmetic", "above", 0.85)', '(QW, "pin", "read_cosmetic", "above", 0.80)'),
    "h4-cell": ('(QW, "pin", "read_cosmetic", "above", 0.85)', '(QW, "pin-foreign", "read_cosmetic", "above", 0.85)'),
    "no-registration-check": ('if data.get("registration") != lock["registration"]:', "if False:"),
    "no-kinds-check": ("if tuple(design[\"kinds\"]) != KINDS:", "if False:"),
    "no-models-check": ('if set(data["models"]) != set(design["models"]):', "if False:"),
    "no-duplicate-check": ("if len(keys) != len(set(keys)) or set(keys) != want:", "if set(keys) != want:"),
    "no-completeness-check": ("if len(keys) != len(set(keys)) or set(keys) != want:", "if len(keys) != len(set(keys)):"),
    "no-values-check": ('if type(r.get("v0")) is not int or type(r.get("v1")) is not int or (r["v0"] == r["v1"]) != (r["kind"] not in STALE):',
                        'if type(r.get("v0")) is not int or type(r.get("v1")) is not int:'),
    "no-turn2-check": ('if not isinstance(r.get("turn1"), str) or ("turn2" in r) != (act(r["turn1"]) == "read"):',
                       'if not isinstance(r.get("turn1"), str):'),
    "wrong-against-v0": ('wrong += answer(r, a) != r["v1"]', 'wrong += answer(r, a) != r["v0"]'),
    "value-anywhere": ('m = VALUE.search(r.get("turn2", "")) if a == "read" else None', 'm = re.search(r"(-?\\d+)", r.get("turn2", "")) if a == "read" else None'),
    "tokens-first": ('"tokens": sum(r["tokens"] for r in rows) / len(rows)', '"tokens": rows[0]["tokens"]'),
}


def main():
    src = (HERE / "analyze.py").read_text()
    (HERE / "runs").mkdir(exist_ok=True)
    bad = []
    for name, (find, repl) in MUTANTS.items():
        if src.count(find) != 1:
            sys.exit(f"{name}: its find text occurs {src.count(find)} times")
        d = Path(tempfile.mkdtemp(dir=HERE / "runs"))
        try:
            (d / "analyze.py").write_text(src.replace(find, repl))
            p = subprocess.run([sys.executable, "-m", "unittest", str(HERE / "tests" / "test_analyze.py")], capture_output=True,
                               text=True, env={**os.environ, "ANALYZE": str(d / "analyze.py")}, cwd=HERE.parent)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        killed = p.returncode != 0 and "AssertionError" in p.stderr
        print(f"{name}: {'killed' if killed else 'SURVIVED'}", flush=True)
        bad += [] if killed else [name]
    sys.exit(f"survivors: {bad}" if bad else 0)


if __name__ == "__main__":
    main()
