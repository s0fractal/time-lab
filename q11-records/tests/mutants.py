"""Each mutant of q11-records/analyze.py must make a control fail on an assertion (not on an import or a crash).

    python -I q11-records/tests/mutants.py"""
import os, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
MUTANTS = {
    "not-stated-off": ('if lines and lines[-1].strip().rstrip(".").lower() == NOT_STATED:', "if False:"),
    "not-stated-case": ('.rstrip(".").lower() == NOT_STATED:', '.rstrip(".") == NOT_STATED:'),
    "not-stated-period": ('.strip().rstrip(".").lower() == NOT_STATED:', '.strip().lower() == NOT_STATED:'),
    "not-stated-other-line": ('if lines and lines[-1].strip().rstrip(".").lower() == NOT_STATED:', 'if lines and lines[0].strip().rstrip(".").lower() == NOT_STATED:'),
    "not-stated-prefix": ('.rstrip(".").lower() == NOT_STATED:', '.rstrip(".").lower().startswith(NOT_STATED[:18]):'),
    "tail-no-space": (r'`?(?:\.(?:\s.*)?)?", re.I)', r'`?(?:\..*)?", re.I)'),
    "tail-off": (r'`?(?:\.(?:\s.*)?)?", re.I)', r'`?\.?", re.I)'),
    "floor-value": ("FLOOR = 0.10 ", "FLOOR = 0.05 "),
    "answer-prefix": ("m = ANSWER.fullmatch(lines[-1].strip()) if lines else None", "m = ANSWER.match(lines[-1].strip()) if lines else None"),
    "answer-search": ("m = ANSWER.fullmatch(lines[-1].strip()) if lines else None", "m = ANSWER.search(lines[-1].strip()) if lines else None"),
    "answer-first-line": ("m = ANSWER.fullmatch(lines[-1].strip()) if lines else None", "m = ANSWER.fullmatch(lines[0].strip()) if lines else None"),
    "answer-no-strip": ("m = ANSWER.fullmatch(lines[-1].strip()) if lines else None", "m = ANSWER.fullmatch(lines[-1]) if lines else None"),
    "answer-empty-crash": ('lines = [x for x in answer.splitlines() if x.strip()]', 'lines = answer.splitlines()'),
    "floor-off": ("if not any(reads.values()):", "if False:"),
    "floor-all": ("if not any(reads.values()):", "if not all(reads.values()):"),
    "floor-strict": ('[0] > FLOOR\n', '[0] > 0\n'),
    "floor-only-other": ('for a in ("records", other):', 'for a in (other,):'),
    "floor-on-changed": ('xs = [s[a, q][0] for q in ids[cls]]', 'xs = [s[a, q][0] for q in ids["changed"]]'),
    "floor-on-control": ('xs = [s[a, q][0] for q in ids[cls]]', 'xs = [s[a, q][0] - s["none", q][0] for q in ids[cls]]'),
    "alpha": ("ALPHA = 0.05", "ALPHA = 0.10"),
    "resamples": ("RESAMPLES = 20000", "RESAMPLES = 2000"),
    "gate-lower": ("GATE = 0.95", "GATE = 0.90"),
    "gate-strict": ("if low < GATE:", "if low <= GATE:"),
    "gate-off": ("if low < GATE:", "if False:"),
    "gate-mean": ("low = min(parsed(rs, a) for rs in rows.values() for a in arms)", "low = sum(parsed(rs, a) for rs in rows.values() for a in arms) / (len(rows) * len(arms))"),
    "gate-other-only": ('arms = ("none",) if kind == "no_leak" else ("records", other)', 'arms = ("none",) if kind == "no_leak" else (other,)'),
    "gate-control-skipped": ('arms = ("none",) if kind == "no_leak" else ("records", other)', 'arms = ("records", other)'),
    "delta-lower": ("DELTA = 0.15 ", "DELTA = 0.05 "),
    "leak-higher": ("LEAK = 0.40 ", "LEAK = 0.60 "),
    "answer-case": ('(?:\\s.*)?)?", re.I)', '(?:\\s.*)?)?")'),
    "answer-no-e": ("([A-E])", "([A-D])"),
    "answer-upper-lost": ("return m.group(1).upper() if m else None", "return m.group(1) if m else None"),
    "e-in-control": ('return "E" if r["arm"] != "none" else None', 'return "E"'),
    "letter-order": ('return r["order"][LETTERS.index(x)] if x else None', 'return sorted(r["order"])[-1 - LETTERS.index(x)] if x else None'),
    "correct-is-old": ('sum(p == item["key"] for p in picks) / len(picks),', 'sum(p == item["old"] for p in picks) / len(picks),'),
    "first-model-only": ('picks = [choice(r) for rs in rows.values() for r in rs if r["arm"] == arm and r["q"] == q]',
                         'picks = [choice(r) for rs in list(rows.values())[:1] for r in rs if r["arm"] == arm and r["q"] == q]'),
    "old-counts-any": ('sum(item["old"] is not None and p == item["old"] for p in picks) / len(picks))', 'sum(p != item["key"] for p in picks) / len(picks))'),
    "lead-reversed": ('lead = lambda c: [s["records", q][0] - s[other, q][0] for q in ids[c]]', 'lead = lambda c: [s[other, q][0] - s["records", q][0] for q in ids[c]]'),
    "lead-on-old": ('lead = lambda c: [s["records", q][0] - s[other, q][0] for q in ids[c]]', 'lead = lambda c: [s["records", q][1] - s[other, q][1] for q in ids[c]]'),
    "interaction-reversed": ("value, ci = mean(a) - mean(b), bootstrap(name, [a, b], lambda g: mean(g[0]) - mean(g[1]))",
                             "value, ci = mean(b) - mean(a), bootstrap(name, [a, b], lambda g: mean(g[1]) - mean(g[0]))"),
    "interaction-pooled": ("value, ci = mean(a) - mean(b), bootstrap(name, [a, b], lambda g: mean(g[0]) - mean(g[1]))",
                           "value, ci = mean(a) - mean(b), bootstrap(name, [a + b], lambda g: mean(g[0][:len(a)]) - mean(g[0][len(a):]))"),
    "fewer-old-reversed": ('xs = [s[other, q][1] - s["records", q][1] for q in ids[cls]]', 'xs = [s["records", q][1] - s[other, q][1] for q in ids[cls]]'),
    "fewer-old-on-correct": ('xs = [s[other, q][1] - s["records", q][1] for q in ids[cls]]', 'xs = [s[other, q][0] - s["records", q][0] for q in ids[cls]]'),
    "lo-at-zero": ('verdict = "supported" if lo > 0 else "contradicted" if hi < DELTA else "inconclusive"', 'verdict = "supported" if lo >= 0 else "contradicted" if hi < DELTA else "inconclusive"'),
    "no-contradiction": ('verdict = "supported" if lo > 0 else "contradicted" if hi < DELTA else "inconclusive"', 'verdict = "supported" if lo > 0 else "inconclusive"'),
    "contradict-at-zero": ('verdict = "supported" if lo > 0 else "contradicted" if hi < DELTA else "inconclusive"', 'verdict = "supported" if lo > 0 else "contradicted" if hi < 0 else "inconclusive"'),
    "within-always": ('verdict = "supported" if hi < DELTA else "contradicted" if lo > DELTA else "inconclusive"', 'verdict = "supported" if hi < DELTA else "inconclusive"'),
    "within-lo": ('verdict = "supported" if hi < DELTA else "contradicted" if lo > DELTA else "inconclusive"', 'verdict = "supported" if lo < DELTA else "contradicted" if lo > DELTA else "inconclusive"'),
    "leak-reads-lo": ('verdict = "supported" if hi < LEAK else "contradicted" if lo > LEAK else "inconclusive"', 'verdict = "supported" if lo < LEAK else "contradicted" if lo > LEAK else "inconclusive"'),
    "any-contradicted-off": ('"contradicted" if "contradicted" in vs else "supported" if set(vs) == {"supported"}', '"supported" if "supported" in vs else "contradicted" if "contradicted" in vs'),
    "one-supported": ('else "supported" if set(vs) == {"supported"} else', 'else "supported" if "supported" in vs else'),
    "h1-on-unchanged": ('"H1": [("ahead", "changed", "prose"), ("ahead", "changed", "log")],', '"H1": [("ahead", "unchanged", "prose"), ("ahead", "unchanged", "log")],'),
    "h1-no-log": ('"H1": [("ahead", "changed", "prose"), ("ahead", "changed", "log")],', '"H1": [("ahead", "changed", "prose")],'),
    "h2-on-changed": ('"H2": [("within", "unchanged", "prose"), ("within", "unchanged", "log")],', '"H2": [("within", "changed", "prose"), ("within", "changed", "log")],'),
    "refuse-registration-off": ('if data.get("registration") != lock["registration"]:', "if False:"),
    "refuse-bank-off": ('if hashlib.sha256(raw).hexdigest() != design["bank_sha256"]:', "if False:"),
    "refuse-arms-off": ('if tuple(design["arms"]) != ARMS:', "if False:"),
    "refuse-models-off": ('if set(data["models"]) != set(design["models"]):', "if False:"),
    "refuse-duplicates-off": ('if len(keys) != len(set(keys)) or set(keys) != want:', "if set(keys) != want:"),
    "refuse-missing-off": ('if len(keys) != len(set(keys)) or set(keys) != want:', "if len(keys) != len(set(keys)):"),
    "refuse-options-off": ('if sorted(r["order"]) != sorted(bank[r["q"]]["options"]):', "if False:"),
    "describe-correct": ('"correct": sum(p == i["key"] for p, i in picks)', '"correct": sum(p != i["key"] for p, i in picks)'),
    "describe-e": ('"not_settled": sum(p == "E" for p, _ in picks)', '"not_settled": sum(p is None for p, _ in picks)'),
}


def main(names=None):
    src = (HERE / "analyze.py").read_text()
    (HERE / "runs").mkdir(exist_ok=True)
    bad = []
    for name, (find, repl) in MUTANTS.items():
        if names and name not in names:
            continue
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
    main(sys.argv[1:])
