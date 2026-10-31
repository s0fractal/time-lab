"""Each mutant of q8-what-counts/analyze.py must make a control fail on an assertion (not on an import or a crash).

    python -I q8-what-counts/tests/mutants.py"""
import os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
MUTANTS = {
    "chars-index-swapped": ('INDEX = {"chars": ("one", "short", "chi"), "tokens": ("many", "one", "tau")}',
                            'INDEX = {"chars": ("short", "one", "chi"), "tokens": ("many", "one", "tau")}'),
    "tokens-on-short": ('INDEX = {"chars": ("one", "short", "chi"), "tokens": ("many", "one", "tau")}',
                        'INDEX = {"chars": ("one", "short", "chi"), "tokens": ("many", "short", "tau")}'),
    "no-log-ratio": ("xs.append(math.log(a / b) / math.log(ratio))", "xs.append(math.log(a / b))"),
    "mean-not-median": ("a, b = np.median(est[(num, n)]), np.median(est[(den, n)])", "a, b = np.mean(est[(num, n)]), np.mean(est[(den, n)])"),
    "no-held-check": ("if abs(held - 1) > HELD or ratio < MIN_RATIO:", "if ratio < MIN_RATIO:"),
    "no-ratio-check": ("if abs(held - 1) > HELD or ratio < MIN_RATIO:", "if abs(held - 1) > HELD:"),
    "no-gate": ("if min(parsed[num], parsed[den]) < GATE:", "if False:"),
    "no-positive-check": ("        if a <= 0 or b <= 0:\n            return None\n", ""),
    "within-one-sided": ('return "supported" if -x < lo and hi < x else', 'return "supported" if hi < x else'),
    "below-bound": ('return "supported" if hi < x else "contradicted" if lo > x', 'return "supported" if lo < x else "contradicted" if lo > x'),
    "above-bound": ('return "supported" if lo > x else "contradicted" if hi < x', 'return "supported" if hi > x else "contradicted" if hi < x'),
    "no-completeness": ("if len(keys) != len(set(keys)) or set(keys) != want:", "if len(keys) != len(set(keys)):"),
    "no-model-check": ("if set(data[\"models\"]) != set(design[\"models\"]):", "if False:"),
    "no-count-check": ('if not isinstance(r.get("prompt_tokens"), int) or not isinstance(r.get("chars"), int):', "if False:"),
    "diff-reversed": ('boots[DIFF].append(math.nan if None in b.values() else b["chars"] - b["tokens"])',
                      'boots[DIFF].append(math.nan if None in b.values() else b["tokens"] - b["chars"])'),
    "diff-from-points": ('boots[DIFF].append(math.nan if None in b.values() else b["chars"] - b["tokens"])',
                         'boots[DIFF].append(math.nan if None in b.values() else point["chars"] - point["tokens"])'),
    "h1-threshold": ('("gemma3:27b", "chars", "above", 0.15)', '("gemma3:27b", "chars", "above", 0.5)'),
    "h2-threshold": ('("gemma3:27b", DIFF, "above", 0.0)', '("gemma3:27b", DIFF, "above", -0.3)'),
    "h3-width": ('("gemma4:26b-a4b-it-q4_K_M", "chars", "within", 0.1)', '("gemma4:26b-a4b-it-q4_K_M", "chars", "within", 0.5)'),
    "h4-threshold": ('("gemma4:26b-a4b-it-q4_K_M", "tokens", "below", -0.2)', '("gemma4:26b-a4b-it-q4_K_M", "tokens", "below", 0.0)'),
    "h5-threshold": ('("qwen3.8:27b-mlx", "chars", "below", 0.0)', '("qwen3.8:27b-mlx", "chars", "below", 0.3)'),
    "h6-threshold": ('("qwen3.8:27b-mlx", "tokens", "below", 0.0)', '("qwen3.8:27b-mlx", "tokens", "below", 0.3)'),
    "no-registration-check": ('if data.get("registration") != lock["registration"]:', "if False:"),
}


def main():
    src = (HERE / "analyze.py").read_text()
    (HERE / "runs").mkdir(exist_ok=True)
    bad = []
    for name, (find, repl) in MUTANTS.items():
        if src.count(find) != 1:
            sys.exit(f"{name}: its find text occurs {src.count(find)} times")
        d = Path(tempfile.mkdtemp(dir=HERE / "runs"))
        (d / "analyze.py").write_text(src.replace(find, repl))
        p = subprocess.run([sys.executable, "-m", "unittest", str(HERE / "tests" / "test_analyze.py")], capture_output=True,
                           text=True, env={**os.environ, "ANALYZE": str(d / "analyze.py")}, cwd=HERE.parent)
        killed = p.returncode != 0 and "AssertionError" in p.stderr
        print(f"{name}: {'killed' if killed else 'SURVIVED'}")
        bad += [] if killed else [name]
    sys.exit(f"survivors: {bad}" if bad else 0)


if __name__ == "__main__":
    main()
