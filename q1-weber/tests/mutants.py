"""Each mutant of analyze.py must make at least one control fail on an assertion (not on an import or a crash).

    python -I q1-weber/tests/mutants.py"""
import os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
MUTANTS = {
    "h1-threshold-zero": ('h1 = "supported" if lo > 0.5', 'h1 = "supported" if lo > 0.0'),
    "no-gate": ("if min(parsed.values()) < GATE:", "if False:"),
    "no-completeness": ("if len(keys) != len(set(keys)) or set(keys) != want:", "if False:"),
    "events-band-wide": ("EVENTS = 0.25 ", "EVENTS = 0.75 "),
    "median-is-mean": ("s, l = np.median(est[(\"short\", n)]), np.median(est[(\"long\", n)])",
                       "s, l = np.mean(est[(\"short\", n)]), np.mean(est[(\"long\", n)]) * 1.3"),
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
