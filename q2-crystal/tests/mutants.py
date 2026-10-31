"""Each mutant of q2-crystal/analyze.py must make a control fail on an assertion (not on an import or a crash).

    python -I q2-crystal/tests/mutants.py"""
import os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
MUTANTS = {
    "h1-any-step": ('"supported" if all(lo[h] > LOG2 for h in sizes)', '"supported" if any(lo[h] > LOG2 for h in sizes)'),
    "h2-threshold": ('"supported" if lo["H2-coupling"] > 1', '"supported" if lo["H2-coupling"] > 0'),
    "h3-threshold": ('"supported" if lo["H3-noise"] > LOG5', '"supported" if lo["H3-noise"] > LOG2'),
    "h4-sign": ('"supported" if hi["H4-gyroscope"] < -LOG2', '"supported" if hi["H4-gyroscope"] < LOG2'),
    "gain-inverted": ("return off.var() / on.var() if on.var() > 0", "return on.var() / off.var() if on.var() > 0"),
    "no-completeness": ("if set(data[\"cells\"]) != set(want):", "if False:"),
    "no-length-check": ('if dJ.shape != (design["trajectories"],) or not np.isfinite(dJ).all():', "if not np.isfinite(dJ).all():"),
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
