"""Each mutant of q3-self/analyze.py must make a control fail on an assertion (not on an import or a crash).

    python -I q3-self/tests/mutants.py"""
import os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
MUTANTS = {
    "single-window": ("return ((phi[2] - phi[0]).var() - (phi[1] - phi[0]).var()) / (2 * (t2 - t1))",
                      "return (phi[2] - phi[0]).var() / (2 * (t2 - t0))"),
    "h1-threshold": ('"H1": ("above", 1.0)', '"H1": ("above", LOG2)'),
    "h3-threshold": ('"H3": ("above", LOG5)', '"H3": ("above", LOG2)'),
    "h6-threshold": ('"H6": ("above", LOG2)', '"H6": ("above", 0.0)'),
    "below-bound": ('return "supported" if hi < x else "contradicted" if lo > x', 'return "supported" if lo < x else "contradicted" if lo > x'),
    "within-one-sided": ('return "supported" if -x < lo and hi < x else', 'return "supported" if hi < x else'),
    "above-on-the-point": ('return "supported" if lo > x else "contradicted" if hi < x', 'return "supported" if (lo + hi) / 2 > x else "contradicted" if hi < x'),
    "both-zero-not-undefined": ("if Da <= 0 and Db <= 0:\n        return math.nan", "if Da <= 0 and Db <= 0:\n        return math.inf"),
    "infinite-flipped": ("if Da <= 0:\n        return math.inf", "if Da <= 0:\n        return -math.inf"),
    "gain-inverted": ("return math.log10(Db / Da)", "return math.log10(Da / Db)"),
    "no-completeness": ('if set(data["cells"]) != set(want):', "if False:"),
    "no-shape-check": ('if J.shape != (3, design["trajectories"]) or not np.isfinite(J).all():', "if not np.isfinite(J).all():"),
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
