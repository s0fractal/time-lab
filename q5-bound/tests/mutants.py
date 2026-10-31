"""Each mutant of q5-bound/analyze.py must make a control fail on an assertion (not on an import or a crash).

    python -I q5-bound/tests/mutants.py"""
import os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
MUTANTS = {
    "single-window": ("D = ((x[2] - x[0]).var() - (x[1] - x[0]).var()) / (2 * (t2 - t1))", "D = (x[2] - x[0]).var() / (2 * (t2 - t0))"),
    "line-heat-uncounted": ("x[0] / (design[\"N\"] * design[\"base\"][\"d\"]), x[1] + x[2]])", "x[0] / (design[\"N\"] * design[\"base\"][\"d\"]), x[1]])"),
    "gain-inverted": ("return math.log10(Db / Da)", "return math.log10(Da / Db)"),
    "eta-without-velocity": ("return math.log10((va * va / (Da * Sa)) / (vb * vb / (Db * Sb)))", "return math.log10((1 / (Da * Sa)) / (1 / (Db * Sb)))"),
    "h1-threshold": ('"H1": ("above", LOG3)', '"H1": ("above", LOG2)'),
    "h2-threshold": ('"H2": ("below", LOG125)', '"H2": ("below", LOG2)'),
    "h3-sign": ('"H3": ("below", 0.0)', '"H3": ("below", LOG2)'),
    "h4-threshold": ('"H4": ("below", -LOG15)', '"H4": ("below", -0.1)'),
    "no-adequacy": ('elif h == "H2" and not near[0] > NEAR:', 'elif False:'),
    "adequacy-on-the-point": ('near = [float(np.quantile(alone, q, method="inverted_cdf")) for q in (0.025, 0.975)]',
                              'near = [eta(cells["alone"], design)] * 2'),
    "below-bound": ('return "supported" if hi < x else "contradicted" if lo > x', 'return "supported" if lo < x else "contradicted" if lo > x'),
    "both-zero-not-undefined": ("if Da <= 0 and Db <= 0:\n        return math.nan", "if Da <= 0 and Db <= 0:\n        return math.inf"),
    "infinite-flipped": ("if Da <= 0:\n        return math.inf", "if Da <= 0:\n        return -math.inf"),
    "no-completeness": ('if set(data["cells"]) != set(want):', "if False:"),
    "no-shape-check": ('if x.shape != (3, 3, design["trajectories"]) or not np.isfinite(x).all():', "if not np.isfinite(x).all():"),
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
