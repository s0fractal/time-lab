"""Each mutant of q6-renewal/analyze.py must make a control fail on an assertion (not on an import or a crash).

    python -I q6-renewal/tests/mutants.py"""
import os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
MUTANTS = {
    "single-window": ("D = ((rows[2] - rows[0]).var() - (rows[1] - rows[0]).var()) / (2 * (t2 - t1))",
                      "D = (rows[2] - rows[0]).var() / (2 * (t2 - t0))"),
    "mse-inverted": ("out[t] = lg(m(a[0], a[2]), m(a[1], a[2]))\n", "out[t] = lg(m(a[1], a[2]), m(a[0], a[2]))\n"),
    "price-without-entropy": ("out[t] = lg(m(a[0], a[2]), m(a[1], a[2])) - lg(S1, S0)", "out[t] = lg(m(a[0], a[2]), m(a[1], a[2]))"),
    "price-entropy-inverted": ("out[t] = lg(m(a[0], a[2]), m(a[1], a[2])) - lg(S1, S0)", "out[t] = lg(m(a[0], a[2]), m(a[1], a[2])) - lg(S0, S1)"),
    "capture-as-anticipation": ('P = WORLD_ROWS.index("sse_perfect")', 'P = WORLD_ROWS.index("sse_none")'),
    "capture-without-ceiling": ("out[t] = lg(mse(worlds[a[1]][P][idx], n[a[1]]), m(a[0], a[1])) - lg(mse(worlds[a[3]][P][idx], n[a[3]]), m(a[2], a[3]))",
                                "out[t] = lg(m(a[2], a[3]), m(a[0], a[1]))"),
    "brownian-without-velocity": ("out[t] = lg(2 * D / v ** 2 * elapsed, m(a[0], a[1]))", "out[t] = lg(2 * D * elapsed, m(a[0], a[1]))"),
    "brownian-at-half-T": ("out[t] = lg(2 * D / v ** 2 * elapsed, m(a[0], a[1]))", "out[t] = lg(2 * D / v ** 2 * design[\"T\"] / 2, m(a[0], a[1]))"),
    "h1-threshold": ('"H1": ("below", LOG11)', '"H1": ("below", LOG125)'),
    "h2-width": ('"H2": ("within", LOG125)', '"H2": ("within", LOG4)'),
    "h3-threshold": ('"H3": ("below", LOG4)', '"H3": ("below", 1.0)'),
    "h4-threshold": ('"H4": ("above", 0.0)', '"H4": ("above", -LOG11)'),
    "h5-width": ('"H5": ("within", LOG11)', '"H5": ("within", LOG125)'),
    "within-one-sided": ('return "supported" if -x < lo and hi < x else', 'return "supported" if hi < x else'),
    "below-bound": ('return "supported" if hi < x else "contradicted" if lo > x', 'return "supported" if lo < x else "contradicted" if lo > x'),
    "undefined-as-zero": ("return math.log10(a / b) if a > 0 and b > 0 else math.nan", "return math.log10(a / b) if a > 0 and b > 0 else 0.0"),
    "no-completeness": ("if set(data[\"clocks\"]) != set(cn) or set(data[\"worlds\"]) != set(wn):", "if False:"),
    "no-length-check": ("any(not isinstance(r, list) or len(r) != T for r in rows)", "False"),
    "no-finite-check": ("if not np.isfinite(x).all():\n        refuse", "if False:\n        refuse"),
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
