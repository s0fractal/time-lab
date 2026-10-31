"""Each mutant of q7-duration/analyze.py must make a control fail on an assertion (not on an import or a crash).

    python -I q7-duration/tests/mutants.py"""
import os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
MUTANTS = {
    "r-inverted": ('R = lambda c, w, m: lg(worlds[w]["count"][m][idx].mean(), clocks[c]["sq"][w]["clock"][m][idx].mean())',
                   'R = lambda c, w, m: lg(clocks[c]["sq"][w]["clock"][m][idx].mean(), worlds[w]["count"][m][idx].mean())'),
    "r-on-fused": ('clocks[c]["sq"][w]["clock"][m][idx].mean())', 'clocks[c]["sq"][w]["fused"][m][idx].mean())'),
    "abs-not-square": ('"count": {m: e * e for m, e in', '"count": {m: abs(e) for m, e in'),
    "formula-without-k": ("math.log10(design[\"T\"] / (2 * kof[w] * star[c]))", "math.log10(design[\"T\"] / (2 * star[c]))"),
    "formula-without-2": ("math.log10(design[\"T\"] / (2 * kof[w] * star[c]))", "math.log10(design[\"T\"] / (kof[w] * star[c]))"),
    "formula-without-diffusion": ("math.log10(design[\"T\"] / (2 * kof[w] * star[c]))", "math.log10(design[\"T\"] / (2 * kof[w]))"),
    "formula-another-clock": ('star = {c["name"]: c["D_star"] for c in design["clocks"]}', 'star = {c["name"]: design["clocks"][1]["D_star"] for c in design["clocks"]}'),
    "ratio-reversed-span": ('"H5-span": ("R-ratio", (("N1", "k256", "16"), ("N1", "k256", "64"))),', '"H5-span": ("R-ratio", (("N1", "k256", "64"), ("N1", "k256", "64"))),'),
    "h1-threshold": ('"H1": ("above", LOG100)', '"H1": ("above", 1.0)'),
    "h2-width": ('"H2": ("within", LOG43)', '"H2": ("within", 0.5)'),
    "h3-width": ('"H3": ("within", LOG15)', '"H3": ("within", 0.5)'),
    "h4-threshold": ('"H4": ("below", -LOG2)', '"H4": ("below", 0.0)'),
    "h5-width": ('"H5": ("within", LOG15)', '"H5": ("within", 0.5)'),
    "within-one-sided": ('return "supported" if -x < lo and hi < x else', 'return "supported" if hi < x else'),
    "below-bound": ('return "supported" if hi < x else "contradicted" if lo > x', 'return "supported" if lo < x else "contradicted" if lo > x'),
    "undefined-as-zero": ("return math.log10(a / b) if a > 0 and b > 0 else math.nan", "return math.log10(a / b) if a > 0 and b > 0 else 0.0"),
    "no-completeness": ("if set(data[\"clocks\"]) != set(cn) or set(data[\"worlds\"]) != set(wn):", "if False:"),
    "no-query-check": ("if not isinstance(got, dict) or set(got) != set(ms):", "if not isinstance(got, dict):"),
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
