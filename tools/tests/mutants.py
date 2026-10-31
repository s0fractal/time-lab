"""Each mutant of tl.py's own checks must make a control fail on an assertion (not on an import or a crash).

    python -I tools/tests/mutants.py"""
import os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
MUTANTS = {
    "code-unchecked": ('.get("match", {}).get("code") != reg.get("analysis")', '.get("match", {}).get("code") == "never"'),
    "withdrawn-unchecked": ('if state == "withdrawn" and any(', 'if False and any('),
    "chain-unordered-counts": ('pos.get(b, (None,))[0] == r["actor"] == pos.get(i, (None,))[0] and', 'True and'),
    "chain-order-reversed": ('pos[i][1] < pos[b][1]', 'pos[i][1] > pos[b][1]'),
    "tolerance-wide": ("return abs(a - b) <= max(rel * max(abs(a), abs(b)), 1e-12)", "return abs(a - b) <= 0.01"),
    "keys-unchecked": ("return a.keys() == b.keys() and all(", "return all(k in b for k in a) and all("),
    "flag-as-number": ("        return a is b\n", "        return a == b\n"),
}


def main():
    src = (HERE / "tl.py").read_text()
    work = HERE.parent / "runs"
    work.mkdir(exist_ok=True)
    bad = []
    with tempfile.TemporaryDirectory(dir=work) as d:
        for name, (find, repl) in MUTANTS.items():
            if src.count(find) != 1:
                sys.exit(f"{name}: its find text occurs {src.count(find)} times")
            (Path(d) / "tl.py").write_text(src.replace(find, repl))
            p = subprocess.run([sys.executable, "-B", "-m", "unittest", str(HERE / "tests" / "test_tl.py")], capture_output=True,
                               text=True, env={**os.environ, "TL": str(Path(d) / "tl.py")}, cwd=HERE.parent)
            killed = p.returncode != 0 and "AssertionError" in p.stderr
            print(f"{name}: {'killed' if killed else 'SURVIVED'}")
            bad += [] if killed else [name]
    sys.exit(f"survivors: {bad}" if bad else 0)


if __name__ == "__main__":
    main()
