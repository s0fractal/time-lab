"""q10-drift calibration: the pilot's rows (calibration/pilot-rows.json, namespace 'q10-calibration', 8 scenarios per
kind, its own scenario per model and cell) for the registered cells, as a data.json the registered analysis can read
(runs/q10-calibration/, ignored), then that analysis on them: calibration/pilot-analysis.json. Not part of any verdict.

    python -I q10-drift/calib.py"""
import copy, json, subprocess, sys
from pathlib import Path

D = Path(__file__).resolve().parent
OUT = D.parent / "runs" / "q10-calibration"

if __name__ == "__main__":
    reg = json.loads((D / "registration.json").read_text())
    pilot = json.loads((D / "calibration" / "pilot-rows.json").read_text())
    cells = reg["design"]["cells"]
    keep = [r for r in pilot["rows"] if r["cond"] in cells]
    per = 1 + max(r["k"] for r in keep)
    cal = copy.deepcopy(reg)
    cal["design"]["scenarios"] = per
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    (OUT / "registration.json").write_text(json.dumps(cal))
    (OUT / "registration.lock.json").write_text(json.dumps({"registration": "calibration"}))
    models = {}
    for r in keep:
        row = {"cell": r["cond"], "kind": r["kind"], "k": r["k"], "v0": r["v0"], "v1": r["v1"], "turn1": r["turn1"], "tokens": r["tokens"]}
        if "turn2" in r:
            row["turn2"] = r["turn2"]
        models.setdefault(r["model"], {"rows": []})["rows"].append(row)
    (OUT / "results" / "data.json").write_text(json.dumps({"registration": "calibration", "models": models}) + "\n")
    p = subprocess.run([sys.executable, "-I", str(D / "analyze.py"), str(OUT)], capture_output=True, text=True, check=True)
    (D / "calibration" / "pilot-analysis.json").write_text(p.stdout)
    for h, v in json.loads(p.stdout)["verdicts"].items():
        print(h, v["verdict"], [(c["model"].split(":")[0], c["cell"], c["statistic"], f"{c['count']}/{c['n']}", c["verdict"]) for c in v["components"]])
