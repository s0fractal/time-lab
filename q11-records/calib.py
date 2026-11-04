"""q11-records calibration: the pilot's rows (calibration/pilot-data.json, namespace 'q11-calibration', the 30 calibration questions)
as a directory the registered analysis can read (runs/q11-calibration/, ignored), then that analysis on them:
calibration/pilot-analysis.json. Not part of any verdict.

    python -I q11-records/calib.py"""
import copy, hashlib, json, subprocess, sys
from pathlib import Path

D = Path(__file__).resolve().parent
OUT = D.parent / "runs" / "q11-calibration"

if __name__ == "__main__":
    reg = copy.deepcopy(json.loads((D / "registration.json").read_text()))
    raw = (D / "calibration" / "bank.json").read_bytes()
    reg["design"]["bank_sha256"] = hashlib.sha256(raw).hexdigest()
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    (OUT / "bank.json").write_bytes(raw)
    (OUT / "registration.json").write_text(json.dumps(reg))
    (OUT / "registration.lock.json").write_text(json.dumps({"registration": "q11-calibration"}))
    (OUT / "results" / "data.json").write_bytes((D / "calibration" / "pilot-data.json").read_bytes())
    p = subprocess.run([sys.executable, "-I", str(D / "analyze.py"), str(OUT)], capture_output=True, text=True, check=True)
    (D / "calibration" / "pilot-analysis.json").write_text(p.stdout)
    r = json.loads(p.stdout)
    for h, v in r["verdicts"].items():
        print(h, v["verdict"], [(c["kind"], c["class"], c["other"], round(c["value"], 3), [round(x, 3) for x in c["ci95"]], c["verdict"]) for c in v["components"]])
    for m, arms in r["arms"].items():
        for a, cl in arms.items():
            print(m.split(":")[0], a, *[f"{c}: {x['correct']}/{x['n']} old {x['old']} E {x['not_settled']} junk {x['unparsed']} tok {x['tokens']:.0f}" for c, x in cl.items()], sep=" | ")
