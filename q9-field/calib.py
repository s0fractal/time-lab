"""q9-field calibration: the registered design on seeds of a separate namespace ('calibration:<cell>'), written as a
data.json the registered analysis can read (runs/q9-calibration/, ignored). Not part of any verdict.

    python -I q9-field/calib.py"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import observe, seed

D = Path(__file__).resolve().parent
OUT = D.parent / "runs" / "q9-calibration"

if __name__ == "__main__":
    design = json.loads((D / "registration.json").read_text())["design"]
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    (OUT / "registration.json").write_text((D / "registration.json").read_text())
    (OUT / "registration.lock.json").write_text(json.dumps({"registration": "calibration"}))
    cells = {}
    for c in design["cells"]:
        cells[c["name"]] = observe({**design["base"], **c["params"]}, c["L"], c["D"], design["trajectories"], design["delta"],
                                   design["warm"], design["t_total"], seed("calibration", "calibration", c["name"]))
        print(c["name"], flush=True)
    (OUT / "results" / "data.json").write_text(json.dumps({"registration": "calibration", "cells": cells}) + "\n")
