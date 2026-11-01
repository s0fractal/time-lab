"""Pilot (calibration seeds): the largest pacemaker detuning the running lattice follows, by dimension."""
import json, sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from lattice import observe
from pilot import BASE

out = {}
for sigma in (0.0, 0.3):
    for D, L in ((1, 64), (2, 32), (3, 16)):
        row = {}
        for delta in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0):
            o = observe({**BASE, "sigma": sigma}, L, D, 4, delta, 200, 1200, seed=2000 + D)
            rate = (o["lead"][:, 1] - o["lead"][:, 0]) / 1000
            row[delta] = round(float(np.mean(rate)), 4)
        out[f"s{sigma}-D{D}"] = row
        print(f"sigma {sigma} D{D}", row, flush=True)
(Path(__file__).parent / "capacity.json").write_text(json.dumps(out, indent=1) + "\n")
