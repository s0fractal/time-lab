"""Pilot (calibration seeds, not part of any verdict): the field of a pacemaker in a running and a resting crystal lattice."""
import json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lattice import observe, green, distance

BASE = {"dt": 1.0, "S": 4, "gamma": 0.5, "g_lat": 1.0, "g_pin": 0.5, "m_pin": 2, "F": 1.3, "sigma": 0.3}


def slope(x, y):
    return float((x * y).sum() / (x * x).sum())


def summary(o, L, D, delta, mu2):
    r = distance(L, D).ravel()
    G = green(L, D, mu2).ravel()
    phi = o["phi"].reshape(o["phi"].shape[0], -1).mean(0)
    pred = delta / BASE["g_lat"] * (G - G[r > 0].mean() if mu2 == 0 else G)
    phi = phi - (phi[r > 0].mean() if mu2 == 0 else 0)
    near, far = (r >= 1) & (r <= 2), (r > 2) & (r <= L / 4)
    var = o["rel_var"].reshape(o["rel_var"].shape[0], -1).mean(0)
    G0 = green(L, D).ravel()
    w = 2 * (G0[0] - G0)
    on_axis = lambda d: int(np.argmin(np.abs(r - d) + 10 * (np.abs(r - d) > 1e-9)))
    c = o["c"].reshape(o["c"].shape[0], -1).mean(0)
    return {"kappa_near": slope(pred[near], phi[near]), "kappa_far": slope(pred[far], phi[far]),
            "var_slope_near": slope(w[near], var[near]), "var_slope_far": slope(w[far], var[far]),
            "var_r1": float(var[on_axis(1)]), "var_rL4": float(var[on_axis(L // 4)]),
            "c0": float(c[0]), "c_rest_mean": float(c[1:].mean()), "lead_drift": (o["lead"][:, 1] - o["lead"][:, 0]).tolist(),
            "speed": o["speed"]}


if __name__ == "__main__":
    out = {}
    for F in (1.3, 0.0):
        for D, L in ((1, 64), (2, 32), (3, 16)):
            t = time.time()
            o = observe({**BASE, "F": F}, L, D, 4, 0.5, 200, 1200, seed=1000 + D)
            mu2 = 0.0 if F else BASE["m_pin"] ** 2 * BASE["g_pin"] / BASE["g_lat"]
            s = summary(o, L, D, 0.5, mu2)
            s["seconds"] = round(time.time() - t, 1)
            out[f"F{F}-D{D}"] = s
            print(f"F{F}-D{D}", json.dumps(s), flush=True)
    (Path(__file__).parent / "pilot.json").write_text(json.dumps(out, indent=1) + "\n")
