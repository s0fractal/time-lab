"""Pilot 2 (calibration seeds): flux through cubes (Gauss), the bond field against the lattice Coulomb field, and borrowed
time's variance by distance, with linear-theory predictions beside each."""
import json, sys, time
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from lattice import observe, green, distance
from pilot import BASE


def cube_sum(a, L, D, R):
    """Sum of a (sites shaped L^D) over the cube of half-width R around the origin (periodic)."""
    idx = np.r_[0:R + 1, L - R:L] if R else np.r_[0:1]
    return float(a[np.ix_(*([idx] * D))].sum())


def lin_flux(L, D, R, mu2, delta):
    """Outward flux predicted by linear theory: delta minus what the pin absorbs inside, mu2 g G^mu summed (mu2 = 0: the
    uniform background delta V / N)."""
    if mu2 == 0:
        V = (2 * R + 1) ** D
        return delta * (1 - V / L ** D)
    G = green(L, D, mu2)
    return delta * (1 - mu2 * cube_sum(G, L, D, R))


out = {}
delta = 0.5
for F, tag in ((1.3, "running"), (0.0, "resting")):
    mu2 = 0.0 if F else BASE["m_pin"] ** 2 * BASE["g_pin"] / BASE["g_lat"]
    for D, L in ((1, 64), (2, 32), (3, 24)):
        t = time.time()
        o = observe({**BASE, "F": F}, L, D, 8, delta, 200, 2200, seed=3000 + D)
        c = o["c"].mean(0)
        R4 = L // 4
        flux = {R: (-cube_sum(c, L, D, R), lin_flux(L, D, R, mu2, delta)) for R in (0, 1, 2, 3, R4)}
        G0 = green(L, D)
        r = distance(L, D)
        var = o["rel_var"].mean(0)
        ax = lambda d: (d,) + (0,) * (D - 1)
        vr = {d: (float(var[ax(d)]), float(2 * (G0[(0,) * D] - G0[ax(d)]))) for d in (1, 2, 4, R4)}
        s = {"flux(R): measured, linear": {R: [round(a, 4), round(b, 4)] for R, (a, b) in flux.items()},
             "var(d) on axis: measured, 2(G0-G)": {d: [round(a, 4), round(b, 4)] for d, (a, b) in vr.items()},
             "var ratio R4/1: measured, G": [round(vr[R4][0] / vr[1][0], 3), round(vr[R4][1] / vr[1][1], 3)],
             "lead drift": np.round(o["lead"][:, 1] - o["lead"][:, 0], 2).tolist(), "seconds": round(time.time() - t, 1)}
        out[f"{tag}-D{D}"] = s
        print(tag, D, json.dumps(s), flush=True)
(Path(__file__).parent / "pilot2.json").write_text(json.dumps(out, indent=1) + "\n")
