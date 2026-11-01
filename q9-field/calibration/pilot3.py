"""Pilot 3 (calibration seeds): is the running lattice's excess relative-phase noise a parametric resonance with the pin?
Mode energy g lam_k <|X_k|^2> / N against lam_k (equipartition: sigma^2 for every mode). The pin turns at m v; a mode of
frequency sqrt(g lam) near m v / 2 is parametrically pumped."""
import json, sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from lattice import Lattice
from pilot import BASE


def spectrum(p, L, D, B, warm, t_total, seed):
    lat = Lattice(p, L, D, B, 0.0, seed)
    k = 2 * np.pi * np.fft.fftfreq(L)
    lam = sum(np.meshgrid(*([2 - 2 * np.cos(k)] * D), indexing="ij"))
    acc, m = 0.0, 0
    ax = tuple(range(1, D + 1))
    for i in range(1, int(t_total / lat.h) + 1):
        lat.step()
        if i * lat.h > warm and i % 4 == 0:
            Xk = np.fft.fftn(lat.X - lat.X.mean(ax, keepdims=True), axes=ax)
            acc = acc + (np.abs(Xk) ** 2).mean(0)
            m += 1
    E = p["g_lat"] * lam * acc / m / L ** D
    bins = np.linspace(0, 4 * D, 4 * D + 1)
    lab = np.digitize(lam, bins)
    return {f"{bins[b - 1]:.0f}-{bins[b]:.0f}": round(float(E[lab == b].mean()), 4) for b in range(1, len(bins)) if (lab == b).any()}


out = {}
for g, D, L in ((1.0, 1, 64), (2.0, 1, 64), (1.0, 2, 32), (1.0, 3, 16)):
    s = spectrum({**BASE, "g_lat": g}, L, D, 4, 200, 1200, 4000 + D)
    out[f"g{g}-D{D}"] = s
    print(f"g {g} D{D} resonance at lam = {(BASE['m_pin'] * 2.6 / 2) ** 2 / g:.2f}", s, flush=True)
(Path(__file__).parent / "pilot3.json").write_text(json.dumps(out, indent=1) + "\n")
