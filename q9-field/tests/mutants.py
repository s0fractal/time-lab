"""Each mutant of q9-field/analyze.py must make a control fail on an assertion (not on an import or a crash).

    python -I q9-field/tests/mutants.py"""
import os, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
MUTANTS = {
    "field-no-weights": ('(self.bw[i] * m["E"][i] * self.pred[i]).sum() / (self.bw[i] * self.pred[i] ** 2).sum()',
                         '(m["E"][i] * self.pred[i]).sum() / (self.pred[i] ** 2).sum()'),
    "field-near-bonds": ("inside.append(2 < min(r0, r1) and max(r0, r1) <= self.L / 4)", "inside.append(max(r0, r1) <= self.L / 4)"),
    "field-no-far-cap": ("inside.append(2 < min(r0, r1) and max(r0, r1) <= self.L / 4)", "inside.append(2 < min(r0, r1))"),
    "field-inward": ("p.append(delta * (at(x0) - at(x1)))", "p.append(delta * (at(x1) - at(x0)))"),
    "field-next-bond": ("x1 = ((k[0] + 1) / 2,) + x0[1:]", "x1 = ((k[0] + 3) / 2,) + x0[1:]"),
    "screening-no-linear": ("return c.flux(m, 1) / f0 / (c.flux_lin[1] / c.flux_lin[0]) if f0 > 0 else math.nan",
                            "return c.flux(m, 1) / f0 if f0 > 0 else math.nan"),
    "flux-no-multiplicity": ('return float(-sum(self.sw[i] * m["c"][i] for i, k in enumerate(self.skeys) if max(k) <= R))',
                             'return float(-sum(m["c"][i] for i, k in enumerate(self.skeys) if max(k) <= R))'),
    "flux-wider-cube": ('return float(-sum(self.sw[i] * m["c"][i] for i, k in enumerate(self.skeys) if max(k) <= R))',
                        'return float(-sum(self.sw[i] * m["c"][i] for i, k in enumerate(self.skeys) if max(k) <= R + 1))'),
    "mu2-not-squared": ('mu2 = self.prm["m_pin"] ** 2 * self.prm["g_pin"] / self.prm["g_lat"]',
                        'mu2 = self.prm["m_pin"] * self.prm["g_pin"] / self.prm["g_lat"]'),
    "time-swapped": ('        return float(m["var"][c.far] / m["var"][c.near])\n    if name == "time_vs_G":',
                     '        return float(m["var"][c.near] / m["var"][c.far])\n    if name == "time_vs_G":'),
    "time-without-G": ('return float(m["var"][c.far] / m["var"][c.near]) / c.time_G', 'return float(m["var"][c.far] / m["var"][c.near])'),
    "heating-fixed-speed": ('v = float(m["speed"].mean())', "v = 2.59"),
    "heating-wide-band": ("return self.band(m, 0.85 * target, 1.15 * target)", "return self.band(m, 0.5 * target, 1.5 * target)"),
    "slow-band-all": ("SLOW = (0.0, 2.0)", "SLOW = (0.0, 4.0)"),
    "band-unweighted": ('return float((m["spectrum"][sel] * self.mcount[sel]).sum() / self.mcount[sel].sum())',
                        'return float(m["spectrum"][sel].mean())'),
    "gain-reversed": ("return top(cells[0], ms[0]) / top(cells[1], ms[1])", "return top(cells[1], ms[1]) / top(cells[0], ms[0])"),
    "no-slip-gate": ("slips = {cc.name: cc.slip for cc in cs if not cc.slip < SLIP}", "slips = {}"),
    "slip-at-pi": ("SLIP = 1.5 * math.pi", "SLIP = math.pi"),
    "no-bootstrap": ("v = statistic(name, cs, [cc.mean(g.integers(0, B, B)) for cc, g in zip(cs, rngs)])",
                     "v = statistic(name, cs, [cc.mean(np.arange(B)) for cc in cs])"),
    "within-one-sided": ('return "supported" if -t < lo and hi < t else', 'return "supported" if hi < t else'),
    "below-bound": ('return "supported" if hi < t else "contradicted" if lo > t', 'return "supported" if lo < t else "contradicted" if lo > t'),
    "above-bound": ('return "supported" if lo > t else "contradicted" if hi < t', 'return "supported" if hi > t else "contradicted" if hi < t'),
    "no-registration-check": ('if data.get("registration") != lock["registration"]:', "if False:"),
    "no-cell-check": ('if sorted(data["cells"]) != sorted(names):', "if False:"),
    "no-site-orbit-check": ('if [tuple(k) for k in x["site_keys"]] != sorted(sites) or x["site_count"] != [sites[k] for k in sorted(sites)]:',
                            'if [tuple(k) for k in x["site_keys"]] != sorted(sites):'),
    "no-bond-orbit-check": ('if [tuple(k) for k in x["bond_keys"]] != sorted(bonds) or x["bond_count"] != [bonds[k] for k in sorted(bonds)]:',
                            'if x["bond_count"] != [bonds[k] for k in sorted(bonds)]:'),
    "no-trajectory-check": ("if len(x[f]) != B or any(len(row) != n for row in x[f]):", "if any(len(row) != n for row in x[f]):"),
    "no-range-check": ('if len(x["speed"]) != B or len(x["range"]) != B:', 'if len(x["speed"]) != B:'),
    "expected-no-minimum-image": ("two_m = min(two_m, 2 * L - two_m)", "two_m = two_m"),
    "h1-width": ('"H1": ("field", ("run-D3",), "within", 1.15)', '"H1": ("field", ("run-D3",), "within", 1.5)'),
    "h2-width": ('"H2": ("field", ("run-D2",), "within", 1.15)', '"H2": ("field", ("run-D2",), "within", 1.5)'),
    "h3-width": ('"H3": ("screening", ("rest-D3",), "within", 1.25)', '"H3": ("screening", ("rest-D3",), "within", 2.0)'),
    "h4-threshold": ('"H4": ("time", ("run-D3",), "below", 1.6)', '"H4": ("time", ("run-D3",), "below", 3.0)'),
    "h5-width": ('"H5": ("time_vs_G", ("run-D1",), "within", 1.25)', '"H5": ("time_vs_G", ("run-D1",), "within", 2.0)'),
    "h6-threshold": ('"H6": ("heating", ("run-D3",), "above", 1.5)', '"H6": ("heating", ("run-D3",), "above", 1.1)'),
    "h7-threshold": ('"H7": ("heating_gain", ("run-D1-g2", "run-D1"), "above", 1.3)',
                     '"H7": ("heating_gain", ("run-D1-g2", "run-D1"), "above", 1.05)'),
}


def main():
    src = (HERE / "analyze.py").read_text()
    (HERE / "runs").mkdir(exist_ok=True)
    bad = []
    for name, (find, repl) in MUTANTS.items():
        if src.count(find) != 1:
            sys.exit(f"{name}: its find text occurs {src.count(find)} times")
        d = Path(tempfile.mkdtemp(dir=HERE / "runs"))
        try:
            (d / "analyze.py").write_text(src.replace(find, repl))
            p = subprocess.run([sys.executable, "-m", "unittest", str(HERE / "tests" / "test_analyze.py")], capture_output=True,
                               text=True, env={**os.environ, "ANALYZE": str(d / "analyze.py")}, cwd=HERE.parent)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        killed = p.returncode != 0 and "AssertionError" in p.stderr
        print(f"{name}: {'killed' if killed else 'SURVIVED'}", flush=True)
        bad += [] if killed else [name]
    sys.exit(f"survivors: {bad}" if bad else 0)


if __name__ == "__main__":
    main()
