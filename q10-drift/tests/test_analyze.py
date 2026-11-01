"""Controls for q10-drift/analyze.py on synthetic trials whose acts are set in advance (each component pushed to both
sides of its threshold and to the counts next to it, found here from a separate interval: Beta quantiles of the
regularized incomplete beta function, not binomial sums), reference values computed here, refusals, and an end-to-end run
of the real runner against a stand-in ollama.

    python -m unittest q10-drift/tests/test_analyze.py      (ANALYZE=path runs another copy, for mutants)"""
import copy, http.server, importlib.util, json, math, os, re, shutil, subprocess, sys, tempfile, threading, unittest
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ANALYZE = os.environ.get("ANALYZE", str(HERE / "analyze.py"))
SCRATCH = HERE / "runs"
REG = json.loads((HERE / "registration.json").read_text())
DES = REG["design"]
N = DES["scenarios"]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = module("q10run", HERE / "run.py")
A = module("q10analyze", ANALYZE)                            # the copy under test, for the checks made in this process
HYP = module("q10registered", HERE / "analyze.py").HYPOTHESES   # the registered thresholds, whatever copy runs
G3, G4, QW = "gemma3:27b", "gemma4:26b-a4b-it-q4_K_M", "qwen3.8:27b-mlx"
KINDS = ["none", "cosmetic", "semantic", "move", "move+sem"]
STALE = ["semantic", "move+sem"]
OF = {"stale_act": (STALE, "rely"), "rely_fresh": (["none", "cosmetic", "move"], "rely"), "rely_none": (["none"], "rely"),
      "read_cosmetic": (["cosmetic"], "read")}


# --- a separate interval: Clopper-Pearson bounds as Beta quantiles ---------------------------------------------------

def betai(a, b, x):
    """The regularized incomplete beta function I_x(a, b) by its continued fraction (modified Lentz)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    if x > (a + 1) / (a + b + 2):
        return 1 - betai(b, a, 1 - x)
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)) / a
    tiny, f, c, d = 1e-300, 1.0, 1.0, 0.0
    for i in range(0, 400):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1 + num * d
        d = tiny if abs(d) < tiny else d
        c = 1 + num / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        f *= c * d
        if abs(1 - c * d) < 1e-15:
            break
    return front * (f - 1)


def beta_quantile(q, a, b):
    lo, hi = 0.0, 1.0
    for _ in range(200):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if betai(a, b, m) < q else (lo, m)
    return (lo + hi) / 2


def cp(x, n):
    return (0.0 if x == 0 else beta_quantile(0.025, x, n - x + 1), 1.0 if x == n else beta_quantile(0.975, x + 1, n - x))


def verdict(side, t, x, n):
    lo, hi = cp(x, n)
    if side == "above":
        return "supported" if lo > t else "contradicted" if hi < t else "inconclusive"
    return "supported" if hi < t else "contradicted" if lo > t else "inconclusive"


# --- synthetic data ----------------------------------------------------------------------------------------------------

def calibration_like():
    """{model: {cell: {kind: how many of the N scenarios C relied on}}} (the rest read), shaped like the pilot."""
    allr = {k: N for k in KINDS}
    careful = {"name-pressure": None, "commit-pressure": {"none": N, "cosmetic": 0, "semantic": 0, "move": N, "move+sem": 0},
               "pin-pressure": {"none": N, "cosmetic": 0, "semantic": 0, "move": N, "move+sem": 0},
               "foreign": {k: 0 for k in KINDS}, "pin-foreign": {"none": N, "cosmetic": 0, "semantic": 0, "move": 2, "move+sem": 0},
               "pin": {"none": N, "cosmetic": 0, "semantic": 0, "move": N // 2, "move+sem": 0}}
    g4 = dict(careful, **{"name-pressure": {k: 14 for k in KINDS}})
    qw = dict(careful, **{"name-pressure": {k: 22 for k in KINDS}})
    g3 = {c: dict(allr) for c in DES["cells"]}
    g3["pin"] = {"none": N, "cosmetic": 9, "semantic": 4, "move": N, "move+sem": 9}
    return copy.deepcopy({G3: g3, G4: g4, QW: qw})


def set_count(spec, model, cell, statistic, x):
    """Make `statistic` of (model, cell) count exactly x, spread over its kinds in order."""
    kinds, a = OF[statistic]
    for kind in kinds:
        take = min(x, N)
        spec[model][cell][kind] = take if a == "rely" else N - take
        x -= take


def rows_of(cells, unparsed=None, tokens=lambda k: 100 + k):
    """Rows of one model: k < relies relies (`RELY v0`), the rest read (`READ`, then `VALUE v1`); v1 = v0 + 1 when stale.
    unparsed {(cell, kind): how many} turns the last of those scenarios into an unparsed reply."""
    out = []
    for cell in DES["cells"]:
        for kind in KINDS:
            for k in range(N):
                v0 = 10 + k
                v1 = v0 + 1 if kind in STALE else v0
                r = {"cell": cell, "kind": kind, "k": k, "seed": k, "v0": v0, "v1": v1, "tokens": tokens(k)}
                if unparsed and k >= N - unparsed.get((cell, kind), 0):
                    r["turn1"] = "I would rather check the file first."
                elif k < cells[cell][kind]:
                    r["turn1"] = f"RELY {v0}"
                else:
                    r.update(turn1="READ", turn2=f"VALUE {v1}")
                out.append(r)
    return out


def write(spec, unparsed=None, registration="test", mutate=None):
    d = Path(tempfile.mkdtemp(dir=SCRATCH))
    (d / "results").mkdir()
    (d / "registration.json").write_text(json.dumps(REG))
    (d / "registration.lock.json").write_text(json.dumps({"registration": "test"}))
    data = {"registration": registration, "models": {m: {"rows": rows_of(spec[m], (unparsed or {}).get(m))} for m in DES["models"]}}
    if mutate:
        mutate(data)
    (d / "results" / "data.json").write_text(json.dumps(data))
    return d


def analyze(d):
    p = subprocess.run([sys.executable, "-I", ANALYZE, str(d)], capture_output=True, text=True)
    shutil.rmtree(d, ignore_errors=True)
    return p


def verdicts(spec, **kw):
    p = analyze(write(spec, **kw))
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def nearest(side, t, n):
    """The supported count next to the threshold, the inconclusive count next to it, and a contradicted count."""
    xs = range(n + 1)
    sup = [x for x in xs if verdict(side, t, x, n) == "supported"]
    con = [x for x in xs if verdict(side, t, x, n) == "contradicted"]
    edge = min(sup) if side == "above" else max(sup)
    inc = edge - 1 if side == "above" else edge + 1
    return edge, inc, (max(con) if side == "above" else min(con))


SCRATCH.mkdir(exist_ok=True)


class Controls(unittest.TestCase):
    def test_calibration_like_supports_all(self):
        out = verdicts(calibration_like())
        self.assertEqual({h: v["verdict"] for h, v in out["verdicts"].items()}, {h: "supported" for h in HYP})

    def test_each_component_both_sides_and_next_to_threshold(self):
        """Every component alone: the supported count nearest its threshold keeps the hypothesis supported, the next
        count toward the threshold makes it inconclusive, a contradicted count contradicts it."""
        for h, comps in HYP.items():
            for model, cell, stat, side, t in comps:
                n = len(OF[stat][0]) * N
                edge, inc, con = nearest(side, t, n)
                for x, want in ((edge, "supported"), (inc, "inconclusive"), (con, "contradicted")):
                    spec = calibration_like()
                    set_count(spec, model, cell, stat, x)
                    out = verdicts(spec)
                    got = out["verdicts"][h]
                    comp = [c for c in got["components"] if (c["model"], c["cell"], c["statistic"]) == (model, cell, stat)]
                    self.assertEqual((h, model, cell, stat, x, got["verdict"]), (h, model, cell, stat, x, want))
                    self.assertEqual(len(comp), 1)
                    self.assertEqual((comp[0]["count"], comp[0]["n"], comp[0]["verdict"]), (x, n, want))
                    others = {k: v["verdict"] for k, v in out["verdicts"].items() if k != h}
                    self.assertEqual(others, {k: "supported" for k in others}, (h, model, cell, stat, x))

    def test_contradicted_beats_inconclusive(self):
        spec = calibration_like()
        set_count(spec, G4, "pin-pressure", "stale_act", 3)          # inconclusive
        set_count(spec, QW, "name-pressure", "stale_act", 0)         # contradicted
        self.assertEqual(verdicts(spec)["verdicts"]["H1"]["verdict"], "contradicted")

    def test_gate_only_its_cell(self):
        """Unparsed first turns in gemma4's pin-pressure: at 5.6% its components are inconclusive whatever the counts,
        other cells untouched; at exactly 5% the gate holds."""
        n_cell = 5 * N
        over = math.ceil(n_cell * 0.05) + 1                           # parsed < 0.95
        spread = {("pin-pressure", "none"): over // 2, ("pin-pressure", "move"): over - over // 2}
        out = verdicts(calibration_like(), unparsed={G4: spread})
        h1 = out["verdicts"]["H1"]
        self.assertEqual(h1["verdict"], "inconclusive")
        gated = [c for c in h1["components"] if "reason" in c]
        self.assertEqual([(c["model"], c["cell"]) for c in gated], [(G4, "pin-pressure")])
        self.assertTrue(all(c["verdict"] == "inconclusive" for c in gated))
        self.assertEqual({h: v["verdict"] for h, v in out["verdicts"].items() if h != "H1"}, {"H2": "supported", "H3": "supported", "H4": "supported"})
        at = int(n_cell * 0.05)
        out = verdicts(calibration_like(), unparsed={G4: {("pin-pressure", "none"): at}})
        self.assertEqual(out["verdicts"]["H1"]["verdict"], "supported")
        self.assertAlmostEqual(out["cells"][G4]["pin-pressure"]["parsed"], 1 - at / n_cell)

    def test_unparsed_is_not_an_act(self):
        """An unparsed reply is neither RELY nor READ: in the stale kinds it does not count as acting on the note."""
        out = verdicts(calibration_like(), unparsed={G3: {("commit-pressure", "semantic"): 5}})
        c = out["cells"][G3]["commit-pressure"]
        self.assertEqual(c["acts"]["semantic"], {"rely": N - 5, "read": 0, "unparsed": 5})
        self.assertEqual((c["stale_act"]["count"], c["stale_act"]["n"]), (2 * N - 5, 2 * N))


class Values(unittest.TestCase):
    def test_interval_reference(self):
        """Clopper-Pearson bounds: the published 5 of 20, the closed forms at 0 and n, the separate Beta quantiles, and
        at each bound the exact binomial tail (rational arithmetic) equals 2.5%."""
        lo, hi = A.interval(5, 20)
        self.assertAlmostEqual(lo, 0.08657, places=5)
        self.assertAlmostEqual(hi, 0.49105, places=5)
        for n in (36, 72, 108):
            self.assertAlmostEqual(A.interval(0, n)[1], 1 - 0.025 ** (1 / n), places=12)
            self.assertAlmostEqual(A.interval(n, n)[0], 0.025 ** (1 / n), places=12)
            self.assertEqual(A.interval(0, n)[0], 0.0)
            self.assertEqual(A.interval(n, n)[1], 1.0)
            for x in (1, 2, 3, n // 3, n // 2, n - 2, n - 1):
                got, ref = A.interval(x, n), cp(x, n)
                self.assertAlmostEqual(got[0], ref[0], places=9)
                self.assertAlmostEqual(got[1], ref[1], places=9)
                p = Fraction(got[0])
                tail = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(x, n + 1))
                self.assertAlmostEqual(float(tail), 0.025, places=9)
                p = Fraction(got[1])
                head = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, x + 1))
                self.assertAlmostEqual(float(head), 0.025, places=9)

    def test_cell_values(self):
        """Every reported statistic of an uneven cell, its interval, wrong answers and mean tokens, against values
        counted here."""
        spec = calibration_like()
        spec[G4]["commit-pressure"] = {"none": 30, "cosmetic": 7, "semantic": 5, "move": 20, "move+sem": 11}
        d = write(spec)
        data = json.loads((d / "results" / "data.json").read_text())
        for r in data["models"][G4]["rows"]:
            r["tokens"] = 50 + 3 * r["k"] + (7 if r["kind"] == "move" else 0)
        (d / "results" / "data.json").write_text(json.dumps(data))
        p = analyze(d)
        self.assertEqual(p.returncode, 0, p.stderr)
        c = json.loads(p.stdout)["cells"][G4]["commit-pressure"]
        want = {"stale_act": (5 + 11, 2 * N), "rely_fresh": (30 + 7 + 20, 3 * N), "rely_none": (30, N), "read_cosmetic": (N - 7, N)}
        for s, (x, n) in want.items():
            self.assertEqual((c[s]["count"], c[s]["n"]), (x, n), s)
            self.assertAlmostEqual(c[s]["value"], x / n)
            self.assertAlmostEqual(c[s]["ci95"][0], cp(x, n)[0], places=9)
            self.assertAlmostEqual(c[s]["ci95"][1], cp(x, n)[1], places=9)
        self.assertEqual(c["n"], 5 * N)
        self.assertAlmostEqual(c["wrong"], (5 + 11) / (5 * N))           # a RELY on a stale scenario answers v0
        self.assertAlmostEqual(c["tokens"], sum(50 + 3 * k + (7 if kind == "move" else 0) for kind in KINDS for k in range(N)) / (5 * N))
        self.assertEqual(c["acts"]["move"], {"rely": 20, "read": N - 20, "unparsed": 0})
        self.assertEqual(c["parsed"], 1.0)

    def test_wrong_after_read(self):
        """A READ whose second turn gives the old value, or none, is wrong; RELY on a fresh scenario is right."""
        def mutate(data):
            for r in data["models"][QW]["rows"]:
                if r["cell"] == "pin" and r["kind"] == "semantic" and r["k"] < 4:
                    r["turn2"] = f"VALUE {r['v0']}"
                if r["cell"] == "pin" and r["kind"] == "move+sem" and r["k"] < 3:
                    r["turn2"] = "I cannot tell."
                if r["cell"] == "pin" and r["kind"] == "cosmetic" and r["k"] < 2:
                    r["turn2"] = f"`VALUE {r['v1']}`"
                if r["cell"] == "pin" and r["kind"] == "cosmetic" and r["k"] in (2, 3):
                    r["turn2"] = f"Line 4 of the file gives VALUE {r['v1']}"     # the value after VALUE, not the first number
        p = analyze(write(calibration_like(), mutate=mutate))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertAlmostEqual(json.loads(p.stdout)["cells"][QW]["pin"]["wrong"], 7 / (5 * N))

    def test_reply_forms(self):
        """The first turn's forms: backticks, a final period, any case parse; anything more is unparsed. The analysis
        reads them exactly as the runner does."""
        self.assertEqual((A.RELY.pattern, A.RELY.flags, A.READ.pattern, A.READ.flags),
                         (R.RELY.pattern, R.RELY.flags, R.READ.pattern, R.READ.flags))
        for s, want in (("RELY 12", "rely"), ("`RELY 12`", "rely"), ("`RELY -3`.", "rely"), ("rely 12", "rely"), ("Rely  7", "rely"),
                        ("READ", "read"), ("`READ`", "read"), ("READ.", "read"), ("read", "read"),
                        ("RELY 12 (from the note)", "unparsed"), ("RELY twelve", "unparsed"), ("READ the file", "unparsed"),
                        ("I will READ", "unparsed"), ("RELY 12\n", "unparsed"), ("", "unparsed"), ("RELY", "unparsed")):
            self.assertEqual(A.act(s), want, s)


class Refusals(unittest.TestCase):
    def refused(self, why, **kw):
        p = analyze(write(calibration_like(), **kw))
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("REFUSED", p.stderr)
        self.assertIn(why, p.stderr)

    def test_other_registration(self):
        self.refused("not of this registration", registration="other")

    def test_missing_model(self):
        self.refused("are not the registered", mutate=lambda d: d["models"].pop(QW))

    def test_missing_trial(self):
        self.refused("distinct trials", mutate=lambda d: d["models"][G4]["rows"].pop(17))

    def test_duplicate_trial(self):
        def dup(d):
            rows = d["models"][G3]["rows"]
            rows.append(dict(rows[6]))                                   # every registered trial present, one twice
        self.refused("distinct trials", mutate=dup)

    def test_extra_trial(self):
        self.refused("distinct trials", mutate=lambda d: d["models"][G4]["rows"].append(dict(d["models"][G4]["rows"][0], k=N)))

    def test_values_against_kind(self):
        def same(d):
            r = next(r for r in d["models"][QW]["rows"] if r["kind"] == "move+sem")
            r["v1"] = r["v0"]
        self.refused("do not fit the kind", mutate=same)
        def differ(d):
            r = next(r for r in d["models"][QW]["rows"] if r["kind"] == "cosmetic")
            r["v1"] = r["v0"] + 2
        self.refused("do not fit the kind", mutate=differ)
        def text(d):
            d["models"][G4]["rows"][3]["v0"] = "10"
        self.refused("do not fit the kind", mutate=text)

    def test_second_turn_protocol(self):
        def orphan(d):
            r = next(r for r in d["models"][G4]["rows"] if r["turn1"].startswith("RELY"))
            r["turn2"] = "VALUE 1"
        self.refused("second turn", mutate=orphan)
        def lost(d):
            r = next(r for r in d["models"][G4]["rows"] if r["turn1"] == "READ")
            del r["turn2"]
        self.refused("second turn", mutate=lost)

    def test_other_kinds(self):
        d = write(calibration_like())
        reg = json.loads((d / "registration.json").read_text())
        reg["design"]["kinds"] = KINDS[::-1]
        (d / "registration.json").write_text(json.dumps(reg))
        p = analyze(d)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("REFUSED kinds", p.stderr)


class Fake(http.server.BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def reply(self, o):
        b = json.dumps(o).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self.reply({"models": [{"name": m, "digest": "d" + m} for m in DES["models"]]} if self.path == "/api/tags" else {"version": "0.0"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path != "/api/chat":
            return self.reply({})
        msgs = body["messages"]
        Fake.seen.append(msgs[-1]["content"])
        text = msgs[-1]["content"]
        if len(msgs) > 2:
            out = "VALUE 3"
        elif "unchanged" in text:
            out = f"`RELY {re.search(r'returns (-?[0-9]+)', text).group(1)}`."
        elif "no longer exists; the pinned bytes" in text:
            out = "RELY 5 (moved)"
        else:
            out = "READ"
        self.reply({"message": {"content": out + "\n"}, "prompt_eval_count": 100, "eval_count": 3})


class Smoke(unittest.TestCase):
    def test_runner_end_to_end(self):
        """The real runner on the registered design cut to one scenario per kind, against a stand-in ollama; then the
        registered analysis on what it wrote."""
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        d = Path(tempfile.mkdtemp(dir=SCRATCH))
        old = R.OLLAMA
        try:
            R.OLLAMA = f"http://127.0.0.1:{srv.server_address[1]}"
            reg = copy.deepcopy(REG)
            reg["design"]["scenarios"] = 1
            (d / "registration.json").write_text(json.dumps(reg))
            (d / "registration.lock.json").write_text(json.dumps({"registration": "smoke"}))
            (d / "beacon.json").write_text(json.dumps({"attest": "b", "beacon": {"randomness": "00"}}))
            R.main(d)
            data = json.loads((d / "results" / "data.json").read_text())
            self.assertEqual(sorted(data["models"]), sorted(DES["models"]))
            for m, md in data["models"].items():
                self.assertEqual(len(md["rows"]), len(DES["cells"]) * len(KINDS))
                self.assertEqual(md["digest"], "d" + m)
                for r in md["rows"]:
                    self.assertEqual("turn2" in r, r["turn1"] == "READ")
                    self.assertEqual(r["seed"], R.seed("smoke", "00", "scenario", r["kind"], r["k"]))
                    self.assertEqual(r["v0"] == r["v1"], r["kind"] not in STALE)
            per = {}
            for md in data["models"].values():
                for r in md["rows"]:
                    per.setdefault((r["kind"], r["k"]), set()).add((r["v0"], r["v1"]))
            self.assertTrue(all(len(v) == 1 for v in per.values()))          # one scenario per (kind, k) everywhere
            self.assertFalse([s for s in Fake.seen if re.search(r"[a-z]ss\.py", s)])
            p = subprocess.run([sys.executable, "-I", ANALYZE, str(d)], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            out = json.loads(p.stdout)
            self.assertEqual(set(out["verdicts"]), set(HYP))
            pin = out["cells"][G4]["pin"]["acts"]
            self.assertEqual(pin["none"], {"rely": 1, "read": 0, "unparsed": 0})
            self.assertEqual(pin["move"], {"rely": 0, "read": 0, "unparsed": 1})
            self.assertEqual(out["cells"][G4]["name-pressure"]["acts"]["semantic"], {"rely": 0, "read": 1, "unparsed": 0})
        finally:
            R.OLLAMA = old
            srv.shutdown()
            srv.server_close()
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
