"""Controls for q11-records/analyze.py on synthetic answers set in advance (each component pushed to both sides of its rule and
next to them), refusals, the runner's retrieval against values computed by hand, and an end-to-end run of the real runner against
a stand-in ollama.

    python -m unittest q11-records/tests/test_analyze.py      (ANALYZE=path runs another copy, for mutants)"""
import hashlib, http.server, importlib.util, json, math, os, shutil, subprocess, sys, tempfile, threading, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ANALYZE = os.environ.get("ANALYZE", str(HERE / "analyze.py"))
SCRATCH = HERE / "runs"
MODELS = ["m1", "m2", "m3"]
ARMS = ["records", "prose", "log", "none"]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


A = module("q11analyze", ANALYZE)
R = module("q11run", HERE / "run.py")


def bank(n=60):
    out = []
    for i in range(n):
        out.append({"id": f"c{i:03}", "class": "changed", "question": f"q{i}", "options": [f"key{i}", f"old{i}", f"w{i}a", f"w{i}b"],
                    "key": f"key{i}", "old": f"old{i}", "value": "''"})
        out.append({"id": f"u{i:03}", "class": "unchanged", "question": f"q{i}", "options": [f"key{i}", f"w{i}c", f"w{i}a", f"w{i}b"],
                    "key": f"key{i}", "old": None, "value": "''"})
    return out


def world(pick, n=60):
    """{model: rows} where pick(model, arm, item, index in its class) is key, old, wrong, E or junk."""
    rows = {}
    for m in MODELS:
        rows[m] = []
        for it in bank(n):
            order = sorted(it["options"])
            for arm in ARMS:
                p = pick(m, arm, it, int(it["id"][1:]))
                opt = {"key": it["key"], "old": it["old"], "wrong": next(o for o in order if o not in (it["key"], it["old"]))}.get(p)
                ans = {"E": "ANSWER E", "junk": "I think so"}.get(p) or f"ANSWER {'ABCD'[order.index(opt)]}"
                rows[m].append({"arm": arm, "q": it["id"], "order": order, "answer": ans, "tokens": 10, "chunks": []})
    return rows


def run(pick, n=60):
    b = {i["id"]: i for i in bank(n)}
    return A.verdicts_of(b, world(pick, n))


def comps(v, h):
    return [c["verdict"] for c in v[h]["components"]]


def share(i, *cuts):
    """The first of `cuts` (lo, hi, value) holding i, else None: a fixed pattern over a class's questions."""
    return next((val for lo, hi, val in cuts if lo <= i < hi), None)


class Parse(unittest.TestCase):
    def test_forms(self):
        for s, want in [("ANSWER A", "A"), ("answer b", "B"), ("`ANSWER C`", "C"), ("ANSWER: D", "D"), ("**E**", "E"), ("C.", "C"), ("a", "A"),
                        ("e.g.\n\nANSWER E", "E"), ("ANSWER A\nANSWER B\n", "B"), ("  ANSWER C  ", "C"), ("ANSWER A\n\n", "A"), ("ANSWER D\n   \n", "D"),
                        ("E. the passages do not settle it", "E"), ("the passages do not settle it", "E"), ("The passages do not settle it.", "E"),
                        ("B. It drops the version", "B"), ("Thinking.\nANSWER: B. It drops the version", "B")]:
            self.assertEqual(A.letter(s), want, s)

    def test_refused_forms(self):
        for s in ["ANSWER F", "ANSWER A or B", "The answer is A", "A because", "", "\n\n", "ANSWER A\nThe passages state", "e.g.", "A.B", "the passages do not settle", "the passages do not settle it\nHmm"]:
            self.assertIsNone(A.letter(s), s)

    def test_choice_reads_the_order_shown(self):
        r = {"arm": "prose", "answer": "ANSWER C", "order": ["w", "x", "y", "z"]}
        self.assertEqual(A.choice(r), "y")

    def test_e_only_where_offered(self):
        self.assertEqual(A.choice({"arm": "prose", "answer": "ANSWER E", "order": list("wxyz")}), "E")
        self.assertIsNone(A.choice({"arm": "none", "answer": "ANSWER E", "order": list("wxyz")}))


class H1(unittest.TestCase):
    def test_records_alone_correct_supported(self):
        v = run(lambda m, a, it, i: "key" if a == "records" else "wrong")
        self.assertEqual(comps(v, "H1"), ["supported", "supported"])
        self.assertEqual(v["H1"]["verdict"], "supported")
        self.assertEqual(v["H1"]["components"][0]["value"], 1.0)

    def test_equal_contradicted(self):
        v = run(lambda m, a, it, i: "key")
        self.assertEqual(v["H1"]["verdict"], "contradicted")

    def test_prose_ahead_contradicted_though_log_behind(self):
        v = run(lambda m, a, it, i: {"records": "key" if i < 30 else "wrong", "prose": "key", "log": "wrong", "none": "wrong"}[a])
        self.assertEqual(comps(v, "H1"), ["contradicted", "supported"])
        self.assertEqual(v["H1"]["verdict"], "contradicted")

    def test_wide_interval_inconclusive(self):
        def p(m, a, it, i):
            if a == "records":
                return "key" if i < 12 else "wrong"
            return "key" if 12 <= i < 21 else "wrong"
        v = run(p)
        self.assertEqual(comps(v, "H1"), ["inconclusive", "inconclusive"])
        self.assertEqual(v["H1"]["verdict"], "inconclusive")

    def test_one_question_is_no_lead(self):
        # lead +1 on one question of 60: a resample misses it, so lo = 0, and that is not above 0
        v = run(lambda m, a, it, i: "key" if a == "records" and i == 0 else "wrong")
        self.assertEqual(comps(v, "H1"), ["contradicted", "contradicted"])

    def test_small_lead_below_delta_contradicted(self):
        # lead 3 of 60 (0.05): a resample can miss all three, so lo = 0, and hi is under 0.15
        def p(m, a, it, i):
            return "key" if a == "records" and i < 3 else "wrong"
        c = run(p)["H1"]["components"][0]
        self.assertTrue(c["ci95"][0] == 0.0 and c["ci95"][1] < 0.15, c)
        self.assertEqual(c["verdict"], "contradicted")

    def test_delta_is_what_separates(self):
        # lead 0.10 with hi between 0.05 and 0.15: contradicted under DELTA 0.15, not under a smaller one; lo stays at or under 0
        def p(m, a, it, i):
            if a == "records":
                return "key" if i < 6 else "wrong"
            return "key" if 6 <= i < 7 else "wrong"
        c = run(p, 200)["H1"]["components"][0]
        self.assertTrue(0.02 < c["value"] < 0.03 and c["ci95"][1] < 0.15, c)
        self.assertEqual(A.DELTA, 0.15)

    def test_mean_over_models(self):
        # one model of three right: a third of a question each
        v = run(lambda m, a, it, i: "key" if a == "records" and m == "m1" else "wrong")
        self.assertAlmostEqual(v["H1"]["components"][0]["value"], 1 / 3)

    def test_classes_are_separate(self):
        v = run(lambda m, a, it, i: "key" if (a == "records" and it["class"] == "unchanged") else "wrong")
        self.assertEqual(v["H1"]["components"][0]["value"], 0.0)


class H2(unittest.TestCase):
    def test_tie_above_floor_supported(self):
        v = run(lambda m, a, it, i: "wrong" if a == "none" else "key")
        self.assertEqual(v["H2"]["verdict"], "supported")

    def test_records_behind_supported(self):
        v = run(lambda m, a, it, i: {"records": "wrong", "none": "wrong"}.get(a, "key"))
        self.assertEqual(v["H2"]["verdict"], "supported")

    def test_records_far_ahead_contradicted(self):
        v = run(lambda m, a, it, i: "key" if a == "records" else "wrong")
        self.assertEqual(comps(v, "H2"), ["contradicted", "contradicted"])

    def test_near_delta_inconclusive(self):
        def p(m, a, it, i):
            if a == "records":
                return "key" if i < 15 else "wrong"
            return "wrong" if a == "none" else "key" if 15 <= i < 21 else "wrong"
        c = run(p)["H2"]["components"][0]
        self.assertTrue(c["ci95"][0] <= 0.15 <= c["ci95"][1], c)
        self.assertEqual(c["verdict"], "inconclusive")

    def test_floor_tests_nothing(self):
        v = run(lambda m, a, it, i: "wrong")
        self.assertEqual(comps(v, "H2"), ["inconclusive", "inconclusive"])
        self.assertEqual(v["H2"]["components"][0]["reason"], "neither arm is above the floor")

    def test_one_arm_above_floor_is_enough(self):
        c = run(lambda m, a, it, i: "key" if a == "prose" else "wrong")["H2"]["components"][0]
        self.assertEqual(c["verdict"], "supported")
        self.assertEqual(c["above_floor"], {"records": False, "prose": True})

    def test_records_alone_above_floor_is_enough(self):
        # 30% on records, none elsewhere: lead 0.3, so hi over DELTA and lo over it too; what matters is no floor reason
        c = run(lambda m, a, it, i: ("key" if i < 18 else "wrong") if a == "records" else "wrong")["H2"]["components"][0]
        self.assertEqual(c["above_floor"], {"records": True, "prose": False})
        self.assertNotIn("reason", c)

    def test_floor_value(self):
        # 15% correct in both arms at n = 60: lo near 0.07, not above FLOOR 0.10; 30%: lo near 0.19, above it
        low = run(lambda m, a, it, i: "key" if a != "none" and i < 9 else "wrong")["H2"]["components"][0]
        self.assertEqual(low["verdict"], "inconclusive")
        high = run(lambda m, a, it, i: "key" if a != "none" and i < 18 else "wrong")["H2"]["components"][0]
        self.assertEqual(high["verdict"], "supported")

    def test_reads_unchanged_only(self):
        # a tie on unchanged while records lead on changed: H2 does not read changed
        v = run(lambda m, a, it, i: "wrong" if a == "none" else "key" if it["class"] == "unchanged" or a == "records" else "wrong")
        self.assertEqual(v["H2"]["verdict"], "supported")
        self.assertEqual(v["H2"]["components"][0]["value"], 0.0)

    def test_floor_is_of_the_class_read(self):
        # the arms are right on changed and at nothing on unchanged: still the floor
        v = run(lambda m, a, it, i: "key" if it["class"] == "changed" and a != "none" else "wrong")
        self.assertEqual(comps(v, "H2"), ["inconclusive", "inconclusive"])

    def test_floor_does_not_subtract_the_control(self):
        # arms and control all at 30%: the arms are above the floor although they do not beat the guess
        c = run(lambda m, a, it, i: "key" if i < 18 else "wrong")["H2"]["components"][0]
        self.assertEqual(c["verdict"], "supported")
        self.assertEqual(c["above_floor"], {"records": True, "prose": True})

    def test_control_does_not_lift_the_floor(self):
        v = run(lambda m, a, it, i: "key" if a == "none" else "wrong")
        self.assertEqual(comps(v, "H2"), ["inconclusive", "inconclusive"])


class H3(unittest.TestCase):
    def test_ahead_only_on_changed_supported(self):
        v = run(lambda m, a, it, i: "key" if a == "records" and it["class"] == "changed" else ("key" if it["class"] == "unchanged" else "wrong"))
        self.assertEqual(v["H3"]["verdict"], "supported")
        self.assertEqual(v["H3"]["components"][0]["value"], 1.0)

    def test_same_lead_in_both_contradicted(self):
        v = run(lambda m, a, it, i: "key" if a == "records" else "wrong")
        self.assertEqual(v["H3"]["verdict"], "contradicted")
        self.assertEqual(v["H3"]["components"][0]["value"], 0.0)

    def test_sign(self):
        v = run(lambda m, a, it, i: "key" if a == "records" and it["class"] == "unchanged" else "wrong")
        self.assertEqual(v["H3"]["components"][0]["value"], -1.0)
        self.assertEqual(v["H3"]["verdict"], "contradicted")

    def test_resamples_each_class_alone(self):
        def p(m, a, it, i):
            if it["class"] == "changed":
                return "key" if a == "records" else "wrong"
            return "key" if a != "records" else "wrong"
        c = run(p)["H3"]["components"][0]
        self.assertEqual(c["value"], 2.0)
        self.assertEqual(c["ci95"], [2.0, 2.0])     # pooled resampling would mix classes and widen this


class H4(unittest.TestCase):
    def test_prose_old_records_key_supported(self):
        v = run(lambda m, a, it, i: "key" if a == "records" else "old" if it["class"] == "changed" else "wrong")
        self.assertEqual(v["H4"]["verdict"], "supported")
        self.assertEqual(v["H4"]["components"][0]["value"], 1.0)

    def test_records_old_contradicted(self):
        v = run(lambda m, a, it, i: "old" if a == "records" and it["class"] == "changed" else "wrong")
        self.assertEqual(v["H4"]["verdict"], "contradicted")
        self.assertEqual(v["H4"]["components"][0]["value"], -1.0)

    def test_wrong_is_not_old(self):
        v = run(lambda m, a, it, i: "key" if a == "records" else "wrong")
        self.assertEqual(v["H4"]["components"][0]["value"], 0.0)
        self.assertEqual(v["H4"]["verdict"], "contradicted")


class H5(unittest.TestCase):
    def test_guess_rate_supported(self):
        v = run(lambda m, a, it, i: "key" if a == "none" and i % 4 == 0 else "wrong")
        self.assertEqual(v["H5"]["verdict"], "supported")
        self.assertAlmostEqual(v["H5"]["components"][0]["value"], 0.25)

    def test_give_away_contradicted(self):
        v = run(lambda m, a, it, i: "key" if a == "none" else "wrong")
        self.assertEqual(v["H5"]["verdict"], "contradicted")

    def test_near_leak_inconclusive(self):
        v = run(lambda m, a, it, i: "key" if a == "none" and i < 24 else "wrong")
        self.assertEqual(comps(v, "H5"), ["inconclusive", "inconclusive"])

    def test_leak_threshold_itself(self):
        self.assertEqual(A.LEAK, 0.40)


class Gate(unittest.TestCase):
    def junk(self, arm, count):
        return lambda m, a, it, i: "junk" if a == arm and m == "m2" and it["class"] == "changed" and i < count else ("key" if a == "records" else "wrong")

    def test_below_gate_inconclusive(self):
        # 7 of 120 in one model's prose: 0.9417 < 0.95
        v = run(self.junk("prose", 7))
        self.assertEqual(comps(v, "H1"), ["inconclusive", "supported"])
        self.assertIn("gate", v["H1"]["components"][0]["reason"])
        self.assertEqual(v["H1"]["verdict"], "inconclusive")

    def test_at_gate_counts(self):
        # 6 of 120: exactly 0.95, not below
        v = run(self.junk("prose", 6))
        self.assertEqual(comps(v, "H1"), ["supported", "supported"])

    def test_gate_reads_records_too(self):
        v = run(lambda m, a, it, i: "junk" if a == "records" and m == "m3" and i < 7 and it["class"] == "changed" else ("key" if a == "records" else "wrong"))
        self.assertEqual(comps(v, "H1"), ["inconclusive", "inconclusive"])

    def test_gate_is_per_model(self):
        # the same 7 spread over three models: each above the gate
        def p(m, a, it, i):
            if a == "prose" and it["class"] == "changed" and i < {"m1": 3, "m2": 2, "m3": 2}[m]:
                return "junk"
            return "key" if a == "records" else "wrong"
        self.assertEqual(comps(run(p), "H1"), ["supported", "supported"])

    def test_control_gate(self):
        v = run(lambda m, a, it, i: "junk" if a == "none" and m == "m1" and i < 7 and it["class"] == "changed" else ("key" if a == "none" and i % 4 == 0 else "wrong"))
        self.assertIn("gate", v["H5"]["components"][0]["reason"])

    def test_control_gate_reads_only_the_control(self):
        v = run(lambda m, a, it, i: "junk" if a == "records" and m == "m1" and i < 7 and it["class"] == "changed" else ("key" if a == "none" and i % 4 == 0 else "wrong"))
        self.assertEqual(comps(v, "H5"), ["supported", "supported"])

    def test_e_counts_as_parsed_and_not_correct(self):
        v = run(lambda m, a, it, i: "E" if a == "prose" else "key" if a == "records" else "wrong")
        self.assertEqual(comps(v, "H1"), ["supported", "supported"])
        self.assertEqual(v["H1"]["components"][0]["value"], 1.0)


class Interval(unittest.TestCase):
    def test_percentile_against_a_binomial_reference(self):
        # the mean of 60 draws of a 0.5 coin: sd sqrt(.25/60); a 95% interval near ±1.96 sd
        xs = [1.0] * 30 + [0.0] * 30
        lo, hi = A.bootstrap("t", [xs], lambda g: A.mean(g[0]))
        sd = math.sqrt(0.25 / 60)
        self.assertAlmostEqual(lo, 0.5 - 1.96 * sd, delta=0.02)
        self.assertAlmostEqual(hi, 0.5 + 1.96 * sd, delta=0.02)

    def test_fixed_by_name(self):
        xs = [1.0] * 30 + [0.0] * 30
        f = lambda g: A.mean(g[0])
        self.assertEqual(A.bootstrap("t", [xs], f), A.bootstrap("t", [xs], f))
        self.assertNotEqual(A.bootstrap("t", [xs], f), A.bootstrap("u", [xs], f))

    def test_resamples_and_alpha(self):
        self.assertEqual((A.RESAMPLES, A.ALPHA, A.GATE), (20000, 0.05, 0.95))


class Describe(unittest.TestCase):
    def test_counts(self):
        b = {i["id"]: i for i in bank(4)}
        d = A.describe(b, world(lambda m, a, it, i: {0: "key", 1: "old", 2: "E", 3: "E"}[i] if it["class"] == "changed" else "junk" if i == 0 else "wrong", 4))
        self.assertEqual(d["m1"]["prose"]["changed"], {"n": 4, "correct": 1, "old": 1, "not_settled": 2, "unparsed": 0, "tokens": 10.0})
        self.assertEqual(d["m1"]["prose"]["unchanged"], {"n": 4, "correct": 0, "old": 0, "not_settled": 0, "unparsed": 1, "tokens": 10.0})


class Refusals(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp(dir=SCRATCH))
        (self.d / "results").mkdir()
        raw = json.dumps(bank(2)).encode()
        (self.d / "bank.json").write_bytes(raw)
        self.reg = {"design": {"arms": ARMS, "models": MODELS, "bank_sha256": hashlib.sha256(raw).hexdigest()}}
        self.data = {"registration": "r", "models": {m: {"rows": rs} for m, rs in world(lambda *x: "wrong", 2).items()}}

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def run_it(self):
        (self.d / "registration.json").write_text(json.dumps(self.reg))
        (self.d / "registration.lock.json").write_text(json.dumps({"registration": "r"}))
        (self.d / "results" / "data.json").write_text(json.dumps(self.data))
        return subprocess.run([sys.executable, "-I", ANALYZE, str(self.d)], capture_output=True, text=True)

    def refused(self, word):
        p = self.run_it()
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("REFUSED", p.stderr)
        self.assertIn(word, p.stderr)

    def test_good(self):
        p = self.run_it()
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(set(json.loads(p.stdout)["verdicts"]), {"H1", "H2", "H3", "H4", "H5"})

    def test_other_registration(self):
        self.data["registration"] = "x"
        self.refused("registration")

    def test_other_bank(self):
        self.reg["design"]["bank_sha256"] = "0" * 64
        self.refused("bank")

    def test_other_arms(self):
        self.reg["design"]["arms"] = ["prose", "records", "log", "none"]
        self.refused("arms")

    def test_missing_model(self):
        del self.data["models"]["m3"]
        self.refused("models")

    def test_missing_row(self):
        self.data["models"]["m1"]["rows"].pop()
        self.refused("distinct")

    def test_duplicate_row(self):
        rs = self.data["models"]["m1"]["rows"]
        rs.append(rs[0])
        self.refused("distinct")

    def test_other_options(self):
        self.data["models"]["m1"]["rows"][0]["order"][0] = "planted"
        self.refused("options")


class Retrieval(unittest.TestCase):
    def setUp(self):
        self.chunks = [{"id": "a", "text": "alpha beta"}, {"id": "b", "text": "alpha alpha gamma delta"}, {"id": "c", "text": "gamma " * 10}]
        self.ix = R.Index(self.chunks, 1.5, 0.75)

    def test_bm25_by_hand(self):
        # alpha in 2 of 3 chunks; lengths 2, 4, 10, mean 16/3
        idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
        s = lambda f, n: idf * f * 2.5 / (f + 1.5 * (1 - 0.75 + 0.75 * n / (16 / 3)))
        self.assertGreater(s(2, 4), s(1, 2))
        self.assertEqual(self.ix.rank("alpha"), [1, 0])

    def test_budget_skips_what_does_not_fit(self):
        got = self.ix.passages("gamma alpha", 40)        # the 60-character chunk ranks in and does not fit; the 10 after it does
        self.assertLessEqual(sum(len(c["text"]) for c in got), 40)
        self.assertEqual({c["id"] for c in got}, {"a", "b"})

    def test_ties_by_id(self):
        ix = R.Index([{"id": "z", "text": "w"}, {"id": "y", "text": "w"}], 1.5, 0.75)
        self.assertEqual([ix.chunks[i]["id"] for i in ix.rank("w")], ["y", "z"])

    def test_order_is_per_question_not_per_arm(self):
        it = bank(1)[0]
        self.assertEqual(R.order_of(it, "n", "r"), R.order_of(it, "n", "r"))
        self.assertEqual(sorted(R.order_of(it, "n", "r")), sorted(it["options"]))

    def test_control_prompt_has_no_passages_and_no_e(self):
        it = bank(1)[0]
        p = R.prompt(it, it["options"], None)
        self.assertNotIn("Passages", p)
        self.assertNotIn("E. ", p)
        self.assertIn("E. ", R.prompt(it, it["options"], []))


class EndToEnd(unittest.TestCase):
    """The real runner against a stand-in ollama that answers A; the real analysis on what it writes."""

    def test_runner_then_analysis(self):
        class Fake(http.server.BaseHTTPRequestHandler):
            calls = []

            def log_message(self, *a):
                pass

            def reply(self, o):
                b = json.dumps(o).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                self.reply({"models": [{"name": m, "digest": "d"} for m in MODELS], "version": "0"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                Fake.calls.append(body)
                self.reply({"message": {"content": "ANSWER A"}, "prompt_eval_count": 3, "eval_count": 1})

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        d = Path(tempfile.mkdtemp(dir=SCRATCH))
        try:
            R.OLLAMA = f"http://127.0.0.1:{srv.server_port}"
            (d / "runs").mkdir()
            (d / "results").mkdir()
            raw, corpus = json.dumps(bank(2)).encode(), json.dumps({a: [{"id": "x", "text": "key0 q0"}] for a in ARMS[:3]}).encode()
            (d / "bank.json").write_bytes(raw)
            (d / "runs" / "corpus.json").write_bytes(corpus)
            reg = {"design": {"arms": ARMS, "models": MODELS, "bank_sha256": hashlib.sha256(raw).hexdigest(), "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
                              "bm25": {"k1": 1.5, "b": 0.75}, "budget": 100, "options": {"temperature": 0}}}
            (d / "registration.json").write_text(json.dumps(reg))
            (d / "registration.lock.json").write_text(json.dumps({"registration": "r"}))
            (d / "beacon.json").write_text(json.dumps({"attest": "a", "beacon": {"randomness": "ff"}}))
            R.main(d)
            chats = [c for c in Fake.calls if "messages" in c]
            self.assertEqual(len(chats), 3 * 4 * 4)
            self.assertEqual(sum("Passages" in c["messages"][1]["content"] for c in chats), 3 * 4 * 3)
            n = len(Fake.calls)
            R.main(d)                                                  # resumes: no trial runs twice
            self.assertEqual(sum("messages" in c for c in Fake.calls[n:]), 0)
            p = subprocess.run([sys.executable, "-I", ANALYZE, str(d)], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("H5", json.loads(p.stdout)["verdicts"])
            bad = json.loads((d / "registration.json").read_text())
            bad["design"]["corpus_sha256"] = "0" * 64
            (d / "registration.json").write_text(json.dumps(bad))
            with self.assertRaises(SystemExit):
                R.main(d)
        finally:
            srv.shutdown()
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
