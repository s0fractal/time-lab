"""What tl.py checks that the core does not (METHOD §4): the closing guard names the registered analysis, and a
withdrawal stands only where its author's chain puts it before the beacon. TL names another tl.py (the mutants)."""
import importlib.util, os, unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("tl", os.environ.get("TL") or Path(__file__).resolve().parents[1] / "tl.py")
TL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TL)
D = "d" * 64


def rec(actor, prev, kind="entity", schema=None, body=None, guard=None):
    return {"actor": actor, "prev": prev, "kind": kind, "schema": schema, "body": body or {}, "guard": guard}


def store(*rows):
    """{id: record}: rows (id, record) in any order."""
    return dict(rows)


REG = ("reg", rec("claude", None, schema="registration@1", body={"analysis": D}))
OPEN = ("p1", rec("claude", "reg", schema="prediction@1", body={"registration": "reg", "hypothesis": "H1", "state": "open"}))


def closing(code, prev="p1", ident="p2"):
    return (ident, rec("claude", prev, schema="prediction@1", guard={"match": {"verdict": "reproduces", "code": code}},
                       body={"registration": "reg", "hypothesis": "H1", "state": "supported"}))


def beacon(prev, ident="b1"):
    return (ident, rec("claude", prev, kind="attest", body={"verdict": "witnessed", "target": "reg"}))


def withdrawn(prev, actor="claude", ident="w1"):
    return (ident, rec(actor, prev, schema="prediction@1", body={"registration": "reg", "hypothesis": "H1", "state": "withdrawn"}))


def codes(rs):
    return [c for c, _, _ in TL.unchecked(rs, {})]


class Unchecked(unittest.TestCase):
    def test_closing_names_the_registered_analysis(self):
        self.assertEqual(codes(store(REG, OPEN, closing(D))), [])

    def test_closing_names_another_code(self):
        self.assertEqual(codes(store(REG, OPEN, closing("e" * 64))), ["CODE_NOT_REGISTERED"])

    def test_closing_names_no_code(self):
        self.assertEqual(codes(store(REG, OPEN, closing(None))), ["CODE_NOT_REGISTERED"])

    def test_withdrawn_before_the_beacon_stands(self):
        self.assertEqual(codes(store(REG, OPEN, withdrawn("p1"), beacon("w1"))), [])

    def test_withdrawn_after_the_beacon(self):
        self.assertEqual(codes(store(REG, OPEN, beacon("p1"), withdrawn("b1"))), ["WITHDRAWN_AFTER_BEACON"])

    def test_withdrawn_by_an_actor_the_beacon_chain_does_not_order(self):
        self.assertEqual(codes(store(REG, OPEN, beacon("p1"), withdrawn(None, actor="codex"))), ["WITHDRAWN_AFTER_BEACON"])

    def test_withdrawn_with_no_beacon_yet(self):
        self.assertEqual(codes(store(REG, OPEN, withdrawn("p1"))), [])


class Same(unittest.TestCase):
    """A closing version stores micro(result) (Warrant has no floats); the decider compares it with its raw rerun."""
    RAW = {"verdict": "supported", "ci": [0.61, 0.83], "iota": 0.7123456789, "n": 3, "ok": True}
    STORED = TL.micro(RAW)

    def test_the_stored_form(self):
        self.assertEqual(self.STORED, {"verdict": "supported", "ci": [610000, 830000], "iota": 712346, "n": 3, "ok": True})

    def test_the_rerun_itself(self):
        self.assertTrue(TL.same(self.STORED, self.RAW))

    def test_the_last_bits_of_a_float_across_a_half(self):
        stored = TL.micro({"x": 0.0000025})                       # 2.5 units: round may go either way elsewhere
        self.assertTrue(TL.same({"x": stored["x"] + 1}, {"x": 0.0000025}) or TL.same({"x": stored["x"]}, {"x": 0.0000025}))
        self.assertTrue(TL.same(self.STORED, dict(self.RAW, iota=0.7123456789 * (1 + 1e-14))))

    def test_a_number_that_moved(self):
        self.assertFalse(TL.same(self.STORED, dict(self.RAW, iota=0.7123486789)))

    def test_an_integer_off_by_one(self):
        self.assertFalse(TL.same(self.STORED, dict(self.RAW, n=4)))

    def test_another_verdict(self):
        self.assertFalse(TL.same(self.STORED, dict(self.RAW, verdict="inconclusive")))

    def test_another_shape(self):
        self.assertFalse(TL.same(self.STORED, dict(self.RAW, extra=1)))
        self.assertFalse(TL.same(self.STORED, dict(self.RAW, ci=[0.61])))
        self.assertFalse(TL.same(dict(self.STORED, extra=1), self.RAW))

    def test_a_flag_is_not_a_number(self):
        self.assertFalse(TL.same(dict(self.STORED, ok=1), self.RAW))

    def test_raw_floats_are_not_stored(self):
        self.assertFalse(TL.same(self.RAW, self.RAW))
        self.assertFalse(TL.same(dict(self.STORED, iota=712346.0), self.RAW))   # micro stores ints only


if __name__ == "__main__":
    unittest.main()
