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


if __name__ == "__main__":
    unittest.main()
