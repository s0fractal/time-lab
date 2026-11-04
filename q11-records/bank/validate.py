#!/usr/bin/env python3
"""Validate a question bank against two code trees. Each item:
{"id": str, "class": "changed"|"unchanged", "question": str, "options": [four distinct str], "key": str, "old": str|null, "value": python expression}
Trees: `git archive 6d6bb700` into A/ and `git archive f61f5567` into H/ of warrant, paths warrant, warrant_agent, tools, integrations,
keeping *.py, *.toml, *.json without test files. f61f5567 and 4167ce77 hold the same code.
`value` is evaluated with read(path) -> text ('' when absent), exists(path) -> bool, re, json; on tree H it must equal `key`;
on tree A it must equal `old` (changed: another option) or `key` (unchanged, where old is null)."""
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
def ev(expr, tree):
    root = HERE / tree
    def read(p):
        f = root / p
        return f.read_text() if f.is_file() else ""
    return eval(expr, {"read": read, "exists": lambda p: (root / p).exists(), "re": re, "json": json})
bad, seen, n = [], set(), {"changed": 0, "unchanged": 0}
for it in json.load(open(sys.argv[1] if len(sys.argv) > 1 else HERE / "bank.json")):
    i = it.get("id")
    try:
        assert set(it) == {"id", "class", "question", "options", "key", "old", "value"}, "fields"
        assert i not in seen, "duplicate id"
        seen.add(i)
        assert it["class"] in n, "class"
        assert len(it["options"]) == 4 and len(set(it["options"])) == 4, "options"
        assert it["key"] in it["options"], "key not an option"
        if it["class"] == "changed":
            assert it["old"] in it["options"] and it["old"] != it["key"], "old"
        else:
            assert it["old"] is None, "old must be null"
        h, a = ev(it["value"], "H"), ev(it["value"], "A")
        assert h == it["key"], f"H gives {h!r}"
        assert a == (it["old"] if it["class"] == "changed" else it["key"]), f"A gives {a!r}"
        n[it["class"]] += 1
    except Exception as e:
        bad.append((i, f"{type(e).__name__}: {e}"))
for b in bad:
    print("BAD", *b)
print(n, "valid;", len(bad), "bad")
sys.exit(1 if bad else 0)
