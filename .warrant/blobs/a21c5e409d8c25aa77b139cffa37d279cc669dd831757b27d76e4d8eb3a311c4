"""q11-records registered analysis: does a fresh model recover more of warrant from its records than from its prose, where what
they speak of changed?

    python -I q11-records/analyze.py DIR        prints {verdicts, arms} as JSON; refuses data that are not the design

An answer is its last non-empty line, matched whole: `ANSWER <letter>`, the letter alone, `<letter>. <its option>`, or E's
sentence. It is correct when its option is the bank's key, old when
it is the bank's `old` (changed questions only). A question's score in an arm is the mean over the models, so the question is
the unit, and arms are compared on the same questions. An interval is the 95% percentile bootstrap over questions (20 000
resamples, a seed fixed by the statistic's name; a difference between classes resamples each class alone).

Components:
    ahead        records minus another arm: supported when lo > 0, contradicted when hi < DELTA
    within       records minus another arm: supported when hi < DELTA, contradicted when lo > DELTA; inconclusive unless a
                 compared arm's correct share has lo > FLOOR: two arms at the floor test nothing (an arm with text may say E,
                 so its share has no guessing in it to speak of, and the control's guess is no floor for it)
    interaction  `ahead` on changed minus on unchanged: supported when lo > 0, contradicted when hi < DELTA
    fewer_old    another arm's old answers minus records' on changed: supported when lo > 0, contradicted when hi < DELTA
    no_leak      the control's correct share: supported when hi < LEAK, contradicted when lo > LEAK
A hypothesis is contradicted if any component is, supported if all are, else inconclusive. Inconclusive by rule: fewer than
95% of the answers of a compared arm parse in some model. Its bytes are the registration's `analysis` blob."""
import hashlib, json, random, re, sys
from pathlib import Path

ALPHA = 0.05
RESAMPLES = 20000
GATE = 0.95                 # parsed answers per (model, arm)
DELTA = 0.15                # the smallest lead worth a name, in share of questions
FLOOR = 0.10                # correct share under which an arm says nothing of a class
LEAK = 0.40                 # above this the control's guess is no guess (chance is 0.25)
ARMS = ("records", "prose", "log", "none")
LETTERS = "ABCD"
ANSWER = re.compile(r"`?(?:ANSWER[: ]+)?\*{0,2}([A-E])\*{0,2}`?(?:\.(?:\s.*)?)?", re.I)
NOT_STATED = "the passages do not settle it"
HYPOTHESES = {
    "H1": [("ahead", "changed", "prose"), ("ahead", "changed", "log")],
    "H2": [("within", "unchanged", "prose"), ("within", "unchanged", "log")],
    "H3": [("interaction", None, "prose"), ("interaction", None, "log")],
    "H4": [("fewer_old", "changed", "prose"), ("fewer_old", "changed", "log")],
    "H5": [("no_leak", "changed", "none"), ("no_leak", "unchanged", "none")],
}


def refuse(why):
    sys.exit(f"REFUSED {why}")


def letter(answer):
    lines = [x for x in answer.splitlines() if x.strip()]
    m = ANSWER.fullmatch(lines[-1].strip()) if lines else None
    if lines and lines[-1].strip().rstrip(".").lower() == NOT_STATED:
        return "E"
    return m.group(1).upper() if m else None


def choice(r):
    """The option text the row's answer names; 'E' for the passages not settling it; None when it does not parse, or names
    E where E was not offered."""
    x = letter(r["answer"])
    if x == "E":
        return "E" if r["arm"] != "none" else None
    return r["order"][LETTERS.index(x)] if x else None


def load(d):
    reg = json.loads((d / "registration.json").read_text())
    lock = json.loads((d / "registration.lock.json").read_text())
    data = json.loads((d / "results" / "data.json").read_text())
    raw = (d / "bank.json").read_bytes()
    design = reg["design"]
    if data.get("registration") != lock["registration"]:
        refuse("data are not of this registration")
    if hashlib.sha256(raw).hexdigest() != design["bank_sha256"]:
        refuse("bank.json is not the registered bank")
    if tuple(design["arms"]) != ARMS:
        refuse(f"arms {design['arms']} are not {list(ARMS)}")
    if set(data["models"]) != set(design["models"]):
        refuse(f"models {sorted(data['models'])} are not the registered {sorted(design['models'])}")
    bank = {i["id"]: i for i in json.loads(raw)}
    want = {(a, q) for a in ARMS for q in bank}
    for m, md in data["models"].items():
        keys = [(r["arm"], r["q"]) for r in md["rows"]]
        if len(keys) != len(set(keys)) or set(keys) != want:
            refuse(f"{m}: {len(set(keys))} distinct trials of {len(want)} registered, {len(keys)} rows")
        for r in md["rows"]:
            if sorted(r["order"]) != sorted(bank[r["q"]]["options"]):
                refuse(f"{m}: {r['arm']} {r['q']}: the options shown are not the bank's")
    return bank, {m: md["rows"] for m, md in data["models"].items()}


def scores(bank, rows):
    """{(arm, q): (share of models correct, share of models giving the old answer)}."""
    out = {}
    for arm in ARMS:
        for q, item in bank.items():
            picks = [choice(r) for rs in rows.values() for r in rs if r["arm"] == arm and r["q"] == q]
            out[arm, q] = (sum(p == item["key"] for p in picks) / len(picks),
                           sum(item["old"] is not None and p == item["old"] for p in picks) / len(picks))
    return out


def mean(xs):
    return sum(xs) / len(xs)


def bootstrap(name, groups, stat):
    """The percentile interval of stat(resampled groups), each group resampled alone."""
    rng = random.Random(name)
    vals = sorted(stat([rng.choices(g, k=len(g)) for g in groups]) for _ in range(RESAMPLES))
    return [vals[int(ALPHA / 2 * RESAMPLES)], vals[int((1 - ALPHA / 2) * RESAMPLES) - 1]]


def parsed(rows, arm):
    sel = [r for r in rows if r["arm"] == arm]
    return sum(choice(r) is not None for r in sel) / len(sel)


def component(kind, cls, other, bank, s, rows):
    ids = {c: sorted(q for q, i in bank.items() if i["class"] == c) for c in ("changed", "unchanged")}
    name = f"{kind}:{cls}:{other}"
    lead = lambda c: [s["records", q][0] - s[other, q][0] for q in ids[c]]
    if kind in ("ahead", "within"):
        xs = lead(cls)
        value, ci = mean(xs), bootstrap(name, [xs], lambda g: mean(g[0]))
    elif kind == "interaction":
        a, b = lead("changed"), lead("unchanged")
        value, ci = mean(a) - mean(b), bootstrap(name, [a, b], lambda g: mean(g[0]) - mean(g[1]))
    elif kind == "fewer_old":
        xs = [s[other, q][1] - s["records", q][1] for q in ids[cls]]
        value, ci = mean(xs), bootstrap(name, [xs], lambda g: mean(g[0]))
    else:
        xs = [s["none", q][0] for q in ids[cls]]
        value, ci = mean(xs), bootstrap(name, [xs], lambda g: mean(g[0]))
    lo, hi = ci
    if kind == "within":
        verdict = "supported" if hi < DELTA else "contradicted" if lo > DELTA else "inconclusive"
    elif kind == "no_leak":
        verdict = "supported" if hi < LEAK else "contradicted" if lo > LEAK else "inconclusive"
    else:
        verdict = "supported" if lo > 0 else "contradicted" if hi < DELTA else "inconclusive"
    c = {"kind": kind, "class": cls, "other": other, "value": value, "ci95": ci, "verdict": verdict}
    if kind == "within":
        reads = {}
        for a in ("records", other):
            xs = [s[a, q][0] for q in ids[cls]]
            reads[a] = bootstrap(f"reads:{cls}:{a}", [xs], lambda g: mean(g[0]))[0] > FLOOR
        c["above_floor"] = reads
        if not any(reads.values()):
            c.update(verdict="inconclusive", reason="neither arm is above the floor")
    arms = ("none",) if kind == "no_leak" else ("records", other)
    low = min(parsed(rs, a) for rs in rows.values() for a in arms)
    if low < GATE:
        c.update(verdict="inconclusive", reason=f"gate: parsed {low:.3f} < {GATE}")
    return c


def verdicts_of(bank, rows):
    s, out = scores(bank, rows), {}
    for h, comps in HYPOTHESES.items():
        res = [component(*c, bank, s, rows) for c in comps]
        vs = [c["verdict"] for c in res]
        out[h] = {"verdict": "contradicted" if "contradicted" in vs else "supported" if set(vs) == {"supported"} else "inconclusive",
                  "components": res}
    return out


def describe(bank, rows):
    out = {}
    for m, rs in rows.items():
        for arm in ARMS:
            for cls in ("changed", "unchanged"):
                sel = [r for r in rs if r["arm"] == arm and bank[r["q"]]["class"] == cls]
                picks = [(choice(r), bank[r["q"]]) for r in sel]
                out.setdefault(m, {}).setdefault(arm, {})[cls] = {
                    "n": len(sel), "correct": sum(p == i["key"] for p, i in picks), "old": sum(i["old"] is not None and p == i["old"] for p, i in picks),
                    "not_settled": sum(p == "E" for p, _ in picks), "unparsed": sum(p is None for p, _ in picks),
                    "tokens": sum(r["tokens"] for r in sel) / len(sel)}
    return out


def main(d):
    bank, rows = load(Path(d))
    print(json.dumps({"verdicts": verdicts_of(bank, rows), "arms": describe(bank, rows)}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
