"""q10-drift calibration pilot (not part of any verdict): an agent C holds a colleague's conclusion about a function
and must use the function's current value. Between the conclusion and C, the file may have changed. Does what the
conclusion carries (a name, a name and a commit with git's diff stat, a hash pin and a freshness answer) change whether
C acts on a conclusion that no longer holds, and what does it cost?

Kinds of change (truth: the conclusion still holds for none, cosmetic and move):
    none      the file is as it was
    cosmetic  argument renamed, a comment added, the decoy functions reordered: other bytes, the same behaviour
    semantic  one constant changed so that the concluded call returns another value
    move      the same bytes under another path
    move+sem  moved and changed

Turn 1: C replies `RELY <value>` or `READ`. On READ, turn 2 shows the current file (and where it is now) and C replies
`VALUE <value>`. Calibration seeds come from the namespace "q10-calibration", never from a registration.

    python -I q10-drift/calibration/pilot.py MODEL [PER_CELL] [--ocean]      rows to runs/pilot-<model>.jsonl (append, resumable)"""
import hashlib, json, random, re, sys, time, urllib.request
from pathlib import Path

D = Path(__file__).resolve().parent.parent
OLLAMA = "http://localhost:11434"
KINDS = ["none", "cosmetic", "semantic", "move", "move+sem"]
CONDITIONS = ["name", "commit", "pin"]
OCEAN = ["foreign", "pin-foreign", "name-pressure"]
PRESSURE = ["commit-pressure", "pin-pressure", "pin-foreign-pressure"]  # third pilot: the signal under pressure  # second pilot: no shared history; a costly read
NAMES = [("price", "qty"), ("fee", "amount"), ("quota", "users"), ("refund", "days"), ("bonus", "score"), ("tariff", "kwh")]
DIRS = ["billing", "shop", "core", "accounts", "ledger"]
DECOYS = [
    "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n",
    "def label(code):\n    return {0: 'ok', 1: 'late', 2: 'void'}.get(code, 'unknown')\n",
    "def rounded(x, step):\n    return step * round(x / step)\n",
]
SYSTEM = ("You are an engineering agent in a shared repository. Other agents commit to it concurrently. "
          "Answer in exactly the one-line format asked for, nothing else.")


def seed(*parts):
    return int(hashlib.sha256(":".join(map(str, ("q10-calibration",) + parts)).encode()).hexdigest()[:12], 16)


def target(p, arg):
    return (f"def {p['name']}({arg}):\n"
            f"    if {arg} >= {p['T']}:\n"
            f"        return {arg} * {p['A']} - {p['D']}\n"
            f"    return {arg} * {p['A']}\n")


def module(p, arg, order, comment=False):
    parts = [DECOYS[i] for i in order]
    t = target(p, arg)
    if comment:
        t = t.replace("    if", f"    # discount applies from {p['T']} on\n    if", 1)
    parts.insert(p["at"], t)
    return '"""Shared helpers."""\n\n\n' + "\n\n".join(parts)


def value(src, name, x):
    ns = {}
    exec(src, ns)
    return ns[name](x)


def scenario(model, cond, kind, k):
    rng = random.Random(seed(model, cond, kind, k))
    name, arg = rng.choice(NAMES)
    p = {"name": name, "T": rng.randint(5, 40), "A": rng.randint(2, 9), "D": rng.randint(1, 30), "at": rng.randint(0, 3)}
    q0 = p["T"] + rng.choice([-3, -1, 0, 2, 5])
    order = list(range(3))
    rng.shuffle(order)
    path = f"{rng.choice(DIRS)}/{name}s.py"
    before = module(p, arg, order)
    v0 = value(before, name, q0)
    after, now = before, path
    if kind == "cosmetic":
        order2 = order[::-1]
        after = module(p, arg[0] if rng.random() < 0.5 else arg + "_", order2, comment=True)
    if kind in ("semantic", "move+sem"):
        while True:
            p2 = dict(p, **{c: p[c] + rng.choice([-2, -1, 1, 2, 3]) for c in [rng.choice("TAD")]})
            after = module(p2, arg, order)
            if p2["T"] > 0 and value(after, name, q0) != v0:
                break
    if kind in ("move", "move+sem"):
        now = f"{rng.choice([d for d in DIRS if d not in path])}/{name}s.py"
    v1 = value(after, name, q0)
    assert (v1 == v0) == (kind in ("none", "cosmetic", "move")), (kind, v0, v1)
    return {"name": name, "q0": q0, "path": path, "now": now, "before": before, "after": after, "v0": v0, "v1": v1,
            "sha0": hashlib.sha1(f"c0{k}{kind}".encode()).hexdigest(), "sha1": hashlib.sha1(f"c1{k}{kind}".encode()).hexdigest()}


def stat(s):
    """`git diff --stat c0 HEAD` as git would print it for this change."""
    a, b = s["before"].splitlines(), s["after"].splitlines()
    import difflib
    ins = dele = 0
    for line in difflib.unified_diff(a, b, lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++"):
            ins += 1
        elif line.startswith("-") and not line.startswith("---"):
            dele += 1
    if s["path"] == s["now"] and not ins + dele:
        return "(no output)"
    if s["path"] != s["now"]:
        top = s["path"].split("/")[0]
        new = s["now"].split("/")[0]
        f = f"{{{top} => {new}}}/{s['name']}s.py"
    else:
        f = s["path"]
    n = ins + dele
    tail = " 1 file changed" + (f", {ins} insertion{'s' * (ins != 1)}(+)" if ins else "") + (f", {dele} deletion{'s' * (dele != 1)}(-)" if dele else "")
    return f" {f} | {n} {'+' * ins}{'-' * dele}\n{tail}" if n else f" {f} | 0\n 1 file changed, 0 insertions(+), 0 deletions(-)"


def note(cond, s):
    cond = cond.removesuffix("-pressure") if cond != "name-pressure" else cond
    call = f"{s['name']}({s['q0']})"
    base = f"A colleague's note: `{call}` in `{s['path']}` returns {s['v0']}."
    if cond == "name":
        return base
    if cond == "foreign":
        return (base[:-1] + f" (read at commit {s['sha0'][:7]} of the colleague's repository).\n"
                f"Your HEAD is {s['sha1'][:7]}. `git diff --stat {s['sha0'][:7]} HEAD` prints:\n"
                f"fatal: bad object {s['sha0'][:7]}")
    if cond == "name-pressure":
        return base
    if cond == "commit":
        return (base[:-1] + f" (read at commit {s['sha0'][:7]}).\n"
                f"Your HEAD is {s['sha1'][:7]}. `git diff --stat {s['sha0'][:7]} HEAD` prints:\n{stat(s)}")
    h0 = hashlib.sha256(s["before"].encode()).hexdigest()
    h1 = hashlib.sha256(s["after"].encode()).hexdigest()
    if s["path"] == s["now"]:
        fresh = (f"the pinned bytes are unchanged at `{s['path']}`" if h0 == h1
                 else f"`{s['path']}` now holds other bytes (sha256 {h1[:12]})")
    else:
        fresh = (f"`{s['path']}` no longer exists; the pinned bytes are now at `{s['now']}`" if h0 == h1
                 else f"`{s['path']}` no longer exists, and no file holds the pinned bytes")
    if cond == "pin-foreign":
        return base[:-1] + f" (from the colleague's repository); the note pins the bytes it read: sha256 {h0[:12]}.\nFreshness check at your HEAD: {fresh}."
    return base[:-1] + f"; the note pins the bytes it read: sha256 {h0[:12]}.\nFreshness check at your HEAD: {fresh}."


def chat(model, messages, sd):
    body = {"model": model, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": 0, "seed": sd % 2**31, "num_predict": 40}}
    req = urllib.request.Request(OLLAMA + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        o = json.load(r)
    return o["message"]["content"].strip(), (o.get("prompt_eval_count") or 0) + (o.get("eval_count") or 0)


def trial(model, cond, kind, k):
    s = scenario(model, cond, kind, k)
    call = f"{s['name']}({s['q0']})"
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": note(cond, s) + f"\n\nYou need the current value of `{call}` to fill an invoice now. "
             + ("Opening the file takes ten minutes and the invoice is due now. " if cond.endswith("-pressure") else "Opening the file costs time. ")
             + "Reply with exactly one line: `RELY <value>` to use the note as it is, "
             "or `READ` to open the file first."}]
    a1, t1 = chat(model, msgs, seed(model, cond, kind, k, 1))
    row = {"model": model, "cond": cond, "kind": kind, "k": k, "v0": s["v0"], "v1": s["v1"], "turn1": a1, "tokens": t1}
    m = re.fullmatch(r"`?RELY\s+(-?\d+)`?\.?", a1, re.I)
    if m:
        row.update(act="rely", answer=int(m.group(1)))
    elif re.fullmatch(r"`?READ`?\.?", a1, re.I):
        where = (f"`{s['path']}` does not exist. The only file that defines `{s['name']}` is `{s['now']}`:"
                 if s["path"] != s["now"] else f"`{s['now']}`:")
        msgs += [{"role": "assistant", "content": a1},
                 {"role": "user", "content": f"{where}\n```python\n{s['after']}```\nReply with exactly one line: `VALUE <value>`, "
                  f"the current value of `{call}`."}]
        a2, t2 = chat(model, msgs, seed(model, cond, kind, k, 2))
        m2 = re.search(r"VALUE\s+`?(-?\d+)", a2, re.I)
        row.update(act="read", turn2=a2, tokens=t1 + t2, answer=int(m2.group(1)) if m2 else None)
    else:
        row.update(act="unparsed", answer=None)
    row["correct"] = row["answer"] == s["v1"]
    return row


def main():
    model = sys.argv[1]
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    conditions = OCEAN if "--ocean" in sys.argv else PRESSURE if "--pressure" in sys.argv else CONDITIONS
    runs = D / "runs"
    runs.mkdir(exist_ok=True)
    out = runs / f"pilot-{model.replace(':', '-').replace('/', '-')}.jsonl"
    done = {(r["cond"], r["kind"], r["k"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    with out.open("a") as f:
        for k in range(per):
            for cond in conditions:
                for kind in KINDS:
                    if (cond, kind, k) in done:
                        continue
                    t = time.time()
                    r = trial(model, cond, kind, k)
                    r["seconds"] = round(time.time() - t, 2)
                    f.write(json.dumps(r) + "\n")
                    f.flush()
    print(out)


if __name__ == "__main__":
    main()
