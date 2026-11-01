"""q10-drift confirmatory runner: an agent C holds a colleague's conclusion about a function and must use the function's
current value. Between the conclusion and C, the file may have changed. Every (model, cell, kind, k) of the registered
design, seeded from the registration and the beacon; calibration/pilot.py's generator and protocol, with the plural of a
file name fixed (bonuses.py, not bonuss.py) and one scenario per (kind, k) shared by every model and cell.

Kinds of change (truth: the conclusion still holds for none, cosmetic and move):
    none      the file is as it was
    cosmetic  argument renamed, a comment added, the decoy functions reordered: other bytes, the same behaviour
    semantic  one constant changed so that the concluded call returns another value
    move      the same bytes under another path
    move+sem  moved and changed

Cells: what the conclusion carries (name, commit with git's diff stat, pin with a freshness answer; foreign: from the
colleague's repository), and with `-pressure` a costly read and an invoice due now.

Turn 1: C replies `RELY <value>` or `READ`. On READ, turn 2 shows the current file (and where it is now) and C replies
`VALUE <value>`. Each row keeps both turns' raw text; classifying them belongs to the registered analysis.

    python -I q10-drift/run.py [DIR]           resumes from DIR/runs/, then writes DIR/results/data.json (DIR: this one)

Models run one at a time and are unloaded after their rows."""
import difflib, hashlib, json, platform, random, re, sys, time, urllib.request
from pathlib import Path

D = Path(__file__).resolve().parent
OLLAMA = "http://localhost:11434"
KINDS = ["none", "cosmetic", "semantic", "move", "move+sem"]
STALE = ("semantic", "move+sem")
NAMES = [("price", "qty"), ("fee", "amount"), ("quota", "users"), ("refund", "days"), ("bonus", "score"), ("tariff", "kwh")]
DIRS = ["billing", "shop", "core", "accounts", "ledger"]
DECOYS = [
    "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n",
    "def label(code):\n    return {0: 'ok', 1: 'late', 2: 'void'}.get(code, 'unknown')\n",
    "def rounded(x, step):\n    return step * round(x / step)\n",
]
SYSTEM = ("You are an engineering agent in a shared repository. Other agents commit to it concurrently. "
          "Answer in exactly the one-line format asked for, nothing else.")
RELY = re.compile(r"`?RELY\s+(-?\d+)`?\.?", re.I)          # the protocol's turn-1 forms, matched whole
READ = re.compile(r"`?READ`?\.?", re.I)


def seed(registration, randomness, *parts):
    # the registered seed rule (registration.json `seeds`, METHOD §3)
    return int(hashlib.sha256(":".join(map(str, (registration, randomness) + parts)).encode()).hexdigest()[:12], 16)


def plural(name):
    return name + ("es" if name.endswith("s") else "s")


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


def scenario(s, kind):
    """The scenario of seed s under one kind of change: the module before and after, the concluded call and both values."""
    rng = random.Random(s)
    name, arg = rng.choice(NAMES)
    p = {"name": name, "T": rng.randint(5, 40), "A": rng.randint(2, 9), "D": rng.randint(1, 30), "at": rng.randint(0, 3)}
    q0 = p["T"] + rng.choice([-3, -1, 0, 2, 5])
    order = list(range(3))
    rng.shuffle(order)
    file = plural(name) + ".py"
    home = rng.choice(DIRS)
    path = f"{home}/{file}"
    before = module(p, arg, order)
    v0 = value(before, name, q0)
    after, now = before, path
    if kind == "cosmetic":
        after = module(p, arg[0] if rng.random() < 0.5 else arg + "_", order[::-1], comment=True)
    if kind in STALE:
        while True:
            p2 = dict(p, **{c: p[c] + rng.choice([-2, -1, 1, 2, 3]) for c in [rng.choice("TAD")]})
            after = module(p2, arg, order)
            if p2["T"] > 0 and value(after, name, q0) != v0:
                break
    if kind in ("move", "move+sem"):
        now = f"{rng.choice([d for d in DIRS if d != home])}/{file}"
    v1 = value(after, name, q0)
    assert (v1 == v0) == (kind not in STALE), (kind, v0, v1)
    return {"name": name, "file": file, "q0": q0, "path": path, "now": now, "before": before, "after": after, "v0": v0, "v1": v1,
            "sha0": hashlib.sha1(f"{s}:c0".encode()).hexdigest(), "sha1": hashlib.sha1(f"{s}:c1".encode()).hexdigest()}


def stat(s):
    """`git diff --stat c0 HEAD` as git would print it for this change."""
    ins = dele = 0
    for line in difflib.unified_diff(s["before"].splitlines(), s["after"].splitlines(), lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++"):
            ins += 1
        elif line.startswith("-") and not line.startswith("---"):
            dele += 1
    if s["path"] == s["now"] and not ins + dele:
        return "(no output)"
    f = (f"{{{s['path'].split('/')[0]} => {s['now'].split('/')[0]}}}/{s['file']}" if s["path"] != s["now"] else s["path"])
    n = ins + dele
    tail = (" 1 file changed" + (f", {ins} insertion{'s' * (ins != 1)}(+)" if ins else "")
            + (f", {dele} deletion{'s' * (dele != 1)}(-)" if dele else ""))
    return f" {f} | {n} {'+' * ins}{'-' * dele}\n{tail}" if n else f" {f} | 0\n 1 file changed, 0 insertions(+), 0 deletions(-)"


def note(cell, s):
    """What the conclusion carries in this cell (pressure adds nothing to the note)."""
    cond = cell.removesuffix("-pressure")
    base = f"A colleague's note: `{s['name']}({s['q0']})` in `{s['path']}` returns {s['v0']}."
    if cond == "name":
        return base
    if cond == "foreign":
        return (base[:-1] + f" (read at commit {s['sha0'][:7]} of the colleague's repository).\n"
                f"Your HEAD is {s['sha1'][:7]}. `git diff --stat {s['sha0'][:7]} HEAD` prints:\n"
                f"fatal: bad object {s['sha0'][:7]}")
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
        return (base[:-1] + f" (from the colleague's repository); the note pins the bytes it read: sha256 {h0[:12]}.\n"
                f"Freshness check at your HEAD: {fresh}.")
    if cond == "pin":
        return base[:-1] + f"; the note pins the bytes it read: sha256 {h0[:12]}.\nFreshness check at your HEAD: {fresh}."
    raise ValueError(cell)


def call(path, body=None):
    req = urllib.request.Request(OLLAMA + path, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def chat(model, messages, sd, options):
    o = call("/api/chat", {"model": model, "messages": messages, "stream": False, "think": False,
                           "options": {**options, "seed": sd % 2**31}})
    return o["message"]["content"].strip(), (o.get("prompt_eval_count") or 0) + (o.get("eval_count") or 0)


def trial(model, cell, kind, k, rid, rnd, options):
    sd = seed(rid, rnd, "scenario", kind, k)
    s = scenario(sd, kind)
    q = f"{s['name']}({s['q0']})"
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": note(cell, s) + f"\n\nYou need the current value of `{q}` to fill an invoice now. "
             + ("Opening the file takes ten minutes and the invoice is due now. " if cell.endswith("-pressure") else "Opening the file costs time. ")
             + "Reply with exactly one line: `RELY <value>` to use the note as it is, or `READ` to open the file first."}]
    a1, t1 = chat(model, msgs, seed(rid, rnd, model, cell, kind, k, 1), options)
    row = {"cell": cell, "kind": kind, "k": k, "seed": sd, "v0": s["v0"], "v1": s["v1"], "turn1": a1, "tokens": t1}
    if READ.fullmatch(a1):
        where = (f"`{s['path']}` does not exist. The only file that defines `{s['name']}` is `{s['now']}`:"
                 if s["path"] != s["now"] else f"`{s['now']}`:")
        msgs += [{"role": "assistant", "content": a1},
                 {"role": "user", "content": f"{where}\n```python\n{s['after']}```\nReply with exactly one line: `VALUE <value>`, "
                  f"the current value of `{q}`."}]
        a2, t2 = chat(model, msgs, seed(rid, rnd, model, cell, kind, k, 2), options)
        row.update(turn2=a2, tokens=t1 + t2)
    return row


def main(d=D):
    reg = json.loads((d / "registration.json").read_text())
    lock = json.loads((d / "registration.lock.json").read_text())
    beacon = json.loads((d / "beacon.json").read_text())
    design, rid, rnd = reg["design"], lock["registration"], beacon["beacon"]["randomness"]
    runs = d / "runs"
    runs.mkdir(exist_ok=True)
    tags = {m["name"]: m["digest"] for m in call("/api/tags")["models"]}
    models = {}
    for model in design["models"]:
        part = runs / f"{model.replace(':', '-').replace('/', '-')}.jsonl"
        done = {(r["cell"], r["kind"], r["k"]) for r in map(json.loads, part.read_text().splitlines())} if part.exists() else set()
        with part.open("a") as out:
            for k in range(design["scenarios"]):
                for cell in design["cells"]:
                    for kind in design["kinds"]:
                        if (cell, kind, k) in done:
                            continue
                        t = time.time()
                        row = trial(model, cell, kind, k, rid, rnd, design["options"])
                        out.write(json.dumps({**row, "seconds": round(time.time() - t, 3)}) + "\n")
                        out.flush()
        call("/api/generate", {"model": model, "keep_alive": 0})
        rows = sorted(map(json.loads, part.read_text().splitlines()), key=lambda r: (r["cell"], r["kind"], r["k"]))
        models[model] = {"digest": tags.get(model), "rows": [{k: v for k, v in r.items() if k != "seconds"} for r in rows]}
        print(model, len(rows), file=sys.stderr, flush=True)
    data = {"registration": rid, "beacon": beacon["attest"], "randomness": rnd,
            "env": {"ollama": call("/api/version")["version"], "python": platform.python_version(), "machine": platform.machine()},
            "models": models}
    (d / "results").mkdir(exist_ok=True)
    (d / "results" / "data.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else D)
