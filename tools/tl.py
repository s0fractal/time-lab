"""time-lab's records: a pre-registration as Warrant 3 records (METHOD.md).

    tl.py schemas                 file registration@1 and prediction@1 (claude, once)
    tl.py register DIR            the runner and analysis as blobs, registration@1, one prediction@1 per hypothesis
    tl.py beacon DIR              the GitHub-witnessed push of the registration, the first drand round after it, as an attest
    tl.py close DIR               run the registered analysis; file each prediction's closing version
    tl.py decide DIR              as time-lab-decider: rerun the analysis and attest each closing version
    tl.py verify DIR              the runner, the analysis and the data are the bytes the records name
    tl.py assess                  each prediction's state, reliance and guard, and what the core cannot check (METHOD §4)

The receiver is .warrant/v3.json. Signers' seeds are read from outside the repository: claude's from $TL_CLAUDE_SEED,
the decider's from $TL_DECIDER_SEED. The decider's key is made and held by GitHub Actions (.github/workflows/
decider-key.yml) and `decide` runs there (decide.yml): no operator holds it."""
import hashlib, json, os, platform, re, subprocess, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE, RECEIVER = ROOT / ".warrant", ROOT / ".warrant/v3.json"
PY = str(ROOT / ".venv/bin/python")
DRAND = "https://api.drand.sh/52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"
REPO = "s0fractal/time-lab"
DECIDED = ("supported", "contradicted", "inconclusive")
DECIDER = "time-lab-decider"
GUARD = {"quorum": 1, "roles": ["prediction/decider"], "match": {"verdict": "reproduces"}, "requires": []}

SCHEMAS = [
    {"name": "registration@1",
     "fields": {"state": "str", "question": "str", "design": "obj", "runner": "ref", "runner_path": "str",
                "analysis": "ref", "analysis_path": "str", "seeds": "str", "beacon": "str", "because": "str?"},
     "states": {"registered": None}, "closed": [], "successors": {}},
    {"name": "prediction@1",
     "fields": {"state": "str", "registration": "ref", "hypothesis": "str", "statement": "str", "rule": "str",
                "result": "obj?", "data": "ref?", "because": "str?"},
     "states": {"open": None, "withdrawn": None, **{s: GUARD for s in DECIDED}},
     "closed": sorted(DECIDED), "successors": {}},
]


def micro(v):
    """A result as Warrant can carry it (no floats, V3 §1): every real number x becomes round(x · 10⁶)."""
    if isinstance(v, float):
        return round(v * 1_000_000)
    if isinstance(v, dict):
        return {k: micro(x) for k, x in v.items()}
    if isinstance(v, list):
        return [micro(x) for x in v]
    return v


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def trust_file() -> str:
    """The receiver's trust as the trust/3 file the CLI signs under: warrant-graph 3.0.0a2 requires `--trust` for put and
    attest, and reads it only inside the working directory. Written for each signing call to the ignored runs/.
    warrant master reads the store's v3.json itself since warrant#236 (unreleased after 3.0.0a2): drop this when the pin
    moves past 3.0.0a2."""
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / "trust.json").write_text(json.dumps(json.loads(RECEIVER.read_text())["trust"]))
    return "runs/trust.json"


def warrant(*args, actor=None, key=None) -> dict:
    cmd = [PY, "-m", "warrant", *args[:1], "--store", str(STORE)]
    if actor:
        cmd += ["--actor", actor, "--key", key, "--trust", trust_file()]
    p = subprocess.run(cmd + list(args[1:]), capture_output=True, text=True, cwd=ROOT)
    if p.returncode or not p.stdout.strip().startswith("{"):
        sys.exit(f"warrant {args[0]} refused: {p.stdout.strip()} {p.stderr.strip()}")
    return json.loads(p.stdout)


def claude():
    return {"actor": "claude", "key": os.environ["TL_CLAUDE_SEED"]}


def decider():
    return {"actor": DECIDER, "key": os.environ["TL_DECIDER_SEED"]}


def put(kind, body, schema=None, lineage=None, refs=(), guard=None, signer=None) -> str:
    args = ["put", "--kind", kind] + (["--schema", schema] if schema else []) + (["--lineage", lineage] if lineage else [])
    args += [x for r in sorted(set(refs)) for x in ("--ref", r)] + (["--guard", json.dumps(guard)] if guard else [])
    return warrant(*args, json.dumps(body), **(signer or claude()))["record"]


def blob(path: Path) -> str:
    return warrant("blob", str(path))["blob"]


def report() -> dict:
    from warrant import engine as E
    from warrant.store import Store
    doc = json.loads(RECEIVER.read_text())
    return E.assess(Store(str(STORE)).snapshot(), doc["trust"], doc["profile"], None)


def records() -> dict:
    from warrant import records as R
    rs = (json.loads(line)["record"] for line in (STORE / "records.jsonl").read_text().splitlines())
    return {R.rid(r): r for r in rs}


def chain_position(rec: dict) -> dict:
    """{id: (actor, n)}: each record's place in its actor's chain, by `prev` (hash-linked, unlike `at`)."""
    nxt = {r["prev"]: i for i, r in rec.items() if r.get("prev")}
    pos = {}
    for i, r in rec.items():
        if not r.get("prev"):
            n, cur = 0, i
            while cur is not None and cur not in pos:
                pos[cur] = (r["actor"], n)
                n, cur = n + 1, nxt.get(cur)
    return pos


def unchecked(rec: dict, rep: dict) -> list:
    """What the core does not check for a prediction (METHOD §4): a closing version whose guard does not name the
    registered analysis (N1: CODE_NOT_REGISTERED), and a withdrawal of a registration that has a beacon attest, unless
    the author's own chain puts the withdrawal before that attest (WITHDRAWN_AFTER_BEACON): a pair the chain does not
    order does not meet the rule (primitives PATTERNS, "Order is witnessed, not claimed")."""
    pos, out = chain_position(rec), []
    beacons = [i for i, r in rec.items() if r["kind"] == "attest" and r["body"].get("verdict") == "witnessed"]
    for i, r in rec.items():
        if r.get("schema") != "prediction@1" or r["kind"] != "entity":
            continue
        reg = rec.get(r["body"]["registration"], {}).get("body", {})
        state = r["body"]["state"]
        if state in DECIDED and (r.get("guard") or {}).get("match", {}).get("code") != reg.get("analysis"):
            out.append(("CODE_NOT_REGISTERED", r["body"]["hypothesis"], i))
        before = lambda b: pos.get(b, (None,))[0] == r["actor"] == pos.get(i, (None,))[0] and pos[i][1] < pos[b][1]
        if state == "withdrawn" and any(rec[b]["body"].get("target") == r["body"]["registration"] and not before(b)
                                        for b in beacons):
            out.append(("WITHDRAWN_AFTER_BEACON", r["body"]["hypothesis"], i))
    return out


def same(stored, raw) -> bool:
    """Whether a closing version's stored result (`micro` of the analysis output) is the rerun's raw output: every
    real number x of the rerun within one unit of the stored round(x · 10⁶) (a rerun on another machine may differ in a
    float's last bits, and round may then fall to the other side of .5), everything else equal and of the same type
    (a flag is no number), the structure the same."""
    if isinstance(raw, float):
        return type(stored) is int and abs(raw * 1_000_000 - stored) <= 1
    if isinstance(raw, dict):
        return isinstance(stored, dict) and stored.keys() == raw.keys() and all(same(stored[k], raw[k]) for k in raw)
    if isinstance(raw, list):
        return isinstance(stored, list) and len(stored) == len(raw) and all(same(x, y) for x, y in zip(stored, raw))
    return type(stored) is type(raw) and stored == raw


def lock(d: Path) -> dict:
    return json.loads((d / "registration.lock.json").read_text())


def analysis(d: Path, lk: dict) -> dict:
    """Run the registered analysis bytes (refused if the file differs from the registered blob) on the committed data."""
    code = (ROOT / lk["analysis_path"]).read_bytes()
    if sha(code) != lk["analysis"]:
        sys.exit(f"{lk['analysis_path']} is not the registered analysis {lk['analysis'][:12]}")
    p = subprocess.run([PY, "-I", str(ROOT / lk["analysis_path"]), str(d)], capture_output=True, text=True, cwd=ROOT)
    if p.returncode:
        sys.exit(f"analysis failed: {p.stderr.strip()[-2000:]}")
    return json.loads(p.stdout)


def env() -> dict:
    import numpy
    return {"python": platform.python_version(), "numpy": numpy.__version__, "machine": platform.machine()}


def head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()


def main(cmd, *a):
    if cmd == "schemas":
        have = {r["body"]["name"] for r in records().values() if r["kind"] == "schema"}
        for s in SCHEMAS:
            if s["name"] not in have:
                print(s["name"], put("schema", s))
    elif cmd == "register":
        d = ROOT / a[0]
        reg = json.loads((d / "registration.json").read_text())
        if (d / "registration.lock.json").exists():
            sys.exit("already registered: a change is a new version of the registration (METHOD §5)")
        runner, ana = blob(ROOT / reg["runner_path"]), blob(ROOT / reg["analysis_path"])
        body = {"state": "registered", "question": reg["question"], "design": micro(reg["design"]), "runner": runner,
                "runner_path": reg["runner_path"], "analysis": ana, "analysis_path": reg["analysis_path"],
                "seeds": reg["seeds"], "beacon": reg["beacon"]}
        rid = put("entity", body, schema="registration@1", refs=[runner, ana])
        preds = {}
        for h in reg["hypotheses"]:
            preds[h["hypothesis"]] = put("entity", {"state": "open", "registration": rid, **h}, schema="prediction@1", refs=[rid])
        out = {"registration": rid, "runner": runner, "runner_path": reg["runner_path"], "analysis": ana,
               "analysis_path": reg["analysis_path"], "predictions": preds}
        (d / "registration.lock.json").write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out, indent=1))
    elif cmd == "beacon":
        d = ROOT / a[0]
        lk = lock(d)
        rel = (d / "registration.lock.json").relative_to(ROOT).as_posix()
        commit = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H", "--", rel], capture_output=True, text=True,
                                cwd=ROOT, check=True).stdout.split()[-1]
        acts = json.loads(subprocess.run(["gh", "api", f"repos/{REPO}/activity?per_page=100"], capture_output=True, text=True,
                                         check=True).stdout)
        pushes = sorted((x["timestamp"], x["after"]) for x in acts if x["activity_type"] in ("push", "branch_creation", "force_push")
                        and subprocess.run(["git", "merge-base", "--is-ancestor", commit, x["after"]], cwd=ROOT).returncode == 0)
        if not pushes:
            sys.exit(f"no GitHub-witnessed push holds {commit[:12]} yet")
        when, pushed = pushes[0]
        from datetime import datetime
        t = int(datetime.fromisoformat(when.replace("Z", "+00:00")).timestamp())
        info = json.load(urllib.request.urlopen(f"{DRAND}/info", timeout=30))
        rnd = (t - info["genesis_time"]) // info["period"] + 2          # round r is at genesis + (r-1)·period: the first at or after t
        b = json.load(urllib.request.urlopen(f"{DRAND}/public/{rnd}", timeout=60))
        body = {"verdict": "witnessed", "witness": {"source": f"GET repos/{REPO}/activity", "push": pushed, "timestamp": when,
                                                     "holds": commit},
                "beacon": {"chain": info["hash"], "round": b["round"], "time": info["genesis_time"] + (b["round"] - 1) * info["period"],
                           "randomness": b["randomness"], "signature": b["signature"]}}
        bid = warrant("attest", lk["registration"], "--body", json.dumps(body), **claude())["record"]
        out = {"attest": bid, **body}
        (d / "beacon.json").write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out, indent=1))
    elif cmd == "close":
        d = ROOT / a[0]
        lk, res = lock(d), analysis(ROOT / a[0], lock(ROOT / a[0]))
        data = blob(d / "results" / "data.json")
        g = dict(GUARD, match={"verdict": "reproduces", "code": lk["analysis"]})
        rec = records()
        closed = {}
        for h, pid in lk["predictions"].items():
            v = micro(res["verdicts"][h])
            body = {**{k: rec[pid]["body"][k] for k in ("registration", "hypothesis", "statement", "rule")},
                    "state": v["verdict"], "result": v, "data": data}
            closed[h] = put("entity", body, schema="prediction@1", lineage=pid, refs=[lk["registration"], data], guard=g)
        (d / "closure.lock.json").write_text(json.dumps({"data": data, "closing": closed}, indent=1) + "\n")
        print(json.dumps(closed, indent=1))
    elif cmd == "verify":
        d = ROOT / a[0]
        lk = lock(d)
        bad = [p for p, want in ((lk["runner_path"], lk["runner"]), (lk["analysis_path"], lk["analysis"]))
               if sha((ROOT / p).read_bytes()) != want]
        if (d / "closure.lock.json").exists():
            cl = json.loads((d / "closure.lock.json").read_text())
            bad += [] if sha((d / "results" / "data.json").read_bytes()) == cl["data"] else [f"{a[0]}/results/data.json"]
        print("\n".join(f"DIFFERS {p}" for p in bad) or "ok: the runner, the analysis and the data are the registered bytes")
        sys.exit(1 if bad else 0)
    elif cmd == "decide":
        d = ROOT / a[0]
        lk, res = lock(d), analysis(ROOT / a[0], lock(ROOT / a[0]))
        cl, rec, at = json.loads((d / "closure.lock.json").read_text()), records(), head()
        if sha((d / "results" / "data.json").read_bytes()) != cl["data"]:
            sys.exit("results/data.json is not the data the closing versions name")
        done = {r["body"].get("target") for r in rec.values() if r["kind"] == "attest" and r["actor"] == DECIDER}
        for h, cid in cl["closing"].items():
            if cid in done:
                print(h, "attested already")
                continue
            b = rec[cid]["body"]
            ok = b["state"] == res["verdicts"][h]["verdict"] and same(b["result"], res["verdicts"][h])
            body = {"verdict": "reproduces" if ok else "does_not_reproduce", "on": cid, "code": lk["analysis"], "env": env(), "commit": at}
            print(h, body["verdict"], warrant("attest", cid, "--body", json.dumps(body), **decider())["record"])
    elif cmd == "assess":
        rep = report()
        rec = records()
        for lid, l in sorted(rep["lineages"].items(), key=lambda x: rec.get(x[0], {}).get("at", 0)):
            r = rec.get(lid)
            if r and r.get("schema") == "prediction@1":
                leaf = rep["versions"].get(l["leaf"], {})
                q = rec.get(r["body"]["registration"], {}).get("body", {}).get("runner_path", "?").split("/")[0]
                print(f"{q + ' ' + r['body']['hypothesis']:<40} {str(l.get('state')):<13} {l.get('reliance'):<10} "
                      f"guard {leaf.get('guard')} {leaf.get('waiting') or ''}")
        print("problems", rep.get("problems"), "rejected", len(rep.get("rejected", [])))
        found = unchecked(rec, rep)
        for code, h, i in found:
            print(code, h, i[:12])
        sys.exit(1 if found else 0)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["help"]))
