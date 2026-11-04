"""q11-records: the three corpora, one file, from a checkout of warrant at commit H (the checkout's tree must be H).

    PYTHONPATH=<warrant checkout> python3 q11-records/build_corpus.py <warrant checkout> H OUT.json

A chunk is {id, text}, at most MAX characters:
    records  the current version of each lineage of .warrant/records.jsonl and each challenge, read through warrant's engine:
             schema, actor, date, the lineage's state and the guard of this version, then its text, what it states first.
             Left out: superseded versions, attests (a verdict says nothing of the thing), blobs, bare ids.
    prose    one paragraph of README.md or docs/**/*.md, with its file and heading.
    log      one commit of `git log H`: date, subject, body.
Nothing else of the repository goes in: no code, no tests, no CHANGELOG."""
import json, re, subprocess, sys, time
from pathlib import Path

MAX = 1200
HEX = re.compile(r"[0-9a-f]{40,64}\Z")


def clip(text):
    return text if len(text) <= MAX else text[:MAX - 1] + "…"


def strings(v, name=""):
    """The readable values of a record's body as `name: value` lines, leaving out bare ids."""
    if isinstance(v, dict):
        return [x for k, y in sorted(v.items()) for x in strings(y, k)]
    if isinstance(v, list):
        return [x for y in v for x in strings(y, name)]
    if isinstance(v, bool) or v is None or (isinstance(v, str) and HEX.match(v)):
        return []
    return [f"{name}: {v}"]


FIRST = ("statement", "text", "name", "title", "established", "residual", "reason")


def records(root):
    from warrant.repo import Repo
    rep = Repo(root).report()
    recs = rep["_recs"]
    current = {lin["leaf"]: L for L, lin in rep["lineages"].items()}
    of = {v: L for L, lin in rep["lineages"].items() for v in lin["path"]}
    rank = lambda line: (FIRST.index(line.split(":")[0]) if line.split(":")[0] in FIRST else len(FIRST), line)
    out = []
    for rid, r in sorted(recs.items(), key=lambda x: (x[1].get("at", 0), x[0])):
        day = time.strftime("%Y-%m-%d", time.gmtime(r.get("at", 0)))
        body = r.get("body") if isinstance(r.get("body"), dict) else {}
        if r["kind"] == "attest" or (r["kind"] == "entity" and rid not in current):
            continue
        if r["kind"] == "entity":
            lin = rep["lineages"].get(of.get(rid), {})
            head = f"[{r.get('schema')} by {r['actor']}, {day}; lineage state {lin.get('state')}; guard {rep['versions'].get(rid, {}).get('guard')}]"
        else:
            on = recs.get(body.get("target"), {})
            about = next(iter(sorted(strings(on.get("body", {})), key=rank)), "")[:120]
            head = f"[{r['kind']} by {r['actor']}, {day}, on {on.get('schema')} «{about}»]"
        out.append({"id": rid[:16], "text": clip(head + "\n" + "\n".join(sorted((x for x in strings(body) if not x.startswith("target:")), key=rank)))})
    return out


def prose(root):
    out = []
    files = [root / "README.md", *sorted((root / "docs").rglob("*.md"))]
    for f in files:
        rel, heading = f.relative_to(root).as_posix(), ""
        for i, para in enumerate(re.split(r"\n\s*\n", f.read_text())):
            para = para.strip()
            if not para:
                continue
            if para.startswith("#"):
                heading = para.splitlines()[0].lstrip("# ")
            out.append({"id": f"{rel}:{i}", "text": clip(f"[{rel} § {heading}]\n{para}")})
    return out


def log(root, h):
    raw = subprocess.run(["git", "-C", str(root), "log", "--date=short", "--format=%H%x1f%ad%x1f%s%x1f%b%x1e", h], capture_output=True,
                         text=True, check=True).stdout
    out = []
    for row in raw.split("\x1e"):
        if row.strip("\n"):
            sha, day, subject, body = row.strip("\n").split("\x1f")   # not bare strip(): \x1f is whitespace to it, and a body may be empty
            out.append({"id": sha[:16], "text": clip(f"[commit {sha[:12]}, {day}] {subject}\n{body.strip()}".strip())})
    return out


def main(root, h, out):
    root = Path(root).resolve()
    tree = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if tree != subprocess.run(["git", "-C", str(root), "rev-parse", h], capture_output=True, text=True, check=True).stdout.strip():
        sys.exit(f"the checkout is at {tree[:12]}, not at {h}")
    corpus = {"warrant": tree, "records": records(root), "prose": prose(root), "log": log(root, h)}
    Path(out).write_text(json.dumps(corpus, ensure_ascii=False, indent=0, sort_keys=True) + "\n")
    print({k: (len(v), sum(len(c["text"]) for c in v)) for k, v in corpus.items() if k != "warrant"})


if __name__ == "__main__":
    main(*sys.argv[1:])
