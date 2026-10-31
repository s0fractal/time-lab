"""q8-what-counts confirmatory runner: every (model, condition, n, k) of the registered design, seeded from the registration
and the beacon. q1-weber's question and prompt, with three vocabularies that pull its long words apart:

    short   q1-weber's short words: one token each, about 5.5 characters
    one     long words that are one token in each model's tokenizer, about 13.4 characters
    many    rare words of the same length that are three to seven tokens in each model's tokenizer

    python -I q8-what-counts/run.py            resumes from runs/, then writes results/data.json

Models run one at a time. Each answer is the model's raw text; parsing belongs to the registered analysis. Each row keeps
the prompt's token count as ollama reports it and the list's characters (letters only, no separators)."""
import hashlib, json, platform, random, sys, time, urllib.request
from pathlib import Path

D = Path(__file__).resolve().parent
OLLAMA = "http://localhost:11434"


def seed(registration, randomness, model, condition, n, k):
    # the registered seed rule (registration.json `seeds`, METHOD §3)
    return int(hashlib.sha256(f"{registration}:{randomness}:{model}:{condition}:{n}:{k}".encode()).hexdigest()[:12], 16)


def call(path, body=None):
    req = urllib.request.Request(OLLAMA + path, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def words(vocabulary, s, n):
    rng = random.Random(s)
    return [rng.choice(vocabulary) for _ in range(n)]


def main():
    reg = json.loads((D / "registration.json").read_text())
    lock = json.loads((D / "registration.lock.json").read_text())
    beacon = json.loads((D / "beacon.json").read_text())
    design, rid, rnd = reg["design"], lock["registration"], beacon["beacon"]["randomness"]
    runs = D / "runs"
    runs.mkdir(exist_ok=True)
    tags = {m["name"]: m["digest"] for m in call("/api/tags")["models"]}
    models = {}
    for model in design["models"]:
        part = runs / f"{model.replace(':', '-').replace('/', '-')}.jsonl"
        done = {(r["condition"], r["n"], r["k"]) for r in map(json.loads, part.read_text().splitlines())} if part.exists() else set()
        with part.open("a") as out:
            for c in design["conditions"]:
                for n in design["grid"]:
                    for k in range(design["lists"]):
                        if (c["name"], n, k) in done:
                            continue
                        s = seed(rid, rnd, model, c["name"], n, k)
                        ws = words(c["vocabulary"], s, n)
                        t = time.time()
                        o = call("/api/generate", {"model": model, "prompt": design["prompt"].replace("{list}", " ".join(ws)),
                                                   "stream": False, "think": False, "options": {**design["options"], "seed": s % 2**31}})
                        out.write(json.dumps({"condition": c["name"], "n": n, "k": k, "seed": s, "answer": o.get("response", ""),
                                              "prompt_tokens": o.get("prompt_eval_count"), "chars": sum(map(len, ws)),
                                              "seconds": round(time.time() - t, 3)}) + "\n")
                        out.flush()
        rows = sorted(map(json.loads, part.read_text().splitlines()), key=lambda r: (r["condition"], r["n"], r["k"]))
        keep = ("condition", "n", "k", "seed", "answer", "prompt_tokens", "chars")
        models[model] = {"digest": tags.get(model), "rows": [{k: r[k] for k in keep} for r in rows]}
        print(model, len(rows), file=sys.stderr, flush=True)
    data = {"registration": rid, "beacon": beacon["attest"], "randomness": rnd,
            "env": {"ollama": call("/api/version")["version"], "python": platform.python_version(), "machine": platform.machine()},
            "models": models}
    (D / "results").mkdir(exist_ok=True)
    (D / "results" / "data.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
