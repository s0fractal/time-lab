"""q11-records runner: a fresh model answers a question about warrant from one kind of text about it.

Arms: records (.warrant/ read through the engine), prose (README and docs), log (git log), none (no text, a guess among the
four options: the control that sees whether the options give the answer away). Each arm with text gets the same budget of
characters, its chunks chosen by the same BM25 over the question and its options, best first (build_corpus.py makes the
chunks). The options' order is seeded per question and is the same in every arm and model, so arms compare on one form.

Each question has four options; an arm with text also has E, "the passages do not settle it". The answer key was fixed from
the code at warrant's commit H and never from any arm (bank/validate.py).

    python -I q11-records/run.py [DIR]                 the registered design, resuming from DIR/runs/, then DIR/results/data.json
    python -I q11-records/run.py DIR --calibration     the calibration bank under the namespace q11-calibration, to calibration/

Models run one at a time and are unloaded after their rows."""
import hashlib, json, math, platform, random, re, sys, time, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent
OLLAMA = "http://localhost:11434"
TOKEN = re.compile(r"[a-z0-9]+")
LETTERS = "ABCD"
SYSTEM = ("You answer questions about a software project. Use only the passages you are given, not what you may remember or "
          "assume. Reply with exactly one line: `ANSWER <letter>`, nothing else.")
GUESS = "You answer questions about a software project. Reply with exactly one line: `ANSWER <letter>`, nothing else."
NOT_STATED = "the passages do not settle it"


def seed(*parts):
    # the registered seed rule (registration.json `seeds`, METHOD §3)
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def tokens(text):
    return TOKEN.findall(text.lower())


class Index:
    """BM25 (k1, b as registered) over one arm's chunks."""

    def __init__(self, chunks, k1, b):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tf = [Counter(tokens(c["text"])) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / len(self.len)
        self.df = Counter(w for t in self.tf for w in t)
        self.post = defaultdict(list)
        for i, t in enumerate(self.tf):
            for w in t:
                self.post[w].append(i)

    def rank(self, query):
        n, score = len(self.chunks), defaultdict(float)
        for w in set(tokens(query)):
            idf = math.log(1 + (n - self.df[w] + 0.5) / (self.df[w] + 0.5))
            for i in self.post.get(w, ()):
                f = self.tf[i][w]
                score[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return sorted(score, key=lambda i: (-score[i], self.chunks[i]["id"]))

    def passages(self, query, budget):
        """The best chunks that fit, in rank order; a chunk that does not fit is skipped."""
        out, used = [], 0
        for i in self.rank(query):
            c = self.chunks[i]
            if used + len(c["text"]) <= budget:
                out.append(c)
                used += len(c["text"])
        return out


def order_of(item, namespace, rnd):
    order = list(item["options"])
    random.Random(seed(namespace, rnd, "order", item["id"])).shuffle(order)
    return order


def prompt(item, order, passages):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(order))
    if passages is None:
        return (f"Question: {item['question']}\n{opts}\n\nYou have no text about this project: give your best guess among A, B, C "
                "and D. Reply with exactly one line: `ANSWER <letter>`.")
    text = "\n\n".join(c["text"] for c in passages) or "(none)"
    return (f"Passages:\n\n{text}\n\nQuestion: {item['question']}\n{opts}\nE. {NOT_STATED}\n\nReply with exactly one line: "
            "`ANSWER <letter>`.")


def call(path, body=None):
    req = urllib.request.Request(OLLAMA + path, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def chat(model, system, user, sd, options):
    o = call("/api/chat", {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                           "stream": False, "think": False, "options": {**options, "seed": sd % 2**31}})
    return o["message"]["content"].strip(), (o.get("prompt_eval_count") or 0) + (o.get("eval_count") or 0)


def main(d=D, calibration=False):
    reg = json.loads((d / "registration.json").read_text())
    design = reg["design"]
    corpus_bytes = (d / "runs" / "corpus.json").read_bytes()
    if hashlib.sha256(corpus_bytes).hexdigest() != design["corpus_sha256"]:
        sys.exit("runs/corpus.json is not the registered corpus: build_corpus.py at the registered commit makes it")
    corpus = json.loads(corpus_bytes)
    if calibration:
        out_dir, namespace, rnd, bank = d / "calibration", "q11-calibration", "q11-calibration", json.loads((d / "calibration" / "bank.json").read_text())
        rid, attest = "q11-calibration", None
    else:
        bank_bytes = (d / "bank.json").read_bytes()
        if hashlib.sha256(bank_bytes).hexdigest() != design["bank_sha256"]:
            sys.exit("bank.json is not the registered bank")
        lock, beacon = json.loads((d / "registration.lock.json").read_text()), json.loads((d / "beacon.json").read_text())
        out_dir, rid, attest, rnd, bank = d / "results", lock["registration"], beacon["attest"], beacon["beacon"]["randomness"], json.loads(bank_bytes)
        namespace = rid
    indexes = {a: Index(corpus[a], design["bm25"]["k1"], design["bm25"]["b"]) for a in design["arms"] if a != "none"}
    runs = out_dir / "runs" if calibration else d / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    tags = {m["name"]: m["digest"] for m in call("/api/tags")["models"]}
    models = {}
    for model in design["models"]:
        part = runs / f"{model.replace(':', '-').replace('/', '-')}.jsonl"
        done = {(r["arm"], r["q"]) for r in map(json.loads, part.read_text().splitlines())} if part.exists() else set()
        with part.open("a") as out:
            for item in bank:
                order = order_of(item, namespace, rnd)
                for arm in design["arms"]:
                    if (arm, item["id"]) in done:
                        continue
                    query = " ".join([item["question"], *item["options"]])
                    passages = None if arm == "none" else indexes[arm].passages(query, design["budget"])
                    t = time.time()
                    answer, used = chat(model, GUESS if arm == "none" else SYSTEM, prompt(item, order, passages),
                                        seed(namespace, rnd, model, arm, item["id"]), design["options"])
                    out.write(json.dumps({"arm": arm, "q": item["id"], "order": order, "chunks": [c["id"] for c in passages or []],
                                          "answer": answer, "tokens": used, "seconds": round(time.time() - t, 3)}, ensure_ascii=False) + "\n")
                    out.flush()
        call("/api/generate", {"model": model, "keep_alive": 0})
        rows = sorted(map(json.loads, part.read_text().splitlines()), key=lambda r: (r["q"], r["arm"]))
        models[model] = {"digest": tags.get(model), "rows": [{k: v for k, v in r.items() if k != "seconds"} for r in rows]}
        print(model, len(rows), file=sys.stderr, flush=True)
    data = {"registration": rid, "beacon": attest, "randomness": rnd,
            "env": {"ollama": call("/api/version")["version"], "python": platform.python_version(), "machine": platform.machine()},
            "models": models}
    out_dir.mkdir(exist_ok=True)
    (out_dir / ("pilot-data.json" if calibration else "data.json")).write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else D, "--calibration" in sys.argv)
