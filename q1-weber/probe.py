"""Calibration probe for q1-weber: does a local model answer an elapsed-count question with a number, and how fast?

    python q1-weber/probe.py MODEL N[,N...] TRIALS OUT.json

Calibration only: its seeds are a fixed namespace, never a registration's; nothing here enters a verdict."""
import hashlib, json, random, sys, time, urllib.request

VOCAB = ("apple river stone cloud lamp horse table green music paper window garden silver train bread "
         "candle forest mirror pencil rocket shadow summer tiger violet winter yellow anchor basket "
         "copper dragon").split()


LONG = ("photosynthesis microorganism thermodynamics electromagnetic counterproductive indistinguishable "
        "uncharacteristically telecommunications hippopotamus kaleidoscope procrastination constellation biodiversity "
        "infrastructure misunderstanding extraordinarily psychotherapist archaeologist entrepreneurship circumnavigation "
        "unconstitutional bureaucratically disproportionate interconnectedness crystallographer chrysanthemum "
        "thermoregulation phosphorescence incomprehensibility otorhinolaryngology").split()


def words(seed, n, vocab=VOCAB):
    rng = random.Random(seed)
    return [rng.choice(vocab) for _ in range(n)]


def ask(model, n, seed, vocab=VOCAB):
    text = " ".join(words(seed, n, vocab))
    prompt = (f"Here is a list of words:\n\n{text}\n\n"
              "How many words are in the list? Do not count step by step. Reply with a single integer only.")
    body = {"model": model, "prompt": prompt, "stream": False, "think": False,
            "options": {"temperature": 0, "seed": seed % (2**31), "num_ctx": 8192, "num_predict": 16}}
    req = urllib.request.Request("http://localhost:11434/api/generate", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    return {"n": n, "seed": seed, "answer": out.get("response", ""), "seconds": round(time.time() - t, 2),
            "prompt_tokens": out.get("prompt_eval_count")}


if __name__ == "__main__":
    model, ns, trials, path = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], int(sys.argv[3]), sys.argv[4]
    vocab = sys.argv[5] if len(sys.argv) > 5 else "short"
    rows = []
    for n in ns:
        for k in range(trials):
            seed = int(hashlib.sha256(f"calibration:{model}:{vocab}:{n}:{k}".encode()).hexdigest()[:12], 16)
            rows.append({**ask(model, n, seed, LONG if vocab == "long" else VOCAB), "vocab": vocab})
            print(json.dumps(rows[-1]), flush=True)
    json.dump({"model": model, "vocab": vocab, "rows": rows}, open(path, "w"), indent=1)
