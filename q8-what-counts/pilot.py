"""Calibration pilot of q8-what-counts (seeds from the namespace calibration:…, never a registration's).

    python -I q8-what-counts/pilot.py MODEL N[,N...] LISTS OUT.json      each vocabulary of calibration/vocabularies.json

Calibration only: nothing here enters a verdict."""
import hashlib, json, random, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPT = ("Here is a list of words:\n\n{list}\n\n"
          "How many words are in the list? Do not count step by step. Reply with a single integer only.")


def ask(model, words, seed):
    body = {"model": model, "prompt": PROMPT.replace("{list}", " ".join(words)), "stream": False, "think": False,
            "options": {"temperature": 0, "seed": seed % 2**31, "num_ctx": 8192, "num_predict": 16}}
    req = urllib.request.Request("http://localhost:11434/api/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    return out.get("response", ""), out.get("prompt_eval_count"), round(time.time() - t, 2)


if __name__ == "__main__":
    model, ns, lists, path = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], int(sys.argv[3]), sys.argv[4]
    vocab = json.loads((HERE / "calibration" / "vocabularies.json").read_text())
    rows = []
    for c, ws in vocab.items():
        for n in ns:
            for k in range(lists):
                s = int(hashlib.sha256(f"calibration:{model}:{c}:{n}:{k}".encode()).hexdigest()[:12], 16)
                rng = random.Random(s)
                answer, tokens, sec = ask(model, [rng.choice(ws) for _ in range(n)], s)
                rows.append({"condition": c, "n": n, "k": k, "answer": answer, "prompt_tokens": tokens, "seconds": sec})
                print(json.dumps(rows[-1]), flush=True)
    json.dump({"model": model, "rows": rows}, open(path, "w"), indent=1)
