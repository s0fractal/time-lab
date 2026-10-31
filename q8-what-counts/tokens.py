"""Calibration probe of q8-what-counts: how many tokens each candidate word takes in each model, as ollama counts them.

    python -I q8-what-counts/tokens.py MODEL WORD [WORD ...]      prints {word: tokens per ' word'} as JSON

A raw prompt of a word repeated 32 times after a fixed head, less the head alone, over 32. Calibration only."""
import json, sys, urllib.request

OLLAMA = "http://localhost:11434"


def count(model, text):
    body = {"model": model, "prompt": text, "raw": True, "stream": False, "think": False,
            "options": {"temperature": 0, "num_predict": 1, "num_ctx": 4096}}
    req = urllib.request.Request(OLLAMA + "/api/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["prompt_eval_count"]


def per_word(model, word, head="List:"):
    return (count(model, head + (" " + word) * 32) - count(model, head)) / 32


if __name__ == "__main__":
    model, words = sys.argv[1], sys.argv[2:]
    print(json.dumps({w: per_word(model, w) for w in words}))
