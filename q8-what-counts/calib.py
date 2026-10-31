"""The registered design of q8-what-counts on calibration seeds (namespace calibration-…, never a registration's), each model
in turn, then the registered analysis over it: the intervals the confirmatory run will be judged by, before choosing its
thresholds (q4-line's lesson). Data under the ignored runs/calibration-full/.

    python -I q8-what-counts/calib.py"""
import json, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run import call, seed, words

reg = json.loads((HERE / "registration.json").read_text())
design, rid, rnd = reg["design"], "calibration-" + "0" * 52, "calibration-full"
d = HERE / "runs" / "calibration-full"
(d / "results").mkdir(parents=True, exist_ok=True)
models = {}
for model in design["models"]:
    part = d / f"{model.replace(':', '-').replace('/', '-')}.jsonl"
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
                    o = call("/api/generate", {"model": model, "prompt": design["prompt"].replace("{list}", " ".join(ws)), "stream": False,
                                               "think": False, "options": {**design["options"], "seed": s % 2**31}})
                    out.write(json.dumps({"condition": c["name"], "n": n, "k": k, "seed": s, "answer": o.get("response", ""),
                                          "prompt_tokens": o.get("prompt_eval_count"), "chars": sum(map(len, ws)),
                                          "seconds": round(time.time() - t, 3)}) + "\n")
    rows = sorted(map(json.loads, part.read_text().splitlines()), key=lambda r: (r["condition"], r["n"], r["k"]))
    models[model] = {"rows": [{k: r[k] for k in ("condition", "n", "k", "seed", "answer", "prompt_tokens", "chars")} for r in rows]}
    print(model, len(rows), flush=True)
shutil.copy(HERE / "registration.json", d)
(d / "registration.lock.json").write_text(json.dumps({"registration": rid}))
(d / "results" / "data.json").write_text(json.dumps({"registration": rid, "models": models}))
if (HERE / "analyze.py").exists():
    a = subprocess.run([sys.executable, "-I", str(HERE / "analyze.py"), str(d)], capture_output=True, text=True)
    (HERE / "calibration" / "full-analysis.json").write_text(a.stdout or a.stderr)
    print(a.stdout[-3000:] or a.stderr[-3000:])
