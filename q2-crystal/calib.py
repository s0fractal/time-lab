"""Calibration probes of the crystal clock (seeds from the namespace calibration:…, never a registration's)."""
import hashlib, json, sys, time
sys.path.insert(0, __import__("os").path.dirname(__file__))
from run import clock

LAB = {"N": 32, "d": 2, "dt": 1.0, "S": 4, "gamma": 0.5, "kappa": 1.0, "g_lat": 1.0, "g_pin": 0.5, "m_pin": 2,
       "j_scale": 0.1, "crystal": 1}


def seed(*a):
    return int(hashlib.sha256(":".join(map(str, ("calibration",) + a)).encode()).hexdigest()[:12], 16)


def probe(name, B=256, t_total=400.0, every=4.0, **kw):
    p = {**LAB, **kw}
    t0 = time.time()
    r = clock(p, B, t_total, every, seed(name, json.dumps(p, sort_keys=True)))
    print(f"{name:<34} v={r['v']:9.3f} D={r['D']:9.3f} S={r["entropy_rate"]:9.3f} W={r["work_rate"] if r["work_rate"] is None else round(r["work_rate"], 3)} "
          f"eta={r['eta'] if r['eta'] is None else round(r['eta'], 3)}  {time.time() - t0:5.1f}s", flush=True)
    return {"name": name, "p": p, "B": B, "t_total": t_total, **r}
