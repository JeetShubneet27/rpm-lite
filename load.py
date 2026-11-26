import json, os, sys, numpy as np
from tm.families import fam_of, cfg_of
from significance import mcnemar_exact
R = sys.argv[1]
FILES = "calib sched probe fix_val fix_sel rel_val rel_sel review_val review_sel rpm_val rpm_sel pitome_val pitome_sel".split()
rows = []
for f in FILES:
    p = f"{R}/{f}.jsonl"
    if os.path.exists(p):
        for l in open(p):
            if l.strip():
                try: rows.append(json.loads(l))
                except Exception: pass
val, sel, base = {}, {}, {}
for r in rows:
    if "acc1" not in r or r.get("img_size", 224) != 224 or (r.get("probe") and "zeroshot" in r["probe"]): continue
    r.setdefault("split", "val"); r.setdefault("schedule", "const")
    if r.get("tag", "") not in ("", "pc-check"): continue
    if r["method"] == "none":
        if r["split"] == "val": base[r["model"]] = r
        continue
    k = fam_of(r)
    if not k or r.get("target_gflops") is None: continue
    b = r["target_ratio"] if r.get("target_ratio") else r["target_gflops"]
    d = val if r["split"] == "val" else (sel if r["split"] == "sel10s1" else None)
    if d is None: continue
    d.setdefault((r["model"], b, k), {})[cfg_of(r)] = r
def chosen(m, b, k):
    v, s = val.get((m, b, k), {}), sel.get((m, b, k), {})
    c = [x for x in v if x in s]
    return v[max(c, key=lambda x: s[x]["acc1"])] if c else None
def pv(a, b):
    if a is b: return 1.0
    pp = lambda p: p if os.path.exists(p) else os.path.join(R, "preds", os.path.basename(p))
    x, y = np.load(pp(a["preds"])), np.load(pp(b["preds"]))
    return mcnemar_exact(int((x & ~y).sum()), int((~x & y).sum()))
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); s = set()
    for k, i in enumerate(o):
        if ps[i] > 0.05 / (len(ps) - k): break
        s.add(i)
    return s
settings = sorted({(m, b) for (m, b, k) in val if k == "rpmlnw"})
fm = lambda m: "dinov2" in m or "clip" in m
BL = ("tome", "sadtm", "pitome", "mctf")
