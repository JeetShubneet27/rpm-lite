import sys, json, numpy as np, collections, pickle
sys.path.insert(0, sys.path[0]); exec(open(sys.path[0] + "/load.py").read())
L = lambda f: [json.loads(l) for l in open(f"{R}/{f}") if l.strip()]
bud = lambda r: r.get("target_ratio") or r["target_gflops"]
acc = collections.defaultdict(list); mism = collections.Counter()
FAMS = ("tome", "mctf", "hybrid", "rpmlnw", "hybrid_lite", "adamerge2", "rpmd")
for f in ("speed_sel.jsonl", "speed_sel_fix.jsonl", "speed_sel_hl.jsonl", "speed_sel_sat.jsonl"):
    for r in L(f):
        b = bud(r)
        for fam in FAMS:
            lab = f"{fam} (compiled)"
            if lab not in r["results"] or fam not in r["configs"]: continue
            o = chosen(r["model"], b, "rpmlnw" if fam == "rpmd" else fam)
            if o is None: continue
            c = r["configs"][fam]
            if (c["tau"], c["schedule"]) != (o["tau"], o.get("schedule", "const")): mism[fam] += 1; continue
            acc[(r["model"], b, fam)].append(r["results"][lab]["x_unmerged"])
spd = {}
for (m, b, f), xs in acc.items():
    o = chosen(m, b, f) if f != "rpmd" else None
    if f == "rpmd":
        rl = chosen(m, b, "rpmlnw")
        mt = [r for (mm, bb, k), v in val.items() if k == "rpmd" and mm == m and bb == b for r in v.values() if r["tau"] == rl["tau"] and r["schedule"] == rl["schedule"]]
        if not mt: continue
        o = mt[0]
    spd[(m, b, f)] = (float(np.median(xs)), o["acc1"])
print("mismatch", dict(mism), collections.Counter(k[2] for k in spd))
core = {s for s in settings if chosen(*s, "rpmf2")}
pts = collections.defaultdict(lambda: collections.defaultdict(list))
for (m, b, f), (s, a) in spd.items(): pts[m][f].append((s, a, b))
def interp(P, x):
    P = sorted(P)
    if x < P[0][0]: return P[0][1]
    if x > P[-1][0]: return None
    for (x0, y0, _), (x1, y1, _) in zip(P, P[1:]):
        if x0 <= x <= x1: return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else max(y0, y1)
for S, nm in ((core, "core"), (None, "all")):
    print("==", nm)
    for ours in ("rpmlnw", "hybrid_lite", "rpmd"):
        for refs in (("tome",), ("mctf",), ("adamerge2",), ("tome", "mctf"), ("tome", "mctf", "adamerge2")):
            d = []
            for m in pts:
                for s, a, b in pts[m][ours]:
                    if S is not None and (m, b) not in S: continue
                    v = [interp(pts[m][k], s) for k in refs if pts[m][k]]; v = [x for x in v if x is not None]
                    if v: d.append(a - max(v))
            d = np.array(d)
            if len(d): print(f"  {ours:11s} vs {'+'.join(refs):20s} n={len(d):2d} better {int((d > .1).sum()):2d} worse {int((d < -.1).sum()):2d} mean {d.mean():+.2f} median {np.median(d):+.2f}")
for f in ("rpmlnw", "hybrid_lite", "mctf", "adamerge2", "rpmd"):
    r_ = [spd[(m, b, f)][0] / spd[(m, b, "tome")][0] for (m, b, ff) in spd if ff == f and (m, b, "tome") in spd]
    r2 = [spd[(m, b, f)][0] / spd[(m, b, "rpmlnw")][0] for (m, b, ff) in spd if ff == f and (m, b, "rpmlnw") in spd]
    print(f"  speed {f}/tome median {np.median(r_):.3f} (n={len(r_)})  /rpmlnw {np.median(r2):.3f}")
pickle.dump(dict(spd=spd, base={m: base[m]["acc1"] for m in base}), open(sys.path[0] + "/spd.pkl", "wb"))
for m in ("vit_base_patch14_dinov2.lvd142m", "vit_base_patch16_clip_224.laion2b", "deit_base_patch16_224"):
    print(m, {f: len(pts[m][f]) for f in pts[m]})
