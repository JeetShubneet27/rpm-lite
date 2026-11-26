import sys, numpy as np, collections, json
from scipy.stats import wilcoxon
sys.path.insert(0, sys.path[0]); exec(open(sys.path[0] + "/load.py").read())
core = [s for s in settings if chosen(*s, "rpmf2")]
def selacc(m, b, k):
    v, s = val.get((m, b, k), {}), sel.get((m, b, k), {})
    c = [x for x in v if x in s]
    return max(s[x]["acc1"] for x in c) if c else None
def ada(m, b):            # AdaMerge variant chosen on the selection images
    c = [(selacc(m, b, k), k) for k in ("adamerge", "adamerge2") if chosen(m, b, k)]
    return chosen(m, b, max(c)[1]) if c else None
def best_bl(m, b, fams, rule):
    c = [(k, chosen(m, b, k)) for k in fams if chosen(m, b, k)]
    if rule == "val": return max(c, key=lambda t: t[1]["acc1"])
    return max(c, key=lambda t: selacc(m, b, t[0]))
B4 = ("tome", "sadtm", "pitome", "mctf")
out = {}
def compare(ours, S, rule, fams=B4):
    comps = []
    for m, b in S:
        o = chosen(m, b, ours)
        if not o: continue
        k, ref = best_bl(m, b, fams, rule)
        comps.append((m, b, o["acc1"] - ref["acc1"], pv(o, ref), k))
    d = np.array([c[2] for c in comps]); p = [c[3] for c in comps]; hs = holm(p)
    w = sum(c[3] < .05 and c[2] > 0 for c in comps); l = sum(c[3] < .05 and c[2] < 0 for c in comps)
    hw = sum(i in hs and comps[i][2] > 0 for i in range(len(comps))); hl = sum(i in hs and comps[i][2] < 0 for i in range(len(comps)))
    F = [c for c in comps if fm(c[0])]; Sv = [c for c in comps if not fm(c[0])]
    wtl = lambda C: f"{sum(c[3]<.05 and c[2]>0 for c in C)}/{sum(c[3]>=.05 for c in C)}/{sum(c[3]<.05 and c[2]<0 for c in C)}"
    return dict(n=len(comps), F=wtl(F), S=wtl(Sv), all=f"{w}/{len(comps)-w-l}/{l}", holm=f"{hw}/{hl}", mean=round(d.mean(), 2),
                median=round(float(np.median(d)), 2), d=d, comps=comps, picked=collections.Counter(c[4] for c in comps))
print("== main table, best baseline chosen on SELECTION images vs on validation")
for o in ("rpmf2", "rpmlnw", "hybrid_lite", "rpmsum"):
    for rule in ("sel", "val"):
        r = compare(o, core, rule)
        print(f"  {o:11s} [{rule}] F {r['F']} S {r['S']} all {r['all']} Holm {r['holm']} mean {r['mean']:+.2f} median {r['median']:+.2f}  picked {dict(r['picked'])}")
        out[(o, rule)] = r
print("== per-baseline head-to-head (core 38)")
for o in ("rpmlnw", "hybrid_lite"):
    for k in B4 + ("ada",):
        comps = []
        for m, b in core:
            x = chosen(m, b, o); y = ada(m, b) if k == "ada" else chosen(m, b, k)
            if x and y: comps.append((x["acc1"] - y["acc1"], pv(x, y)))
        d = np.array([c[0] for c in comps]); hs = holm([c[1] for c in comps])
        w = sum(c[1] < .05 and c[0] > 0 for c in comps); l = sum(c[1] < .05 and c[0] < 0 for c in comps)
        hw = sum(i in hs and comps[i][0] > 0 for i in range(len(comps))); hl = sum(i in hs and comps[i][0] < 0 for i in range(len(comps)))
        print(f"  {o:11s} vs {k:7s} n={len(d)} {w}/{len(d)-w-l}/{l} Holm {hw}/{hl} mean {d.mean():+.2f} median {np.median(d):+.2f}")
ad = collections.Counter(max((selacc(m, b, k), k) for k in ("adamerge", "adamerge2") if chosen(m, b, k))[1] for m, b in core if ada(m, b))
print("  AdaMerge variant chosen on selection images:", dict(ad))
print("== clustered / hierarchical bootstrap of the mean gain (best baseline on selection images)")
rng = np.random.default_rng(0)
fam = lambda m: "clip" if "clip" in m else ("dinov2" if "dinov2" in m else "deit")
for o in ("rpmlnw", "hybrid_lite"):
    r = out[(o, "sel")]
    by = collections.defaultdict(list)
    for c in r["comps"]: by[c[0]].append(c[2])
    models = list(by)
    def hb():
        ms = rng.choice(len(models), len(models))
        vals = np.concatenate([rng.choice(by[models[i]], len(by[models[i]])) for i in ms])
        return vals.mean()
    bs = np.array([hb() for _ in range(10000)])
    mm = np.array([np.mean(by[m]) for m in models])
    print(f"  {o}: settings n={len(r['d'])} models={len(models)} mean {r['d'].mean():+.2f} "
          f"setting-bootstrap CI ... hierarchical CI [{np.quantile(bs,.025):+.2f},{np.quantile(bs,.975):+.2f}] "
          f"model-level: {int((mm>0).sum())}/{len(mm)} models positive, Wilcoxon over model means p={wilcoxon(mm).pvalue:.3g}, "
          f"setting-level Wilcoxon p={wilcoxon(r['d']).pvalue:.2g}")
    fb = collections.defaultdict(list)
    for c in r["comps"]: fb[fam(c[0])].append(c[2])
    print("    by family:", {f: (len(v), round(float(np.mean(v)), 2)) for f, v in fb.items()})
print("== severity: gain by budget level")
def level(m, b):
    if fm(m) or "augreg" in m: return {0.75: "mild", 0.6: "mid", 0.45: "aggr"}.get(b)
    bs = sorted({bb for mm, bb in core if mm == m}, reverse=True)
    return "mild" if b == bs[0] else ("aggr" if b == bs[-1] else "mid")
for o, ref in (("rpmlnw", "rpmsum"), ("rpmlnw", "BEST"), ("hybrid_lite", "BEST"), ("rpmlnw", "tome")):
    g = collections.defaultdict(list)
    for m, b in core:
        x = chosen(m, b, o)
        y = best_bl(m, b, B4, "sel")[1] if ref == "BEST" else chosen(m, b, ref)
        if x and y and level(m, b): g[level(m, b)].append(x["acc1"] - y["acc1"])
    print(f"  {o} - {ref}: " + "  ".join(f"{k} n={len(g[k])} mean {np.mean(g[k]):+.2f} median {np.median(g[k]):+.2f}" for k in ("mild", "mid", "aggr")))
print("== SAD-TM: published configuration vs size-weighted variant (supervised)")
for m, b in core:
    if fm(m): continue
    p, s = chosen(m, b, "sadtm_pub"), chosen(m, b, "sadtm")
    t = chosen(m, b, "tome")
    print(f"  {m[:24]:24s} {b:5}  pub {p['acc1'] if p else None}  ours {s['acc1'] if s else None}  tome {t['acc1']}")
