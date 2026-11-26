import sys, numpy as np, pickle, collections
sys.path.insert(0, sys.path[0]); exec(open(sys.path[0] + "/load.py").read())
core = {s for s in settings if chosen(*s, "rpmf2")}
print("sadtm_pub coverage:", sum(1 for s in core if chosen(*s, "sadtm_pub")), "of", len(core))
for k in ("rpm_wo", "rpm_all", "rpmf", "rpmf2"):
    d = []
    for m, b in core:
        x, y = chosen(m, b, k), chosen(m, b, "rpm_all")
        if x and y and k != "rpm_all": d.append((x["acc1"] - y["acc1"], pv(x, y)))
    if d: print(k, "vs rpm_all", len(d), sum(p<.05 and v>0 for v,p in d), sum(p>=.05 for v,p in d), sum(p<.05 and v<0 for v,p in d), round(np.mean([v for v,_ in d]),2))
S = pickle.load(open(sys.path[0] + "/spd.pkl", "rb")); spd = S["spd"]
pts = collections.defaultdict(lambda: collections.defaultdict(list))
for (m, b, f), (s, a) in spd.items(): pts[m][f].append((s, a, b))
def interp(P, x):
    P = sorted(P)
    if x < P[0][0] or x > P[-1][0]: return None          # no extrapolation: excluded
    for (x0, y0, _), (x1, y1, _) in zip(P, P[1:]):
        if x0 <= x <= x1: return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else max(y0, y1)
rng = np.random.default_rng(0)
for ours in ("rpmlnw", "hybrid_lite"):
    for refs in (("tome",), ("mctf",), ("tome", "mctf")):
        d, excl, tot = [], 0, 0
        for m in pts:
            for s, a, b in pts[m][ours]:
                if (m, b) not in core: continue
                tot += 1
                v = [interp(pts[m][k], s) for k in refs if pts[m][k]]
                if any(x is None for x in v) or not v: excl += 1; continue
                d.append(a - max(v))
        d = np.array(d); bs = rng.choice(d, (10000, len(d))).mean(1)
        print(f"{ours:11s} vs {'+'.join(refs):10s} total {tot} valid {len(d)} excluded {excl}  better {int((d>.1).sum())} tie(|d|<=0.1) {int((abs(d)<=.1).sum())} worse {int((d<-.1).sum())} mean {d.mean():+.2f} CI [{np.quantile(bs,.025):+.2f},{np.quantile(bs,.975):+.2f}] median {np.median(d):+.2f}")
