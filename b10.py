import sys, numpy as np, collections, json
from scipy.stats import friedmanchisquare, rankdata, wilcoxon, studentized_range
sys.path.insert(0, sys.path[0]); exec(open(sys.path[0] + "/b6.py").read().split('print("== main table')[0])
def ada_sel(m, b): return ada(m, b)
METH = [("ToMe", lambda m, b: chosen(m, b, "tome")), ("SAD-TM", lambda m, b: chosen(m, b, "sadtm")),
        ("PiToMe", lambda m, b: chosen(m, b, "pitome")), ("MCTF", lambda m, b: chosen(m, b, "mctf")),
        ("AdaMerge", ada_sel), ("RPM-lite", lambda m, b: chosen(m, b, "rpmlnw")),
        ("Probe-guided", lambda m, b: chosen(m, b, "hybrid_lite"))]
out = {}
for nm, S in (("core38", core), ("all", settings)):
    rows = []
    for m, b in S:
        r = [f(m, b) for _, f in METH]
        if all(r): rows.append([x["acc1"] for x in r])
    X = np.array(rows); n, k = X.shape
    RK = np.array([rankdata(-x) for x in X]); mr = RK.mean(0)
    st, p = friedmanchisquare(*X.T)
    q = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    print(f"{nm}: n={n} k={k} Friedman chi2={st:.1f} p={p:.1e}  Nemenyi CD={cd:.2f}")
    print("   mean ranks:", {METH[i][0]: round(float(mr[i]), 2) for i in np.argsort(mr)})
    out[nm] = dict(n=n, cd=float(cd), ranks={METH[i][0]: float(mr[i]) for i in range(k)}, p=float(p), chi2=float(st))
json.dump(out, open(sys.path[0] + "/cd.json", "w"))
# effect sizes by regime
def level(m, b):
    if fm(m): return {0.75: "mild", 0.6: "moderate", 0.45: "aggressive"}[b]
    bs = sorted({bb for mm, bb in core if mm == m}, reverse=True)
    return "mild" if b == bs[0] else ("aggressive" if b == bs[-1] else "moderate")
def rb(d):                                       # matched-pairs rank-biserial correlation
    d = np.asarray([x for x in d if x != 0]); r = rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()
import os
def counts(a, b):
    pp = lambda p: p if os.path.exists(p) else os.path.join(R_, "preds", os.path.basename(p))
    x, y = np.load(pp(a["preds"])), np.load(pp(b["preds"]))
    return int((x & ~y).sum()), int((~x & y).sum())
R_ = sys.argv[1]
print("== effect sizes (RPM-lite vs reference)")
for ref in ("BEST", "tome", "mctf"):
    for lv in ("mild", "moderate", "aggressive", "all"):
        d, n10, n01 = [], 0, 0
        ors = []
        for m, b in core:
            if lv != "all" and level(m, b) != lv: continue
            o = chosen(m, b, "rpmlnw"); y = best_bl(m, b, B4, "sel")[1] if ref == "BEST" else chosen(m, b, ref)
            d.append(o["acc1"] - y["acc1"]); a10, a01 = counts(o, y); n10 += a10; n01 += a01; ors.append((a10 + .5) / (a01 + .5))
        p = wilcoxon(d).pvalue if len(d) > 5 and any(d) else float("nan")
        print(f"  vs {ref:5s} {lv:10s} n={len(d):2d} mean {np.mean(d):+.2f} rank-biserial r={rb(d):+.2f} Wilcoxon p={p:.2g}  "
              f"pooled OR n10/n01={n10/max(n01,1):.2f} ({n10}/{n01})  median per-setting OR {np.median(ors):.2f}")
