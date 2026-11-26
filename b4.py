import sys, numpy as np
from scipy.stats import wilcoxon, binomtest
sys.path.insert(0, sys.path[0]); exec(open(sys.path[0] + "/load.py").read())
core = [s for s in settings if chosen(*s, "rpmf2")]
B4 = ("tome", "sadtm", "pitome", "mctf")
rng = np.random.default_rng(0)
for ours in ("rpmlnw", "hybrid_lite", "rpmf2"):
    for S, nm in ((core, "core"), (settings, "all")):
        d = []
        for m, b in S:
            o = chosen(m, b, ours); refs = [r for r in (chosen(m, b, k) for k in B4) if r]
            if o and refs: d.append(o["acc1"] - max(r["acc1"] for r in refs))
        d = np.array(d); bs = rng.choice(d, (10000, len(d))).mean(1)
        print(f"{ours:11s} {nm} n={len(d)} mean {d.mean():+.2f} CI [{np.quantile(bs,.025):+.2f},{np.quantile(bs,.975):+.2f}] wilcoxon p={wilcoxon(d).pvalue:.1e} sign {int((d>0).sum())}+/{int((d<0).sum())}- p={binomtest(int((d>0).sum()), int((d!=0).sum())).pvalue:.1e}")
NAME = {"deit_tiny_patch16_224": "DeiT-T", "deit_small_patch16_224": "DeiT-S", "deit_base_patch16_224": "DeiT-B",
        "deit3_small_patch16_224.fb_in1k": "DeiT-III-S", "deit3_base_patch16_224.fb_in1k": "DeiT-III-B",
        "vit_small_patch16_224.augreg_in21k_ft_in1k": "AugReg-S", "vit_base_patch16_224.augreg_in21k_ft_in1k": "AugReg-B",
        "vit_small_patch14_dinov2.lvd142m": "DINOv2-S", "vit_small_patch14_reg4_dinov2.lvd142m": "DINOv2-S+reg",
        "vit_base_patch14_dinov2.lvd142m": "DINOv2-B", "vit_base_patch14_reg4_dinov2.lvd142m": "DINOv2-B+reg",
        "vit_large_patch14_dinov2.lvd142m": "DINOv2-L", "vit_large_patch14_reg4_dinov2.lvd142m": "DINOv2-L+reg",
        "vit_base_patch16_clip_224.laion2b": "CLIP-B"}
order = list(NAME)
def selacc(m, b, k):
    v, s_ = val.get((m, b, k), {}), sel.get((m, b, k), {}); c = [x for x in v if x in s_]
    return max(s_[x]["acc1"] for x in c) if c else None
rowsT = []
for m in order:
    bs = sorted({b for mm, b in settings if mm == m}, reverse=True)
    for b in bs:
        cols = [chosen(m, b, "tome"), chosen(m, b, "sadtm_pub"), chosen(m, b, "sadtm"), chosen(m, b, "pitome"), chosen(m, b, "mctf")]
        am = [(selacc(m, b, k), chosen(m, b, k)) for k in ("adamerge", "adamerge2") if chosen(m, b, k)]
        cols.append(max(am, key=lambda t: t[0])[1] if am else None)
        ours = [chosen(m, b, "rpmlnw"), chosen(m, b, "hybrid_lite")]
        cand = [(selacc(m, b, k), chosen(m, b, k)) for k in B4 if chosen(m, b, k)]
        ref = max(cand, key=lambda t: t[0])[1]
        allv = [r["acc1"] for r in cols + ours if r]; top = max(allv)
        def fmt(r, mark=False):
            if r is None: return "--"
            s = f"{r['acc1']:.2f}"
            if round(r["acc1"], 2) == round(top, 2): s = r"\textbf{" + s + "}"
            if mark:
                p = pv(r, ref)
                if p < .05: s += r"$^{\uparrow}$" if r["acc1"] > ref["acc1"] else r"$^{\downarrow}$"
            return s
        bud = f"{int(round(b*100))}\\%" if (fm(m) or "augreg" in m) else f"{b:g}G"
        star = "" if (m, b) in core else "$^{*}$"
        rowsT.append((fm(m), " & ".join([NAME[m], bud + star, f"{base[m]['acc1']:.2f}"] + [fmt(r) for r in cols] + [fmt(r, True) for r in ours]) + r" \\"))
with open(sys.path[0] + "/tab_full.tex", "w") as f:
    f.write("\\begin{tabular}{@{}llr|rrrrrr|rr@{}}\\toprule\nModel & Budget & Unmerged & ToMe & SAD-TM (pub.) & SAD-TM (mod.) & PiToMe & MCTF & AdaMerge & RPM-lite & Probe-guided \\\\\\midrule\n")
    sup = [r for g, r in rowsT if not g]; fo = [r for g, r in rowsT if g]
    f.write("\n".join(sup) + "\n\\midrule\n" + "\n".join(fo) + "\n\\bottomrule\n\\end{tabular}\n")
print(len(rowsT), "rows")
