import sys, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D_, OUT, AN = sys.argv[1], sys.argv[2], sys.argv[3]
import pickle; res = pickle.load(open(f"{AN}/rd1.pkl", "rb"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 300})
NAME = {"deit_tiny_patch16_224": "DeiT-T", "deit_small_patch16_224": "DeiT-S", "deit_base_patch16_224": "DeiT-B",
        "deit3_small_patch16_224.fb_in1k": "DeiT-III-S", "deit3_base_patch16_224.fb_in1k": "DeiT-III-B",
        "vit_small_patch16_224.augreg_in21k_ft_in1k": "AugReg-S", "vit_base_patch16_224.augreg_in21k_ft_in1k": "AugReg-B",
        "vit_small_patch14_dinov2.lvd142m": "DINOv2-S", "vit_small_patch14_reg4_dinov2.lvd142m": "DINOv2-S+reg",
        "vit_base_patch14_dinov2.lvd142m": "DINOv2-B", "vit_base_patch14_reg4_dinov2.lvd142m": "DINOv2-B+reg",
        "vit_large_patch14_dinov2.lvd142m": "DINOv2-L", "vit_large_patch14_reg4_dinov2.lvd142m": "DINOv2-L+reg",
        "vit_base_patch16_clip_224.laion2b": "CLIP-B"}
fam = lambda m: "CLIP" if "clip" in m else ("DINOv2" if "dinov2" in m else "supervised")
FC = {"supervised": "#2a78d6", "DINOv2": "#eb6834", "CLIP": "#1baf7a"}
C3 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
# ---------------- fig_theory --------------------------------------------------------------------------------
fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
gs = fig.add_gridspec(2, 3)
m = "vit_base_patch14_dinov2.lvd142m"
z = np.load(f"{D_}/{m}.npz"); X = z["data"]; c = {k: i for i, k in enumerate(z["fields"])}
for j, (tgt, ttl) in enumerate((("act_ideal", "(a) idealised merge (DINOv2-B)"), ("act_sum", "(b) implemented merge (DINOv2-B)"))):
    a = fig.add_subplot(gs[0, j])
    x, y = np.log10(X[:, c["pred_sum"]] + 1e-9), np.log10(X[:, c[tgt]] + 1e-9)
    hb = a.hexbin(x, y, gridsize=55, bins="log", cmap="Blues", mincnt=1, linewidths=0)
    lo, hi = np.quantile(np.r_[x, y], [0.001, 0.999]); a.plot([lo, hi], [lo, hi], color="#52514e", lw=0.7, ls="--")
    a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_xlabel("predicted change, Prop. 1 (log$_{10}$)"); a.set_ylabel("measured change (log$_{10}$)")
    a.set_title(f"{ttl}", fontsize=7.5, loc="left"); a.grid(False)
# (c) ratio implemented / idealised by depth third, per family
a = fig.add_subplot(gs[0, 2])
for f_, col in FC.items():
    ms = [mm for mm in res if fam(mm) == f_]
    v = np.array([[res[mm]["by_depth"][d]["ratio_act_ideal"] for d in ("early", "middle", "late")] for mm in ms])
    for row in v: a.plot([0, 1, 2], row, color=col, lw=0.8, alpha=0.5)
    a.plot([0, 1, 2], np.median(v, 0), color=col, lw=2, marker="o", ms=3.5, label=f_)
a.axhline(1, color="#52514e", lw=0.6, ls="--"); a.set_xticks([0, 1, 2]); a.set_xticklabels(["early", "middle", "late"])
a.set_ylabel("measured / idealised change\n(median over merges)"); a.legend(fontsize=6)
a.set_title("(c) LayerNorm + MLP effect", fontsize=7.5, loc="left")
# (d) per-model Spearman
a = fig.add_subplot(gs[1, :])
ms = list(NAME); x = np.arange(len(ms)); w = 0.2
for i, (key, lab) in enumerate((("pred_sum~act_ideal", "prediction vs idealised merge (expansion only)"),
                                ("pred_sum~act_sum", "prediction vs implemented merge"),
                                ("pred_proj~act_proj", "prediction vs implemented merge, after $W_o$"),
                                ("lite~act_sum", "RPM-lite from block $l$ vs implemented merge"))):
    mu = [res[mm][key][0] for mm in ms]; lo = [res[mm][key][1] for mm in ms]; hi = [res[mm][key][2] for mm in ms]
    a.bar(x + (i - 1.5) * w, mu, w * 0.9, color=C3[i], label=lab)
    a.errorbar(x + (i - 1.5) * w, mu, yerr=[np.array(mu) - lo, np.array(hi) - mu], fmt="none", ecolor="#52514e", lw=0.5)
a.set_xticks(x); a.set_xticklabels([NAME[mm] for mm in ms], rotation=30, ha="right", fontsize=6.5)
a.set_ylabel("Spearman correlation\n(within block, mean over images)")
a.set_ylim(0, 1.3); a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
a.legend(fontsize=6, ncol=2, loc="upper center", handlelength=1.2); a.grid(axis="x", visible=False)
a.set_title("(d) ranking of the measured change of the next attention layer", fontsize=7.5, loc="left")
fig.savefig(f"{OUT}/fig_theory.pdf"); fig.savefig(f"{OUT}/fig_theory.png"); plt.close(fig)
# ---------------- fig_assump --------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.5), constrained_layout=True)
for mm, r in res.items():
    L = r["depth"]; ls = "--" if "reg4" in mm else "-"
    for a, key in zip(ax[:3], ("level_cv", "pr", "rho_cls_level")):
        b = r[key + "/blk"]; xs = [(int(k) + 0.5) / L for k in b]; a.plot(xs, list(b.values()), ls, color=FC[fam(mm)], lw=1.0, alpha=0.85)
for a, t in zip(ax[:3], ("(a) A1: CV of the pair's\nlevel across queries", "(b) A2: query isotropy\n(particip. ratio / d)",
                         "(c) A3: [CLS] vs. all-\nquery level (Spearman)")):
    a.set_title(t, fontsize=7, loc="left"); a.set_xlim(0, 1); a.set_xlabel("relative depth")
ax[0].set_yscale("log"); from matplotlib.ticker import ScalarFormatter, NullFormatter
ax[0].yaxis.set_major_formatter(ScalarFormatter()); ax[0].yaxis.set_minor_formatter(NullFormatter()); ax[0].set_yticks([0.3, 1, 3])
for mm, r in res.items():
    ax[3].plot(r["outlier_mass"], r["rho_cls_level"], "o", ms=4, color=FC[fam(mm)], mfc="white" if "reg4" in mm else FC[fam(mm)])
ax[3].set_xlabel("[CLS] attn. on top-2%\nnorm tokens"); ax[3].set_ylabel("mean A3 agreement")
ax[3].set_title("(d) outliers vs. A3", fontsize=7, loc="left")
from matplotlib.lines import Line2D
ax[1].legend(handles=[Line2D([], [], color=c_, lw=1.5, label=f_) for f_, c_ in FC.items()] + [Line2D([], [], color="#52514e", ls="--", label="registers")], fontsize=6)
fig.savefig(f"{OUT}/fig_assump.pdf"); fig.savefig(f"{OUT}/fig_assump.png"); plt.close(fig)
# ---------------- fig_damage2: criteria vs output damage, random partners -----------------------------------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True, sharey=True)
crit = (("tome", "ToMe"), ("attn", "[CLS] attn."), ("exact", "exact RPM"), ("lite", "RPM-lite"))
cc = {"ToMe": "#eb6834", "[CLS] attn.": "#6250d6", "exact RPM": "#1baf7a", "RPM-lite": "#2a78d6"}
for a, (tgt, ttl) in zip(ax, (("d_cos", "(a) feature damage, 1 − cos"), ("d_kl", "(b) task damage, KL of class distribution"))):
    for i, (k, lab) in enumerate(crit):
        mu = [res[mm][f"rand/{k}/{tgt}"][0] for mm in ms]; lo = [res[mm][f"rand/{k}/{tgt}"][1] for mm in ms]; hi = [res[mm][f"rand/{k}/{tgt}"][2] for mm in ms]
        a.errorbar(x + (i - 1.5) * 0.18, mu, yerr=[np.array(mu) - lo, np.array(hi) - mu], fmt="o", ms=3, color=cc[lab], lw=0.7, label=lab)
    a.set_xticks(x); a.set_xticklabels([NAME[mm] for mm in ms], rotation=35, ha="right", fontsize=6)
    a.set_title(ttl, fontsize=7.5, loc="left"); a.axhline(0, color="#52514e", lw=0.5)
ax[0].set_ylabel("Spearman with measured damage\n(random partners, 95% CI)"); ax[0].legend(fontsize=6, ncol=2)
fig.savefig(f"{OUT}/fig_damage2.pdf"); fig.savefig(f"{OUT}/fig_damage2.png"); plt.close(fig)
print("ok")
