import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
F, OUT = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 300})
NM = {"deit_small_patch16_224": "DeiT-S", "deit_base_patch16_224": "DeiT-B", "vit_base_patch14_dinov2.lvd142m": "DINOv2-B",
      "vit_small_patch14_reg4_dinov2.lvd142m": "DINOv2-S+reg", "vit_base_patch16_clip_224.laion2b": "CLIP-B"}
CL = dict(zip(NM.values(), ["#2a78d6", "#6250d6", "#eb6834", "#eda100", "#1baf7a"]))
R = [json.loads(l) for l in open(F)]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3), constrained_layout=True)
for r in R:
    if r["ratio"] != 0.45: continue
    nm = NM[r["model"]]; B = r["blocks"]; L = r["depth"]
    x = [(b["block"] + 0.5) / L for b in B]
    lr = [np.log10(np.array(b["t1_sel"]) / np.array(b["t2_sel"])) for b in B]
    ax[0].plot(x, [np.median(v) for v in lr], color=CL[nm], marker="o", ms=2.5, lw=1.2, label=nm)
    ax[0].fill_between(x, [np.quantile(v, .1) for v in lr], [np.quantile(v, .9) for v in lr], color=CL[nm], alpha=0.12, lw=0)
    ax[1].plot(x, [np.median(b["t1_sel"]) / np.median(B[0]["t1_sel"]) for b in B], color=CL[nm], lw=1.2, ls="-")
    ax[1].plot(x, [np.median(b["t2_sel"]) / np.median(B[0]["t2_sel"]) for b in B], color=CL[nm], lw=1.2, ls="--")
    ax[2].plot(x, [b["size_max"] for b in B], color=CL[nm], lw=1.2)
ax[0].set_title("(a) log$_{10}$(attention / value factor)\nmedian and 10--90% range over heads", fontsize=7, loc="left")
ax[1].set_title("(b) factor scale relative to the first\nmerging block (solid: attention, dashed: value)", fontsize=7, loc="left")
ax[2].set_title("(c) largest token size", fontsize=7, loc="left"); ax[1].set_yscale("log"); ax[2].set_yscale("log")
from matplotlib.ticker import ScalarFormatter, NullFormatter
for a in ax[1:]: a.yaxis.set_major_formatter(ScalarFormatter()); a.yaxis.set_minor_formatter(NullFormatter())
ax[1].set_yticks([1, 2, 4, 6]); ax[2].set_yticks([1, 10, 100])
for a in ax: a.set_xlabel("relative depth")
ax[0].legend(fontsize=6, loc="lower left")
fig.savefig(f"{OUT}/fig_factors.pdf"); fig.savefig(f"{OUT}/fig_factors.png"); plt.close(fig)
