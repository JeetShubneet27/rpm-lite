import sys, json, numpy as np, collections, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, sys.path[0]); exec(open(sys.path[0] + "/load.py").read())
OUT = sys.argv[2]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 300})
C = {"RPM-lite": "#2a78d6", "ToMe": "#eb6834", "MCTF": "#1baf7a", "AdaMerge": "#eda100", "Probe-guided": "#e87ba4",
     "RPM-dense": "#6250d6", "weighted sum": "#8a8984"}
MK = {"RPM-lite": "o", "ToMe": "s", "MCTF": "^", "AdaMerge": "v", "Probe-guided": "D", "RPM-dense": "P", "weighted sum": "x"}
core = [s for s in settings if chosen(*s, "rpmf2")]
B4 = ("tome", "sadtm", "pitome", "mctf")
def selacc(m, b, k):
    v, s = val.get((m, b, k), {}), sel.get((m, b, k), {}); c = [x for x in v if x in s]
    return max(s[x]["acc1"] for x in c) if c else None
def best_sel(m, b):
    c = [(k, chosen(m, b, k)) for k in B4 if chosen(m, b, k)]
    return max(c, key=lambda t: selacc(m, b, t[0]))[1]
NAME = {"deit_tiny_patch16_224": "DeiT-T", "deit_small_patch16_224": "DeiT-S", "deit_base_patch16_224": "DeiT-B",
        "deit3_small_patch16_224.fb_in1k": "DeiT-III-S", "deit3_base_patch16_224.fb_in1k": "DeiT-III-B",
        "vit_small_patch14_dinov2.lvd142m": "DINOv2-S", "vit_small_patch14_reg4_dinov2.lvd142m": "DINOv2-S+reg",
        "vit_base_patch14_dinov2.lvd142m": "DINOv2-B", "vit_base_patch14_reg4_dinov2.lvd142m": "DINOv2-B+reg",
        "vit_large_patch14_dinov2.lvd142m": "DINOv2-L", "vit_large_patch14_reg4_dinov2.lvd142m": "DINOv2-L+reg",
        "vit_base_patch16_clip_224.laion2b": "CLIP-B"}
# ---- G1 per-setting gains (forest) ------------------------------------------------------
frows = []
for m in NAME:
    for b in sorted({bb for mm, bb in core if mm == m}, reverse=True):
        o, h, ref = chosen(m, b, "rpmlnw"), chosen(m, b, "hybrid_lite"), best_sel(m, b)
        lab = f"{NAME[m]} {int(round(b*100))}%" if fm(m) else f"{NAME[m]} {b:g}G"
        frows.append((lab, o["acc1"] - ref["acc1"], pv(o, ref), h["acc1"] - ref["acc1"], pv(h, ref), fm(m)))
fig, ax = plt.subplots(figsize=(3.5, 6.4), constrained_layout=True)
y = np.arange(len(frows))[::-1]
for yy, r in zip(y, frows):
    for dx, val_, p, lab in ((0.18, r[1], r[2], "RPM-lite"), (-0.18, r[3], r[4], "Probe-guided")):
        ax.plot([val_], [yy + dx], marker=MK[lab], ms=4, color=C[lab], mfc=C[lab] if p < .05 else "white", mew=1.0, ls="none")
ax.axvline(0, color="#52514e", lw=0.7)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in frows], fontsize=6)
ax.set_xscale("symlog", linthresh=1); ax.set_xticks([-1, 0, 1, 3, 10]); ax.set_xticklabels(["−1", "0", "1", "3", "10"])
ax.set_xlabel("gain over the best baseline (pp)")
sep = sum(1 for r in frows if not r[5]); ax.axhline(y[sep - 1] - 0.5, color="#c9c8c2", lw=0.6)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker=MK[k], color=C[k], ls="none", ms=4, label=k) for k in ("RPM-lite", "Probe-guided")] +
          [Line2D([], [], marker="o", color="#52514e", mfc="white", ls="none", ms=4, label="not significant (McNemar)")],
          fontsize=6, loc="lower right")
ax.grid(axis="y", visible=False)
fig.savefig(f"{OUT}/fig_gains.pdf"); fig.savefig(f"{OUT}/fig_gains.png"); plt.close(fig)
# ---- G2 severity -------------------------------------------------------------------------
def level(m, b):
    if fm(m): return {0.75: 0, 0.6: 1, 0.45: 2}[b]
    bs = sorted({bb for mm, bb in core if mm == m}, reverse=True)
    return 0 if b == bs[0] else (2 if b == bs[-1] else 1)
g = collections.defaultdict(lambda: collections.defaultdict(list))
for m, b in core:
    o = chosen(m, b, "rpmlnw")
    for lab, ref in (("best baseline", best_sel(m, b)), ("ToMe", chosen(m, b, "tome")), ("weighted sum", chosen(m, b, "rpmsum")),
                     ("MCTF", chosen(m, b, "mctf"))):
        if ref: g[lab][level(m, b)].append(o["acc1"] - ref["acc1"])
fig, ax = plt.subplots(figsize=(3.5, 2.4), constrained_layout=True)
cols = {"best baseline": "#2a78d6", "ToMe": C["ToMe"], "MCTF": C["MCTF"], "weighted sum": C["weighted sum"]}
w = 0.2
for i, lab in enumerate(cols):
    med = [np.median(g[lab][k]) for k in range(3)]; q1 = [np.quantile(g[lab][k], .25) for k in range(3)]; q3 = [np.quantile(g[lab][k], .75) for k in range(3)]
    x = np.arange(3) + (i - 1.5) * w
    ax.bar(x, med, w * 0.9, color=cols[lab], label=f"RPM-lite − {lab}")
    ax.errorbar(x, med, yerr=[np.array(med) - q1, np.array(q3) - med], fmt="none", ecolor="#52514e", lw=0.7, capsize=1.5)
ax.set_xticks(range(3)); ax.set_xticklabels(["mildest budget", "middle", "most aggressive"])
ax.set_ylabel("accuracy gain (pp)\nmedian and IQR over settings"); ax.legend(fontsize=6, loc="upper left")
ax.grid(axis="x", visible=False)
fig.savefig(f"{OUT}/fig_severity.pdf"); fig.savefig(f"{OUT}/fig_severity.png"); plt.close(fig)
print({lab: [round(float(np.median(g[lab][k])), 2) for k in range(3)] for lab in cols}, {k: len(g["ToMe"][k]) for k in range(3)})
# ---- G3 segmentation ---------------------------------------------------------------------
seg = collections.defaultdict(dict)
for l in open(f"{R}/seg_ade20k.jsonl"):
    o = json.loads(l)
    if o.get("img", 448) == 448: seg[o["model"]][(o["method"], o["ratio"])] = o["miou"]
SM = {"tome": "ToMe", "mctf": "MCTF", "adamerge": "AdaMerge", "rpmlnw": "RPM-lite", "rpmd": "RPM-dense"}
mods = [m for m in ("vit_small_patch14_dinov2.lvd142m", "vit_base_patch14_dinov2.lvd142m", "vit_large_patch14_dinov2.lvd142m") if m in seg]
fig, ax = plt.subplots(1, len(mods), figsize=(7.2, 2.3), constrained_layout=True, sharey=False)
for a, m in zip(ax, mods):
    for k, lab in SM.items():
        p = sorted((r, v) for (mm, r), v in seg[m].items() if mm == k)
        if p: a.plot([x for x, _ in p], [v for _, v in p], marker=MK[lab], ms=3.5, color=C[lab], lw=1.4 if lab in ("RPM-lite", "RPM-dense") else 1.0, label=lab)
    if ("none", 1.0) in seg[m]: a.axhline(seg[m][("none", 1.0)], color="#52514e", ls=":", lw=0.6)
    a.set_title(NAME[m], fontsize=8, loc="left"); a.invert_xaxis(); a.set_xticks([0.75, 0.6, 0.45]); a.set_xticklabels(["75%", "60%", "45%"])
ax[0].set_ylabel("ADE20K mIoU (%)"); fig.supxlabel("FLOPs relative to the unmerged backbone", fontsize=8)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="outside upper center", ncol=5, fontsize=7)
fig.savefig(f"{OUT}/fig_seg.pdf"); fig.savefig(f"{OUT}/fig_seg.png"); plt.close(fig)
# ---- G4 robustness scatter ----------------------------------------------------------------
sel2 = {}
for r in rows:
    if r.get("split") != "sel10s2" or "acc1" not in r or r.get("img_size", 224) != 224: continue
    r.setdefault("schedule", "const"); k = fam_of(r)
    if not k or r.get("target_gflops") is None: continue
    b = r["target_ratio"] if r.get("target_ratio") else r["target_gflops"]
    sel2.setdefault((r["model"], b, k), {})[cfg_of(r)] = r
def chosen2(m, b, k):
    v, s2 = val.get((m, b, k), {}), sel2.get((m, b, k), {}); c = [x for x in v if x in s2]
    return v[max(c, key=lambda x: s2[x]["acc1"])] if c else None
S2 = sorted({(m, b) for (m, b, k) in sel2})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
for k, lab in (("tome", "ToMe"), ("mctf", "MCTF"), ("rpmlnw", "RPM-lite"), ("hybrid_lite", "Probe-guided")):
    d = [(chosen(m, b, k)["acc1"], chosen2(m, b, k)["acc1"]) for m, b in S2 if chosen(m, b, k) and chosen2(m, b, k)]
    ch = [c2 - c1 for c1, c2 in d]
    ax[0].plot(range(len(ch)), sorted(ch), marker=MK[lab], ms=3.5, color=C[lab], lw=1.0, label=lab)
gain = []
for m, b in S2:
    for ch_, nm in ((chosen, 1), (chosen2, 2)):
        o = ch_(m, b, "rpmlnw"); refs = [ch_(m, b, k) for k in ("tome", "mctf") if ch_(m, b, k)]
        gain.append((nm, o["acc1"] - max(r["acc1"] for r in refs)))
g1 = [v for n, v in gain if n == 1]; g2 = [v for n, v in gain if n == 2]
ax[1].plot(g1, g2, "o", ms=4, color=C["RPM-lite"])
lim = [min(g1 + g2) - 0.3, max(g1 + g2) + 0.3]; ax[1].plot(lim, lim, color="#8a8984", lw=0.7, ls="--")
ax[1].set_xscale("symlog", linthresh=1); ax[1].set_yscale("symlog", linthresh=1)
for f_ in (ax[1].set_xticks, ax[1].set_yticks): f_([0, 1, 3, 10])
ax[1].set_xticklabels(["0", "1", "3", "10"]); ax[1].set_yticklabels(["0", "1", "3", "10"])
ax[0].axhline(0, color="#52514e", lw=0.6)
ax[0].set_xlabel("settings (sorted)"); ax[0].set_ylabel("validation accuracy change (pp)\nwhen selecting on image set 2")
ax[0].set_title("(a) effect of the selection images", fontsize=8, loc="left"); ax[0].legend(fontsize=6)
ax[1].set_xlabel("gain over best of ToMe/MCTF, set 1 (pp)"); ax[1].set_ylabel("gain, set 2 (pp)")
ax[1].set_title("(b) RPM-lite's gain under both selections", fontsize=8, loc="left")
fig.savefig(f"{OUT}/fig_robust.pdf"); fig.savefig(f"{OUT}/fig_robust.png"); plt.close(fig)
