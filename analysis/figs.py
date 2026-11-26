import json, pickle, collections, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import rankdata
R, OUT, AN = sys.argv[1], sys.argv[2], sys.argv[3]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 300})
C = {"RPM-lite": "#2a78d6", "ToMe": "#eb6834", "MCTF": "#1baf7a", "AdaMerge": "#eda100", "Probe-guided": "#e87ba4",
     "[CLS] attn.": "#6250d6"}
MK = {"RPM-lite": "o", "ToMe": "s", "MCTF": "^", "AdaMerge": "v", "Probe-guided": "D", "[CLS] attn.": "x"}
# ---------- F1 reliability
rel = {}
for l in open(f"{R}/reliability.jsonl"):
    o = json.loads(l); rel[o["model"]] = o
fam = lambda m: "DeiT / DeiT-III" if "deit" in m else ("CLIP" if "clip" in m else "DINOv2")
FC = {"DeiT / DeiT-III": "#2a78d6", "DINOv2": "#eb6834", "CLIP": "#1baf7a"}
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3), constrained_layout=True)
for m, o in rel.items():
    x = (np.arange(o["depth"]) + 0.5) / o["depth"]; ls = "--" if "reg4" in m else "-"
    for a, key in zip(ax, ("rho_attn_loo", "rho_sim_loo", "outlier_mass")):
        y = np.array(o[key], float) * (100 if key == "outlier_mass" else 1)
        a.plot(x, y, ls, color=FC[fam(m)], lw=1.2, alpha=0.85)
for a, t in zip(ax, ("(a) [CLS] attention vs. importance", "(b) key uniqueness vs. importance", "(c) [CLS] attn. on top-2% norm tokens (%)")):
    a.set_title(t, fontsize=7.5, loc="left"); a.set_xlim(0, 1)
fig.supxlabel("relative depth", fontsize=8); ax[0].set_ylabel("Spearman correlation")
ax[0].axhline(0, color="#52514e", lw=0.6); ax[1].axhline(0, color="#52514e", lw=0.6)
from matplotlib.lines import Line2D
h = [Line2D([], [], color=c, lw=1.5, label=f) for f, c in FC.items()] + [Line2D([], [], color="#52514e", lw=1.2, ls="--", label="with registers")]
ax[2].legend(handles=h, fontsize=6.5, loc="upper left")
fig.savefig(f"{OUT}/fig_reliability.png"); fig.savefig(f"{OUT}/fig_reliability.pdf"); plt.close(fig)
print("reliability: min late rho_sim DINOv2", {m[:22]: round(min(o["rho_sim_loo"][-3:]), 2) for m, o in rel.items() if "dinov2" in m},
      "max outlier", {m[:22]: round(max(o["outlier_mass"]), 2) for m, o in rel.items()})
# ---------- F2 damage vs predicted cost (deciles within block)
NM = {"deit_small_patch16_224": "DeiT-S", "deit3_small_patch16_224.fb_in1k": "DeiT-III-S", "deit_base_patch16_224": "DeiT-B",
      "vit_base_patch14_dinov2.lvd142m": "DINOv2-B", "vit_base_patch14_reg4_dinov2.lvd142m": "DINOv2-B+reg", "vit_base_patch16_clip_224.laion2b": "CLIP-B"}
fig, ax = plt.subplots(1, 6, figsize=(7.2, 1.9), sharey=True, constrained_layout=True)
for a, (m, nm) in zip(ax, NM.items()):
    P = json.load(open(f"{R}/pair_points_v4/{m}.json"))
    for sig, lab in (("tome", "ToMe"), ("attn", "[CLS] attn."), ("lite", "RPM-lite")):
        acc = np.zeros(10); cnt = np.zeros(10)
        for p in P:
            act = np.array(p["actual"]); act = act / max(np.median(act), 1e-12)
            dec = np.minimum((rankdata(p[sig]) - 1) / len(act) * 10, 9.999).astype(int)
            for d in range(10): acc[d] += act[dec == d].mean(); cnt[d] += 1
        a.plot(np.arange(1, 11), acc / cnt, marker=MK[lab], ms=3, lw=1.3, color=C[lab], label=lab)
    a.set_title(nm, fontsize=7.5); a.set_xticks([1, 5, 10])
    a.set_yscale("log")
ax[0].set_ylabel("measured damage\n(relative to block median)"); fig.supxlabel("decile of predicted cost (1 = cheapest 10% of merges)", fontsize=8)
ax[0].legend(fontsize=6, loc="upper left")
fig.savefig(f"{OUT}/fig_damage.png"); fig.savefig(f"{OUT}/fig_damage.pdf"); plt.close(fig)
# ---------- F3 accuracy vs measured speed
S = pickle.load(open(f"{AN}/spd.pkl", "rb")); spd, base = S["spd"], S["base"]
FM = {"tome": "ToMe", "mctf": "MCTF", "adamerge2": "AdaMerge", "rpmlnw": "RPM-lite", "hybrid_lite": "Probe-guided"}
MODELS = {"vit_base_patch14_dinov2.lvd142m": "DINOv2-B", "vit_base_patch16_clip_224.laion2b": "CLIP-B", "deit_base_patch16_224": "DeiT-B"}
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
for a, (m, nm) in zip(ax, MODELS.items()):
    for f, lab in FM.items():
        p = sorted((s, acc) for (mm, b, ff), (s, acc) in spd.items() if mm == m and ff == f)
        if len(p) < 3: continue
        x, y = zip(*p)
        a.plot(x, y, marker=MK[lab], ms=3.5, lw=1.4 if lab in ("RPM-lite", "Probe-guided") else 1.0, color=C[lab], label=lab)
    a.axhline(base[m], color="#52514e", lw=0.6, ls=":"); a.text(0.99, base[m], "unmerged", fontsize=6, va="bottom", ha="right", color="#52514e", transform=a.get_yaxis_transform())
    a.set_title(nm, fontsize=8, loc="left")
fig.supxlabel("measured throughput speed-up over the unmerged model (compiled, batch 256)", fontsize=8)
ax[0].set_ylabel("ImageNet top-1 (%)")
hs = {}
for a in ax:
    for hh, ll in zip(*a.get_legend_handles_labels()): hs[ll] = hh
fig.legend([hs[k] for k in FM.values() if k in hs], [k for k in FM.values() if k in hs], loc="outside upper center", ncol=5, fontsize=7)
fig.savefig(f"{OUT}/fig_speed.png"); fig.savefig(f"{OUT}/fig_speed.pdf"); plt.close(fig)
# ---------- F4 merge visualisation
z = np.load(f"{R}/vis/vit_base_patch14_dinov2.lvd142m.npz"); g = int(z["grid"]); pix = 224 // g
IMG = [0, 3, 7]
fig, ax = plt.subplots(len(IMG), 4, figsize=(5.4, 4.2), constrained_layout=True)
for r, i in enumerate(IMG):
    im = z["images"][i]; ax[r, 0].imshow(im)
    for c, (k, lab) in enumerate((("tome", "ToMe"), ("mctf", "MCTF"), ("rpmlnw", "RPM-lite")), 1):
        src = z[f"src_{k}"][i].reshape(g, g); mask = np.kron(src, np.ones((pix, pix), int))[:224, :224]
        out = im.astype(float).copy()
        for t in np.unique(src):
            sel = mask == t
            if sel.sum() > pix * pix: out[sel] = 0.35 * out[sel] + 0.65 * im[sel].mean(0)
        edge = (np.diff(mask, axis=0, prepend=mask[:1]) != 0) | (np.diff(mask, axis=1, prepend=mask[:, :1]) != 0)
        out[edge] = [255, 255, 255]
        ax[r, c].imshow(out.astype(np.uint8))
        if r == 0: ax[r, c].set_title(f"{lab} ({len(np.unique(z[f'src_{k}'][i]))} groups)" if False else lab, fontsize=8)
    if r == 0: ax[r, 0].set_title("image", fontsize=8)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([]); a.grid(False); [s.set_visible(False) for s in a.spines.values()]
fig.savefig(f"{OUT}/fig_vis.png"); fig.savefig(f"{OUT}/fig_vis.pdf"); plt.close(fig)
print("vis ratio", float(z["ratio"]), {k: int(z[f"ntok_{k}"]) for k in ("tome", "mctf", "rpmlnw")},
      {k: [len(np.unique(z[f"src_{k}"][i])) for i in IMG] for k in ("tome", "mctf", "rpmlnw")})
