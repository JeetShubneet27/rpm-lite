import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
f, OUT, B = sys.argv[1], sys.argv[2], int(sys.argv[3])
z = np.load(f); g = int(z["grid"]); S = z["images"].shape[1]; pix = S // g
IMG = [0, 2, 5, 7]
M = (("tome", "ToMe"), ("mctf", "MCTF"), ("rpmlnw", "RPM-lite"))
fig, ax = plt.subplots(len(IMG), 4, figsize=(5.6, 5.6), constrained_layout=True)
for r, i in enumerate(IMG):
    im = z["images"][i][:g * pix, :g * pix]; ax[r, 0].imshow(im)
    for c, (k, lab) in enumerate(M, 1):
        src = z[f"src_{k}_b{B}"][i].reshape(g, g)
        mask = np.kron(src, np.ones((pix, pix), int))
        out = im.astype(float).copy()
        for t in np.unique(src):
            sel = mask == t
            if sel.sum() > pix * pix:
                out[sel] = 0.25 * out[sel] + 0.75 * im[sel].mean(0)
        edge = (np.diff(mask, axis=0, prepend=mask[:1]) != 0) | (np.diff(mask, axis=1, prepend=mask[:, :1]) != 0)
        out[edge] = [255, 255, 255]
        ax[r, c].imshow(out.astype(np.uint8))
        if r == 0: ax[r, c].set_title(lab, fontsize=8)
    if r == 0: ax[r, 0].set_title("image", fontsize=8)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([]); [s.set_visible(False) for s in a.spines.values()]
fig.savefig(f"{OUT}/fig_vis.pdf", dpi=300); fig.savefig(f"{OUT}/fig_vis.png", dpi=300); plt.close(fig)
print({k: len(np.unique(z[f'src_{k}_b{B}'][0])) for k, _ in M}, "groups after block", B, "of", g * g, "patches")
