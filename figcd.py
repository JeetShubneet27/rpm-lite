import json, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open(sys.argv[1]))["core38"]; OUT = sys.argv[2]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "savefig.dpi": 300})
R = sorted(d["ranks"].items(), key=lambda t: t[1]); cd = d["cd"]; k = len(R)
X = lambda r: k + 1 - r                                   # best rank on the right
fig, ax = plt.subplots(figsize=(5.6, 2.0))
ax.set_xlim(-1.6, k + 2.6); ax.set_ylim(-0.05, 1.3); ax.axis("off")
ax.plot([X(k), X(1)], [1, 1], color="#52514e", lw=0.8)
for t in range(1, k + 1):
    ax.plot([X(t), X(t)], [1, 1.04], color="#52514e", lw=0.8); ax.text(X(t), 1.07, str(t), ha="center", fontsize=7)
ax.plot([X(k), X(k) + cd], [1.2, 1.2], color="#0b0b0b", lw=1.5); ax.text(X(k) + cd / 2, 1.24, f"CD = {cd:.2f}", ha="center", fontsize=7)
ours = {"RPM-lite", "Probe-guided"}
best, worst = R[:(k + 1) // 2], R[(k + 1) // 2:][::-1]
for side, items in (("right", best), ("left", worst)):
    for i, (nm, r) in enumerate(items):
        y = 0.62 - 0.15 * i; xt = X(1) + 0.4 if side == "right" else X(k) - 0.4
        ax.plot([X(r), X(r)], [1, y], color="#52514e", lw=0.7); ax.plot([X(r), xt], [y, y], color="#52514e", lw=0.7)
        ax.text(xt + (0.05 if side == "right" else -0.05), y, f"{nm} ({r:.2f})", ha="left" if side == "right" else "right",
                va="center", fontsize=7, fontweight="bold" if nm in ours else "normal", color="#2a78d6" if nm in ours else "#0b0b0b")
rs = [r for _, r in R]; groups = []
for i in range(k):
    j = max(jj for jj in range(i, k) if rs[jj] - rs[i] <= cd)
    if j > i and not any(a <= i and j <= b for a, b in groups): groups.append((i, j))
for g_i, (a, b) in enumerate(groups):
    y = 0.9 - 0.06 * g_i
    ax.plot([X(rs[a]) + 0.05, X(rs[b]) - 0.05], [y, y], color="#0b0b0b", lw=2.2, solid_capstyle="round")
fig.tight_layout(pad=0.2)
fig.savefig(f"{OUT}/fig_cd.pdf"); fig.savefig(f"{OUT}/fig_cd.png"); plt.close(fig)
