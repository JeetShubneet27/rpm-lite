"""Paired significance tests between runs saved with --save-preds.

    # every run in a results file against one reference config (e.g. uniform prior)
    python significance.py results/full.jsonl --ref-prior uniform

For each pair (run, reference) evaluated on the SAME images, with the same model, method, tau and
GFLOPs, prints the accuracy difference, discordant counts, an exact two-sided McNemar p-value and
a 95% bootstrap CI of the difference.
"""
import argparse, json, math
import numpy as np


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided exact binomial test with p = 0.5, computed in log space
    logp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    m = max(logp)
    p = 2 * math.exp(m) * sum(math.exp(v - m) for v in logp)
    return min(1.0, p)


def boot_ci(x, y, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = x.astype(np.float32) - y.astype(np.float32)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(1) * 100
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--ref-prior", default="uniform")
    ap.add_argument("--ref-method", default="sadtm")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    rows = [r for r in rows if r.get("preds")]

    def cfg(r):
        return (r["model"], r["tau"], r["gflops"], r.get("weight_mode"), r.get("prop_attn"))

    refs = {cfg(r): r for r in rows if r["method"] == a.ref_method and r["prior"] == a.ref_prior
            and r.get("prior_gain", 1.0) == 1.0}
    print(f"{'model':24s} {'tau':>3s} {'GF':>5s} {'run':34s} {'acc':>6s} {'ref':>6s} {'diff':>6s} "
          f"{'95% CI':>16s} {'b':>5s} {'c':>5s} {'p':>8s}")
    for r in rows:
        ref = refs.get(cfg(r))
        if ref is None or ref is r:
            continue
        x, y = np.load(r["preds"]), np.load(ref["preds"])
        if len(x) != len(y):
            continue
        b = int((x & ~y).sum())   # run right, ref wrong
        c = int((~x & y).sum())   # run wrong, ref right
        lo, hi = boot_ci(x, y)
        name = f"{r['method']}:{r['prior']}" + (f" s{r['seed']}" if r["prior"] in ("random", "saliency_shuffled") else "") \
            + (f" gain{r['prior_gain']:g}" if r.get("prior_gain", 1.0) != 1.0 else "")
        p = mcnemar_exact(b, c)
        star = "*" if p < 0.05 else " "
        print(f"{r['model'][:24]:24s} {r['tau']:>3d} {r['gflops']:5.2f} {name:34s} {x.mean()*100:6.2f} "
              f"{y.mean()*100:6.2f} {100*(x.mean()-y.mean()):+6.2f} [{lo:+6.2f},{hi:+6.2f}] {b:5d} {c:5d} {p:8.4f}{star}")


if __name__ == "__main__":
    main()
