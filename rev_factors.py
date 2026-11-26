"""Mentor item 4: how do the two factors of the cost change as merging proceeds?

Runs RPM-lite merging (the implemented rule, ToMe schedule with one deferred block) at several budgets and records,
in every merging block, the per-head median of the attention factor  a_h = sqrt(u_0i u_0j) ||dk_h||  and of the value
factor  b_h = ||dv_h||  over all candidate pairs and over the pairs RPM-lite selects, together with the token sizes.
A sum normalised once per block weights a_h and b_h by fixed constants; the spread of a_h / b_h across heads and
blocks shows how far such a sum is from the product's implicit weighting.

    python rev_factors.py --data $IMAGENET/val --out results/rev_factors.jsonl
"""
import argparse, json, os
import numpy as np
import torch
import tm.merge as tmm
from tm.data import load_model, make_loader, per_class_indices
from tm.patch import apply_patch, configure, simulate_counts

MODELS = ["deit_small_patch16_224", "deit_base_patch16_224", "vit_base_patch14_dinov2.lvd142m",
          "vit_small_patch14_reg4_dinov2.lvd142m", "vit_base_patch16_clip_224.laion2b"]
RATIOS = (0.75, 0.6, 0.45)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--images", type=int, default=256)
    ap.add_argument("--out", default="results/rev_factors.jsonl")
    ap.add_argument("--r", type=int, default=None, help="testing: fixed r instead of a FLOP ratio")
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    done = set()
    if os.path.exists(a.out):
        done = {(json.loads(l)["model"], json.loads(l)["ratio"]) for l in open(a.out) if l.strip()}
    orig = tmm.rpm_lite_cost

    def logged(extra, size, num_prefix, *args, **kw):                      # record factors, then the usual cost
        if tmm.FACTOR_LOG is not None and "k" in extra:
            tmm.rpm_sum_cost(extra, size, num_prefix, 0.5, "head")
        return orig(extra, size, num_prefix, *args, **kw)
    tmm.rpm_lite_cost = logged
    for name in a.models:
        model = apply_patch(load_model(name, dev, pretrained=not a.no_pretrained, headless=True))
        depth = len(model.blocks)
        idx = per_class_indices(a.data, 1, seed=3)
        idx = idx[::max(1, len(idx) // a.images)][:a.images]
        dl = make_loader(model, a.data, batch_size=64, workers=4, indices=idx)
        for ratio in RATIOS if a.r is None else (0.0,):
            if (name, ratio) in done:
                continue
            if a.r is None:
                from tm.flops import find_r_for_gflops, vit_gflops
                base = vit_gflops(model, simulate_counts(model, "none", [0] * depth))
                r, _ = find_r_for_gflops(model, "rpmlnw", ratio * base, 1)
            else:
                r = a.r
            configure(model, "rpmlnw", r, tau=1)
            merge_blocks = [i for i, rr in enumerate(model._tm["r_list"]) if rr > 0]
            per_block = {}
            with torch.no_grad():
                for x, _ in dl:
                    tmm.FACTOR_LOG = []
                    model.forward_features(x.to(dev))
                    for b, st in zip(merge_blocks, tmm.FACTOR_LOG):
                        per_block.setdefault(b, []).append(st)
            tmm.FACTOR_LOG = None
            blocks = []
            for b in sorted(per_block):
                L = per_block[b]
                avg = {k: (np.mean([s[k] for s in L], 0).tolist() if isinstance(L[0][k], list) else float(np.mean([s[k] for s in L])))
                       for k in L[0]}
                blocks.append(dict(block=b, **avg))
            rec = dict(model=name, ratio=ratio, r=r if isinstance(r, int) else list(r), depth=depth, blocks=blocks)
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            sp = [np.std(np.log(np.array(bb["t1_sel"]) / np.array(bb["t2_sel"]))) for bb in blocks]
            print(f"  {name} {ratio}: {len(blocks)} merge blocks, spread of log(a/b) across heads {np.mean(sp):.2f}")
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
