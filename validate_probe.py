"""Does the reliability probe measure what merging actually costs? (reviewer W1, W2, Q1, Q5)

For every model, on the SAME unlabeled training images the probe uses (seed 0):

1. LOO probe, per image: Spearman of class attention / key uniqueness with leave-one-out importance
   (identical to measure_reliability.py), kept per image so we can bootstrap over images.

2. Merge-cost ground truth AT THE LAYER (no input-patch removal, no depth mixing): in block l, right
   where merging happens (after the attention residual), merge token i into its most key-similar
   token (size-weighted average, proportional attention afterwards), run the rest of the network and
   measure  mc_l[i] = 1 - cos(final CLS feature, unmerged final CLS feature).
   We then score the same two signals against mc_l. This answers: is LOO a valid proxy for merge cost,
   and are the late-layer "failures" an artefact of LOO perturbing the input?

3. Block-wise operator cost (direct test of the probe's per-block decision): merge 25% of the tokens
   ONLY in block l, once with ToMe matching and once with attention-guided merging; cost = mean
   1 - cos of the final CLS feature. The probe "chooses right" in block l when the operator it prefers
   (rho_sim > rho_attn -> ToMe) is the cheaper one.

4. Bootstrap over images (1000 resamples): 95% CIs of the per-block reliabilities and the probability
   that the probe's operator choice in block l flips.

    python validate_probe.py --data $IMAGENET/train --out results/probe_validation.jsonl
"""
import argparse, json, math, os, time
import torch
import torch.nn.functional as F

from measure_reliability import DEFAULT_MODELS, build, loader_for, embed, leave_one_out, spearman
from tm.patch import MergeAttention
from tm.merge import tome_merge, sadtm_merge
from tm.device import pick_device


def blk_attn(blk, h, size):
    out, k, ca = MergeAttention.forward_with_stats(blk.attn, blk.norm1(h), size, size is not None)
    return h + blk.ls1(out), k, ca


def blk_mlp(blk, h):
    return h + blk.ls2(blk.mlp(blk.norm2(h)))


def finish(model, h, size, start):
    """Run blocks start.. with proportional attention (size may be None = all ones)."""
    for blk in model.blocks[start:]:
        h, _, _ = blk_attn(blk, h, size)
        h = blk_mlp(blk, h)
    return model.norm(h)[:, 0]


def uniq(k, P):
    kp = F.normalize(k[:, P:].float(), dim=-1)
    s = kp @ kp.transpose(1, 2)
    s.diagonal(dim1=1, dim2=2).fill_(-2)
    return -s.amax(-1), s.argmax(-1)          # uniqueness (high = keep), nearest neighbour index


@torch.no_grad()
def analyse_image(model, x, do_mc, frac, chunk):
    """x: (1, 3, H, W). Returns per-block signals, LOO, merge-cost and operator costs for one image."""
    P = model.num_prefix_tokens
    L = len(model.blocks)
    h = embed(model, x)
    ref = finish(model, h, None, 0)
    loo = leave_one_out(model, x, ref, chunk=chunk)[0]                   # (Np,)
    out = dict(attn=[], uniq=[], mc=[], cost_tome=[], cost_attn=[])
    for l, blk in enumerate(model.blocks):
        hm, k, ca = blk_attn(blk, h, None)                                # after attention residual
        u, nn = uniq(k, P)
        a = ca[:, P:].float()
        out["attn"].append(a[0])
        out["uniq"].append(u[0])
        N, C = hm.shape[1], hm.shape[2]
        Np = N - P
        if do_mc:
            # (2) merge token i into its nearest neighbour, for every i at once (chunked)
            ar = torch.arange(Np, device=x.device)
            j = nn[0]
            mc = torch.empty(Np, device=x.device)
            for s0 in range(0, Np, chunk):
                ii, jj = ar[s0:s0 + chunk], j[s0:s0 + chunk]
                n = ii.numel()
                hv = hm.expand(n, N, C).clone()
                sz = torch.ones(n, N, device=x.device)
                rows = torch.arange(n, device=x.device)
                hv[rows, P + jj] = 0.5 * (hm[0, P + ii] + hm[0, P + jj])
                sz[rows, P + jj] = 2.0
                keep = torch.ones(n, N, dtype=torch.bool, device=x.device)
                keep[rows, P + ii] = False
                hv, sz = hv[keep].view(n, N - 1, C), sz[keep].view(n, N - 1)
                hv = blk_mlp(blk, hv)
                f = finish(model, hv, sz, l + 1)
                mc[s0:s0 + n] = 1 - F.cosine_similarity(f.float(), ref.float(), dim=-1)
            out["mc"].append(mc)
            # (3) merge `frac` of the tokens only in this block, ToMe vs attention-guided
            r = int(round(frac * Np))
            size = torch.ones(1, N, device=x.device)
            src = torch.arange(Np, device=x.device)[None]
            prior = torch.zeros(1, N, device=x.device)
            for op in ("tome", "attn"):
                if op == "tome":
                    xm, _, sm, _ = tome_merge(hm, k, prior, size, src, r, P)
                else:
                    xm, _, sm, _ = sadtm_merge(hm, k, ca, prior, size, src, r, P, weight_mode="size", signal_w=1.0)
                f = finish(model, blk_mlp(blk, xm), sm, l + 1)
                out[f"cost_{op}"].append(float(1 - F.cosine_similarity(f.float(), ref.float(), dim=-1)))
        h = blk_mlp(blk, hm)
    return out, loo


def boot(per_img, B=1000, seed=0):
    """per_img: (n, L) tensor -> mean, lo, hi (95%) per block."""
    g = torch.Generator().manual_seed(seed)
    n = per_img.shape[0]
    idx = torch.randint(0, n, (B, n), generator=g)
    m = per_img[idx].mean(1)                                             # (B, L)
    return per_img.mean(0), m.quantile(0.025, 0), m.quantile(0.975, 0), m


def run_model(name, a, dev):
    model = build(name, pretrained=not a.no_pretrained, img_size=a.img_size).to(dev)
    L = len(model.blocks)
    loader = loader_for(model, a.data, a.loo_images, 1, a.workers, a.seed)   # same images as the probe
    per = {k: [] for k in ("ra_loo", "rs_loo", "ra_mc", "rs_mc", "loo_mc")}
    cost_t, cost_a = [], []
    t0 = time.time()
    for n, (x, _) in enumerate(loader):
        x = x.to(dev)
        do_mc = n < a.mc_images
        o, loo = analyse_image(model, x, do_mc, a.frac, a.chunk)
        per["ra_loo"].append(torch.stack([spearman(t[None], loo[None])[0] for t in o["attn"]]))
        per["rs_loo"].append(torch.stack([spearman(t[None], loo[None])[0] for t in o["uniq"]]))
        if do_mc:
            per["ra_mc"].append(torch.stack([spearman(t[None], m[None])[0] for t, m in zip(o["attn"], o["mc"])]))
            per["rs_mc"].append(torch.stack([spearman(t[None], m[None])[0] for t, m in zip(o["uniq"], o["mc"])]))
            per["loo_mc"].append(torch.stack([spearman(loo[None], m[None])[0] for m in o["mc"]]))
            cost_t.append(o["cost_tome"])
            cost_a.append(o["cost_attn"])
        print(f"\r  {name}: {n + 1} imgs  {time.time() - t0:.0f}s", end="", flush=True)
    print()
    per = {k: torch.stack(v).cpu().float() for k, v in per.items() if v}
    res = dict(model=name, img_size=int(model.patch_embed.img_size[0]), depth=L, n_loo=per["ra_loo"].shape[0],
               n_mc=per["ra_mc"].shape[0], frac=a.frac)
    for k, v in per.items():
        mu, lo, hi, _ = boot(v)
        res[k] = [round(float(t), 4) for t in mu]
        res[k + "_lo"] = [round(float(t), 4) for t in lo]
        res[k + "_hi"] = [round(float(t), 4) for t in hi]
    # operator choice of the probe (as in eval_imagenet: attention if [ra]+ >= [rs]+) and its stability
    _, _, _, ba = boot(per["ra_loo"])
    _, _, _, bs = boot(per["rs_loo"])
    choice = (per["ra_loo"].mean(0).clamp_min(0) >= per["rs_loo"].mean(0).clamp_min(0))
    bchoice = ba.clamp_min(0) >= bs.clamp_min(0)
    res["op_probe"] = ["A" if c else "S" for c in choice.tolist()]
    res["op_flip_prob"] = [round(float(t), 3) for t in (bchoice != choice[None]).float().mean(0)]
    # the same decision from the merge-cost ground truth
    mc_choice = (per["ra_mc"].mean(0).clamp_min(0) >= per["rs_mc"].mean(0).clamp_min(0))
    res["op_mc"] = ["A" if c else "S" for c in mc_choice.tolist()]
    # block-wise operator costs
    ct, ca = torch.tensor(cost_t).mean(0), torch.tensor(cost_a).mean(0)
    res["cost_tome"] = [round(float(t), 5) for t in ct]
    res["cost_attn"] = [round(float(t), 5) for t in ca]
    res["op_cheaper"] = ["A" if a_ < t_ else "S" for a_, t_ in zip(ca.tolist(), ct.tolist())]
    del model
    torch.cuda.empty_cache() if dev == "cuda" else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="ImageFolder root (train; labels unused)")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--loo-images", type=int, default=128, help="images for the LOO probe + bootstrap")
    ap.add_argument("--mc-images", type=int, default=32, help="of those, images for merge cost / operator cost")
    ap.add_argument("--frac", type=float, default=0.25, help="fraction merged in the single-block operator test")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--no-pretrained", action="store_true", help="random weights (pipeline test only)")
    ap.add_argument("--out", default="results/probe_validation.jsonl")
    a = ap.parse_args()
    dev = pick_device()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["model"] for l in open(a.out) if l.strip()}
    for name in a.models:
        if name in done:
            print("skip", name)
            continue
        try:
            res = run_model(name, a, dev)
        except Exception as e:
            print(f"\n  FAILED {name}: {type(e).__name__}: {e}")
            continue
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")
        print("  probe op :", "".join(res["op_probe"]))
        print("  mc op    :", "".join(res["op_mc"]))
        print("  cheaper  :", "".join(res["op_cheaper"]))
        print("  flip prob:", " ".join(f"{p:.2f}" for p in res["op_flip_prob"]))


if __name__ == "__main__":
    main()
