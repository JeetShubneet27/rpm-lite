"""Day 1 of Part 2: where is class attention a reliable token-importance signal?

For every block l of a pretrained ViT we compare three per-token signals on unlabeled images:
  attn_l : head-averaged CLS attention to each patch token  (SAD-TM / EViT-style importance)
  sim_l  : -(max cosine similarity of the token's key to any other patch key)
           (ToMe-style: a token with a near-duplicate is redundant, so low sim_l = mergeable)
  final  : CLS attention at the last block
against a ground-truth importance obtained by leave-one-out (LOO):
  loo_i  = 1 - cos( CLS_out(full image), CLS_out(image without patch i) )
which is label-free and works for any backbone with a CLS token (DeiT, DeiT-III, DINOv2, CLIP).

Per block we report (mean over images):
  rho_attn_loo  Spearman(attn_l, loo)          <- how reliable attention is as importance
  rho_sim_loo   Spearman(sim_l, loo)           <- how reliable similarity is
  rho_attn_last Spearman(attn_l, attn_L)       <- label-free proxy we can use at test time
  p30_attn / p30_sim : precision of the 30% tokens each signal would merge first, w.r.t. the 30%
                       least important tokens by LOO (chance = 0.30)
  outlier_mass  share of CLS attention on the top-2% highest-norm patch tokens
  norm_ratio    max / median patch-token norm (high-norm artifact indicator)

    python measure_reliability.py --data $IMAGENET/train --out results/reliability.jsonl
"""
import argparse, json, math, os, time
import torch
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config, create_transform
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from tm.patch import MergeAttention
from tm.device import pick_device

DEFAULT_MODELS = [
    "deit_tiny_patch16_224", "deit_small_patch16_224", "deit_base_patch16_224",
    "deit3_small_patch16_224.fb_in1k", "deit3_base_patch16_224.fb_in1k",
    "vit_small_patch14_dinov2.lvd142m", "vit_small_patch14_reg4_dinov2.lvd142m",
    "vit_base_patch14_dinov2.lvd142m", "vit_base_patch14_reg4_dinov2.lvd142m",
    "vit_large_patch14_dinov2.lvd142m", "vit_large_patch14_reg4_dinov2.lvd142m",
    "vit_base_patch16_clip_224.laion2b",
]


def build(name, pretrained, img_size=224):
    kw = {"num_classes": 0}
    if "dinov2" in name:
        kw["img_size"] = img_size          # 518 -> 224 (pos-embed interpolated), 16x16 patches
    m = timm.create_model(name, pretrained=pretrained, **kw)
    return m.eval()


def loader_for(model, root, n, bs, workers, seed=0):
    cfg = resolve_data_config({"input_size": (3,) + tuple(model.patch_embed.img_size)}, model=model)
    tf = create_transform(**cfg, is_training=False)
    ds = ImageFolder(root, transform=tf)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False, num_workers=workers)


def embed(model, x):
    x = model.patch_embed(x)
    x = model._pos_embed(x)
    x = model.patch_drop(x)
    return model.norm_pre(x)


def cls_out(model, x):
    for blk in model.blocks:
        x = blk(x)
    return model.norm(x)[:, 0]


@torch.no_grad()
def trace(model, x):
    """Run block by block; return per-block CLS attention, key-similarity score and token norms
    for the PATCH tokens only, plus the final CLS feature."""
    P = model.num_prefix_tokens
    h = embed(model, x)
    attn_l, sim_l, norm_l = [], [], []
    for blk in model.blocks:
        out, k, ca = MergeAttention.forward_with_stats(blk.attn, blk.norm1(h), None, False)
        h = h + blk.ls1(out)
        kp = F.normalize(k[:, P:].float(), dim=-1)
        s = kp @ kp.transpose(1, 2)
        s.diagonal(dim1=1, dim2=2).fill_(-2)
        sim_l.append(-s.amax(-1))                    # high = unique = important
        attn_l.append(ca[:, P:].float())
        h = h + blk.ls2(blk.mlp(blk.norm2(h)))
        norm_l.append(h[:, P:].float().norm(dim=-1))
    return attn_l, sim_l, norm_l, model.norm(h)[:, 0]


@torch.no_grad()
def leave_one_out(model, x, ref, chunk=512):
    """loo[b, i] = 1 - cos(CLS(full), CLS(without patch i)) for one forward per (image, patch)."""
    P = model.num_prefix_tokens
    e = embed(model, x)
    B, N, C = e.shape
    Np = N - P
    out = torch.empty(B, Np, device=x.device)
    keep = ~torch.eye(Np, dtype=torch.bool, device=x.device)            # (Np, Np-1) masks
    for b in range(B):
        pref = e[b, :P].expand(Np, -1, -1)
        pat = e[b, P:].expand(Np, -1, -1)[keep].view(Np, Np - 1, C)
        seqs = torch.cat([pref, pat], 1)
        feats = torch.cat([cls_out(model, seqs[i:i + chunk]) for i in range(0, Np, chunk)])
        out[b] = 1 - F.cosine_similarity(feats.float(), ref[b:b + 1].float(), dim=-1)
    return out


def ranks(t):
    return t.argsort(-1).argsort(-1).float()


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ra = ra - ra.mean(-1, keepdim=True)
    rb = rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1)).clamp_min(1e-12)


def bottom_precision(sig, truth, frac=0.3):
    k = max(1, int(round(frac * sig.shape[-1])))
    a = sig.argsort(-1)[:, :k]
    t = truth.argsort(-1)[:, :k]
    hit = torch.zeros_like(sig, dtype=torch.bool).scatter_(1, a, True)
    hit_t = torch.zeros_like(sig, dtype=torch.bool).scatter_(1, t, True)
    return (hit & hit_t).sum(-1).float() / k


def run_model(name, a, dev):
    model = build(name, pretrained=not a.no_pretrained, img_size=a.img_size).to(dev)
    L = len(model.blocks)
    loader = loader_for(model, a.data, a.num_images, a.batch_size, a.workers, a.seed)
    acc = {k: torch.zeros(L) for k in ["rho_attn_loo", "rho_sim_loo", "rho_attn_last", "rho_attn_next",
                                        "p30_attn", "p30_sim", "outlier_mass", "norm_ratio", "attn_entropy"]}
    n_img = n_loo = 0
    t0 = time.time()
    for x, _ in loader:
        x = x.to(dev)
        attn_l, sim_l, norm_l, ref = trace(model, x)
        Np = attn_l[0].shape[-1]
        do_loo = n_loo < a.loo_images
        loo = leave_one_out(model, x, ref, chunk=max(32, 512 * 256 // Np)) if do_loo else None
        for l in range(L):
            nr = norm_l[l]
            k2 = max(1, int(math.ceil(0.02 * Np)))
            top = nr.topk(k2, dim=-1).indices
            acc["outlier_mass"][l] += attn_l[l].gather(1, top).sum(-1).sum().cpu()
            acc["norm_ratio"][l] += (nr.amax(-1) / nr.median(-1).values).sum().cpu()
            p = attn_l[l] / attn_l[l].sum(-1, keepdim=True)
            acc["attn_entropy"][l] += (-(p * p.clamp_min(1e-12).log()).sum(-1) / math.log(Np)).sum().cpu()
            acc["rho_attn_last"][l] += spearman(attn_l[l], attn_l[-1]).sum().cpu()
            if l < L - 1:
                acc["rho_attn_next"][l] += spearman(attn_l[l], attn_l[l + 1]).sum().cpu()
            if do_loo:
                acc["rho_attn_loo"][l] += spearman(attn_l[l], loo).sum().cpu()
                acc["rho_sim_loo"][l] += spearman(sim_l[l], loo).sum().cpu()
                acc["p30_attn"][l] += bottom_precision(attn_l[l], loo).sum().cpu()
                acc["p30_sim"][l] += bottom_precision(sim_l[l], loo).sum().cpu()
        n_img += x.shape[0]
        n_loo += x.shape[0] if do_loo else 0
        print(f"\r  {name}: {n_img} imgs ({n_loo} LOO)  {time.time() - t0:.0f}s", end="", flush=True)
    print()
    res = dict(model=name, img_size=int(model.patch_embed.img_size[0]), depth=L, n_images=n_img, n_loo=n_loo, num_prefix=model.num_prefix_tokens,
               n_patch=Np, pretrained=not a.no_pretrained)
    for k, v in acc.items():
        denom = n_loo if k in ("rho_attn_loo", "rho_sim_loo", "p30_attn", "p30_sim") else n_img
        res[k] = [round(float(t) / max(denom, 1), 4) for t in v]
    del model
    torch.cuda.empty_cache() if dev == "cuda" else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="ImageFolder root (use train: labels are not used)")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--num-images", type=int, default=512)
    ap.add_argument("--loo-images", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--img-size", type=int, default=224, help="input resolution (DINOv2 only)")
    ap.add_argument("--out", default="results/reliability.jsonl")
    a = ap.parse_args()
    dev = pick_device()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for name in a.models:
        try:
            res = run_model(name, a, dev)
        except Exception as e:  # keep going if one checkpoint fails to download
            print(f"\n  FAILED {name}: {type(e).__name__}: {e}")
            continue
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")
        L = res["depth"]
        print(f"  layer:        " + " ".join(f"{i + 1:5d}" for i in range(L)))
        for k in ("rho_attn_loo", "rho_sim_loo", "rho_attn_last", "outlier_mass", "norm_ratio"):
            print(f"  {k:13s} " + " ".join(f"{v:5.2f}" for v in res[k]))


if __name__ == "__main__":
    main()
