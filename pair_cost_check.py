"""Go / no-go test for a NEW merging criterion derived from the attention readout.

Derivation. A query q reads  out_q = sum_t A_qt v_t. Merge tokens i and j into one token with
v = (v_i + v_j)/2, k = (k_i + k_j)/2 and size 2 (proportional attention adds log 2 to its logit).
If q.k_i = q.k_j the merged token receives exactly A_qi + A_qj and the readout is unchanged. To first
order in the logit difference, the readout error is

        delta out_q  =  1/2 (A_qi - A_qj) (v_i - v_j)                               (*)

so the cost of merging the PAIR (i, j) is  C(i,j) = 1/2 sum_heads sum_q |A_qi - A_qj| * ||v_i - v_j||.
It is zero when the two tokens get the same attention OR carry the same value. ToMe uses only key
similarity (a proxy for A_qi = A_qj); attention-based merging uses only A_0i (and ignores the partner).
(*) says the right quantity is the product of the attention difference and the value difference.

This script measures, for every token i and its key-nearest neighbour j in block l, the true cost of
merging them (final CLS feature change, as in validate_probe.py) and the Spearman correlation of that
cost with:
  tome     : 1 - cos(k_i, k_j)                         (ToMe's pair score)
  attn     : A_0i, head-averaged CLS attention          (SAD-TM / EViT importance)
  value    : sum_h ||v_i - v_j||                        (value distance only)
  rp_cls   : sum_h |A_0i - A_0j| ||v_i - v_j||          (*) with the CLS query only   [cheap]
  rp_all   : sum_h sum_q |A_qi - A_qj| ||v_i - v_j||    (*) with every query         [full attention]
  rp_wo    : rp_all with values pushed through the output projection and layer-scale (what the residual stream gets)
  rp_next  : rp_all with the attention/values of block l+1 (the layer that actually reads the merged token)

    python pair_cost_check.py --data $IMAGENET/train --images 16 --out results/pair_cost.jsonl
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F

from measure_reliability import build, loader_for, embed, spearman
from validate_probe import blk_mlp, finish
from tm.device import pick_device

MODELS = ["deit_small_patch16_224", "deit3_small_patch16_224.fb_in1k", "deit_base_patch16_224",
          "vit_base_patch14_dinov2.lvd142m", "vit_base_patch14_reg4_dinov2.lvd142m", "vit_base_patch16_clip_224.laion2b"]
SIGS = ("tome", "attn", "value", "rp_cls", "rp_all", "rp_wo", "rp_next")


def attn_parts(attn_mod, x):
    """Full attention A (B,H,N,N), values v (B,H,N,d), keys k (B,H,N,d) and the block's output."""
    B, N, _ = x.shape
    qkv = attn_mod.qkv(x).reshape(B, N, 3, attn_mod.num_heads, attn_mod.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)
    A = ((q * attn_mod.scale) @ k.transpose(-2, -1)).float().softmax(-1)
    o = (A.to(v.dtype) @ v).transpose(1, 2).reshape(B, N, attn_mod.attn_dim)
    o = attn_mod.proj(attn_mod.norm(o))
    return A, v.float(), k.float(), o


def out_values(blk, v):
    """Per-head values as they reach the residual stream: gamma * W_o^h v^h  -> (H, N, C)."""
    W = blk.attn.proj.weight.float()                                       # (C, H*d)
    H, N, d = v.shape
    vo = torch.einsum("hnd,chd->hnc", v, W.view(W.shape[0], H, d))
    g = getattr(blk.ls1, "gamma", None)
    return vo * g.float() if g is not None else vo


def rp_costs(A, v, P, nn):
    """(*) for every patch token i and its partner nn[i]: CLS-only and all-query versions. A: (H,N,N)."""
    H, N, _ = A.shape
    Np = N - P
    ii = torch.arange(Np, device=A.device) + P
    jj = nn + P
    dv = (v[:, ii] - v[:, jj]).norm(dim=-1)                                # (H, Np)
    dA_cls = (A[:, 0, ii] - A[:, 0, jj]).abs()                             # (H, Np)
    dA_all = (A[:, :, ii] - A[:, :, jj]).abs().sum(1)                      # (H, Np)  sum over queries
    return (dA_cls * dv).sum(0), (dA_all * dv).sum(0), dv.sum(0)


@torch.no_grad()
def run_image(model, x, chunk):
    P = model.num_prefix_tokens
    h = embed(model, x)
    ref = finish(model, h, None, 0)
    blocks = list(model.blocks)
    hs = [h]
    parts = []
    for blk in blocks:                                                     # unmerged pass, keep everything
        A, v, k, o = attn_parts(blk.attn, blk.norm1(hs[-1]))
        hm = hs[-1] + blk.ls1(o)
        parts.append((A[0], v[0], k[0], hm, out_values(blk, v[0])))
        hs.append(blk_mlp(blk, hm))
    out = {s: [] for s in SIGS}
    for l, blk in enumerate(blocks[:-1]):                                  # last block: merging cannot matter
        A, v, k, hm, vo = parts[l]
        km = F.normalize(k.mean(0)[P:], dim=-1)                            # ToMe metric (head-mean keys)
        s = km @ km.T
        s.fill_diagonal_(-2)
        best, nn = s.max(-1)
        N, C = hm.shape[1], hm.shape[2]
        Np = N - P
        # true cost of merging i into nn[i] right here (as validate_probe.py)
        mc = torch.empty(Np, device=x.device)
        ar = torch.arange(Np, device=x.device)
        for s0 in range(0, Np, chunk):
            i, j = ar[s0:s0 + chunk], nn[s0:s0 + chunk]
            n = i.numel()
            hv = hm.expand(n, N, C).clone()
            sz = torch.ones(n, N, device=x.device)
            rows = torch.arange(n, device=x.device)
            hv[rows, P + j] = 0.5 * (hm[0, P + i] + hm[0, P + j])
            sz[rows, P + j] = 2.0
            keep = torch.ones(n, N, dtype=torch.bool, device=x.device)
            keep[rows, P + i] = False
            hv, sz = hv[keep].view(n, N - 1, C), sz[keep].view(n, N - 1)
            f = finish(model, blk_mlp(blk, hv), sz, l + 1)
            mc[s0:s0 + n] = 1 - F.cosine_similarity(f.float(), ref.float(), dim=-1)
        c_cls, c_all, c_v = rp_costs(A, v, P, nn)
        _, c_wo, _ = rp_costs(A, vo, P, nn)
        A2, v2, _, _, _ = parts[l + 1]
        _, c_next, _ = rp_costs(A2, v2, P, nn)
        sig = dict(tome=1 - best, attn=A[:, 0, P:].mean(0), value=c_v, rp_cls=c_cls, rp_all=c_all, rp_wo=c_wo,
                   rp_next=c_next)
        for name, t in sig.items():
            out[name].append(float(spearman(t[None].float(), mc[None])[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--images", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--out", default="results/pair_cost.jsonl")
    a = ap.parse_args()
    dev = pick_device()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for name in a.models:
        model = build(name, pretrained=not a.no_pretrained).to(dev)
        loader = loader_for(model, a.data, a.images, 1, a.workers, a.seed)
        acc = None
        t0 = time.time()
        for n, (x, _) in enumerate(loader):
            o = run_image(model, x.to(dev), a.chunk)
            acc = o if acc is None else {s: [p + q for p, q in zip(acc[s], o[s])] for s in SIGS}
            print(f"\r  {name}: {n + 1} imgs {time.time() - t0:.0f}s", end="", flush=True)
        print()
        res = dict(model=name, n=n + 1, **{s: [round(v / (n + 1), 4) for v in acc[s]] for s in SIGS})
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")
        L = len(res["tome"])
        thirds = [(0, L // 3), (L // 3, 2 * L // 3), (2 * L // 3, L)]
        print(f"  {'signal':8s}  mean   early   mid   late   (Spearman with the true merge cost)")
        for s in SIGS:
            v = res[s]
            print(f"  {s:8s} {sum(v) / L:6.3f}  " + "  ".join(f"{sum(v[a_:b_]) / max(b_ - a_, 1):6.3f}" for a_, b_ in thirds))
        del model
        torch.cuda.empty_cache() if dev == "cuda" else None


if __name__ == "__main__":
    main()
