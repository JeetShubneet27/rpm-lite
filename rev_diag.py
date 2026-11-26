"""Revision diagnostics (mentor items 1, 2, 5, 6). Self-contained: only timm + tm.data.

For every image, every block l < L-1 and three kinds of candidate pairs (key-nearest = ToMe-favoured,
value-nearest, uniformly random), the merge is performed EXACTLY as in the implementation: tokens i and j of the
residual stream after block l's attention are replaced by their size-weighted mean (size 2), block l's MLP runs on the
merged sequence, and the remaining blocks run with proportional attention. We record

  item 1/2  ACTUAL change of block (l+1)'s attention output for every other query, per head (before W_o), as the sum of
            per-head norms, as the norm of the concatenated heads, and after W_o and layer-scale; together with the
            first-order PREDICTION of Prop. 1 from the unmerged block-(l+1) attention and values (same three norms), and
            how far the merged token's block-(l+1) key/value are from the size-weighted mean of the originals
            (LayerNorm + MLP non-linearity).
  item 5/6  the criteria computed at block l (ToMe, [CLS] attention, exact RPM over all queries, exact RPM with the
            [CLS] query only, RPM-lite, RPM-lite without attention / without value term) and the ACTUAL damage at the
            output: 1 - cos(final feature), KL divergence of the class distribution, change of the top-1 margin and
            whether the top-1 prediction flips (all label-free: the reference is the unmerged prediction).
  item 5    per block and head: query isotropy (participation ratio of the query second moment, spectrum), how much
            the pair's attention level varies across queries, how well the [CLS] row estimates it, and the share of
            [CLS] attention on the top-2% norm tokens.

    python rev_diag.py --data $IMAGENET/train --images 64 --out results/rev_diag
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from tm.data import load_model, make_loader, per_class_indices

MODELS = ["deit_tiny_patch16_224", "deit_small_patch16_224", "deit_base_patch16_224", "deit3_small_patch16_224.fb_in1k",
          "deit3_base_patch16_224.fb_in1k", "vit_small_patch16_224.augreg_in21k_ft_in1k",
          "vit_base_patch16_224.augreg_in21k_ft_in1k", "vit_small_patch14_dinov2.lvd142m",
          "vit_small_patch14_reg4_dinov2.lvd142m", "vit_base_patch14_dinov2.lvd142m",
          "vit_base_patch14_reg4_dinov2.lvd142m", "vit_base_patch16_clip_224.laion2b",
          "vit_large_patch14_dinov2.lvd142m", "vit_large_patch14_reg4_dinov2.lvd142m"]
PARTNERS = ("key", "val", "rand")
FIELDS = ("block", "ptype", "img", "act_ideal",
          "act_sum", "act_cat", "act_proj", "pred_sum", "pred_cat", "pred_proj", "dev_k", "dev_v",
          "tome", "attn", "exact", "exact_cls", "lite", "lite_noatt", "lite_noval", "kdist_white", "kdist",
          "d_cos", "d_kl", "d_margin", "d_flip")


def attn_full(attn, x, logsize=None):
    B, N, _ = x.shape
    qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    lg = (q * attn.scale) @ k.transpose(-2, -1)
    if logsize is not None:
        lg = lg + logsize[:, None, None, :]
    A = lg.softmax(-1)
    oh = A @ v                                                             # (B, H, N, d)
    o = oh.transpose(1, 2).reshape(B, N, -1)
    o = attn.proj(getattr(attn, "norm", torch.nn.Identity())(o))
    return q, k, v, lg, A, oh, o


def block_fwd(blk, x, logsize=None):
    *_, o = attn_full(blk.attn, blk.norm1(x), logsize)
    h = x + blk.ls1(o)
    return h + blk.ls2(blk.mlp(blk.norm2(h)))


def readout(model, x, probe):
    f = model.forward_head(model.norm(x), pre_logits=True)
    if probe is not None:
        z = ((f.float() - probe["mean"]) / probe["std"]) @ probe["W"].T + probe["b"]
    else:
        z = model.head(f)
    return f.float(), z.float()


def spearman_np(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def run_image(model, probe, x, n_pairs, chunk, g, img_id, blockstats):
    P = model.num_prefix_tokens
    blocks = list(model.blocks)
    L = len(blocks)
    h = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(x))))
    cache = []
    for blk in blocks:                                                     # clean pass
        q, k, v, lg, A, oh, o = attn_full(blk.attn, blk.norm1(h))
        hm = h + blk.ls1(o)
        cache.append(dict(x=h, q=q[0], k=k[0], v=v[0], lg=lg[0], A=A[0], oh=oh[0], o=blk.ls1(o)[0], hm=hm))
        h = hm + blk.ls2(blk.mlp(blk.norm2(hm)))
    f0, z0 = readout(model, h, probe)
    p0 = z0.log_softmax(-1)
    c0 = int(z0.argmax())
    m0 = float(z0[0, c0] - z0[0].clone().index_fill_(0, torch.tensor([c0], device=z0.device), -1e9).max())
    N = cache[0]["hm"].shape[1]
    Np = N - P
    rows = []
    for l in range(L - 1):
        C, Cn = cache[l], cache[l + 1]
        H, _, d = C["k"].shape
        # ---- block statistics (item 5) -------------------------------------------------------
        if img_id < 16:
            q = C["q"].float()                                             # (H, N, d)
            S = q.transpose(1, 2) @ q                                      # (H, d, d) query second moment
            ev = torch.linalg.eigvalsh(S).clamp_min(0).flip(-1)            # descending
            pr = (ev.sum(-1) ** 2 / (ev ** 2).sum(-1).clamp_min(1e-12)) / d
            spec = (ev / ev.sum(-1, keepdim=True)).mean(0)
            nrm = C["x"][0, P:].float().norm(dim=-1)
            top = nrm.topk(max(1, math.ceil(0.02 * Np))).indices + P
            om = float(C["A"][:, 0, top].sum(-1).mean() / C["A"][:, 0, P:].sum(-1).mean())
            # attention level of random pairs across queries vs the [CLS] estimate
            ii = torch.randint(0, Np, (64,), generator=g).to(x.device) + P
            jj = (ii - P + torch.randint(1, Np, (64,), generator=g).to(x.device)) % Np + P
            A, lg = C["A"].float(), C["lg"].float()
            dl = lg[:, :, ii] - lg[:, :, jj]                               # (H, N, 64)
            du = A[:, :, ii] - A[:, :, jj]
            ut = torch.where(dl.abs() > 1e-3, du / dl.where(dl.abs() > 1e-3, torch.ones_like(dl)),
                             0.5 * (A[:, :, ii] + A[:, :, jj]))            # mean-value-theorem level per query
            cv = (ut.std(1) / ut.mean(1).abs().clamp_min(1e-12)).median()
            ucls = (A[:, 0, ii] * A[:, 0, jj]).sqrt()                      # (H, 64)
            umean = ut.mean(1)
            rho_cls = float(np.mean([spearman_np(ucls[hh].cpu().numpy(), umean[hh].cpu().numpy()) for hh in range(H)]))
            logratio = float((ucls.clamp_min(1e-12).log() - umean.clamp_min(1e-12).log()).abs().median())
            blockstats.append(dict(img=img_id, block=l, pr=float(pr.mean()), pr_min=float(pr.min()),
                                   spec=spec[:min(d, 64)].cpu().tolist(), outlier_mass=om, level_cv=float(cv),
                                   rho_cls_level=rho_cls, cls_level_logratio=logratio))
        # ---- candidate pairs ---------------------------------------------------------------
        ii_all = torch.randperm(Np, generator=g)[:n_pairs].to(x.device)
        km = F.normalize(C["k"].mean(0)[P:].float(), dim=-1)
        sim = km @ km.T
        sim.fill_diagonal_(-2)
        vc = C["v"][:, P:].float().transpose(0, 1).reshape(Np, -1)
        vd = torch.cdist(vc, vc)
        vd.fill_diagonal_(float("inf"))
        partners = dict(key=sim.argmax(-1), val=vd.argmin(-1),
                        rand=(torch.arange(Np, device=x.device) + torch.randint(1, Np, (Np,), generator=g).to(x.device)) % Np)
        A, k, v = C["A"].float(), C["k"].float(), C["v"].float()
        An, vn = Cn["A"].float(), Cn["v"].float()
        W = blocks[l + 1].attn.proj.weight.float()                          # (C, H*d)
        gam = getattr(blocks[l + 1].ls1, "gamma", None)
        for pt_i, pt in enumerate(PARTNERS):
            ii = ii_all
            jj = partners[pt][ii]
            I, J = ii + P, jj + P
            n = ii.numel()
            # criteria at block l (what a merging rule sees)
            dk = (k[:, I] - k[:, J]).norm(dim=-1)                          # (H, n)
            dv = (v[:, I] - v[:, J]).norm(dim=-1)
            lvl = (A[:, 0, I].clamp_min(1e-12) * A[:, 0, J].clamp_min(1e-12)).sqrt()
            dA = (A[:, :, I] - A[:, :, J]).norm(dim=1)                      # (H, n) all queries
            q = C["q"].float()
            S = q.transpose(1, 2) @ q
            kdw = torch.einsum("hnd,hde,hne->hn", k[:, I] - k[:, J], S, k[:, I] - k[:, J]).clamp_min(0).sqrt()
            crit = dict(tome=1 - (km[ii] * km[jj]).sum(-1), attn=A[:, 0, I].mean(0),
                        exact=(0.5 * dA * dv).sum(0), exact_cls=(0.5 * (A[:, 0, I] - A[:, 0, J]).abs() * dv).sum(0),
                        lite=(lvl * dk * dv).sum(0), lite_noatt=(dk * dv).sum(0), lite_noval=(lvl * dk).sum(0),
                        kdist_white=kdw.mean(0), kdist=dk.mean(0))
            # first-order prediction at block l+1 (unmerged quantities of the next block)
            keepq = torch.ones(n, N, dtype=torch.bool, device=x.device)
            keepq[torch.arange(n), I] = False
            keepq[torch.arange(n), J] = False
            c = 0.5 * (An[:, :, J] - An[:, :, I])                          # (H, N, n)
            dvn = vn[:, I] - vn[:, J]                                      # (H, n, d)
            mq = keepq.T.float()                                           # (N, n)
            pred_h = (c.pow(2) * mq).sum(1).sqrt() * dvn.norm(dim=-1)      # (H, n)
            ph = torch.einsum("chd,hnd->hnc", W.view(W.shape[0], H, d), dvn)   # (H, n, C)
            if gam is not None:
                ph = ph * gam.float()
            pq = torch.einsum("hqn,hnc->nqc", c, ph)                       # (n, N, C)
            pred_proj = ((pq.pow(2).sum(-1)) * keepq.float()).sum(-1).sqrt()
            # actual merge, chunked
            out = {key: torch.empty(n, device=x.device) for key in
                   ("act_ideal", "act_sum", "act_cat", "act_proj", "dev_k", "dev_v", "d_cos", "d_kl", "d_margin", "d_flip")}
            for s0 in range(0, n, chunk):
                sl = slice(s0, min(n, s0 + chunk))
                i_, j_ = I[sl], J[sl]
                b = i_.numel()
                rows_ = torch.arange(b, device=x.device)
                hm = C["hm"].expand(b, N, -1).clone()
                hm[rows_, j_] = 0.5 * (C["hm"][0, i_] + C["hm"][0, j_])
                keep = torch.ones(b, N, dtype=torch.bool, device=x.device)
                keep[rows_, i_] = False
                hm = hm[keep].view(b, N - 1, -1)
                sz = torch.ones(b, N, device=x.device)
                sz[rows_, j_] = 2.0
                sz = sz[keep].view(b, N - 1)
                ls = sz.log()
                blk = blocks[l]
                xm = hm + blk.ls2(blk.mlp(blk.norm2(hm)))
                qn, kn2, vn2, _, _, ohm, om_ = attn_full(blocks[l + 1].attn, blocks[l + 1].norm1(xm), ls)
                # map clean token index -> merged index
                pos = torch.cumsum(keep.long(), 1) - 1                    # (b, N)
                jm = pos[rows_, j_]
                kq = keepq[sl]                                             # (b, N) queries other than i, j
                idx = pos.masked_fill(~kq, 0)
                ohm_q = ohm.permute(0, 2, 1, 3).gather(1, idx[..., None, None].expand(-1, -1, H, d))   # (b, N, H, d)
                d_oh = (ohm_q - Cn["oh"].permute(1, 0, 2)[None].float()) * kq[..., None, None]
                per_head = d_oh.pow(2).sum((1, 3)).sqrt()                  # (b, H)
                out["act_sum"][sl] = per_head.sum(-1)
                out["act_cat"][sl] = per_head.pow(2).sum(-1).sqrt()
                om2 = blocks[l + 1].ls1(om_).gather(1, idx[..., None].expand(-1, -1, om_.shape[-1]))
                out["act_proj"][sl] = ((om2 - Cn["o"][None].float()).pow(2).sum(-1) * kq).sum(-1).sqrt()
                # merged key/value vs size-weighted mean of the originals (block l+1)
                kmean = 0.5 * (Cn["k"][:, i_] + Cn["k"][:, j_]).float()   # (H, b, d)
                vmean = 0.5 * (Cn["v"][:, i_] + Cn["v"][:, j_]).float()
                kmer = kn2[rows_, :, jm].float().transpose(0, 1)           # (H, b, d)
                vmer = vn2[rows_, :, jm].float().transpose(0, 1)
                out["dev_k"][sl] = ((kmer - kmean).norm(dim=-1) / (Cn["k"][:, i_] - Cn["k"][:, j_]).float().norm(dim=-1).clamp_min(1e-6)).mean(0)
                out["dev_v"][sl] = ((vmer - vmean).norm(dim=-1) / (Cn["v"][:, i_] - Cn["v"][:, j_]).float().norm(dim=-1).clamp_min(1e-6)).mean(0)
                # idealised merge inside block l+1 (keys/values exactly the size-weighted means, clean queries):
                # isolates the first-order expansion from LayerNorm / MLP / projection effects
                kk = Cn["k"].float()[None].expand(b, -1, -1, -1).clone()
                vv = Cn["v"].float()[None].expand(b, -1, -1, -1).clone()
                kk[rows_, :, j_] = kmean.transpose(0, 1)
                vv[rows_, :, j_] = vmean.transpose(0, 1)
                kk = kk.permute(0, 2, 1, 3)[keep].view(b, N - 1, H, d).permute(0, 2, 1, 3)
                vv = vv.permute(0, 2, 1, 3)[keep].view(b, N - 1, H, d).permute(0, 2, 1, 3)
                lgi = (Cn["q"].float()[None] * blocks[l + 1].attn.scale) @ kk.transpose(-2, -1) + ls[:, None, None, :]
                ohi = lgi.softmax(-1) @ vv                                 # (b, H, N, d) for all clean queries
                d_i = (ohi - Cn["oh"].float()[None]) * kq[:, None, :, None]
                out["act_ideal"][sl] = d_i.pow(2).sum((2, 3)).sqrt().sum(-1)
                # rest of the network
                hcur = xm
                for blk2 in blocks[l + 1:]:
                    hcur = block_fwd(blk2, hcur, ls)
                f1, z1 = readout(model, hcur, probe)
                out["d_cos"][sl] = 1 - F.cosine_similarity(f1, f0, dim=-1)
                lp1 = z1.log_softmax(-1)
                out["d_kl"][sl] = (p0.exp() * (p0 - lp1)).sum(-1)
                zc = z1[:, c0]
                zo = z1.clone()
                zo[:, c0] = -1e9
                out["d_margin"][sl] = m0 - (zc - zo.max(-1).values)
                out["d_flip"][sl] = (z1.argmax(-1) != c0).float()
            for t in range(n):
                rows.append([l, pt_i, img_id, float(out["act_ideal"][t]), float(out["act_sum"][t]), float(out["act_cat"][t]), float(out["act_proj"][t]),
                             float(pred_h[:, t].sum()), float(pred_h[:, t].pow(2).sum().sqrt()), float(pred_proj[t]),
                             float(out["dev_k"][t]), float(out["dev_v"][t])] +
                            [float(crit[c_][t]) for c_ in ("tome", "attn", "exact", "exact_cls", "lite", "lite_noatt",
                                                            "lite_noval", "kdist_white", "kdist")] +
                            [float(out[c_][t]) for c_ in ("d_cos", "d_kl", "d_margin", "d_flip")])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--images", type=int, default=64)
    ap.add_argument("--pairs", type=int, default=32, help="tokens i per block and partner type")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/rev_diag")
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for name in a.models:
        fn = os.path.join(a.out, f"{name}.npz")
        if os.path.exists(fn):
            continue
        pf = f"probes/{name}.pt"
        probe = torch.load(pf, map_location=dev) if (("dinov2" in name or "clip" in name) and os.path.exists(pf)) else None
        if ("dinov2" in name or "clip" in name) and probe is None and not a.no_pretrained:
            print(f"  {name}: no probe file {pf}, skipped"); continue
        model = load_model(name, dev, pretrained=not a.no_pretrained, headless=probe is not None).float().eval()
        if probe is None and a.no_pretrained and ("dinov2" in name or "clip" in name):
            probe = dict(mean=torch.zeros(model.num_features, device=dev), std=torch.ones(model.num_features, device=dev),
                         W=torch.randn(1000, model.num_features, device=dev), b=torch.zeros(1000, device=dev))
        idx = per_class_indices(a.data, 1, seed=7)
        idx = idx[::max(1, len(idx) // a.images)][:a.images]
        dl = make_loader(model, a.data, batch_size=1, workers=a.workers, indices=idx)
        g = torch.Generator().manual_seed(0)
        rows, bstats = [], []
        t0 = time.time()
        for n_, (x, _) in enumerate(dl):
            rows += run_image(model, probe, x.to(dev).float(), a.pairs, a.chunk, g, n_, bstats)
            print(f"\r  {name}: {n_ + 1}/{len(idx)} images {time.time() - t0:.0f}s", end="", flush=True)
        print()
        np.savez_compressed(fn, data=np.array(rows, dtype=np.float32), fields=np.array(FIELDS),
                            depth=len(model.blocks), num_prefix=model.num_prefix_tokens)
        with open(os.path.join(a.out, f"{name}_blocks.json"), "w") as f:
            json.dump(bstats, f)
        D = np.array(rows, dtype=np.float32)
        col = {k: i for i, k in enumerate(FIELDS)}
        msg = []
        for c in ("act_sum", "act_proj"):
            pr = "pred_sum" if c == "act_sum" else "pred_proj"
            msg.append(f"rho(pred,{c})={spearman_np(D[:, col[pr]], D[:, col[c]]):.2f}")
        for c in ("tome", "exact", "lite"):
            msg.append(f"rho({c},cos)={spearman_np(D[:, col[c]], D[:, col['d_cos']]):.2f}")
        print("   ", "  ".join(msg))
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
