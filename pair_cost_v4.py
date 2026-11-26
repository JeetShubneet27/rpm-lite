"""v4 = v3 + "cheapest-pairs precision": merging only uses the CHEAPEST pairs, so besides the rank correlation over
all tokens we report, per block, the fraction of the 10% truly least damaging merges that each signal also puts in its
10% predicted cheapest (bp10). This is where a product (zero when either factor is zero) and a sum differ most.

Does the derived cost predict the damage a merge does? (mentor points 1 and 3)

For every block l < L-1, every patch token i and three partners (ToMe's key-nearest token, the token exact RPM
would choose, a random token) we merge (i, partner) inside block l, run the rest of the network, and record the
ACTUAL damage: 1 - cos(final feature, unmerged final feature). Spearman correlation with each predicted cost:

  tome        1 - cos(mean key)                              ToMe's criterion
  attn        A_0i                                           SAD-TM / attention-based importance
  exact       sum_h ||dA_col||_2 * ||dv||                    exact first-order readout error (RPM)
  lite        sum_h sqrt(u_0i u_0j) ||dk|| * ||dv||          RPM-lite (closed form, no attention matrix)
  lite_noatt  sum_h ||dk|| * ||dv||                          RPM-lite without the attention-level term
  lite_noval  sum_h sqrt(u_0i u_0j) ||dk||                   RPM-lite without the value term
  sumX_a      sum_h [a * T1/mean(T1) + (1-a) * T2/mean(T2)]  WEIGHTED SUM of the same two factors instead of
              their product, for a in ALPHAS; X = lite (T1 = sqrt(u u)||dk||) or exact (T1 = ||dA_col||),
              T2 = ||dv||. Both factors are normalised so the sum is unit-free (the best case for a sum).
Also saves raw (predicted, actual) points of 4 images for scatter plots.

    python pair_cost_v3.py --data $IMAGENET/train --images 16 --out results/pair_cost_v3.jsonl
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F
from measure_reliability import build, loader_for, embed, spearman
from validate_probe import blk_mlp, finish
from pair_cost_check import attn_parts, MODELS
from tm.device import pick_device

ALPHAS = (0.1, 0.25, 0.5, 0.75, 0.9)
SIGS = ("tome", "attn", "exact", "lite", "lite_noatt", "lite_noval") + \
       tuple(f"sumlite_{a}" for a in ALPHAS) + tuple(f"sumexact_{a}" for a in ALPHAS)
PAIRS = ("nn", "rpm", "rand")


def merge_cost(model, blk, hm, ref, P, ii, jj, l, chunk):
    N, C = hm.shape[1], hm.shape[2]
    out = torch.empty(ii.numel(), device=hm.device)
    for s0 in range(0, ii.numel(), chunk):
        i, j = ii[s0:s0 + chunk], jj[s0:s0 + chunk]
        n = i.numel()
        rows = torch.arange(n, device=hm.device)
        hv = hm.expand(n, N, C).clone()
        sz = torch.ones(n, N, device=hm.device)
        hv[rows, P + j] = 0.5 * (hm[0, P + i] + hm[0, P + j])
        sz[rows, P + j] = 2.0
        keep = torch.ones(n, N, dtype=torch.bool, device=hm.device)
        keep[rows, P + i] = False
        hv, sz = hv[keep].view(n, N - 1, C), sz[keep].view(n, N - 1)
        f = finish(model, blk_mlp(blk, hv), sz, l + 1)
        out[s0:s0 + n] = 1 - F.cosine_similarity(f.float(), ref.float(), dim=-1)
    return out


def pair_signals(A, v, k, P, jj):
    """A (H,N,N), v and k (H,N,d); partner jj (Np,) indexes patch tokens. Returns {signal: (Np,)}."""
    Np = A.shape[-1] - P
    ii = torch.arange(Np, device=A.device) + P
    j = jj + P
    dv = (v[:, ii] - v[:, j]).norm(dim=-1)                                # (H, Np)
    dk = (k[:, ii] - k[:, j]).norm(dim=-1)
    dA = (A[:, :, ii] - A[:, :, j]).norm(dim=1)                           # (H, Np) attention-column difference
    u0 = A[:, 0, :].clamp_min(1e-12)                                      # sizes are 1 here, so u = A
    lvl = (u0[:, ii] * u0[:, j]).sqrt()
    km = F.normalize(k.mean(0), dim=-1)
    t_lite = lvl * dk
    out = dict(tome=1 - (km[ii] * km[j]).sum(-1), attn=A[:, 0, ii].mean(0),
               exact=(dA * dv).sum(0), lite=(t_lite * dv).sum(0), lite_noatt=(dk * dv).sum(0),
               lite_noval=t_lite.sum(0))
    nrm = lambda t: t / t.mean(-1, keepdim=True).clamp_min(1e-12)
    for a in ALPHAS:
        out[f"sumlite_{a}"] = (a * nrm(t_lite) + (1 - a) * nrm(dv)).sum(0)
        out[f"sumexact_{a}"] = (a * nrm(dA) + (1 - a) * nrm(dv)).sum(0)
    return out


@torch.no_grad()
def run_image(model, x, chunk, g, keep_points):
    P = model.num_prefix_tokens
    h = embed(model, x)
    ref = finish(model, h, None, 0)
    parts, hs = [], [h]
    for blk in model.blocks:
        A, v, k, o = attn_parts(blk.attn, blk.norm1(hs[-1]))
        hm = hs[-1] + blk.ls1(o)
        parts.append((A[0], v[0], k[0], hm))
        hs.append(blk_mlp(blk, hm))
    res = {(p, s): [] for p in PAIRS for s in SIGS}
    res.update({(p, s, "bp10"): [] for p in PAIRS for s in SIGS})
    points = []
    for l, blk in enumerate(list(model.blocks)[:-1]):
        A, v, k, hm = parts[l]
        Np = hm.shape[1] - P
        km = F.normalize(k.mean(0)[P:], dim=-1)
        sim = km @ km.T
        sim.fill_diagonal_(-2)
        nn = sim.argmax(-1)
        Ac = A[:, :, P:]
        cf = (torch.cdist(Ac.transpose(1, 2), Ac.transpose(1, 2)) * torch.cdist(v[:, P:], v[:, P:])).sum(0)
        cf.fill_diagonal_(float("inf"))
        rp = cf.argmin(-1)
        rnd = (torch.arange(Np, device=x.device) + torch.randint(1, Np, (Np,), generator=g).to(x.device)) % Np
        ii = torch.arange(Np, device=x.device)
        for name, jj in (("nn", nn), ("rpm", rp), ("rand", rnd)):
            mc = merge_cost(model, blk, hm, ref, P, ii, jj, l, chunk)
            sig = pair_signals(A, v, k, P, jj)
            k10 = max(1, int(0.1 * Np))
            truth = set(mc.argsort()[:k10].tolist())
            for s in SIGS:
                res[(name, s)].append(float(spearman(sig[s][None].float(), mc[None])[0]))
                pred = set(sig[s].float().argsort()[:k10].tolist())
                res[(name, s, "bp10")].append(len(truth & pred) / k10)
            if keep_points and name == "rand":
                points.append(dict(block=l, actual=mc.cpu().tolist(),
                                   **{s: sig[s].float().cpu().tolist() for s in ("tome", "attn", "exact", "lite")}))
    return res, points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--images", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--out", default="results/pair_cost_v4.jsonl")
    a = ap.parse_args()
    dev = pick_device()
    g = torch.Generator().manual_seed(0)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["model"] for l in open(a.out) if l.strip()}
    for name in a.models:
        if name in done:
            continue
        model = build(name, pretrained=not a.no_pretrained).to(dev)
        per_img, pts = [], []
        t0 = time.time()
        for n, (x, _) in enumerate(loader_for(model, a.data, a.images, 1, a.workers, 0)):
            r, p = run_image(model, x.to(dev), a.chunk, g, keep_points=n < 4)
            per_img.append(r)
            pts += [dict(img=n, **q) for q in p]
            print(f"\r  {name}: {n + 1} imgs {time.time() - t0:.0f}s", end="", flush=True)
        print()
        out = dict(model=name, n=len(per_img))
        for p in PAIRS:
            for s in SIGS:
                M = torch.tensor([r[(p, s)] for r in per_img])                  # (images, blocks)
                img_mean = M.mean(1)
                bs = img_mean[torch.randint(0, len(img_mean), (1000, len(img_mean)), generator=g)].mean(1)
                L = M.shape[1]
                Q = torch.tensor([r[(p, s, "bp10")] for r in per_img]).mean(1)
                out[f"{p}/{s}/bp10"] = round(float(Q.mean()), 4)
                out[f"{p}/{s}"] = dict(mean=round(float(img_mean.mean()), 4), lo=round(float(bs.quantile(0.025)), 4),
                                       hi=round(float(bs.quantile(0.975)), 4),
                                       thirds=[round(float(M[:, a_:b_].mean()), 4) for a_, b_ in
                                               ((0, L // 3), (L // 3, 2 * L // 3), (2 * L // 3, L))])
        with open(a.out, "a") as f:
            f.write(json.dumps(out) + "\n")
        os.makedirs("results/pair_points_v4", exist_ok=True)
        with open(f"results/pair_points_v4/{name}.json", "w") as f:
            json.dump(pts, f)
        for p in PAIRS:
            best_sum = max((out[f"{p}/sum{x}_{al}"]["mean"], f"sum{x}_{al}") for x in ("lite", "exact") for al in ALPHAS)
            print(f"  [{p:4s}] " + "  ".join(f"{s} {out[f'{p}/{s}']['mean']:.3f}" for s in SIGS[:6])
                  + f"  | best weighted sum {best_sum[0]:.3f} ({best_sum[1]})")
        del model
        torch.cuda.empty_cache() if dev == "cuda" else None


if __name__ == "__main__":
    main()
