"""Saliency-outlier prior of SAD-TM (Eq. 5-7) and the control priors used in Part 1.

Convention everywhere: a prior is a vector o of length Np (one value per patch position,
raster order). Higher o  ->  higher importance  ->  less likely to be merged.
"""
import math
import torch


@torch.enable_grad()
def saliency_outlier_scores(model, images, labels=None):
    """Per-image saliency-outlier scores o_i (Eq. 5-7). Returns (B, Np).

    Eq. 5: s = max_c |dL/dx|, with L the logit of the predicted class (or the true label if
    given) -- the vanilla Simonyan et al. saliency map the paper cites.
    Eq. 6: Pearson correlation between the flattened saliency vectors of every patch pair.
    Eq. 7: o_i = mean_{j != i} r_ij.
    The model must run WITHOUT token merging here.
    """
    was_training = model.training
    model.eval()
    x = images.detach().clone().requires_grad_(True)
    logits = model(x)
    idx = logits.argmax(-1) if labels is None else labels
    score = logits.gather(1, idx[:, None]).sum()
    grad, = torch.autograd.grad(score, x)
    model.train(was_training)

    s = grad.abs().amax(dim=1)  # (B, H, W)
    ph, pw = model.patch_embed.patch_size
    B, H, W = s.shape
    gh, gw = H // ph, W // pw
    patches = s[:, : gh * ph, : gw * pw].reshape(B, gh, ph, gw, pw).permute(0, 1, 3, 2, 4).reshape(B, gh * gw, ph * pw)
    z = patches - patches.mean(-1, keepdim=True)
    z = z / (z.pow(2).mean(-1, keepdim=True).sqrt() + 1e-12)
    r = z @ z.transpose(1, 2) / (ph * pw)          # (B, Np, Np) Pearson
    Np = gh * gw
    o = (r.sum(-1) - r.diagonal(dim1=1, dim2=2)) / (Np - 1)
    return o.detach()


def grid_shape(model):
    return tuple(model.patch_embed.grid_size)


def center_prior(grid, sigma_frac=0.25):
    gh, gw = grid
    yy, xx = torch.meshgrid(torch.arange(gh, dtype=torch.float32), torch.arange(gw, dtype=torch.float32), indexing="ij")
    cy, cx = (gh - 1) / 2, (gw - 1) / 2
    sig = sigma_frac * max(gh, gw)
    return torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2)).flatten()


def build_prior(kind, model, saliency_mean=None, seed=0, sigma_frac=0.25):
    """kind in {none/uniform, saliency, saliency_shuffled, center, anticenter, random}."""
    grid = grid_shape(model)
    Np = grid[0] * grid[1]
    g = torch.Generator().manual_seed(seed)
    if kind in ("none", "uniform"):
        return torch.zeros(Np)
    if kind == "saliency":
        assert saliency_mean is not None, "--prior-file required for saliency prior"
        return saliency_mean.float().flatten()
    if kind == "saliency_shuffled":
        assert saliency_mean is not None
        return saliency_mean.float().flatten()[torch.randperm(Np, generator=g)]
    if kind == "center":
        return center_prior(grid, sigma_frac)
    if kind == "anticenter":
        return -center_prior(grid, sigma_frac)
    if kind == "random":
        return torch.rand(Np, generator=g)
    raise ValueError(kind)


def rank_corr(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


def pearson(a, b):
    return torch.corrcoef(torch.stack([a.float(), b.float()]))[0, 1].item()
