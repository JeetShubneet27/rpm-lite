"""Analytic GFLOPs (multiply-accumulates, the DeiT/ToMe/SAD-TM convention: DeiT-S = 4.6G)."""
from .patch import simulate_counts, make_r_list


def vit_gflops(model, counts):
    """counts: list of (tokens entering MHSA, tokens entering FFN) per block."""
    D = model.embed_dim
    blk = model.blocks[0]
    hidden = blk.mlp.fc1.out_features
    attn_dim = blk.attn.attn_dim
    pe = model.patch_embed
    in_ch = pe.proj.in_channels
    ps = pe.patch_size[0] * pe.patch_size[1]
    macs = pe.num_patches * ps * in_ch * D                      # patch embedding
    for n_attn, n_mlp in counts:
        macs += n_attn * D * 3 * attn_dim                       # qkv
        macs += 2 * n_attn * n_attn * attn_dim                  # q k^T and attn @ v
        macs += n_attn * attn_dim * D                           # proj
        macs += 2 * n_mlp * D * hidden                          # fc1 + fc2
    if getattr(model, "head", None) is not None and hasattr(model.head, "in_features"):
        macs += model.head.in_features * model.head.out_features
    return macs / 1e9


def find_r_for_gflops(model, method, target, tau=0, layer_ops=None):
    """Smallest r whose simulated GFLOPs <= target (constant r per block from tau on)."""
    depth = len(model.blocks)
    best = None
    for r in range(0, model.patch_embed.num_patches):
        g = vit_gflops(model, simulate_counts(model, method, make_r_list(depth, r, tau), layer_ops))
        if g <= target + 1e-9:
            best = (r, g)
            break
    if best is None:
        raise ValueError(f"cannot reach {target} GFLOPs with method={method}, tau={tau}")
    return best


def geo_r_list(model, method, keep, tau=0, layer_ops=None):
    """Geometric schedule: from block tau on, each block keeps a fraction `keep` of its patch tokens."""
    from .merge import effective_r
    depth = len(model.blocks)
    n = model.patch_embed.num_patches
    out = []
    for l in range(depth):
        if l < tau:
            out.append(0)
            continue
        r = int(round(n * (1.0 - keep)))
        r = effective_r(layer_ops[l] if (method == "hybrid" and layer_ops) else method, r, n)
        out.append(r)
        n -= r
    return out


def find_geo_schedule(model, method, target, tau=0, iters=40, layer_ops=None):
    """Largest keep ratio (least merging) whose geometric schedule meets the GFLOPs target."""
    lo, hi = 0.0, 1.0            # keep=0 merges everything; keep=1 merges nothing
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        rl = geo_r_list(model, method, mid, tau, layer_ops)
        g = vit_gflops(model, simulate_counts(model, method, rl, layer_ops))
        if g <= target + 1e-9:
            best = (mid, rl, g)
            lo = mid
        else:
            hi = mid
    if best is None:
        raise ValueError(f"cannot reach {target} GFLOPs with a geometric schedule (method={method}, tau={tau})")
    return best


def rel_r_list(model, method, scale, rel, kappa=2.0, layer_ops=None, max_frac=0.95):
    """Reliability-aware schedule: block l merges a fraction min(scale * s_l, max_frac) of its patch
    tokens, with s_l = (q_l / max_l q_l) ** kappa and q_l = max(rho_attn, rho_sim, 0) from the probe.
    Blocks where neither signal is reliable merge little or nothing; no tau, no schedule choice."""
    from .merge import effective_r
    q = [max(a, s, 0.0) for a, s in rel]
    top = max(q) if max(q) > 0 else 1.0
    n = model.patch_embed.num_patches
    out = []
    for l, ql in enumerate(q):
        frac = min(scale * (ql / top) ** kappa, max_frac)
        r = int(round(n * frac))
        r = effective_r(layer_ops[l] if (method == "hybrid" and layer_ops) else method, r, n)
        out.append(r)
        n -= r
    return out


def find_rel_schedule(model, method, target, rel, kappa=2.0, layer_ops=None, iters=40):
    """Smallest scale whose reliability-aware schedule meets the GFLOPs target (least merging)."""
    def g(scale):
        rl = rel_r_list(model, method, scale, rel, kappa, layer_ops)
        return vit_gflops(model, simulate_counts(model, method, rl, layer_ops)), rl
    hi = 1.0
    while g(hi)[0] > target + 1e-9:
        hi *= 2
        if hi > 1e4:
            raise ValueError(f"cannot reach {target} GFLOPs with a reliability-aware schedule (method={method})")
    lo = 0.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if g(mid)[0] <= target + 1e-9:
            hi = mid
        else:
            lo = mid
    gf, rl = g(hi)
    return hi, rl, gf
