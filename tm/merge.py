"""Token merging operators: ToMe (Bolya et al., ICLR'23) and SAD-TM (Xie et al., CVPR'26).

All operators act on patch tokens only. Prefix tokens (CLS, distillation, registers)
are never merged. Every operator also returns updated per-token side information:
  size  : number of original patches each token represents (ToMe-style tracking)
  prior : per-token prior score o_i carried along with the token
  src   : (B, Np) map from every ORIGINAL patch position to its current token index
          (used for visualisation and outlier-overlap analysis in later parts)
"""
import torch
import torch.nn.functional as F


def _minmax(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-row min-max normalisation to [0, 1]; constant rows map to 0."""
    lo = t.amin(dim=-1, keepdim=True)
    hi = t.amax(dim=-1, keepdim=True)
    rng = hi - lo
    return torch.where(rng > eps, (t - lo) / rng.clamp_min(eps), torch.zeros_like(t))


def _remap_src(src: torch.Tensor, new_index_of_old: torch.Tensor) -> torch.Tensor:
    """src holds current token indices (patch-space); remap through old->new table (B, Np_old)."""
    return new_index_of_old.gather(1, src)


# ---------------------------------------------------------------------------
# SAD-TM  (Sec. 3.2, Eq. 8-12 of the paper)
# ---------------------------------------------------------------------------
def sadtm_merge(x, metric, cls_attn, prior, size, src, r, num_prefix,
                alpha=0.5, beta=0.5, weight_mode="importance", norm="attn", signal_w=None):
    """One SAD-TM merging step.

    x        : (B, N, C)   tokens after MHSA residual (Z'_l)
    metric   : (B, N, Ck)  similarity features (keys averaged over heads)
    cls_attn : (B, N)      head-averaged CLS attention a'_l (Eq. 9); prefix entries ignored
    prior    : (B, N)      saliency-outlier prior o_i carried with tokens; prefix entries ignored
    r        : number of patch tokens to merge away in this layer

    Importance m = alpha * f(a') + beta * g(o)  (Eq. 10). The paper's Fig. 1 shows a
    normalisation box only on the attention path, so the default `norm="attn"` min-max
    normalises a' per image and uses o raw. `norm="both"` also min-max normalises o (this
    stretches a near-flat dataset prior into a strong positional ranking); `norm="none"`
    uses both raw. The r lowest-importance tokens form the
    merging set M; each is merged into its most similar token in the candidate set C, weighted
    by importance (Eq. 12). Several tokens of M may merge into the same candidate.
    """
    B, N, C = x.shape
    P = num_prefix
    Np = N - P
    r = min(r, Np - 1)
    if r <= 0:
        return x, prior, size, src

    xp, mp, sp, op = x[:, P:], metric[:, P:], size[:, P:], prior[:, P:]
    a_raw, o_raw = cls_attn[:, P:].float(), op.float()
    if signal_w is not None:
        # Calibrated merging: blend the rank of CLS attention with the rank of key uniqueness
        # (-max cosine similarity to any other patch token, the ToMe redundancy signal).
        kn = F.normalize(mp.float(), dim=-1)
        ss = kn @ kn.transpose(1, 2)
        ss.diagonal(dim1=1, dim2=2).fill_(-2)
        uniq = -ss.amax(-1)
        rk = lambda t: t.argsort(-1).argsort(-1).float() / max(Np - 1, 1)
        m = signal_w * rk(a_raw) + (1.0 - signal_w) * rk(uniq)
    elif norm == "attn":
        m = alpha * _minmax(a_raw) + beta * o_raw
    elif norm == "both":
        m = alpha * _minmax(a_raw) + beta * _minmax(o_raw)
    elif norm == "none":
        m = alpha * a_raw + beta * o_raw
    else:
        raise ValueError(norm)

    order = m.argsort(dim=-1, descending=True)
    cand = order[:, : Np - r].sort(dim=-1).values  # keep original spatial order for determinism
    mrg = order[:, Np - r:]

    mn = F.normalize(mp.float(), dim=-1)
    sim = mn.gather(1, mrg[..., None].expand(-1, -1, mn.shape[-1])) @ \
        mn.gather(1, cand[..., None].expand(-1, -1, mn.shape[-1])).transpose(1, 2)  # (B, r, Nc)
    dst = sim.argmax(dim=-1)  # index into cand

    if weight_mode == "importance":          # Eq. 12 as written: raw importance (clamped > 0)
        w = m.clamp_min(1e-6)
    elif weight_mode == "importance_shift":  # v2 behaviour: shifted so min -> 0 (acts like pruning)
        w = (m - m.amin(dim=-1, keepdim=True)) + 1e-6
    elif weight_mode == "uniform":           # plain average of the merged pair(s)
        w = torch.ones_like(m)
    elif weight_mode == "size":
        w = sp.float()
    else:
        raise ValueError(weight_mode)

    def g(t, idx):
        return t.gather(1, idx[..., None].expand(-1, -1, t.shape[-1]) if t.dim() == 3 else idx)

    wc, wm = g(w, cand), g(w, mrg)
    xc, xm = g(xp, cand).float(), g(xp, mrg).float()
    num = xc * wc[..., None]
    num = num.scatter_add(1, dst[..., None].expand(-1, -1, C), xm * wm[..., None])
    den = wc.scatter_add(1, dst, wm)
    x_new = (num / den[..., None]).to(x.dtype)

    s_new = g(sp, cand).scatter_add(1, dst, g(sp, mrg))
    o_new = g(op, cand)  # merged token keeps the candidate's prior (assumption; unspecified in paper)

    # old patch index -> new patch index
    table = torch.empty(B, Np, dtype=torch.long, device=x.device)
    table.scatter_(1, cand, torch.arange(Np - r, device=x.device).expand(B, -1))
    table.scatter_(1, mrg, dst)
    src = _remap_src(src, table)

    x = torch.cat([x[:, :P], x_new], 1)
    prior = torch.cat([prior[:, :P], o_new], 1)
    size = torch.cat([size[:, :P], s_new], 1)
    return x, prior, size, src


# ---------------------------------------------------------------------------
# ToMe bipartite soft matching (size-weighted average, Bolya et al. 2023)
# ---------------------------------------------------------------------------
def tome_merge(x, metric, prior, size, src, r, num_prefix, scores=None, wts=None):
    """ToMe bipartite soft matching. `scores` (B, |A|, |B|), higher = better to merge, replaces the
    key cosine similarity (used by rpm_merge)."""
    B, N, C = x.shape
    P = num_prefix
    Np = N - P
    na = (Np + 1) // 2
    r = min(r, Np - na)  # cannot merge more than |B side| ... and at most |A side|
    r = min(r, na)
    if r <= 0:
        return x, prior, size, src

    mp = F.normalize(metric[:, P:].float(), dim=-1)
    a, b = mp[:, ::2], mp[:, 1::2]
    if scores is None:
        scores = a @ b.transpose(1, 2)
    node_max, node_idx = scores.max(dim=-1)
    edge = node_max.argsort(dim=-1, descending=True)
    unm, srci = edge[:, r:], edge[:, :r]           # indices into A
    dsti = node_idx.gather(1, srci)                 # indices into B
    nb = b.shape[1]

    xp, sp, op = x[:, P:].float(), size[:, P:].float(), prior[:, P:]
    xa, xb = xp[:, ::2], xp[:, 1::2]
    sa, sb = sp[:, ::2], sp[:, 1::2]

    def g(t, idx):
        return t.gather(1, idx[..., None].expand(-1, -1, t.shape[-1]) if t.dim() == 3 else idx)

    if wts is None:
        num_b = (xb * sb[..., None]).scatter_add(1, dsti[..., None].expand(-1, -1, C),
                                                 g(xa, srci) * g(sa, srci)[..., None])
        s_b = sb.scatter_add(1, dsti, g(sa, srci))
        x_b = num_b / s_b[..., None]
    else:                                         # weighted average with per-token weights (sizes still summed)
        wp = wts.float()
        wa, wb = wp[:, ::2], wp[:, 1::2]
        num_b = (xb * wb[..., None]).scatter_add(1, dsti[..., None].expand(-1, -1, C),
                                                 g(xa, srci) * g(wa, srci)[..., None])
        x_b = num_b / wb.scatter_add(1, dsti, g(wa, srci))[..., None]
        s_b = sb.scatter_add(1, dsti, g(sa, srci))
    x_new = torch.cat([g(xa, unm), x_b], 1).to(x.dtype)
    s_new = torch.cat([g(sa, unm), s_b], 1).to(size.dtype)
    o_new = torch.cat([g(op[:, ::2], unm), op[:, 1::2]], 1)

    # old patch index -> new index.  A-side even positions, B-side odd positions.
    n_unm = na - r
    table = torch.empty(B, Np, dtype=torch.long, device=x.device)
    a_pos = torch.arange(0, Np, 2, device=x.device)
    b_pos = torch.arange(1, Np, 2, device=x.device)
    table[:, b_pos] = (n_unm + torch.arange(nb, device=x.device)).expand(B, -1)
    table.scatter_(1, a_pos[unm], torch.arange(n_unm, device=x.device).expand(B, -1))
    table.scatter_(1, a_pos[srci], n_unm + dsti)
    src = _remap_src(src, table)

    x = torch.cat([x[:, :P], x_new], 1)
    prior = torch.cat([prior[:, :P], o_new], 1)
    size = torch.cat([size[:, :P], s_new], 1)
    return x, prior, size, src


# ---------------------------------------------------------------------------
# Readout-preserving merging (ours)
# ---------------------------------------------------------------------------
def rpm_cost(extra, size, num_prefix, full, wo=False, second=False, fro=False, sqrtq=False):
    """Pairwise cost of merging A-token i with B-token j (bipartite split as in ToMe).

    A query q reads out_q = sum_t A_qt v_t (A includes log size, proportional attention). Merging i and j
    into their size-weighted average (attention mass A_qi + A_qj to first order) changes the readout by
        (A_qi s_j - A_qj s_i) / (s_i + s_j) * (v_i - v_j) = s_i s_j / (s_i + s_j) * (u_qi - u_qj) * (v_i - v_j)
    with u = A / s the per-unit attention. Cost = sum over heads (and over queries, weighted by their size,
    if full) of |u_qi - u_qj| * ||v_i - v_j||, times s_i s_j / (s_i + s_j).
      full=False: CLS query only (cheap; one attention row)    full=True: every query (needs the attention matrix)
    """
    P = num_prefix
    v = extra["v"][:, :, P:].float()                                       # (B, H, Np, d)
    s = size[:, P:].float()                                                # (B, Np)
    B, H, Np, d = v.shape
    if wo:     # values as they reach the residual stream: gamma * W_o^h v^h  (B, H, Np, C)
        W = extra["wo"].float()
        v = torch.einsum("bhnd,chd->bhnc", v, W.view(W.shape[0], H, d))
        if extra.get("gamma") is not None:
            v = v * extra["gamma"].float()
        d = v.shape[-1]
    if full:
        A = extra["A"].float()                                             # (B, H, Q, N): all (Q = N) or sampled queries
        qs = size.float() if "qidx" not in extra else size.float()[:, extra["qidx"]]
        qw = qs[:, None, :, None]                                          # query weights (size)
        u = A[..., P:] / s[:, None, None, :]                               # (B, H, N, Np) per-unit attention
        u_w = u * (qw.sqrt() if sqrtq else qw)        # Frobenius over patches: a size-s query counts s times -> sqrt(s)
        ua, ub = u_w[..., ::2].transpose(-1, -2), u_w[..., 1::2].transpose(-1, -2)   # (B, H, Na|Nb, N)
        if fro:   # Frobenius norm of the rank-1 readout error over all queries: ||dA_col||_2 * ||dv||_2 (matmul-based)
            du = torch.cdist(ua.reshape(B * H, ua.shape[2], -1), ub.reshape(B * H, ub.shape[2], -1), p=2,
                             compute_mode="use_mm_for_euclid_dist")
        else:
            du = torch.cdist(ua.reshape(B * H, ua.shape[2], -1), ub.reshape(B * H, ub.shape[2], -1), p=1)
    else:
        u = extra["cls_h"][:, :, P:].float() / s[:, None, :]               # (B, H, Np)
        du = (u[..., ::2, None] - u[..., None, 1::2]).abs().reshape(B * H, (Np + 1) // 2, Np // 2)
    dv = torch.cdist(v[:, :, ::2].reshape(B * H, -1, d), v[:, :, 1::2].reshape(B * H, -1, d),
                     compute_mode="use_mm_for_euclid_dist")
    cost = (du * dv).view(B, H, du.shape[1], du.shape[2]).sum(1)          # (B, Na, Nb)
    sa, sb = s[:, ::2], s[:, 1::2]
    hm = (sa[:, :, None] * sb[:, None, :]) / (sa[:, :, None] + sb[:, None, :])
    cost = cost * hm
    if second:
        cost = cost + rpm_mass_cost(extra, size, P, u, sa, sb)
    return cost


_SKETCH = {}


def _sketch_matrix(n, k, device, dtype, seed=0):
    """Fixed Gaussian sketch (k, n) / sqrt(k), same for every layer, image and run (given the seed)."""
    key = (n, k, str(device), dtype, seed)
    if key not in _SKETCH:
        g = torch.Generator().manual_seed(seed)
        _SKETCH[key] = (torch.randn(k, n, generator=g) / k ** 0.5).to(device=device, dtype=dtype)
    return _SKETCH[key]


def _pair_dist(a, b):
    """Euclidean distances between the rows of a (n, Na, k) and b (n, Nb, k): one batched matmul and in-place
    elementwise ops (torch.cdist pads and copies its inputs and makes several passes over the output)."""
    d = torch.baddbmm(b.pow(2).sum(-1).unsqueeze(1), a, b.transpose(1, 2), alpha=-2.0)
    return d.add_(a.pow(2).sum(-1, keepdim=True)).clamp_min_(0).sqrt_()


def rpm_sketch_cost(extra, size, num_prefix, k=16, headavg=False, seed=0, impl=None):
    """C_fast with the query dimension compressed by a Gaussian sketch (Johnson-Lindenstrauss):
    ||sqrt(s_q) (u_qi - u_qj)||_2 over all N queries  ~=  ||G diag(sqrt(s)) (u_i - u_j)||_2 in k dimensions,
    exact in expectation for the squared norm. Reads the attention matrix once (one batched matmul, k x N x N per
    head). headavg: one sketch of the head-averaged attention and the value distance over all heads concatenated.
    impl: "cdist" (v34/v35 runs) or "mm" (default: same arithmetic, less memory traffic)."""
    impl = impl or SKETCH_IMPL
    P = num_prefix
    A = extra["A"]                                                         # (B, H, N, N)
    B, H, N, _ = A.shape
    s = size[:, P:].float()
    G = _sketch_matrix(N, k, A.device, A.dtype, seed)
    W = G[None] * size.float().sqrt()[:, None, :].to(A.dtype)              # (B, k, N) query weights folded in
    if headavg:
        Z = torch.matmul(W, A.mean(1))[:, None]                            # (B, 1, k, N)
    else:
        Z = torch.matmul(W[:, None], A)                                    # (B, H, k, N)
    Z = Z[..., P:].float() / s[:, None, None, :]                           # per-unit attention, sketched
    Zt = Z.transpose(-1, -2)                                               # (B, h, Np, k)
    h = Zt.shape[1]
    v = extra["v"][:, :, P:].float()                                       # (B, H, Np, d)
    if headavg:
        v = v.permute(0, 2, 1, 3).reshape(B, 1, v.shape[2], -1)            # (B, 1, Np, H*d)
    d = v.shape[-1]
    za, zb = Zt[:, :, ::2].reshape(B * h, -1, k), Zt[:, :, 1::2].reshape(B * h, -1, k)
    va, vb = v[:, :, ::2].reshape(B * h, -1, d), v[:, :, 1::2].reshape(B * h, -1, d)
    if impl == "cdist":
        du = torch.cdist(za, zb, compute_mode="use_mm_for_euclid_dist")
        dv = torch.cdist(va, vb, compute_mode="use_mm_for_euclid_dist")
        cost = (du * dv).view(B, h, du.shape[1], du.shape[2]).sum(1)
    else:
        du = _pair_dist(za, zb)
        cost = du.mul_(_pair_dist(va, vb)).view(B, h, du.shape[1], du.shape[2]).sum(1)
    sa, sb = s[:, ::2], s[:, 1::2]
    return cost * (sa[:, :, None] * sb[:, None, :]) / (sa[:, :, None] + sb[:, None, :])


SKETCH_IMPL = "mm"


def rpm_lite_cost_cand(extra, metric, size, num_prefix, whiten=True, attn=True, m=8, eps=1e-4):
    """RPM-lite evaluated only for each A-token's m most key-similar B-tokens (ToMe's similarity proposes, the
    readout cost decides). Per-head work drops from |A| x |B| to |A| x m pairs. Returns SCORES (higher = merge),
    -inf outside the candidates."""
    P = num_prefix
    mp = F.normalize(metric[:, P:].float(), dim=-1)
    sim = mp[:, ::2] @ mp[:, 1::2].transpose(1, 2)                         # (B, Na, Nb)  ToMe's similarity
    B, Na, Nb = sim.shape
    m = min(m, Nb)
    cand = sim.topk(m, dim=-1).indices                                     # (B, Na, m)
    k = extra["k"][:, :, P:].float()                                       # (B, H, Np, d)
    v = extra["v"][:, :, P:].float()
    s = size[:, P:].float()
    H, d = k.shape[1], k.shape[-1]
    if whiten:
        C = extra["qcov"].float()
        C = C + eps * C.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * torch.eye(d, device=C.device)
        k = k @ torch.linalg.cholesky(C)
    idx = cand.reshape(B, 1, Na * m, 1).expand(B, H, Na * m, d)
    def pair_norm(t):
        ta, tb = t[:, :, ::2], t[:, :, 1::2]
        tb = tb.gather(2, idx).view(B, H, Na, m, d)
        return (ta[:, :, :, None, :] - tb).norm(dim=-1)                    # (B, H, Na, m)
    c = pair_norm(k).mul_(pair_norm(v))
    if attn:
        u = (extra["cls_h"][:, :, P:].float() / s[:, None, :]).clamp_min(1e-12).sqrt()   # (B, H, Np)
        ub = u[:, :, 1::2].gather(2, cand.reshape(B, 1, Na * m).expand(B, H, Na * m)).view(B, H, Na, m)
        c.mul_(u[:, :, ::2, None]).mul_(ub)
    c = c.sum(1)                                                           # (B, Na, m)
    sa = s[:, ::2, None]
    sb = s[:, 1::2].gather(1, cand.reshape(B, Na * m)).view(B, Na, m)
    c = c * (sa * sb) / (sa + sb)
    return torch.full((B, Na, Nb), float("-inf"), device=sim.device, dtype=c.dtype).scatter_(2, cand, -c)


def adamerge_scores(metric, num_prefix, eps=1e-6):
    """AdaMerge (arXiv 2605.27465), matching part, re-implemented from the paper. Salience s = Ahat^T 1 with
    Ahat = softmax(X X^T) (rows), min-max normalised; pair score s_i^A * cos(x_i^A, x_j^B); merged tokens are the
    salience-weighted average. Choices the paper leaves open: X = the attention keys averaged over heads (ToMe's
    metric), logits scaled by 1/sqrt(d), ToMe's alternating bipartite split, top-r pairs by score. Its per-image
    adaptive merge count is replaced by the per-layer schedules every method chooses from under our protocol."""
    P = num_prefix
    x = metric[:, P:].float()
    aff = (x @ x.transpose(1, 2)) / x.shape[-1] ** 0.5
    sal = aff.softmax(-1).sum(1)                                          # column sums of the row-softmaxed affinity
    lo, hi = sal.min(-1, keepdim=True).values, sal.max(-1, keepdim=True).values
    sal = (sal - lo) / (hi - lo).clamp_min(eps)
    xn = F.normalize(x, dim=-1)
    scores = sal[:, ::2, None] * (xn[:, ::2] @ xn[:, 1::2].transpose(1, 2))
    return scores, sal + eps


def rpm_sum_cost(extra, size, num_prefix, alpha=0.5, mode="head"):
    """Control for the product: the SAME two factors as RPM-lite (attention-level-weighted key distance and value
    distance, per head), ADDED with weight alpha instead of multiplied. Each factor is divided by its mean over all
    pairs (per image and head) so the sum is unit-free, which is the most favourable form for a weighted sum."""
    P = num_prefix
    k = extra["k"][:, :, P:].float()
    v = extra["v"][:, :, P:].float()
    s = size[:, P:].float()
    B, H, Np, d = k.shape
    ka, kb = k[:, :, ::2].reshape(B * H, -1, d), k[:, :, 1::2].reshape(B * H, -1, d)
    va, vb = v[:, :, ::2].reshape(B * H, -1, d), v[:, :, 1::2].reshape(B * H, -1, d)
    t1 = _pair_dist(ka, kb)
    Na, Nb = t1.shape[1], t1.shape[2]
    t1 = t1.view(B, H, Na, Nb)
    u = (extra["cls_h"][:, :, P:].float() / s[:, None, :]).clamp_min(1e-12).sqrt()
    t1.mul_(u[:, :, ::2, None]).mul_(u[:, :, None, 1::2])
    t2 = _pair_dist(va, vb).view(B, H, Na, Nb)
    # mode (reviewer controls for the product):
    #   head  : each factor divided by its mean over pairs, per image AND head (original control), then weighted sum
    #   layer : each factor divided by ONE mean per image over all heads and pairs (keeps the heads' relative scale)
    #   geo   : sum_h sqrt(t1_h * t2_h)  (geometric mean; same per-head ranking as the product, different head pooling)
    #   nprod : sum_h (t1_h / m1_h) * (t2_h / m2_h)  (product with per-head normalised factors)
    if mode == "geo":
        cost = (t1 * t2).clamp_min(0).sqrt().sum(1)
    else:
        dims = (-1, -2) if mode in ("head", "nprod") else (1, -1, -2)
        m1 = t1.mean(dims, keepdim=True).clamp_min(1e-12)
        m2 = t2.mean(dims, keepdim=True).clamp_min(1e-12)
        if mode == "nprod":
            cost = ((t1 / m1) * (t2 / m2)).sum(1)
        else:
            cost = (alpha * t1 / m1 + (1 - alpha) * t2 / m2).sum(1)
    if FACTOR_LOG is not None:
        FACTOR_LOG.append(_factor_stats(t1, t2, s))
    sa, sb = s[:, ::2], s[:, 1::2]
    return cost * (sa[:, :, None] * sb[:, None, :]) / (sa[:, :, None] + sb[:, None, :])


FACTOR_LOG = None          # set to a list to record factor scales during merging (rev_factors.py)


def _factor_stats(t1, t2, s):
    """Scale of the attention factor t1 and value factor t2 (B, H, Na, Nb) in one block: per-head medians over all
    candidate pairs and over each A-token's cheapest RPM-lite pair, plus the token-size distribution."""
    prod = (t1 * t2).sum(1)                                           # RPM-lite cost without size factor
    j = prod.argmin(-1)                                               # (B, Na) cheapest partner
    sel = lambda t: t.gather(-1, j[:, None, :, None].expand(-1, t.shape[1], -1, 1))[..., 0]   # (B, H, Na)
    med = lambda t: t.flatten(2).median(-1).values.mean(0)            # (H,)
    return dict(t1_all=med(t1).tolist(), t2_all=med(t2).tolist(), t1_sel=med(sel(t1)).tolist(),
                t2_sel=med(sel(t2)).tolist(), size_mean=float(s.mean()), size_max=float(s.max()),
                n_tok=int(s.shape[1]))


def rpm_dense_cost(extra, size, num_prefix):
    """RPM-lite plus the REPLACEMENT error, for outputs that need every patch (dense prediction).
    Merging i and j changes (1) what every query reads (the readout error of Prop. 1) and (2) the merged tokens' own
    values, which are replaced by their size-weighted mean: summed over the s_i + s_j patches this second error is
    w ||v_i - v_j||^2 with w = s_i s_j / (s_i + s_j). Squared errors add, so per head
        cost_h = ||dv_h|| * sqrt( (w * ubar * sigma_h * sqrt(S) * ||dk_h||)^2 + w ),
    with ubar = sqrt(u_0i u_0j) (per-patch [CLS] attention level), sigma_h^2 = tr(Sigma_Q)/d the mean squared scaled
    query norm per dimension and S = sum of token sizes (number of patches), i.e. the isotropic form of Prop. 3.
    With the replacement term dropped this ranks pairs exactly like RPM-lite (rpmlnw). No free parameter."""
    P = num_prefix
    k = extra["k"][:, :, P:].float()
    v = extra["v"][:, :, P:].float()
    s = size[:, P:].float()
    B, H, Np, d = k.shape
    ka, kb = k[:, :, ::2].reshape(B * H, -1, d), k[:, :, 1::2].reshape(B * H, -1, d)
    va, vb = v[:, :, ::2].reshape(B * H, -1, d), v[:, :, 1::2].reshape(B * H, -1, d)
    dk = _pair_dist(ka, kb)
    Na, Nb = dk.shape[1], dk.shape[2]
    dk, dv = dk.view(B, H, Na, Nb), _pair_dist(va, vb).view(B, H, Na, Nb)
    u = (extra["cls_h"][:, :, P:].float() / s[:, None, :]).clamp_min(1e-12).sqrt()
    sa, sb = s[:, ::2], s[:, 1::2]
    w = ((sa[:, :, None] * sb[:, None, :]) / (sa[:, :, None] + sb[:, None, :]))[:, None]          # (B, 1, Na, Nb)
    sig2 = extra["qcov"].float().diagonal(dim1=-2, dim2=-1).mean(-1)                             # (B, H)
    S = size.float().sum(-1)                                                                      # (B,)
    scale = (sig2 * S[:, None]).sqrt()[:, :, None, None]
    readout = w * u[:, :, ::2, None] * u[:, :, None, 1::2] * scale * dk
    return (dv * (readout.square() + w).sqrt()).sum(1)


def rpm_lite_cost(extra, size, num_prefix, whiten=True, attn=True, eps=1e-4, allq=False):
    """Closed-form RPM without the attention matrix. To first order in the logit difference,
        u_qi - u_qj  ~=  u_q,ij * (q . (k_i - k_j))          (u = A / s, per-unit attention)
    Taking the per-unit attention level of the pair as query-independent (estimated from the [CLS] row,
    geometric mean of u_0i and u_0j), the size-weighted norm over all queries becomes
        ||sqrt(s_q) (u_qi - u_qj)||_2  ~=  u_ij * sqrt(sum_q s_q (q . dk)^2)  =  u_ij * ||L^T dk||,
    with L L^T the size-weighted second moment of the queries (one d x d matrix per head). So
        cost(i, j) = s_i s_j / (s_i + s_j) * sum_h  u_ij^h * ||L_h^T (k_i - k_j)^h|| * ||(v_i - v_j)^h||.
    whiten=False uses ||dk|| (isotropic queries); attn=False drops u_ij. With both off this is ToMe's key distance
    times the value distance, per head."""
    P = num_prefix
    k = extra["k"][:, :, P:].float()                                       # (B, H, Np, d)
    v = extra["v"][:, :, P:].float()
    s = size[:, P:].float()
    B, H, Np, d = k.shape
    if whiten:
        C = extra["qcov"].float()
        C = C + eps * C.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * torch.eye(d, device=C.device)
        k = k @ torch.linalg.cholesky(C)                                   # ||L^T dk|| = ||dk^T L||
    ka, kb = k[:, :, ::2].reshape(B * H, -1, d), k[:, :, 1::2].reshape(B * H, -1, d)
    va, vb = v[:, :, ::2].reshape(B * H, -1, d), v[:, :, 1::2].reshape(B * H, -1, d)
    cost = _pair_dist(ka, kb).mul_(_pair_dist(va, vb))                     # (B*H, Na, Nb)
    Na, Nb = cost.shape[1], cost.shape[2]
    cost = cost.view(B, H, Na, Nb)
    if attn:
        lvl = extra["A"].float().mean(2) if allq else extra["cls_h"].float()   # all queries (diagnostic) or [CLS]
        qc = extra.get("u0_clip")
        if qc:                                 # CLIP diagnosis: cap attention spikes (e.g. high-norm outlier tokens)
            cap = lvl[:, :, P:].quantile(qc, dim=-1, keepdim=True)
            lvl = torch.cat([lvl[:, :, :P], torch.minimum(lvl[:, :, P:], cap)], -1)
        u = (lvl[:, :, P:] / s[:, None, :]).clamp_min(1e-12).sqrt()        # (B, H, Np)
        cost.mul_(u[:, :, ::2, None]).mul_(u[:, :, None, 1::2])            # geometric mean of u_0i, u_0j
    cost = cost.sum(1)
    sa, sb = s[:, ::2], s[:, 1::2]
    return cost * (sa[:, :, None] * sb[:, None, :]) / (sa[:, :, None] + sb[:, None, :])


def rpm_mass_cost(extra, size, P, u, sa, sb, eps=1e-3):
    """Second-order term: the merged token's attention mass. With merged key = size-weighted mean key,
    the merged token receives (s_i + s_j) u_i^{w_i} u_j^{w_j} instead of s_i u_i + s_j u_j (w = s/(s_i+s_j)),
    i.e. it loses ~ (s_i u_i + s_j u_j) w_i w_j (log u_i - log u_j)^2 / 2 of attention, which the softmax
    hands to the other tokens; the readout moves by that mass times ~||v - out_q||. Summed over queries
    (weighted by their size) with r_q = sum_t A_qt ||v_t - out_q||. Computed with matmuls (no N^3 tensor)."""
    A = extra["A"].float()                                                 # (B, H, N, N)
    v = extra["v"].float()                                                 # (B, H, N, d)
    B, H, N, _ = A.shape
    o = A @ v                                                              # readouts (B, H, N, d)
    r = (A * torch.cdist(o.reshape(B * H, N, -1), v.reshape(B * H, N, -1)).view(B, H, N, N)).sum(-1)
    om = r * size.float()[:, None, :]                                      # (B, H, N) query weights
    l = torch.log(u + eps / N)                                             # (B, H, N, Np) log per-unit attention
    ua, ub, la, lb = u[..., ::2], u[..., 1::2], l[..., ::2], l[..., 1::2]
    wa, wb = om[..., None] * ua, om[..., None] * ub                        # (B, H, N, Na|Nb)
    mm = lambda x, y: x.transpose(-1, -2) @ y                              # sum over queries -> (B, H, Na, Nb)
    side_a = (wa * la ** 2).sum(2)[..., :, None] - 2 * mm(wa * la, lb) + mm(wa, lb ** 2)
    side_b = (wb * lb ** 2).sum(2)[..., None, :] - 2 * mm(la, wb * lb) + mm(la ** 2, wb)
    T = (sa[:, None, :, None] * side_a + sb[:, None, None, :] * side_b).sum(1)   # (B, Na, Nb)
    w = (sa[:, :, None] * sb[:, None, :]) / (sa[:, :, None] + sb[:, None, :]) ** 2
    return 0.5 * w * T.clamp_min(0)


def mctf_scores(extra, metric, size, P, tau_sim=1.0, tau_info=20.0, tau_size=40.0, drop=None):
    """MCTF (Lee et al., CVPR 2024) multi-criteria score: similarity x informativeness x size, with the
    official exponents. Re-implemented training-free inside our bipartite merger (size-weighted average;
    without its one-step-ahead attention and bidirectional matching)."""
    mp = F.normalize(metric[:, P:].float(), dim=-1)
    a, b = mp[:, ::2], mp[:, 1::2]
    w_sim = ((a @ b.transpose(1, 2) + 1) / 2).clamp_min(0) ** (1 / tau_sim)
    inv = 1 / extra["A"].float().mean(dim=(1, 2))[:, P:].clamp_min(1e-12)      # 1 / attention received
    inv = inv / inv.amax(1, keepdim=True)
    w_info = (inv[:, ::2, None] * inv[:, None, 1::2]) ** (1 / tau_info)
    isz = 1 / size[:, P:].float()
    isz = isz / isz.amax(1, keepdim=True)
    w_size = (isz[:, ::2, None] * isz[:, None, 1::2]) ** (1 / tau_size)
    if drop == "info":                         # component ablation (CLIP diagnosis)
        w_info = torch.ones_like(w_info)
    if drop == "size":
        w_size = torch.ones_like(w_size)
    return w_sim * w_info * w_size


def mctf_merge(x, metric, extra, prior, size, src, r, num_prefix):
    if effective_r("mctf", r, x.shape[1] - num_prefix) <= 0:
        return x, prior, size, src
    return tome_merge(x, metric, prior, size, src, r, num_prefix,
                      scores=mctf_scores(extra, metric, size, num_prefix, drop=extra.get("mctf_drop")))


def rpm_cost_topk(extra, metric, size, num_prefix, k=4):
    """RPM cost (all queries) evaluated only for each A-token's k most key-similar B-tokens (ToMe's matrix
    proposes, the readout cost decides). Same formula as rpm_cost(full=True); pairs outside the k candidates
    get -inf score. With k = |B| it equals the full RPM."""
    P = num_prefix
    mp = F.normalize(metric[:, P:].float(), dim=-1)
    a, b = mp[:, ::2], mp[:, 1::2]
    sim = a @ b.transpose(1, 2)                                            # (B, Na, Nb)
    Bsz, Na, Nb = sim.shape
    k = min(k, Nb)
    cand = sim.topk(k, dim=-1).indices                                     # (B, Na, k)
    A = extra["A"].float()                                                 # (B, H, N, N)
    v = extra["v"][:, :, P:].float()                                       # (B, H, Np, d)
    s = size[:, P:].float()
    H, N = A.shape[1], A.shape[2]
    qw = size.float()[:, None, :, None]
    u = A[..., P:] / s[:, None, None, :] * qw                              # (B, H, N, Np) size-weighted per-unit
    ua, ub = u[..., ::2], u[..., 1::2]                                     # (B, H, N, Na|Nb)
    idx = cand.reshape(Bsz, 1, 1, Na * k).expand(-1, H, N, -1)
    ubc = ub.gather(3, idx).view(Bsz, H, N, Na, k)
    du = (ua[..., None] - ubc).abs().sum(2)                                # (B, H, Na, k)
    va, vb = v[:, :, ::2], v[:, :, 1::2]
    vbc = vb.gather(2, cand.reshape(Bsz, 1, Na * k, 1).expand(-1, H, -1, v.shape[-1])).view(Bsz, H, Na, k, -1)
    dv = (va[:, :, :, None] - vbc).norm(dim=-1)                            # (B, H, Na, k)
    cost = (du * dv).sum(1)                                                # (B, Na, k)
    sa, sb = s[:, ::2], s[:, 1::2].gather(1, cand.reshape(Bsz, -1)).view(Bsz, Na, k)
    cost = cost * sa[..., None] * sb / (sa[..., None] + sb)
    scores = torch.full(sim.shape, float("-inf"), device=sim.device, dtype=cost.dtype)   # bf16-autocast safe
    return scores.scatter(2, cand, -cost)


def rpm_merge(x, metric, extra, prior, size, src, r, num_prefix, full=False, wo=False, second=False, topk=0, fro=False, sqrtq=False, sketch=0, headavg=False, seed=0, lite=None, cand=0, sum_alpha=None, allq=False, dense=False, sum_mode="head"):
    """Merge the r A-tokens whose cheapest merge (into some B-token) changes the attention readout least."""
    Np = x.shape[1] - num_prefix
    if effective_r("rpm", r, Np) <= 0:
        return x, prior, size, src
    if dense:
        return tome_merge(x, metric, prior, size, src, r, num_prefix, scores=-rpm_dense_cost(extra, size, num_prefix))
    if sum_alpha is not None:
        return tome_merge(x, metric, prior, size, src, r, num_prefix,
                          scores=-rpm_sum_cost(extra, size, num_prefix, sum_alpha, sum_mode))
    if lite is not None and cand:
        return tome_merge(x, metric, prior, size, src, r, num_prefix,
                          scores=rpm_lite_cost_cand(extra, metric, size, num_prefix, *lite, m=cand))
    if lite is not None:
        return tome_merge(x, metric, prior, size, src, r, num_prefix,
                          scores=-rpm_lite_cost(extra, size, num_prefix, *lite, allq=allq))
    if sketch:
        return tome_merge(x, metric, prior, size, src, r, num_prefix,
                          scores=-rpm_sketch_cost(extra, size, num_prefix, sketch, headavg, seed))
    if topk:
        return tome_merge(x, metric, prior, size, src, r, num_prefix,
                          scores=rpm_cost_topk(extra, metric, size, num_prefix, topk))
    return tome_merge(x, metric, prior, size, src, r, num_prefix,
                      scores=-rpm_cost(extra, size, num_prefix, full, wo, second, fro, sqrtq))


# ---------------------------------------------------------------------------
# PiToMe  (Tran et al., NeurIPS 2024): energy-based protection + bipartite merging
# ---------------------------------------------------------------------------
def pitome_merge(x, metric, prior, size, src, r, num_prefix, margin, alpha=1.0):
    """One PiToMe step (off-the-shelf), following the official implementation (hchautran/PiToMe,
    algo/pitome/merge.py: pitome_vision): energy E_i = mean_j ELU(cos(k_i, k_j) - m, alpha), the mean
    taken over all patch tokens including i. The 2r highest-energy tokens (large,
    redundant regions) are merge candidates, split alternately by energy rank into A and B; every
    A token merges into its most similar B token (size-weighted). Low-energy tokens are protected."""
    B, N, C = x.shape
    P = num_prefix
    Np = N - P
    r = min(r, Np // 2)
    if r <= 0:
        return x, prior, size, src
    kn = F.normalize(metric[:, P:].float(), dim=-1)
    sim = kn @ kn.transpose(1, 2)
    energy = F.elu(sim - margin, alpha=alpha).mean(-1)
    order = energy.argsort(dim=-1, descending=True)
    cand, protect = order[:, :2 * r], order[:, 2 * r:]
    ia, ib = cand[:, ::2], cand[:, 1::2]                     # r tokens each

    def g(t, idx):
        return t.gather(1, idx[..., None].expand(-1, -1, t.shape[-1]) if t.dim() == 3 else idx)

    dst = (g(kn, ia) @ g(kn, ib).transpose(1, 2)).argmax(-1)  # (B, r) index into B set
    xp, sp, op = x[:, P:].float(), size[:, P:].float(), prior[:, P:]
    xa, xb, sa, sb = g(xp, ia), g(xp, ib), g(sp, ia), g(sp, ib)
    num_b = (xb * sb[..., None]).scatter_add(1, dst[..., None].expand(-1, -1, C), xa * sa[..., None])
    s_b = sb.scatter_add(1, dst, sa)
    x_new = torch.cat([g(xp, protect), num_b / s_b[..., None]], 1).to(x.dtype)
    s_new = torch.cat([g(sp, protect), s_b], 1).to(size.dtype)
    o_new = torch.cat([g(op, protect), g(op, ib)], 1)
    n_prot = protect.shape[1]
    table = torch.empty(B, Np, dtype=torch.long, device=x.device)
    ar = lambda n: torch.arange(n, device=x.device).expand(B, -1)
    table.scatter_(1, protect, ar(n_prot))
    table.scatter_(1, ib, n_prot + ar(r))
    table.scatter_(1, ia, n_prot + dst)
    src = _remap_src(src, table)
    return (torch.cat([x[:, :P], x_new], 1), torch.cat([prior[:, :P], o_new], 1),
            torch.cat([size[:, :P], s_new], 1), src)


def evit_merge(x, cls_attn, prior, size, src, r, num_prefix):
    """Training-free EViT (Liang et al., ICLR 2022) in our framework: the r+1 patch tokens with the lowest [CLS]
    attention are fused into ONE token (average weighted by attention x size), the rest are kept, so the token count
    drops by r. The fused token is an ordinary patch token afterwards."""
    B, N, C = x.shape
    P = num_prefix
    Np = N - P
    k = min(r + 1, Np - 1)
    if k < 2:
        return x, prior, size, src
    a = cls_attn[:, P:].float()
    order = a.argsort(dim=-1)                                              # ascending attention
    fuse, keep = order[:, :k], order[:, k:].sort(dim=-1).values            # keep original order
    xp, sp, op = x[:, P:].float(), size[:, P:].float(), prior[:, P:]
    g = lambda t, idx: t.gather(1, idx[..., None].expand(-1, -1, t.shape[-1]) if t.dim() == 3 else idx)
    w = (g(a, fuse) * g(sp, fuse)).clamp_min(1e-12)
    xf = (g(xp, fuse) * w[..., None]).sum(1, keepdim=True) / w.sum(1, keepdim=True)[..., None]
    x_new = torch.cat([g(xp, keep), xf], 1).to(x.dtype)
    s_new = torch.cat([g(sp, keep), g(sp, fuse).sum(1, keepdim=True)], 1).to(size.dtype)
    o_new = torch.cat([g(op, keep), op.new_zeros(B, 1)], 1)
    nk = keep.shape[1]
    table = torch.empty(B, Np, dtype=torch.long, device=x.device)
    table.scatter_(1, keep, torch.arange(nk, device=x.device).expand(B, -1))
    table.scatter_(1, fuse, torch.full_like(fuse, nk))
    src = _remap_src(src, table)
    return torch.cat([x[:, :P], x_new], 1), torch.cat([prior[:, :P], o_new], 1), torch.cat([size[:, :P], s_new], 1), src


def effective_r(method: str, r: int, n_patch_tokens: int) -> int:
    """Number of tokens a method will actually remove given current patch count (for FLOP simulation)."""
    if r <= 0:
        return 0
    if method in ("sadtm", "calib"):
        return min(r, n_patch_tokens - 1)
    if method in ("tome", "rpm", "rpm_all", "rpm_wo", "rpm2", "mctf", "rpmq", "rpmk", "rpmf", "rpmf2", "rpms", "rpmsh", "rpml", "rpmlkv", "rpmlnw", "rpmlc", "rpmlnwc", "rpmlkvc", "rpmsum", "adamerge", "rpmlnwq", "adamerge2", "rpmd"):
        na = (n_patch_tokens + 1) // 2
        return max(0, min(r, na, n_patch_tokens - na))
    if method == "pitome":
        return max(0, min(r, n_patch_tokens // 2))
    if method == "evit":
        return max(0, min(r + 1, n_patch_tokens - 1) - 1)
    return 0
