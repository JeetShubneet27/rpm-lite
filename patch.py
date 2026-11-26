"""Patch a timm VisionTransformer in place so every block can merge tokens.

Merging happens between MHSA and FFN, exactly as in SAD-TM (Eq. 2-3) and ToMe:
    Z'_l  = Z_{l-1} + MHSA(LN(Z_{l-1}))
    Z''_l = Merge(Z'_l)
    Z_l   = Z''_l + FFN(LN(Z''_l))

Usage:
    model = timm.create_model("deit_small_patch16_224", pretrained=True)
    apply_patch(model)
    configure(model, method="sadtm", r=8, tau=0, alpha=0.5, prior=prior_tensor)
"""
import math
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, Block

from .merge import evit_merge, sadtm_merge, tome_merge, pitome_merge, rpm_merge, mctf_merge, effective_r, adamerge_scores

EXPLICIT_ATTENTION = False   # True reproduces the original explicit-softmax implementation


def _mean_query(q, size, P):
    """Size-weighted mean of the patch-token queries (B, H, 1, d): the attention-level estimate for ViTs that pool
    with global average pooling and have no [CLS] token. O(N d) per head, no N x N attention matrix."""
    w = size[:, P:].float() if size is not None else torch.ones(q.shape[0], q.shape[2] - P, device=q.device)
    return ((q[:, :, P:].float() * w[:, None, :, None]).sum(2, keepdim=True) / w.sum(-1)[:, None, None, None]).to(q.dtype)


class MergeAttention(nn.Module):
    """Explicit attention that also returns head-averaged CLS attention (Eq. 8-9) and keys."""

    @staticmethod
    def forward_with_stats(attn_mod, x, size: Optional[torch.Tensor], prop_attn: bool, extra=None):
        """extra: optional dict; receives per-head CLS attention 'cls_h', values 'v' and, if
        extra['need'] == 'full', the attention matrix 'A' (explicit attention)."""
        B, N, _ = x.shape
        gate = attn_mod.gate(x).sigmoid() if getattr(attn_mod, "gate", None) is not None else None
        qkv = attn_mod.qkv(x).reshape(B, N, 3, attn_mod.num_heads, attn_mod.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)
        bias = None
        if prop_attn and size is not None:
            bias = size.float().log()[:, None, None, :].to(q.dtype)          # (B, 1, 1, N)
        full = extra is not None and extra.get("need") == "full"
        if EXPLICIT_ATTENTION or full:
            attn = (q * attn_mod.scale) @ k.transpose(-2, -1)
            if bias is not None:
                attn = attn + bias
            attn = attn.softmax(dim=-1)
            gp = getattr(attn_mod, "_tm_gap", None)
            if gp is not None:                 # no [CLS] token: attention row of the size-weighted mean query
                qm = _mean_query(q, size, gp)
                crow = ((qm * attn_mod.scale) @ k.transpose(-2, -1) + (0 if bias is None else bias)).float().softmax(-1)[:, :, 0, :]
            else:
                crow = attn[:, :, 0, :]
            cls_attn = crow.mean(1)
            if extra is not None:
                extra["cls_h"] = crow
                if full:
                    extra["A"] = attn
            out = attn_mod.attn_drop(attn) @ v
        else:
            # fused kernel for the full attention; only the [CLS] query row is computed explicitly
            # (N dot products per head), which is all that attention-guided merging needs.
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None if bias is None else bias.expand(B, 1, N, N),
                dropout_p=attn_mod.attn_drop.p if attn_mod.training else 0.0)
            if extra is not None and extra.get("need") == "rows":
                # explicit attention rows for a few queries only: [CLS] + K evenly spaced tokens
                K = extra.get("k", 16)
                idx = torch.unique(torch.cat([torch.zeros(1, dtype=torch.long, device=x.device),
                                              torch.linspace(1, N - 1, K, device=x.device).round().long()]))
                rows = (q[:, :, idx] * attn_mod.scale) @ k.transpose(-2, -1)
                if bias is not None:
                    rows = rows + bias
                extra["A"] = rows.float().softmax(dim=-1)                  # (B, H, Q, N)
                extra["qidx"] = idx
            gp = getattr(attn_mod, "_tm_gap", None)
            q0 = _mean_query(q, size, gp) if gp is not None else q[:, :, :1]
            row = (q0 * attn_mod.scale) @ k.transpose(-2, -1)               # (B, H, 1, N)
            if bias is not None:
                row = row + bias
            cls_h = row.float().softmax(dim=-1)[:, :, 0, :]                # (B, H, N)
            cls_attn = cls_h.mean(1)                                       # (B, N), head-averaged
            if extra is not None:
                extra["cls_h"] = cls_h
        if extra is not None:
            extra["v"] = v
            if extra.get("need") == "lite" or extra.get("want_k"):
                # RPM-lite: keys and the size-weighted second moment of the queries, per head (B, H, d, d)
                qs = q.float() * attn_mod.scale
                w = size.float() if size is not None else torch.ones(B, N, device=x.device)
                extra["k"] = k
                extra["qcov"] = (qs * w[:, None, :, None]).transpose(-1, -2) @ qs / w.sum(-1)[:, None, None, None]
        out = out.transpose(1, 2).reshape(B, N, attn_mod.attn_dim)
        out = attn_mod.norm(out)
        if gate is not None:
            out = out * gate
        out = attn_mod.proj_drop(attn_mod.proj(out))
        return out, k.mean(1), cls_attn


class MergeBlock(Block):
    def forward(self, x, attn_mask=None, is_causal=False):
        st = self._tm
        if not st["active"]:
            return super().forward(x, attn_mask=attn_mask, is_causal=is_causal)

        B, N, _ = x.shape
        P = st["num_prefix"]
        if st["size"] is None:  # first block of this forward pass: initialise side info
            st["size"] = torch.ones(B, N, device=x.device, dtype=torch.float32)
            pr = st["prior"]
            if pr is None:
                pr = torch.zeros(N - P, device=x.device)
            pr = pr.to(x.device).float()
            if pr.dim() == 1:
                pr = pr.expand(B, -1)
            st["prior_tok"] = torch.cat([torch.zeros(B, P, device=x.device), pr], 1)
            st["src"] = torch.arange(N - P, device=x.device).expand(B, -1).clone()
            st["counts"] = []

        r = st["r_list"][self._tm_idx]
        op = st["method"]
        if op == "hybrid":                     # per-layer operator chosen by calibration
            op = st["layer_ops"][self._tm_idx]
        extra = None
        if r > 0 and op in ("rpm", "rpm_all", "rpm_wo", "rpm2", "mctf", "rpmq", "rpmk", "rpmf", "rpmf2", "rpms", "rpmsh", "rpml", "rpmlkv", "rpmlnw", "rpmlc", "rpmlnwc", "rpmlkvc", "rpmsum", "rpmlnwq", "rpmd"):
            extra = {"need": "cls" if op == "rpm" else ("rows" if op == "rpmq" else ("full" if op == "rpmlnwq" else ("lite" if op.startswith("rpml") or op in ("rpmsum", "rpmd") else "full")))}
            if op == "rpmlnwq":
                extra["want_k"] = True
            extra["u0_clip"] = st.get("u0_clip")
            extra["mctf_drop"] = st.get("mctf_drop")
            if op == "rpm_wo":
                extra["wo"] = self.attn.proj.weight
                extra["gamma"] = getattr(self.ls1, "gamma", None)
        attn_out, metric, cls_attn = MergeAttention.forward_with_stats(
            self.attn, self.norm1(x), st["size"], st["prop_attn"], extra)
        om = st.get("omask", 0.0)
        if om > 0 and N - P > 1:
            # outlier-masked class attention: zero the CLS attention of the top-om highest-norm patch
            # tokens (same definition as the probe's outlier_mass) and renormalise over patch tokens
            nrm = x[:, P:].float().norm(dim=-1)
            k = max(1, int(math.ceil(om * (N - P))))
            idx = nrm.topk(k, dim=-1).indices + P
            cls_attn = cls_attn.clone().scatter_(1, idx, 0.0)
            cls_attn[:, P:] = cls_attn[:, P:] / cls_attn[:, P:].sum(-1, keepdim=True).clamp_min(1e-12)
        x = x + self.drop_path1(self.ls1(attn_out))
        n_attn = x.shape[1]

        if r > 0:
            if op in ("rpm", "rpm_all", "rpm_wo", "rpm2", "rpmq", "rpmk", "rpmf", "rpmf2", "rpms", "rpmsh", "rpml", "rpmlkv", "rpmlnw", "rpmlc", "rpmlnwc", "rpmlkvc", "rpmsum", "rpmlnwq", "rpmd"):
                x, st["prior_tok"], st["size"], st["src"] = rpm_merge(
                    x, metric, extra, st["prior_tok"], st["size"], st["src"], r, P, full=op != "rpm",
                    wo=op == "rpm_wo", second=op == "rpm2", topk=4 if op == "rpmk" else 0, fro=op in ("rpmf", "rpmf2"),
                    sqrtq=op == "rpmf2", sketch=st.get("sketch_k", 16) if op in ("rpms", "rpmsh") else 0,
                    headavg=op == "rpmsh", seed=st.get("sketch_seed", 0),
                    lite=None if not op.startswith("rpml") else dict(rpml=(True, True), rpmlnw=(False, True), rpmlkv=(False, False), rpmlnwq=(False, True))[op if op == "rpmlnwq" else op.rstrip("c")],
                    allq=op == "rpmlnwq",
                    cand=st.get("cand_m", 8) if op.startswith("rpml") and op.endswith("c") else 0,
                    sum_alpha=st.get("sum_alpha", 0.5) if op == "rpmsum" else None, dense=op == "rpmd",
                    sum_mode=st.get("sum_mode", "head"))
            elif op in ("adamerge", "adamerge2"):
                sc, w = adamerge_scores(metric, P)
                x, st["prior_tok"], st["size"], st["src"] = tome_merge(
                    x, metric, st["prior_tok"], st["size"], st["src"], r, P, scores=sc, wts=w if op == "adamerge" else None)
            elif op == "mctf":
                x, st["prior_tok"], st["size"], st["src"] = mctf_merge(
                    x, metric, extra, st["prior_tok"], st["size"], st["src"], r, P)
            elif op in ("sadtm", "calib"):
                x, st["prior_tok"], st["size"], st["src"] = sadtm_merge(
                    x, metric, cls_attn, st["prior_tok"], st["size"], st["src"], r, P,
                    alpha=st["alpha"], beta=1.0 - st["alpha"], weight_mode=st["weight_mode"],
                    norm=st["norm"],
                    signal_w=st["w_list"][self._tm_idx] if op == "calib" else None)
            elif op == "pitome":
                depth = len(st["r_list"])
                x, st["prior_tok"], st["size"], st["src"] = pitome_merge(
                    x, metric, st["prior_tok"], st["size"], st["src"], r, P,
                    margin=st.get("pitome_m0", 0.75) * (1 - self._tm_idx / depth))
            elif op == "tome":
                x, st["prior_tok"], st["size"], st["src"] = tome_merge(
                    x, metric, st["prior_tok"], st["size"], st["src"], r, P)
            elif op == "evit":
                x, st["prior_tok"], st["size"], st["src"] = evit_merge(
                    x, cls_attn, st["prior_tok"], st["size"], st["src"], r, P)

        if st["record_cls_attn"]:
            st.setdefault("cls_attn_trace", []).append(cls_attn.detach())
        st["counts"].append((n_attn, x.shape[1]))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class MergeVisionTransformer(VisionTransformer):
    def forward_features(self, x, *args, **kwargs):
        self._tm["size"] = None
        self._tm.pop("cls_attn_trace", None)
        return super().forward_features(x, *args, **kwargs)


def apply_patch(model: VisionTransformer):
    assert isinstance(model, VisionTransformer), "only timm VisionTransformer is supported"
    st = dict(active=False, method="none", r_list=[0] * len(model.blocks), alpha=0.5,
              prior=None, prop_attn=False, weight_mode="importance", norm="attn", size=None,
              num_prefix=model.num_prefix_tokens, record_cls_attn=False)
    gap = getattr(model, "cls_token", None) is None and model.global_pool == "avg"
    if model.global_pool not in ("token", "map", "", "avg"):
        print(f"[warn] global_pool='{model.global_pool}': pooled output is not size-weighted after merging")
    model.__class__ = MergeVisionTransformer
    model._tm = st
    for i, blk in enumerate(model.blocks):
        blk.__class__ = MergeBlock
        blk._tm = st
        blk._tm_idx = i
        if gap:
            blk.attn._tm_gap = model.num_prefix_tokens
    if model.global_pool == "avg":
        def pool(x, pool_type=None, _m=model):          # size-weighted average of the patch tokens after merging
            P = _m.num_prefix_tokens
            s = _m._tm.get("size")
            if not _m._tm["active"] or s is None or s.shape[1] != x.shape[1]:
                return x[:, P:].mean(1)
            w = s[:, P:].to(x.dtype)
            return (x[:, P:] * w[..., None]).sum(1) / w.sum(1, keepdim=True)
        model.pool = pool
    return model


def make_r_list(depth: int, r: int, tau: int = 0) -> list:
    """Constant r per block, no merging in blocks with index < tau (SAD-TM-DM, Eq. 13)."""
    return [0 if i < tau else r for i in range(depth)]


def configure(model, method="none", r: Union[int, Sequence[int]] = 0, tau: int = 0, alpha=0.5,
              prior: Optional[torch.Tensor] = None, prop_attn=None, weight_mode="importance", norm="attn",
              w_list: Optional[Sequence[float]] = None, layer_ops: Optional[Sequence[str]] = None,
              outlier_mask: float = 0.0):
    st = model._tm
    depth = len(model.blocks)
    st["method"] = method
    st["active"] = method != "none"
    st["r_list"] = list(r) if isinstance(r, (list, tuple)) else make_r_list(depth, int(r), tau)
    st["alpha"] = alpha
    st["prior"] = prior
    st["weight_mode"] = weight_mode
    st["norm"] = norm
    st["w_list"] = list(w_list) if w_list is not None else [0.5] * depth
    st["layer_ops"] = list(layer_ops) if layer_ops is not None else [method] * depth
    # ToMe uses proportional attention by default; SAD-TM paper does not mention it.
    st["prop_attn"] = (method in ("tome", "hybrid", "pitome", "rpm", "rpm_all", "rpm_wo", "rpm2", "mctf", "rpmq", "rpmk", "rpmf", "rpmf2", "rpms", "rpmsh", "rpml", "rpmlkv", "rpmlnw", "rpmlc", "rpmlnwc", "rpmlkvc", "rpmsum", "rpmlnwq", "adamerge", "adamerge2", "rpmd", "evit")) if prop_attn is None else prop_attn
    st["omask"] = float(outlier_mask)
    return model


def simulate_counts(model, method, r_list, layer_ops=None):
    """Token counts (attn_in, mlp_in) per block without running the model."""
    P = model.num_prefix_tokens
    N = model.patch_embed.num_patches + P
    out = []
    for i, r in enumerate(r_list):
        n_attn = N
        if method != "none":
            m = layer_ops[i] if (method == "hybrid" and layer_ops is not None) else method
            N = N - effective_r(m, r, N - P)
        out.append((n_attn, N))
    return out
