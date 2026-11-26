import sys, json, glob, os, numpy as np, collections
from scipy.stats import rankdata, pearsonr
D_ = sys.argv[1]
NAME = {"deit_tiny_patch16_224": "DeiT-T", "deit_small_patch16_224": "DeiT-S", "deit_base_patch16_224": "DeiT-B",
        "deit3_small_patch16_224.fb_in1k": "DeiT-III-S", "deit3_base_patch16_224.fb_in1k": "DeiT-III-B",
        "vit_small_patch16_224.augreg_in21k_ft_in1k": "AugReg-S", "vit_base_patch16_224.augreg_in21k_ft_in1k": "AugReg-B",
        "vit_small_patch14_dinov2.lvd142m": "DINOv2-S", "vit_small_patch14_reg4_dinov2.lvd142m": "DINOv2-S+reg",
        "vit_base_patch14_dinov2.lvd142m": "DINOv2-B", "vit_base_patch14_reg4_dinov2.lvd142m": "DINOv2-B+reg",
        "vit_large_patch14_dinov2.lvd142m": "DINOv2-L", "vit_large_patch14_reg4_dinov2.lvd142m": "DINOv2-L+reg",
        "vit_base_patch16_clip_224.laion2b": "CLIP-B"}
def sp(a, b):
    if np.std(a) == 0 or np.std(b) == 0: return np.nan
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]
rng = np.random.default_rng(0)
res = {}
for m in NAME:
    z = np.load(f"{D_}/{m}.npz"); X = z["data"]; c = {k: i for i, k in enumerate(z["fields"])}; L = int(z["depth"])
    col = lambda k: X[:, c[k]]
    g = collections.defaultdict(list)
    for i, key in enumerate(zip(X[:, c["img"]].astype(int), X[:, c["block"]].astype(int), X[:, c["ptype"]].astype(int))): g[key].append(i)
    def grp(a, b, sel=None):
        """mean within-(image, block, partner) Spearman, per image means for CI"""
        per_img = collections.defaultdict(list); per_blk = collections.defaultdict(list)
        for (im, bl, pt), idx in g.items():
            if sel is not None and not sel(im, bl, pt): continue
            idx = np.array(idx); r = sp(col(a)[idx], col(b)[idx])
            if np.isfinite(r): per_img[im].append(r); per_blk[bl].append(r)
        v = np.array([np.mean(x) for x in per_img.values()])
        bs = rng.choice(v, (2000, len(v))).mean(1)
        return float(v.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975)), {b: float(np.mean(x)) for b, x in per_blk.items()}, v
    r = dict(depth=L)
    # --- theory: prediction vs actual change at block l+1
    for a, b in (("pred_sum", "act_ideal"), ("act_ideal", "act_sum"), ("pred_sum", "act_sum"), ("pred_proj", "act_proj"),
                 ("act_sum", "act_proj"), ("act_cat", "act_proj"), ("exact", "act_sum"), ("lite", "act_sum"), ("tome", "act_sum"),
                 ("lite", "exact")):
        mu, lo, hi, blk, _ = grp(a, b); r[f"{a}~{b}"] = (mu, lo, hi); r[f"{a}~{b}/blk"] = blk
    lp = np.log(np.clip(col("pred_sum"), 1e-12, None)); la = np.log(np.clip(col("act_sum"), 1e-12, None)); li = np.log(np.clip(col("act_ideal"), 1e-12, None))
    ok = np.isfinite(lp) & np.isfinite(la)
    r["pearson_log_pred_act"] = float(pearsonr(lp[ok], la[ok])[0]); r["R2_log_pred_act"] = r["pearson_log_pred_act"] ** 2
    r["pearson_log_pred_ideal"] = float(pearsonr(lp[ok], li[ok])[0])
    r["ratio_ideal_pred"] = float(np.median(col("act_ideal") / np.clip(col("pred_sum"), 1e-12, None)))
    r["ratio_act_pred"] = float(np.median(col("act_sum") / np.clip(col("pred_sum"), 1e-12, None)))
    r["ratio_act_ideal"] = float(np.median(col("act_sum") / np.clip(col("act_ideal"), 1e-12, None)))
    r["dev_k"] = float(np.median(col("dev_k"))); r["dev_v"] = float(np.median(col("dev_v")))
    # by key-distance tercile (dissimilarity) and depth third
    kd = col("kdist"); t1, t2 = np.quantile(kd, [1/3, 2/3])
    blk = col("block")
    r["by_dissim"] = {}
    for nm, s_ in (("low", kd <= t1), ("mid", (kd > t1) & (kd <= t2)), ("high", kd > t2)):
        r["by_dissim"][nm] = dict(rho=sp(col("pred_sum")[s_], col("act_sum")[s_]), rho_ideal=sp(col("pred_sum")[s_], col("act_ideal")[s_]),
                                  ratio_act_ideal=float(np.median(col("act_sum")[s_] / col("act_ideal")[s_])))
    r["by_depth"] = {}
    for nm, s_ in (("early", blk < L / 3), ("middle", (blk >= L / 3) & (blk < 2 * L / 3)), ("late", blk >= 2 * L / 3)):
        r["by_depth"][nm] = dict(ratio_act_ideal=float(np.median(col("act_sum")[s_] / col("act_ideal")[s_])),
                                 dev_k=float(np.median(col("dev_k")[s_])))
    # --- damage (item 6): criteria vs output damage, by partner type
    for pt_i, pt in enumerate(("key", "val", "rand")):
        for crit in ("tome", "attn", "exact", "exact_cls", "lite", "lite_noatt", "lite_noval"):
            for tgt in ("d_cos", "d_kl", "d_margin"):
                mu, lo, hi, _, v = grp(crit, tgt, sel=lambda im, bl, p, pt_i=pt_i: p == pt_i)
                r[f"{pt}/{crit}/{tgt}"] = (mu, lo, hi); r[f"{pt}/{crit}/{tgt}/img"] = v.tolist()
    r["flip_rate"] = float(col("d_flip").mean())
    # --- assumptions (item 5)
    B = json.load(open(f"{D_}/{m}_blocks.json"))
    for k in ("pr", "pr_min", "level_cv", "rho_cls_level", "cls_level_logratio", "outlier_mass"):
        r[k] = float(np.mean([b[k] for b in B]))
        r[k + "/blk"] = {bl: float(np.mean([b[k] for b in B if b["block"] == bl])) for bl in sorted({b["block"] for b in B})}
    r["spec"] = np.mean([b["spec"] for b in B], 0).tolist()
    res[m] = r
import pickle; pickle.dump(res, open(sys.path[0] + "/rd1.pkl", "wb"))
print(f"{'model':13s} rho(pred,ideal) rho(ideal,act) rho(pred,act) rho_proj  act/ideal  dev_k  R2log | PR  levelCV rhoCLS outl | lite~exact")
for m, r in res.items():
    print(f"{NAME[m]:13s} {r['pred_sum~act_ideal'][0]:.3f}  {r['act_ideal~act_sum'][0]:.3f}  {r['pred_sum~act_sum'][0]:.3f}  {r['pred_proj~act_proj'][0]:.3f}  "
          f"{r['ratio_act_ideal']:.2f}  {r['dev_k']:.2f}  {r['R2_log_pred_act']:.2f} | {r['pr']:.2f} {r['level_cv']:.2f} {r['rho_cls_level']:.2f} {r['outlier_mass']:.2f} | {r['lite~exact'][0]:.2f}")
print("aggregation: act_sum~act_proj, act_cat~act_proj, exact(l)~act(l+1), lite(l)~act(l+1), tome~act")
for m, r in res.items():
    print(f"  {NAME[m]:13s} {r['act_sum~act_proj'][0]:.3f} {r['act_cat~act_proj'][0]:.3f} {r['exact~act_sum'][0]:.3f} {r['lite~act_sum'][0]:.3f} {r['tome~act_sum'][0]:.3f}  ratio ideal/pred {r['ratio_ideal_pred']:.3f}")
