"""Evaluate a timm ViT on ImageNet-val with no merging / ToMe / SAD-TM and a chosen prior.

Examples
  # baseline
  python eval_imagenet.py --model deit_small_patch16_224 --data /data/imagenet/val
  # SAD-TM at 3.0 GFLOPs with the paper's saliency prior
  python eval_imagenet.py --model deit_small_patch16_224 --data /data/imagenet/val \
      --method sadtm --prior saliency --prior-file priors/deit_small.pt --target-gflops 3.0
  # same budget, center-Gaussian prior instead of saliency
  ... --prior center
  # SAD-TM-DM (deferred merging, rho = 0.25)
  ... --rho 0.25
Every run appends one JSON line to --out.
"""
import argparse, json, math, os, time
import torch
from tm.device import pick_device, sync
from tm.data import load_model, make_loader
from tm.patch import apply_patch, configure, make_r_list, simulate_counts
from tm.flops import vit_gflops, find_r_for_gflops, find_geo_schedule, find_rel_schedule
from tm.priors import build_prior, saliency_outlier_scores


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="ImageNet val ImageFolder root")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--probe", default=None,
                    help="linear probe from train_probe.py: the backbone is loaded headless and classified "
                         "with this frozen head (DINOv2, CLIP)")
    ap.add_argument("--img-size", type=int, default=None,
                    help="input resolution (DINOv2 default 224); e.g. 448 for the high-resolution experiment")
    ap.add_argument("--target-ratio", type=float, default=None,
                    help="budget as a fraction of the unmerged GFLOPs (overrides --target-gflops)")
    ap.add_argument("--method", choices=["none", "tome", "sadtm", "calib", "hybrid", "pitome", "rpm", "rpm_all", "rpm_wo", "rpm2", "mctf", "rpmq", "rpmk", "rpmf", "rpmf2", "rpms", "rpmsh", "rpml", "rpmlkv", "rpmlnw", "rpmlc", "rpmlnwc", "rpmlkvc", "rpmsum", "adamerge", "rpmlnwq", "adamerge2", "rpmd", "evit"], default="none")
    ap.add_argument("--reliability", default="results/reliability.jsonl",
                    help="calib: per-layer reliabilities from measure_reliability.py")
    ap.add_argument("--blend", choices=["soft", "hard", "attn", "sim"], default="soft",
                    help="calib: soft = w_l = a+^g/(a+^g + s+^g); hard = pick the better signal per layer; "
                         "attn / sim = always that signal (ablations)")
    ap.add_argument("--gamma", type=float, default=1.0, help="calib soft-blend sharpness")
    ap.add_argument("--prior", default="uniform",
                    choices=["uniform", "saliency", "saliency_shuffled", "center", "anticenter", "random", "oracle"],
                    help="oracle = per-image saliency computed online with a backward pass (analysis only)")
    ap.add_argument("--prior-file", default=None)
    ap.add_argument("--sigma-frac", type=float, default=0.25)
    ap.add_argument("--no-match-stats", action="store_true",
                    help="by default center/anticenter/random priors are rescaled to the saliency prior's "
                         "mean and std (needs --prior-file) so all priors perturb the ranking equally")
    ap.add_argument("--prior-gain", type=float, default=1.0,
                    help="scale the prior's deviation from its mean: o <- mean + gain * (o - mean)")
    ap.add_argument("--seed", type=int, default=0, help="seed for random / shuffled priors")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of CLS attention; beta = 1 - alpha")
    ap.add_argument("--r", type=int, default=None)
    ap.add_argument("--target-gflops", type=float, default=None)
    ap.add_argument("--gflops-tol", type=float, default=0.0499,
                    help="papers report GFLOPs to 1 decimal; accept target+tol")
    ap.add_argument("--tau", type=int, default=None)
    ap.add_argument("--schedule", choices=["const", "geo", "rel"], default="const",
                    help="const: same r every block after tau (paper); geo: each block keeps a fixed fraction; "
                         "rel: reliability-aware budget allocation from the probe (no tau)")
    ap.add_argument("--rel-kappa", type=float, default=2.0,
                    help="rel schedule: merge fraction of block l proportional to (q_l / max q) ** kappa")
    ap.add_argument("--rho", type=float, default=None, help="SAD-TM-DM: tau = ceil(rho * depth)")
    ap.add_argument("--prop-attn", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--weight-mode", choices=["importance", "importance_shift", "uniform", "size"], default="importance")
    ap.add_argument("--norm", choices=["attn", "both", "none"], default="attn",
                    help="SAD-TM Eq.10 normalisation: attn = min-max CLS attention only (Fig. 1), both, none")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="evaluate on a subset (debug)")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (paper used fp32)")
    ap.add_argument("--throughput", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="random weights (pipeline test only)")
    ap.add_argument("--out", default="results/part1.jsonl")
    ap.add_argument("--tag", default="")
    ap.add_argument("--per-class", type=int, default=None,
                    help="evaluate on a class-balanced subset of --data (e.g. 10 train images per class for "
                         "hyper-parameter selection)")
    ap.add_argument("--split-seed", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true",
                    help="do nothing if an identical configuration is already in --out")
    ap.add_argument("--fixed-ops", choices=["edge2", "sw0.25", "sw0.33", "sw0.5"], default=None,
                    help="hybrid with a FIXED per-block operator rule instead of the probe (reviewer control): "
                         "edge2 = ToMe in the first 2 and last 2 blocks, attention elsewhere; "
                         "swF = ToMe before block ceil(F*depth), attention after (RAPID-style switch)")
    ap.add_argument("--outlier-mask", type=float, default=0.0,
                    help="zero the CLS attention of this fraction of highest-norm tokens before merging")
    ap.add_argument("--sel-exclude", type=int, default=0,
                    help="selection split: first remove the N/class linear-probe training images (seed 0)")
    ap.add_argument("--sketch-k", type=int, default=16, help="rpms/rpmsh: Gaussian sketch size over queries")
    ap.add_argument("--sum-alpha", type=float, default=0.5, help="rpmsum: weight of the attention factor in the weighted sum")
    ap.add_argument("--u0-clip", type=float, default=None,
                    help="RPM-lite diagnosis: cap the attention level at this quantile over patch tokens (e.g. 0.98)")
    ap.add_argument("--mctf-drop", choices=["info", "size"], default=None, help="MCTF component ablation")
    ap.add_argument("--sum-mode", default="head", choices=["head", "layer", "geo", "nprod"],
                    help="rpmsum normalisation/combination (reviewer controls): head (default), layer, geo, nprod")
    ap.add_argument("--cand-m", type=int, default=8, help="rpml*c: candidate partners per token (key-similar)")
    ap.add_argument("--sketch-seed", type=int, default=0, help="rpms/rpmsh: seed of the (fixed) Gaussian sketch")
    ap.add_argument("--attn-op", choices=["calib", "rpms", "rpmlnw"], default="calib",
                    help="hybrid: operator of the non-ToMe blocks (calib = class attention, rpms = sketched RPM)")
    ap.add_argument("--explicit-attn", action="store_true", help="diagnostic: explicit softmax attention for every method")
    ap.add_argument("--save-config", default=None,
                    help="save the configured merging state (no data, no timing) to this file and exit; used by bench_selected.py")
    ap.add_argument("--speed-only", action="store_true",
                    help="no data: measure fp32 and bf16 throughput of this configuration (batch 256)")
    ap.add_argument("--also-check", nargs="*", default=[],
                    help="with --skip-existing: also skip if the same run is in any of these files")
    ap.add_argument("--save-preds", action="store_true",
                    help="save the per-image correct/incorrect vector (for McNemar tests in significance.py)")
    return ap.parse_args()


@torch.no_grad()
def measure_throughput(model, dev, res, bs=256, iters=30, amp=False):
    x = torch.randn(bs, 3, res, res, device=dev)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp and dev == "cuda" else torch.autocast("cpu", enabled=False)
    with ctx:
        for _ in range(5):
            model(x)
        sync(dev)
        t = time.time()
        for _ in range(iters):
            model(x)
        sync(dev)
    return bs * iters / (time.time() - t)


def reliability_entry(a, depth):
    """Per-layer probe results for this model and resolution (last matching run wins)."""
    rel = None
    for l in open(a.reliability):
        if not l.strip():
            continue
        r = json.loads(l)
        if (r["model"] == a.model or r["model"].split(".")[0] == a.model.split(".")[0]) and \
                r.get("img_size", 224) == (a.img_size or 224):
            rel = r                                      # last matching run wins
    if rel is None or rel["depth"] != depth:
        raise SystemExit(f"no reliability entry for {a.model} at {a.img_size or 224}px in {a.reliability}")
    return rel


def calib_weights(a, depth):
    """Per-layer weight of the attention signal from calibrated reliabilities."""
    if a.blend == "attn":
        return [1.0] * depth
    if a.blend == "sim":
        return [0.0] * depth
    rel = reliability_entry(a, depth)
    ws = []
    for ra, rs in zip(rel["rho_attn_loo"], rel["rho_sim_loo"]):
        ra, rs = max(ra, 0.0), max(rs, 0.0)
        if a.blend == "hard":
            ws.append(1.0 if ra > rs else 0.0)
        else:
            ga, gs = ra ** a.gamma, rs ** a.gamma
            ws.append(0.5 if ga + gs == 0 else ga / (ga + gs))
    return ws


def main():
    a = parse()
    dev = pick_device()
    model = apply_patch(load_model(a.model, dev, pretrained=not a.no_pretrained, checkpoint=a.checkpoint,
                                   headless=a.probe is not None, img_size=a.img_size))
    img_size = int(model.patch_embed.img_size[0])
    depth = len(model.blocks)
    head = None
    if a.probe:
        pr = torch.load(a.probe, map_location=dev)
        head = lambda f: ((f.float() - pr["mean"]) / pr["std"]) @ pr["W"].T + pr["b"]
    if a.target_ratio is not None:
        a.target_gflops = round(a.target_ratio * vit_gflops(model, simulate_counts(model, "none", [0] * depth)), 3)
        a.gflops_tol = 0.0

    keep = None
    if a.fixed_ops:
        if a.method != "hybrid" or a.schedule == "rel":
            raise SystemExit("--fixed-ops needs --method hybrid and a const/geo schedule")
        if a.fixed_ops == "edge2":
            layer_ops = ["tome" if (i < 2 or i >= depth - 2) else "calib" for i in range(depth)]
        else:
            k = math.ceil(float(a.fixed_ops[2:]) * depth)
            layer_ops = ["tome" if i < k else "calib" for i in range(depth)]
        w_list = [1.0] * depth                       # attention blocks use class attention only (no probe)
    else:
        w_list = calib_weights(a, depth) if a.method in ("calib", "hybrid") else None
        layer_ops = ["calib" if w >= 0.5 else "tome" for w in w_list] if a.method == "hybrid" else None
    if layer_ops is not None and a.attn_op != "calib":
        layer_ops = [a.attn_op if o == "calib" else o for o in layer_ops]
    tau = a.tau if a.tau is not None else (math.ceil(a.rho * depth) if a.rho is not None else 0)
    if a.method == "none":
        r = 0
    elif a.r is not None:
        r = a.r
    elif a.target_gflops is not None:
        if a.schedule == "rel":
            ent = reliability_entry(a, depth)
            keep, r_geo, _ = find_rel_schedule(model, a.method, a.target_gflops + a.gflops_tol,
                                               list(zip(ent["rho_attn_loo"], ent["rho_sim_loo"])),
                                               a.rel_kappa, layer_ops=layer_ops)
            tau, r = 0, None
        elif a.schedule == "geo":
            keep, r_geo, _ = find_geo_schedule(model, a.method, a.target_gflops + a.gflops_tol, tau,
                                               layer_ops=layer_ops)
            r = None
        else:
            r, _ = find_r_for_gflops(model, a.method, a.target_gflops + a.gflops_tol, tau, layer_ops=layer_ops)
    else:
        raise SystemExit("give --r or --target-gflops")
    keep = keep if a.schedule in ("geo", "rel") and a.method != "none" else None
    if a.schedule == "rel":
        a.schedule = f"rel-k{a.rel_kappa:g}"         # recorded name, e.g. rel-k2
    r_list = r_geo if keep is not None else make_r_list(depth, r or 0, tau)
    gflops = vit_gflops(model, simulate_counts(model, a.method, r_list, layer_ops))
    base_gflops = vit_gflops(model, simulate_counts(model, "none", [0] * depth))

    sal = torch.load(a.prior_file, map_location="cpu")["mean"].float() if a.prior_file else None
    prior = None if a.prior == "oracle" else build_prior(a.prior, model, sal, seed=a.seed, sigma_frac=a.sigma_frac)
    matched = False
    if prior is not None and sal is not None and a.prior in ("center", "anticenter", "random") and not a.no_match_stats:
        prior = (prior - prior.mean()) / prior.std().clamp_min(1e-12) * sal.std() + sal.mean()
        matched = True
    if prior is not None and a.prior_gain != 1.0:
        prior = prior.mean() + a.prior_gain * (prior - prior.mean())
    prop = None if a.prop_attn == "auto" else (a.prop_attn == "on")
    # PiToMe margins as in the official patches: m_l = m0 (1 - l/L), m0 = 0.75 for image classification
    # (patch/deit.py) and 0.9 for CLIP (patch/clip_hf.py)
    model._tm["pitome_m0"] = 0.9 if "clip" in a.model else 0.75
    model._tm["sketch_k"] = a.sketch_k
    model._tm["sketch_seed"] = a.sketch_seed
    model._tm["cand_m"] = a.cand_m
    model._tm["sum_alpha"] = a.sum_alpha
    model._tm["sum_mode"] = a.sum_mode
    model._tm["u0_clip"] = a.u0_clip
    model._tm["mctf_drop"] = a.mctf_drop
    if a.explicit_attn:
        import tm.patch as _tp
        _tp.EXPLICIT_ATTENTION = True
    configure(model, a.method, r_list, alpha=a.alpha, prior=prior, prop_attn=prop, weight_mode=a.weight_mode,
              norm=a.norm, w_list=w_list, layer_ops=layer_ops, outlier_mask=a.outlier_mask)

    split = (f"sel{a.per_class}s{a.split_seed}" + (f"x{a.sel_exclude}" if a.sel_exclude else "")) if a.per_class else "val"
    if a.speed_only:
        split = "speed"
    prior_name = (a.prior if a.method == "sadtm" else
                  (("hyb-" if a.method == "hybrid" else "") + a.blend +
                   (f"-g{a.gamma:g}" if a.blend == "soft" and a.gamma != 1 else "")) if a.method in ("calib", "hybrid")
                  else "-")
    if a.method == "pitome":
        prior_name = "official" if model._tm["prop_attn"] else "official-noprop"            # v24: energy and margins as in the official code (v22 rows have "-")
    if a.u0_clip:
        prior_name = f"u0c{a.u0_clip:g}"
    if a.mctf_drop:
        prior_name = f"no{a.mctf_drop}"
    if a.method == "rpmsum":
        prior_name = {"head": f"a{a.sum_alpha:g}", "layer": f"L{a.sum_alpha:g}", "geo": "geo", "nprod": "nprod"}[a.sum_mode]
    if a.method in ("rpmlc", "rpmlnwc", "rpmlkvc"):
        prior_name = f"m{a.cand_m}"
    if a.method in ("rpms", "rpmsh"):
        prior_name = f"k{a.sketch_k}" + (f"-s{a.sketch_seed}" if a.sketch_seed else "")
    if a.explicit_attn:
        prior_name += "-explicit"
    if a.fixed_ops:
        prior_name = f"fix-{a.fixed_ops}"
    if a.method == "hybrid" and a.attn_op == "rpms":
        prior_name = ("r" + prior_name if a.fixed_ops else prior_name) + f"-k{a.sketch_k}"
    if a.method == "hybrid" and a.attn_op == "rpmlnw":
        prior_name += "-rpmlnw"
    if a.outlier_mask > 0:
        prior_name += f"-om{a.outlier_mask:g}"
    ident = dict(model=a.model, method=a.method, prior=prior_name, tau=tau, schedule=a.schedule,
                 target_gflops=a.target_gflops, split=split, probe=a.probe, weight_mode=a.weight_mode,
                 img_size=img_size)
    if a.save_config:
        st = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in model._tm.items()
              if k not in ("size", "src", "prior_tok", "counts")}
        os.makedirs(os.path.dirname(os.path.abspath(a.save_config)), exist_ok=True)
        torch.save(dict(model=a.model, img_size=img_size, headless=a.probe is not None, ident=ident,
                        gflops=round(gflops, 3), base_gflops=round(base_gflops, 3), target_ratio=a.target_ratio,
                        tm=st), a.save_config)
        print("saved", a.save_config)
        return
    if a.skip_existing:
        for fn in [a.out] + list(a.also_check):
          if not os.path.exists(fn):
            continue
          for l in open(fn):
            if l.strip():
                o = json.loads(l)
                o.setdefault("split", "val")
                o.setdefault("schedule", "const")
                o.setdefault("probe", None)
                o.setdefault("img_size", 224)          # every run before the high-res experiment was 224px
                if all(o.get(k) == v for k, v in ident.items()):
                    print("skip (already in", fn, ")")
                    return

    if a.speed_only:
        res = dict(ident, layer_ops=layer_ops, r=r if r is not None else r_list, gflops=round(gflops, 3),
                   base_gflops=round(base_gflops, 3), target_ratio=a.target_ratio, tag=a.tag,
                   prop_attn=model._tm["prop_attn"],
                   img_per_s_fp32=round(measure_throughput(model, dev, img_size, amp=False), 1),
                   img_per_s_bf16=round(measure_throughput(model, dev, img_size, amp=True), 1))
        print(json.dumps(res))
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")
        return

    if a.per_class:
        from tm.data import per_class_indices
        loader = make_loader(model, a.data, a.batch_size, a.workers,
                             indices=per_class_indices(a.data, a.per_class, seed=a.split_seed,
                                                       exclude_per_class=a.sel_exclude))
    else:
        loader = make_loader(model, a.data, a.batch_size, a.workers, limit=a.limit)
    correct = total = 0
    hits = []
    measured = None
    t0 = time.time()
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        if a.prior == "oracle":
            model._tm["active"] = False
            for p in model.parameters():
                p.requires_grad_(False)
            model._tm["prior"] = saliency_outlier_scores(model, x)
            model._tm["active"] = a.method != "none"
        with torch.no_grad(), torch.autocast("cuda" if dev == "cuda" else "cpu", dtype=torch.bfloat16, enabled=a.amp and dev == "cuda"):
            out = model(x)
            if head is not None:
                out = head(out)
        if measured is None and a.method != "none":
            measured = vit_gflops(model, model._tm["counts"])
        hit = out.argmax(-1) == y
        hits.append(hit.cpu())
        correct += hit.sum().item()
        total += y.numel()
        print(f"\r{total} imgs  acc={100 * correct / total:.2f}  {time.time() - t0:.0f}s", end="", flush=True)
    print()

    res = dict(model=a.model, img_size=img_size, method=a.method, prior=prior_name, split=split, layer_ops=layer_ops,
               w_list=[round(w, 3) for w in w_list] if w_list else None,
               alpha=a.alpha if a.method == "sadtm" else None, seed=a.seed, r=r if r is not None else r_list, tau=tau, schedule=a.schedule,
               keep=None if keep is None else round(keep, 4),
               gflops=round(gflops, 3), gflops_measured=None if measured is None else round(measured, 3),
               base_gflops=round(base_gflops, 3), flops_red=round(100 * (1 - gflops / base_gflops), 2),
               acc1=round(100 * correct / total, 3), n=total, prop_attn=model._tm["prop_attn"],
               weight_mode=a.weight_mode, norm=a.norm if a.method == "sadtm" else ("rank" if a.method in ("calib", "hybrid") else None),
               amp=a.amp, tag=a.tag, target_gflops=a.target_gflops, target_ratio=a.target_ratio,
               probe=a.probe,
               prior_gain=a.prior_gain, prior_matched=matched,
               prior_std=None if prior is None else round(prior.std().item(), 6))
    if a.save_preds:
        import numpy as np
        name = (f"{a.model}_{a.method}_{res['prior']}_s{a.seed}_g{res['gflops']}_t{tau}_{a.weight_mode}"
                f"_pa{int(model._tm['prop_attn'])}_n{a.norm}_gain{a.prior_gain:g}{('_' + a.tag) if a.tag else ''}"
                f"{'' if img_size == 224 else f'_px{img_size}'}"
                f"{'_' + a.schedule if a.schedule.startswith('rel') else ''}")
        pdir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "preds")
        os.makedirs(pdir, exist_ok=True)
        res["preds"] = os.path.join(pdir, name + ".npy")
        np.save(res["preds"], torch.cat(hits).numpy())
    if a.throughput and dev in ("cuda", "mps"):
        res["img_per_s"] = round(measure_throughput(model, dev, model.patch_embed.img_size[0], amp=a.amp), 1)
    print(json.dumps(res))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
