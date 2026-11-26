"""Compute SAD-TM's offline saliency-outlier prior (Eq. 5-7) and save it.

Use images that are NOT the evaluation set (e.g. a subset of ImageNet train), so the prior
does not peek at val. The paper does not say which images it used; we record ours.

    python compute_prior.py --model deit_small_patch16_224 --data /data/imagenet/train \
        --num-images 10000 --out priors/deit_small.pt
"""
import argparse, os, time
import torch
from tm.data import load_model, make_loader
from tm.priors import saliency_outlier_scores, grid_shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="ImageFolder root")
    ap.add_argument("--num-images", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target", choices=["pred", "label"], default="pred")
    ap.add_argument("--keep-per-image", type=int, default=2000, help="store this many per-image o vectors")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true", help="random weights (pipeline test only)")
    args = ap.parse_args()

    from tm.device import pick_device
    dev = pick_device()
    model = load_model(args.model, dev, pretrained=not args.no_pretrained, checkpoint=args.checkpoint)
    for p in model.parameters():
        p.requires_grad_(False)
    loader = make_loader(model, args.data, args.batch_size, args.workers,
                         limit=args.num_images, seed=args.seed, shuffle_subset=True)

    s1 = s2 = None
    n = 0
    kept = []
    t0 = time.time()
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        o = saliency_outlier_scores(model, x, labels=y if args.target == "label" else None).cpu().double()
        s1 = o.sum(0) if s1 is None else s1 + o.sum(0)
        s2 = (o ** 2).sum(0) if s2 is None else s2 + (o ** 2).sum(0)
        n += o.shape[0]
        if sum(k.shape[0] for k in kept) < args.keep_per_image:
            kept.append(o.float())
        print(f"\r{n} images  {time.time() - t0:.0f}s", end="", flush=True)
    print()
    mean = (s1 / n).float()
    std = (s2 / n - (s1 / n) ** 2).clamp_min(0).sqrt().float()
    per_img = torch.cat(kept)[: args.keep_per_image]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(dict(mean=mean, std=std, per_image=per_img, n=n, grid=grid_shape(model),
                    model=args.model, target=args.target, data=args.data), args.out)
    # Signal-to-noise of the position effect: how much of the per-image variance is explained by position
    total_var = per_img.var().item()
    pos_var = mean.var().item()
    print(f"saved {args.out}: grid={grid_shape(model)} n={n}")
    print(f"o mean over positions: {mean.mean():.4f}  position-variance / total-variance = {pos_var / total_var:.3f}")


if __name__ == "__main__":
    main()
