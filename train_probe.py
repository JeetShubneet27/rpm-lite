"""Linear probe on frozen CLS features of the UNMERGED backbone (for DINOv2 / CLIP, which have no
ImageNet classifier). The probe is trained once and then shared by every merging method, so all
methods are compared through the same fixed classifier and merging itself stays training-free.

    python train_probe.py --model vit_base_patch14_dinov2.lvd142m --data $IMAGENET/train \
        --per-class 100 --out probes/vit_base_patch14_dinov2.lvd142m.pt
"""
import argparse, os, time
import torch
from tm.device import pick_device
import torch.nn.functional as F
from tm.data import load_model, make_loader, per_class_indices


@torch.no_grad()
def features(model, loader, dev):
    fs, ys = [], []
    t0 = time.time()
    for x, y in loader:
        fs.append(model(x.to(dev, non_blocking=True)).float().cpu())
        ys.append(y)
        print(f"\r  features {sum(len(v) for v in ys)}  {time.time() - t0:.0f}s", end="", flush=True)
    print()
    return torch.cat(fs), torch.cat(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="ImageNet train ImageFolder")
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--holdout", type=int, default=10, help="per-class images held out to pick weight decay")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--img-size", type=int, default=None, help="input resolution (DINOv2 default 224)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = pick_device()
    model = load_model(a.model, dev, pretrained=not a.no_pretrained, headless=True, img_size=a.img_size)
    idx = per_class_indices(a.data, a.per_class + a.holdout)
    X, Y = features(model, make_loader(model, a.data, a.batch_size, a.workers, indices=idx), dev)

    # split: last `holdout` images of each class -> validation for the weight-decay choice
    n_cls = int(Y.max()) + 1
    pos = torch.zeros_like(Y)
    seen = torch.zeros(n_cls, dtype=torch.long)
    for i, y in enumerate(Y.tolist()):
        pos[i] = seen[y]
        seen[y] += 1
    tr = pos < a.per_class
    mean, std = X[tr].mean(0), X[tr].std(0).clamp_min(1e-6)
    Xn = ((X - mean) / std).to(dev)
    Yd = Y.to(dev)

    def fit(mask, wd):
        torch.manual_seed(0)
        lin = torch.nn.Linear(X.shape[1], n_cls).to(dev)
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=wd)
        xs, ys = Xn[mask], Yd[mask]
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        for _ in range(a.epochs):
            perm = torch.randperm(len(xs), device=dev)
            for i in range(0, len(xs), 4096):
                b = perm[i:i + 4096]
                loss = F.cross_entropy(lin(xs[b]), ys[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        return lin

    trm, hom = tr.to(dev), (~tr).to(dev)
    best = None
    for wd in (1e-4, 1e-2, 1e-1, 1.0):
        lin = fit(trm, wd)
        with torch.no_grad():
            acc = (lin(Xn[hom]).argmax(-1) == Yd[hom]).float().mean().item() * 100
        print(f"  wd={wd:g}  holdout acc {acc:.2f}")
        if best is None or acc > best[0]:
            best = (acc, wd)
    lin = fit(torch.ones_like(trm), best[1])      # refit on everything with the chosen weight decay
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(dict(W=lin.weight.detach(), b=lin.bias.detach(), mean=mean.to(dev), std=std.to(dev),
                    model=a.model, img_size=int(model.patch_embed.img_size[0]), per_class=a.per_class, wd=best[1], holdout_acc=best[0]), a.out)
    print(f"saved {a.out}  (wd={best[1]:g}, holdout acc {best[0]:.2f})")


if __name__ == "__main__":
    main()
