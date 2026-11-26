"""Semantic segmentation on ADE20K with token merging at test time (mentor point 5: generalisation).

Protocol (training-free for the merging methods):
  1. A linear head (BatchNorm + 1x1 conv on the last-layer patch features, the DINOv2 "linear" protocol) is trained
     once per backbone on frozen, UNMERGED features.
  2. At test time each merging method runs at the same FLOP budget; after the last block every merged token is
     copied back to all patch positions it absorbed ("unmerging", from the tracked token sources), so the head sees
     a full-resolution feature grid. All methods use the same schedule (one deferred block, constant r).
  3. mIoU over the 150 classes on the 2000 validation images (label 0 = ignore), logits upsampled to the original
     mask size.

    python seg_ade20k.py --model vit_base_patch14_dinov2.lvd142m --data ~/data/ade20k/ADEChallengeData2016
"""
import argparse, glob, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tm.data import load_model
from tm.patch import apply_patch, configure, simulate_counts
from tm.flops import find_r_for_gflops, vit_gflops

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
NCLS = 150


class ADE(torch.utils.data.Dataset):
    def __init__(self, root, split, size, train):
        sub = "training" if split == "train" else "validation"
        self.imgs = sorted(glob.glob(os.path.join(root, "images", sub, "*.jpg")))
        self.root, self.sub, self.size, self.train = root, sub, size, train
        assert self.imgs, f"no images under {root}/images/{sub}"

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        img = Image.open(p).convert("RGB")
        mask = Image.open(os.path.join(self.root, "annotations", self.sub, os.path.basename(p)[:-4] + ".png"))
        x = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255
        y = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy()) - 1          # 0 (other) -> -1 = ignore
        if self.train:                                                             # random scale crop + flip
            H, W = y.shape
            s = self.size * np.random.uniform(0.5, 2.0) / min(H, W)
            h, w = max(self.size, round(H * s)), max(self.size, round(W * s))
            x = F.interpolate(x[None], (h, w), mode="bilinear", align_corners=False)[0]
            y = F.interpolate(y[None, None].float(), (h, w), mode="nearest")[0, 0].long()
            t, l = np.random.randint(0, h - self.size + 1), np.random.randint(0, w - self.size + 1)
            x, y = x[:, t:t + self.size, l:l + self.size], y[t:t + self.size, l:l + self.size]
            if np.random.rand() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            yt = y
        else:                                                                      # full image resized; mask kept
            yt = y
            x = F.interpolate(x[None], (self.size, self.size), mode="bilinear", align_corners=False)[0]
        x = (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]
        return x, yt


def collate_val(b):
    return torch.stack([x for x, _ in b]), [y for _, y in b]


def patch_grid(model, x):
    """Full-resolution patch features (B, C, h, w); merged tokens are copied back to every patch they absorbed."""
    tok = model.forward_features(x)
    st, P = model._tm, model.num_prefix_tokens
    pt = tok[:, P:]
    if st["active"] and st.get("src") is not None:
        pt = pt.gather(1, st["src"][..., None].expand(-1, -1, pt.shape[-1]))
    B, N, C = pt.shape
    h = w = int(math.isqrt(N))
    return pt.transpose(1, 2).reshape(B, C, h, w)


class Head(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.bn, self.conv = nn.BatchNorm2d(C), nn.Conv2d(C, NCLS, 1)

    def forward(self, f):
        return self.conv(self.bn(f))


def train_head(model, a, dev, path):
    ds = ADE(a.data, "train", a.img, True)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                                     pin_memory=dev == "cuda", persistent_workers=a.workers > 0)
    head = Head(model.embed_dim).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    configure(model, "none", 0)
    t0, it = time.time(), 0
    for ep in range(a.epochs):
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                f = patch_grid(model, x)
            logits = F.interpolate(head(f.float()), size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, y, ignore_index=-1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); it += 1
            if it % 100 == 0:
                print(f"\r  head epoch {ep + 1}/{a.epochs} step {it}/{steps} loss {loss.item():.3f} {time.time() - t0:.0f}s",
                      end="", flush=True)
    print()
    torch.save(head.state_dict(), path)
    return head


@torch.no_grad()
def evaluate(model, head, a, dev):
    ds = ADE(a.data, "val", a.img, False)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers, collate_fn=collate_val)
    conf = torch.zeros(NCLS, NCLS, dtype=torch.long, device=dev)
    for x, ys in dl:
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            f = patch_grid(model, x.to(dev))
        lg = head(f.float())
        for k, y in enumerate(ys):
            y = y.to(dev)
            pred = F.interpolate(lg[k:k + 1], size=y.shape, mode="bilinear", align_corners=False)[0].argmax(0)
            m = y >= 0
            conf += torch.bincount(y[m] * NCLS + pred[m], minlength=NCLS * NCLS).view(NCLS, NCLS)
    tp = conf.diag().float()
    iou = tp / (conf.sum(0) + conf.sum(1) - conf.diag()).float().clamp_min(1)
    present = conf.sum(1) > 0
    return round(100 * float(iou[present].mean()), 3), round(100 * float(tp.sum() / conf.sum()), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--img", type=int, default=448)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--methods", nargs="+", default=["tome", "mctf", "adamerge", "rpmlnw"])
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.75, 0.6, 0.45])
    ap.add_argument("--tau", type=int, default=1)
    ap.add_argument("--out", default="results/seg_ade20k.jsonl")
    ap.add_argument("--no-pretrained", action="store_true", help="testing only")
    a = ap.parse_args()
    a.data = os.path.expanduser(a.data)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = apply_patch(load_model(a.model, dev, pretrained=not a.no_pretrained, headless=True, img_size=a.img))
    depth = len(model.blocks)
    base = vit_gflops(model, simulate_counts(model, "none", [0] * depth))
    os.makedirs("results/seg_heads", exist_ok=True)
    hp = f"results/seg_heads/{a.model}_{a.img}.pt"
    head = Head(model.embed_dim).to(dev)
    if os.path.exists(hp):
        head.load_state_dict(torch.load(hp, map_location=dev))
    else:
        head = train_head(model, a, dev, hp)
    head.eval()
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            if l.strip():
                o = json.loads(l); done.add((o["model"], o["img"], o["method"], o["ratio"]))
    runs = [("none", 1.0)] + [(m, r) for r in a.ratios for m in a.methods]
    for meth, ratio in runs:
        if (a.model, a.img, meth, ratio) in done:
            continue
        if meth == "none":
            configure(model, "none", 0); g = base
        else:
            r, _ = find_r_for_gflops(model, meth, ratio * base, a.tau)
            configure(model, meth, r, tau=a.tau)
            g = vit_gflops(model, simulate_counts(model, meth, model._tm["r_list"]))
        miou, acc = evaluate(model, head, a, dev)
        rec = dict(model=a.model, img=a.img, method=meth, ratio=ratio, tau=a.tau, gflops=round(g, 2),
                   base_gflops=round(base, 2), miou=miou, pixel_acc=acc)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  {meth:9s} {ratio:4.2f}  {g:6.1f} GFLOPs  mIoU {miou:6.2f}  pixel acc {acc:6.2f}", flush=True)


if __name__ == "__main__":
    main()
