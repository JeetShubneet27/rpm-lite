"""Output fidelity of merged models (mentor roadmap, section 2): for each saved configuration, on the 50,000
ImageNet validation images, the KL and Jensen-Shannon divergence between the unmerged and the merged class
distributions, the top-1 flip rate (prediction differs from the unmerged model), accuracy, and the expected
calibration error (15 equal-width bins) of the merged and the unmerged model. One pass over the data per model:
the unmerged model and every configuration of that model run on the same batch.

    python fidelity.py --list results/cfgs/list_fid.txt --data $IMAGENET/val --out results/fidelity.jsonl
    (list lines: family|path/to/config.pt, as written by eval_imagenet.py --save-config)
"""
import argparse, collections, json, os, time
import torch
from tm.data import load_model, make_loader
from tm.patch import apply_patch, configure

NB = 15


def ece_bins():
    return dict(n=torch.zeros(NB, dtype=torch.float64), conf=torch.zeros(NB, dtype=torch.float64),
                acc=torch.zeros(NB, dtype=torch.float64))


def add_ece(b, p, y):
    conf, pred = p.max(-1)
    idx = (conf * NB).long().clamp_max(NB - 1).cpu()
    b["n"] += torch.bincount(idx, minlength=NB).double()
    b["conf"] += torch.bincount(idx, weights=conf.double().cpu(), minlength=NB)
    b["acc"] += torch.bincount(idx, weights=(pred == y).double().cpu(), minlength=NB)


def ece_of(b):
    n = b["n"].sum()
    return float(((b["acc"] - b["conf"]).abs()).sum() / n) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/fidelity.jsonl")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="testing: number of batches")
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["config"] for l in open(a.out) if l.strip()}
    groups = collections.defaultdict(list)
    for line in open(a.list):
        if not line.strip():
            continue
        fam, path = line.strip().split("|")
        if path in done or not os.path.exists(path):
            continue
        cfg = torch.load(path, map_location="cpu", weights_only=False)
        groups[cfg["model"]].append((fam, path, cfg))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for name, cfgs in groups.items():
        headless = cfgs[0][2]["headless"]
        img = cfgs[0][2]["img_size"]
        model = apply_patch(load_model(name, dev, pretrained=not a.no_pretrained, headless=headless,
                                       img_size=img if ("dinov2" in name or img != 224) else None)).eval()
        if headless:
            pr = torch.load(f"probes/{name}.pt", map_location=dev)
            head = lambda f: ((f.float() - pr["mean"]) / pr["std"]) @ pr["W"].T + pr["b"]
        else:
            head = lambda f: f.float()
        dl = make_loader(model, a.data, batch_size=a.batch_size, workers=a.workers)
        acc = {p: dict(kl=0.0, js=0.0, flip=0, correct=0, n=0, ece=ece_bins()) for _, p, _ in cfgs}
        base = dict(correct=0, n=0, ece=ece_bins())
        t0 = time.time()
        with torch.no_grad():
            for it, (x, y) in enumerate(dl):
                if a.limit and it >= a.limit:
                    break
                x, y = x.to(dev, non_blocking=True), y.to(dev)
                configure(model, "none", 0)
                z0 = head(model(x))
                lp0 = z0.float().log_softmax(-1)
                p0 = lp0.exp()
                pred0 = p0.argmax(-1)
                base["correct"] += int((pred0 == y).sum()); base["n"] += len(y); add_ece(base["ece"], p0, y)
                for fam, path, cfg in cfgs:
                    model._tm.update(u0_clip=None, mctf_drop=None, sum_mode="head")
                    model._tm.update({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in cfg["tm"].items()})
                    model._tm["size"] = None
                    z = head(model(x))
                    lp = z.float().log_softmax(-1)
                    p = lp.exp()
                    m = 0.5 * (p0 + p)
                    lm = m.clamp_min(1e-12).log()
                    s = acc[path]
                    s["kl"] += float((p0 * (lp0 - lp)).sum())
                    s["js"] += float((0.5 * (p0 * (lp0 - lm)).sum(-1) + 0.5 * (p * (lp - lm)).sum(-1)).sum())
                    pred = p.argmax(-1)
                    s["flip"] += int((pred != pred0).sum()); s["correct"] += int((pred == y).sum()); s["n"] += len(y)
                    add_ece(s["ece"], p, y)
                print(f"\r  {name}: {base['n']} images, {len(cfgs)} configs, {time.time() - t0:.0f}s", end="", flush=True)
        print()
        with open(a.out, "a") as f:
            for fam, path, cfg in cfgs:
                s = acc[path]
                rec = dict(model=name, family=fam, config=path, target_ratio=cfg.get("target_ratio"), gflops=cfg["gflops"],
                           base_gflops=cfg["base_gflops"], n=s["n"], acc1=round(100 * s["correct"] / s["n"], 3),
                           kl=round(s["kl"] / s["n"], 5), js=round(s["js"] / s["n"], 5),
                           flip=round(100 * s["flip"] / s["n"], 3), ece=round(ece_of(s["ece"]), 3),
                           base_acc1=round(100 * base["correct"] / base["n"], 3), base_ece=round(ece_of(base["ece"]), 3))
                f.write(json.dumps(rec) + "\n")
                print(f"    {fam:12s} {cfg['gflops']:6.2f} GF  acc {rec['acc1']:6.2f}  KL {rec['kl']:.4f}  flip {rec['flip']:5.2f}%  ECE {rec['ece']:.2f}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
