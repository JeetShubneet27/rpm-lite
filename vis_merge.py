"""Which patches does each method merge? Saves, for a few ImageNet validation images, the final token every patch
was merged into (and the image), for ToMe, MCTF and RPM-lite at the same budget. Figures are rendered from the .npz.

    python vis_merge.py --data $IMAGENET/val --out results/vis
"""
import argparse, glob, os
import numpy as np
import torch
from PIL import Image
from timm.data import resolve_data_config, create_transform
from tm.data import load_model
from tm.patch import apply_patch, configure, simulate_counts
from tm.flops import find_r_for_gflops, vit_gflops

SETUPS = [("deit_base_patch16_224", 0.5), ("vit_base_patch14_dinov2.lvd142m", 0.45), ("vit_base_patch14_dinov2.lvd142m", 0.6)]
METHODS = ("tome", "mctf", "rpmlnw")
CLASSES = (1, 89, 207, 291, 340, 409, 530, 717, 817, 948)       # fixed, varied ImageNet classes (sorted folders)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/vis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    folders = sorted(glob.glob(os.path.join(a.data, "*")))
    paths = [sorted(glob.glob(os.path.join(folders[c], "*")))[0] for c in CLASSES]
    for name, ratio in SETUPS:
        model = apply_patch(load_model(name, "cuda", pretrained=True, headless=True))
        cfg = resolve_data_config({"input_size": (3,) + tuple(model.patch_embed.img_size)}, model=model)
        tf = create_transform(**cfg, is_training=False)
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths]).cuda()
        mean, std = torch.tensor(cfg["mean"])[:, None, None], torch.tensor(cfg["std"])[:, None, None]
        imgs = ((x.cpu() * std + mean).clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()
        depth = len(model.blocks)
        MID = sorted({depth // 4, depth // 3, depth // 2})
        base = vit_gflops(model, simulate_counts(model, "none", [0] * depth))
        out = dict(images=imgs, paths=np.array(paths), ratio=ratio, grid=int(model.patch_embed.grid_size[0]))
        out["mid_blocks"] = np.array(sorted({depth // 4, depth // 3, depth // 2}))
        for m in METHODS:
            r, _ = find_r_for_gflops(model, m, ratio * base, 1)
            configure(model, m, r, tau=1)
            snaps = {}
            hooks = [model.blocks[b].register_forward_hook(
                         lambda mod, i, o, b=b: snaps.__setitem__(b, model._tm["src"].clone() if model._tm.get("src") is not None else None))
                     for b in MID]
            with torch.no_grad():
                model.forward_features(x)
            for hk in hooks:
                hk.remove()
            for b in MID:
                if snaps.get(b) is not None:
                    out[f"src_{m}_b{b}"] = snaps[b].cpu().numpy()
            out[f"src_{m}"] = model._tm["src"].cpu().numpy()
            out[f"ntok_{m}"] = int(model._tm["counts"][-1][1])
        np.savez_compressed(os.path.join(a.out, f"{name}_{ratio}.npz"), **out)
        print(name, {m: out[f"ntok_{m}"] for m in METHODS})


if __name__ == "__main__":
    main()
