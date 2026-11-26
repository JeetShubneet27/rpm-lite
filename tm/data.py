import os
import torch
import timm
from timm.data import resolve_data_config, create_transform
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torchvision.datasets.vision import VisionDataset


class _CachedImageFolder(ImageFolder):
    """ImageFolder built from a saved file list (same order as ImageFolder's own scan)."""

    def __init__(self, root, d, transform=None):
        VisionDataset.__init__(self, root, transform=transform)
        self.loader, self.extensions = default_loader, IMG_EXTENSIONS
        self.classes, self.class_to_idx = d["classes"], d["class_to_idx"]
        self.samples = [tuple(x) for x in d["samples"]]
        self.targets = [y for _, y in self.samples]
        self.imgs = self.samples


def image_folder(root, transform=None):
    """ImageFolder(root), with the file list cached in ~/.cache/tokenmerge: scanning the 1.28M training
    files on every selection run is slow. The cache is keyed by the folder path and its class list.
    TM_NO_CACHE=1 disables it."""
    import hashlib, pickle
    if os.environ.get("TM_NO_CACHE"):
        return ImageFolder(root, transform=transform)
    cache = os.path.join(os.path.expanduser("~/.cache/tokenmerge"),
                         hashlib.md5(os.path.abspath(root).encode()).hexdigest() + ".pkl")
    try:
        classes = sorted(e.name for e in os.scandir(root) if e.is_dir())
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                d = pickle.load(f)
            if d["classes"] == classes:
                return _CachedImageFolder(root, d, transform)
    except Exception:
        pass
    ds = ImageFolder(root, transform=transform)
    try:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache + ".tmp", "wb") as f:
            pickle.dump(dict(classes=ds.classes, class_to_idx=ds.class_to_idx, samples=ds.samples), f)
        os.replace(cache + ".tmp", cache)
    except Exception:
        pass
    return ds


def load_model(name, device, pretrained=True, checkpoint=None, headless=False, img_size=None):
    """headless=True returns the CLS feature (num_classes=0), used with a linear probe.
    DINOv2 checkpoints are built at 224px (16x16 patches, interpolated position embeddings)."""
    kw = {}
    if headless:
        kw["num_classes"] = 0
    if "dinov2" in name:
        kw["img_size"] = img_size or 224
    elif img_size:
        kw["img_size"] = img_size          # timm interpolates the position embeddings
    model = timm.create_model(name, pretrained=pretrained and checkpoint is None, **kw)
    if checkpoint:
        sd = torch.load(checkpoint, map_location="cpu")
        sd = sd.get("model", sd)
        model.load_state_dict(sd)
    return model.to(device).eval()


def make_loader(model, root, batch_size=128, workers=8, limit=None, seed=0, shuffle_subset=False, indices=None):
    """root: ImageFolder directory (e.g. imagenet/val with 1000 class sub-folders)."""
    size = tuple(model.patch_embed.img_size)
    cfg = resolve_data_config({"input_size": (3,) + size}, model=model)
    tf = create_transform(**cfg, is_training=False)
    ds = image_folder(root, transform=tf)
    if indices is not None:
        # keep only the selected samples in the dataset itself (same order as Subset): every
        # DataLoader worker gets a copy of the dataset, and the full 1.28M-file list costs ~0.5 GB each
        ds.samples = [ds.samples[i] for i in indices]
        ds.targets = [t for _, t in ds.samples]
        ds.imgs = ds.samples
    elif limit is not None and limit < len(ds):
        if shuffle_subset:
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(ds), generator=g)[:limit].tolist()
        else:
            idx = list(range(0, len(ds), max(1, len(ds) // limit)))[:limit]
        ds = Subset(ds, idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


def per_class_indices(root, per_class, seed=0, exclude_per_class=0, exclude_seed=0):
    """Indices of `per_class` random images from every class of an ImageFolder (no image loading).
    exclude_per_class > 0 first removes the images that per_class_indices(root, exclude_per_class,
    exclude_seed) would return (e.g. the linear-probe training images), so the two sets are disjoint.
    With exclude_per_class == 0 the result is identical to earlier versions."""
    ds = image_folder(root)
    by = {}
    for i, (_, y) in enumerate(ds.samples):
        by.setdefault(y, []).append(i)
    ex = set()
    if exclude_per_class:
        ge = torch.Generator().manual_seed(exclude_seed)
        for y in sorted(by):
            idx = by[y]
            ex.update(idx[j] for j in torch.randperm(len(idx), generator=ge)[:exclude_per_class].tolist())
    g = torch.Generator().manual_seed(seed)
    out = []
    for y in sorted(by):
        idx = [i for i in by[y] if i not in ex] if ex else by[y]
        perm = torch.randperm(len(idx), generator=g)[:per_class].tolist()
        out += [idx[j] for j in perm]
    return out
