"""Turn the Kaggle 'ImageNet Object Localization Challenge' download into the layout our scripts use:
    <out>/train/<synset>/*.JPEG   (link to the Kaggle train folder, unchanged)
    <out>/val/<synset>/*.JPEG     (symlinks to the flat Kaggle val images, sorted by class)

    python prepare_imagenet.py --kaggle ~/imagenet_kaggle --out ~/imagenet
"""
import argparse, csv, os, shutil, sys


def link(src, dst):
    """Symlink; on Windows without symlink rights, move instead (same drive: instant)."""
    try:
        os.symlink(src, dst, target_is_directory=os.path.isdir(src))
    except (OSError, NotImplementedError):
        shutil.move(src, dst)

ap = argparse.ArgumentParser()
ap.add_argument("--kaggle", required=True, help="folder where the Kaggle zip was extracted (contains ILSVRC/ and LOC_val_solution.csv)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
k = os.path.abspath(os.path.expanduser(a.kaggle))
out = os.path.abspath(os.path.expanduser(a.out))
src_train = os.path.join(k, "ILSVRC", "Data", "CLS-LOC", "train")
src_val = os.path.join(k, "ILSVRC", "Data", "CLS-LOC", "val")
sol = os.path.join(k, "LOC_val_solution.csv")
for p in (src_train, src_val, sol):
    if not os.path.exists(p) and not os.path.exists(os.path.join(out, "val")):
        sys.exit(f"not found: {p}")
os.makedirs(out, exist_ok=True)
if not os.path.exists(os.path.join(out, "train")):
    link(src_train, os.path.join(out, "train"))
n = 0
with open(sol) as f:
    for row in csv.DictReader(f):
        syn = row["PredictionString"].split()[0]
        d = os.path.join(out, "val", syn)
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, row["ImageId"] + ".JPEG")
        if not os.path.exists(dst):
            link(os.path.join(src_val, row["ImageId"] + ".JPEG"), dst)
        n += 1
classes = sorted(os.listdir(os.path.join(out, "val")))
tr = os.path.join(out, "train")
n_train = len([c for c in os.listdir(tr) if os.path.isdir(os.path.join(tr, c))])
print(f"val: {n} images in {len(classes)} classes; train: {n_train} classes")
print("OK" if n == 50000 and len(classes) == 1000 and n_train == 1000 else "WARNING: unexpected counts")
