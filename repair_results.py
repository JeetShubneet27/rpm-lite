"""Repair results after the disk ran full: drop result lines that were cut off (invalid JSON) and evaluation rows whose
saved per-image predictions cannot be read, so that those runs are simply redone. Every changed file is backed up as
<file>.bak first.

    python repair_results.py
"""
import glob, json, os, shutil
import numpy as np

total_bad = 0
for fn in sorted(glob.glob("results/*.jsonl")):
    lines = open(fn, errors="replace").read().split("\n")
    good, bad = [], 0
    for l in lines:
        if not l.strip():
            continue
        try:
            o = json.loads(l)
        except Exception:
            bad += 1
            continue
        p = o.get("preds") if isinstance(o, dict) else None
        if p:
            q = p if os.path.exists(p) else os.path.join("results", "preds", os.path.basename(p))
            try:
                np.load(q)
            except Exception:
                bad += 1
                if os.path.exists(q):
                    os.replace(q, q + ".broken")
                continue
        good.append(l)
    if bad:
        shutil.copy(fn, fn + ".bak")
        with open(fn, "w") as f:
            f.write("\n".join(good) + "\n")
        print(f"{fn}: removed {bad} broken line(s), kept {len(good)}")
        total_bad += bad
for fn in glob.glob("results/rev_diag/*.npz"):
    try:
        np.load(fn)["data"].shape
    except Exception:
        os.replace(fn, fn + ".broken")
        print(f"{fn}: unreadable, will be recomputed")
        total_bad += 1
print("done," if total_bad else "done, nothing to repair;", f"{total_bad} item(s) removed")
