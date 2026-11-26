"""Print the eval_imagenet.py arguments of the configuration each method family selected on the
training subset, so that only those are evaluated on val (used by run_hires.sh).

    python pick_selected.py results/hires_sel.jsonl --model M --img-size 448
"""
import argparse, json

from tm.families import fam_of, args_of

ap = argparse.ArgumentParser()
ap.add_argument("sel", nargs="+")
ap.add_argument("--family", default=None, help="only this method family")
ap.add_argument("--model", required=True)
ap.add_argument("--img-size", type=int, default=224)
ap.add_argument("--split", default=None, help="only selection rows of this split (e.g. sel10s1x110)")
ap.add_argument("--no-rel", action="store_true", help="ignore probe-allocation (rel) schedules")
ap.add_argument("--probe", default=None, help="only rows with this probe file (default: any probe except zero-shot heads)")
a = ap.parse_args()
best = {}
lines = [l for f in a.sel for l in open(f) if l.strip()]
for l in lines:
    r = json.loads(l)
    if r["model"] != a.model or r.get("img_size", 224) != a.img_size or r.get("split", "val") in ("val", "speed"):
        continue
    if a.split and r.get("split") != a.split:
        continue
    pr = r.get("probe") or ""
    if (a.probe and pr != a.probe) or (not a.probe and "zeroshot" in pr):
        continue
    k = fam_of(r)
    if k is None or (a.family and k != a.family):
        continue
    if a.no_rel and str(r.get("schedule", "")).startswith("rel"):
        continue
    bud = (f"--target-ratio {r['target_ratio']}" if r.get("target_ratio")
           else f"--target-gflops {r['target_gflops']}")
    key = (bud, k)
    if key not in best or r["acc1"] > best[key]["acc1"]:
        best[key] = r
for (bud, k), r in sorted(best.items()):
    print(f"{bud} {args_of(k, r)}")
