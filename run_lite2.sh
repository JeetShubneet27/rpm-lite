#!/usr/bin/env bash
# RPM-lite on key-similar candidates: per-head cost only for each token's m most similar partners.  Resumable.
# Waits for run_lite.sh to finish.
#   cd ~/S_VIT/tokenmerge_study && (nohup bash run_lite2.sh > results/lite2.log 2>&1 &)
#  C1 fair speed (alternating rounds), 4 models                                              ~15 min
#  C2 accuracy at RPM-f's selected configuration, 6 settings x 4 variants                     ~2 h
set -uo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/VIT_5/data/imagenet}
W=${WORKERS:-6}
FAIL=results/lite2_failed.txt
while pgrep -f "^bash run_lite.sh" >/dev/null 2>&1; do echo "waiting for run_lite.sh  $(date +%H:%M)"; sleep 120; done
echo "== start $(date)"
ev() { python eval_imagenet.py --data $IMAGENET/val --workers $W --out results/rpm_val.jsonl --save-preds \
       --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED val: $*" >> $FAIL; }

echo "== C1 fair speed $(date)"
python bench_speed.py --rounds 7 --only mctf "rpms32 (new mm)" "rpmlkv (" rpmlc rpmlnwc rpmlkvc 2>&1 | grep -v -i warn

echo "== C2 accuracy $(date)"
for MB in "vit_small_patch14_reg4_dinov2.lvd142m --probe probes/vit_small_patch14_reg4_dinov2.lvd142m.pt|vit_small_patch14_reg4_dinov2.lvd142m|0.45" \
          "vit_base_patch14_dinov2.lvd142m --probe probes/vit_base_patch14_dinov2.lvd142m.pt|vit_base_patch14_dinov2.lvd142m|0.45" \
          "vit_base_patch14_dinov2.lvd142m --probe probes/vit_base_patch14_dinov2.lvd142m.pt|vit_base_patch14_dinov2.lvd142m|0.6" \
          "vit_base_patch16_clip_224.laion2b --probe probes/vit_base_patch16_clip_224.laion2b.pt|vit_base_patch16_clip_224.laion2b|0.6" \
          "deit3_small_patch16_224.fb_in1k|deit3_small_patch16_224.fb_in1k|2.2" \
          "deit_base_patch16_224|deit_base_patch16_224|9.0"; do
  IFS='|' read -r m name bud <<< "$MB"; m="--model $m"
  python pick_selected.py results/rpm_sel.jsonl --family rpmf2 --model $name | while read -r c; do
    case "$c" in *"-ratio $bud"*|*"-gflops $bud"*) ;; *) continue;; esac
    for MC in "rpmlc --cand-m 8" "rpmlc --cand-m 16" "rpmlnwc --cand-m 8" "rpmlkvc --cand-m 8"; do ev $m ${c/--method rpmf2/--method $MC}; done
  done
done
python - <<'PY'
import json, statistics as st
rows = [json.loads(l) for l in open("results/rpm_val.jsonl") if l.strip()]
bud = lambda r: r.get("target_ratio") or r["target_gflops"]
key = lambda r: (r["model"], bud(r), r["tau"], r.get("schedule", "const"))
ref = {key(r): r["acc1"] for r in rows if r["method"] == "rpmf2"}
sk = {key(r): r["acc1"] for r in rows if r["method"] == "rpms" and r.get("prior") == "k32"}
bl = {}
for r in rows:
    if r["method"] in ("tome", "sadtm", "pitome", "mctf") and r.get("img_size", 224) == 224 and not r.get("tag"):
        k2 = (r["model"], bud(r)); bl[k2] = max(bl.get(k2, 0), r["acc1"])
d = {}
print(f"  {'model':30s} {'budget':>6s}  {'variant':8s}  acc     RPM-f   sketch32  best baseline")
for r in rows:
    if r["method"] in ("rpml", "rpmlnw", "rpmlkv", "rpmlc", "rpmlnwc", "rpmlkvc") and key(r) in ref:
        k = key(r)
        d.setdefault(r["method"] + ("-" + r["prior"] if r.get("prior", "-") != "-" else ""), []).append((r["acc1"] - ref[k], r["acc1"] - bl.get((k[0], k[1]), float("nan"))))
        print(f"  {k[0][:30]:30s} {k[1]:>6}  {r['method'] + ('-' + r['prior'] if r.get('prior', '-') != '-' else ''):11s}  {r['acc1']:6.2f}  {ref[k]:6.2f}  {sk.get(k, float('nan')):6.2f}    {bl.get((k[0], k[1]), float('nan')):6.2f}")
for m, v in d.items():
    print(f"  {m:11s}: mean change vs RPM-f {st.mean(x for x, _ in v):+.2f} pp, vs best baseline {st.mean(y for _, y in v):+.2f} pp ({len(v)} settings)")
PY
echo "== LITE2 DONE $(date)"
