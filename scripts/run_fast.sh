#!/usr/bin/env bash
# Fast RPM variants: rpmf (Frobenius norm over all queries, matmul-based) and rpmk (4 key-nearest candidates).
#  1. GPU speed (alone on the GPU)  2. accuracy, same protocol: rpmf on all 12 models (foundation first)
#   cd ~/S_VIT/tokenmerge_study && (nohup bash run_fast.sh > results/fast.log 2>&1 &)
set -uo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/VIT_5/data/imagenet}
W=${WORKERS:-8}
FAIL=results/fast_failed.txt
echo "== start $(date)"
echo "== 1 speed $(date)"
SP=results/speed_fast.jsonl
for MB in "deit_base_patch16_224|--target-gflops 9.0" "deit_small_patch16_224|--target-gflops 2.6" \
          "vit_base_patch14_dinov2.lvd142m --probe probes/vit_base_patch14_dinov2.lvd142m.pt|--target-ratio 0.6" \
          "vit_base_patch16_clip_224.laion2b --probe probes/vit_base_patch16_clip_224.laion2b.pt|--target-ratio 0.6"; do
  m="--model ${MB%%|*}"; b=${MB#*|}
  for F in none tome rpmf rpmk rpm_all; do
    if [ $F = none ]; then a="--method none"; else a="$b --tau 1 --method $F"; fi
    python eval_imagenet.py --data $IMAGENET/val --speed-only --out $SP $m $a >/dev/null 2>&1 || echo "FAILED speed $m $a" >> $FAIL
  done
done
python - <<'PY' > results/speed_fast.txt 2>&1
import json
rows = [json.loads(l) for l in open("results/speed_fast.jsonl") if l.strip()]
base = {}
for r in rows:
    if r["method"] == "none": base[r["model"]] = r
for r in rows:
    b = base.get(r["model"])
    if b: print(f"{r['model'][:30]:30s} {r['method']:8s} fp32 {r['img_per_s_fp32'] / b['img_per_s_fp32']:.2f}x  bf16 {r['img_per_s_bf16'] / b['img_per_s_bf16']:.2f}x")
PY
cat results/speed_fast.txt

echo "== 2 protocol $(date)"
sel() { python eval_imagenet.py --data $IMAGENET/train --per-class 10 --workers $W --out results/rpm_sel.jsonl \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED sel: $*" >> $FAIL; }
ev()  { python eval_imagenet.py --data $IMAGENET/val --workers $W --out results/rpm_val.jsonl --save-preds \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED val: $*" >> $FAIL; }
report() { python select_report.py results/{calib,sched,probe,fix_val,fix_sel,hires_val,hires_sel,rel_val,rel_sel,review_val,review_sel,rpm_val,rpm_sel}.jsonl \
             > results/rpm_report.txt 2>&1; }
{
  for M in vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m vit_base_patch16_clip_224.laion2b \
           vit_large_patch14_dinov2.lvd142m vit_small_patch14_dinov2.lvd142m vit_base_patch14_reg4_dinov2.lvd142m \
           vit_large_patch14_reg4_dinov2.lvd142m; do echo "--model $M --probe probes/$M.pt|--target-ratio|0.45 0.6 0.75"; done
  echo "--model deit3_small_patch16_224.fb_in1k|--target-gflops|2.2 2.6 3.0"
  echo "--model deit_base_patch16_224|--target-gflops|9.0 10.1 11.5"
  echo "--model deit3_base_patch16_224.fb_in1k|--target-gflops|9.0 10.1 11.5"
  echo "--model deit_small_patch16_224|--target-gflops|2.2 2.4 2.6 3.0 3.6"
  echo "--model deit_tiny_patch16_224|--target-gflops|0.7 0.8 0.9"
} | while IFS='|' read -r m flag budgets; do
  name=$(echo $m | awk '{print $2}')
  echo "== rpmf $name $(date +%H:%M)"
  for B in $budgets; do
    for T in 0 1 2 3; do sel $m $flag $B --tau $T --method rpmf; done
    for T in 1 2 3; do sel $m $flag $B --tau $T --method rpmf --schedule geo; done
    for K in 1 2 4; do sel $m $flag $B --method rpmf --schedule rel --rel-kappa $K; done
  done
  python pick_selected.py results/rpm_sel.jsonl --family rpmf --model $name | while read -r c; do ev $m $c; done
  report; grep -E "rpmf +(found|super)" results/rpm_report.txt
done
echo "== FAST DONE $(date)"
