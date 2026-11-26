#!/usr/bin/env bash
# Full selection protocol for RPM-lite without whitening (rpmlnw), 38 settings.  Resumable.
#   cd ~/S_VIT/tokenmerge_study && (nohup bash run_rpmlnw.sh > results/rpmlnw.log 2>&1 &)
set -uo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/data/imagenet}
W=${WORKERS:-6}
FAIL=results/rpmlnw_failed.txt
echo "== start $(date)"
sel() { python eval_imagenet.py --data $IMAGENET/train --per-class 10 --workers $W --out results/rpm_sel.jsonl \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED sel: $*" >> $FAIL; }
ev()  { python eval_imagenet.py --data $IMAGENET/val --workers $W --out results/rpm_val.jsonl --save-preds \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED val: $*" >> $FAIL; }
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
  echo "== rpmlnw $name $(date +%H:%M)"
  for B in $budgets; do
    for T in 0 1 2 3; do sel $m $flag $B --tau $T --method rpmlnw; done
    for T in 1 2 3; do sel $m $flag $B --tau $T --method rpmlnw --schedule geo; done
    for R in 1 2 4; do sel $m $flag $B --method rpmlnw --schedule rel --rel-kappa $R; done
  done
  python pick_selected.py results/rpm_sel.jsonl --family rpmlnw --model $name | while read -r c; do ev $m $c; done
done
echo "== RPMLNW DONE $(date)"
