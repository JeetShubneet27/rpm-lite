#!/usr/bin/env bash
# Fixes for the paper's weak points (~10 h, resumable: re-running skips finished configurations).
#   IMAGENET=/path/to/imagenet nohup bash run_fix.sh > results/fix.log 2>&1 &
#  A. hybrid calibrated merging (ToMe matching in layers where similarity is more reliable)
#  B. a practical 75% budget for every foundation model
#  C. the fair protocol: every method selects tau + schedule on 10 TRAIN images/class (10k images)
set -uo pipefail
: "${IMAGENET:?set IMAGENET=/path/to/imagenet}"
W=${WORKERS:-16}
VAL=results/fix_val.jsonl
SEL=results/fix_sel.jsonl
CFG="--weight-mode size --prop-attn on"
ev()  { python eval_imagenet.py --data $IMAGENET/val --workers $W --out $VAL --save-preds --skip-existing "$@" \
        || echo "FAILED val: $*" >> results/fix_failed.txt; }
sel() { python eval_imagenet.py --data $IMAGENET/train --per-class 10 --workers $W --out $SEL --skip-existing "$@" \
        || echo "FAILED sel: $*" >> results/fix_failed.txt; }

declare -A DEIT=(
  [deit_small_patch16_224]="3.6 3.0 2.6 2.4 2.2"
  [deit_tiny_patch16_224]="0.9 0.8 0.7"
  [deit3_small_patch16_224.fb_in1k]="3.0 2.6 2.2"
  [deit_base_patch16_224]="11.5 10.1 9.0"
  [deit3_base_patch16_224.fb_in1k]="11.5 10.1 9.0"
)
FM="vit_small_patch14_dinov2.lvd142m vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m
    vit_base_patch14_reg4_dinov2.lvd142m vit_base_patch16_clip_224.laion2b
    vit_large_patch14_dinov2.lvd142m vit_large_patch14_reg4_dinov2.lvd142m"

# candidate configurations of every family (same grids as the earlier runs)
deit_candidates() {   # $1 = "--model M --target-gflops G"
  for T in 0 1 2 3; do echo "$1 --tau $T --method tome"; done
  for T in 1 2 3; do echo "$1 --tau $T --method tome --schedule geo"; done
  for T in 1 2 3; do echo "$1 --tau $T --method sadtm $CFG --prior uniform"; done
  for T in 1 2 3; do echo "$1 --tau $T --method sadtm $CFG --prior uniform --schedule geo"; done
  for T in 0 1 2 3; do echo "$1 --tau $T --method calib $CFG --blend soft --gamma 4"; done
  for T in 1 2 3; do echo "$1 --tau $T --method calib $CFG --blend soft --gamma 4 --schedule geo"; done
  for T in 0 1 2 3; do echo "$1 --tau $T --method hybrid $CFG --blend soft --gamma 4"; done
  for T in 1 2 3; do echo "$1 --tau $T --method hybrid $CFG --blend soft --gamma 4 --schedule geo"; done
}
fm_candidates() {
  for T in 0 1 2 3; do echo "$1 --tau $T --method tome"; done
  for T in 1 2; do echo "$1 --tau $T --method tome --schedule geo"; done
  for T in 1 2 3; do echo "$1 --tau $T --method sadtm $CFG --prior uniform"; done
  echo "$1 --tau 2 --method sadtm $CFG --prior uniform --schedule geo"
  for T in 0 1 2 3; do echo "$1 --tau $T --method calib $CFG --blend soft --gamma 4"; done
  for T in 1 2; do echo "$1 --tau $T --method calib $CFG --blend soft --gamma 4 --schedule geo"; done
  for T in 0 1 2 3; do echo "$1 --tau $T --method hybrid $CFG --blend soft --gamma 4"; done
  for T in 1 2 3; do echo "$1 --tau $T --method hybrid $CFG --blend soft --gamma 4 --schedule geo"; done
}

# ---- A + B: new val runs (hybrid everywhere; every family at the new 75% budget) ----------------
for M in "${!DEIT[@]}"; do for G in ${DEIT[$M]}; do
  deit_candidates "--model $M --target-gflops $G" | grep -- "--method hybrid" | while read -r c; do ev $c; done
done; done
for M in $FM; do
  P="--probe probes/$M.pt"
  for R in 0.6 0.45; do
    fm_candidates "--model $M $P --target-ratio $R" | grep -- "--method hybrid" | while read -r c; do ev $c; done
  done
  fm_candidates "--model $M $P --target-ratio 0.75" | while read -r c; do ev $c; done
done

# ---- C: selection runs on 10 train images per class, every candidate of every family ------------
for M in "${!DEIT[@]}"; do for G in ${DEIT[$M]}; do
  deit_candidates "--model $M --target-gflops $G" | while read -r c; do sel $c; done
done; done
for M in $FM; do for R in 0.75 0.6 0.45; do
  fm_candidates "--model $M --probe probes/$M.pt --target-ratio $R" | while read -r c; do sel $c; done
done; done

python select_report.py results/calib.jsonl results/sched.jsonl results/probe.jsonl $VAL $SEL \
  | tee results/select_report.txt
echo DONE
