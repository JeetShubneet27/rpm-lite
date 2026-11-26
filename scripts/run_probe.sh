#!/usr/bin/env bash
# Part 2, step 5: DINOv2 (with / without registers) and CLIP, via frozen linear probes (~4 h).
#   IMAGENET=/path/to/imagenet nohup bash run_probe.sh > results/probe.log 2>&1 &
# Waits for run_sched.sh to finish first (so the GPU is not shared between our own jobs).
set -uo pipefail
: "${IMAGENET:?set IMAGENET=/path/to/imagenet}"
W=${WORKERS:-16}
OUT=results/probe.jsonl
while [ -f results/sched.log ] && ! grep -q DONE results/sched.log; do sleep 60; done

MODELS="vit_base_patch14_dinov2.lvd142m vit_base_patch14_reg4_dinov2.lvd142m
        vit_large_patch14_dinov2.lvd142m vit_large_patch14_reg4_dinov2.lvd142m
        vit_base_patch16_clip_224.laion2b
        vit_small_patch14_dinov2.lvd142m vit_small_patch14_reg4_dinov2.lvd142m"
CFG="--weight-mode size --prop-attn on"
run() { python eval_imagenet.py --data $IMAGENET/val --workers $W --out $OUT --save-preds "$@" \
        || echo "FAILED: $*" >> results/probe_failed.txt; }

for M in $MODELS; do
  P=probes/$M.pt
  [ -f $P ] || python train_probe.py --model $M --data $IMAGENET/train --per-class 100 --workers $W --out $P \
      || { echo "FAILED probe $M" >> results/probe_failed.txt; continue; }
  run --model $M --probe $P --method none
  for R in 0.6 0.45; do
    S="--model $M --probe $P --target-ratio $R"
    for T in 0 1 2 3; do run $S --tau $T --method tome; done
    for T in 1 2 3; do run $S --tau $T --method sadtm $CFG --prior uniform; done
    for T in 0 1 2 3; do run $S --tau $T --method calib $CFG --blend soft --gamma 4; done
    for T in 1 2; do
      run $S --tau $T --method calib $CFG --blend soft --gamma 4 --schedule geo
      run $S --tau $T --method tome --schedule geo
    done
    run $S --tau 2 --method sadtm $CFG --prior uniform --schedule geo
  done
done
python compare_calib.py $OUT > results/probe_compare.txt
python compare_sched.py $OUT > results/probe_sched_compare.txt
echo DONE
