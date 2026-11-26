#!/usr/bin/env bash
# Part 2, step 3: calibrated selection AFTER deferral (tau = 1..3), ~50 min.
#   IMAGENET=/path/to/imagenet bash run_calib2.sh
set -uo pipefail
: "${IMAGENET:?set IMAGENET=/path/to/imagenet}"
W=${WORKERS:-16}
OUT=results/calib.jsonl          # same file: compare_calib.py sees baselines and all calib runs together
CFG="--weight-mode size --prop-attn on"
run() { python eval_imagenet.py --data $IMAGENET/val --workers $W --out $OUT --save-preds "$@" \
        || echo "FAILED: $*" >> results/calib_failed.txt; }
declare -A BUD=(
  [deit_tiny_patch16_224]="0.9 0.8 0.7"
  [deit_small_patch16_224]="3.6 3.0 2.6 2.4 2.2"
  [deit_base_patch16_224]="11.5 10.1 9.0"
  [deit3_base_patch16_224.fb_in1k]="11.5 10.1 9.0"
)
for M in deit_small_patch16_224 deit_tiny_patch16_224 deit_base_patch16_224 deit3_base_patch16_224.fb_in1k; do
  for G in ${BUD[$M]}; do
    for T in 1 2 3; do
      run --model $M --target-gflops $G --tau $T --method calib $CFG --blend hard
      run --model $M --target-gflops $G --tau $T --method calib $CFG --blend soft --gamma 4
    done
  done
done
python compare_calib.py $OUT > results/calib_compare2.txt
echo DONE
