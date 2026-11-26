#!/usr/bin/env bash
# Part 2, step 2: calibrated merging vs. ToMe and SAD-TM at every tau (~2 h).
#   IMAGENET=/path/to/imagenet bash run_calib.sh
set -uo pipefail
: "${IMAGENET:?set IMAGENET=/path/to/imagenet}"
W=${WORKERS:-16}
OUT=results/calib.jsonl
CFG="--weight-mode size --prop-attn on"
run() { python eval_imagenet.py --data $IMAGENET/val --workers $W --out $OUT --save-preds "$@" \
        || echo "FAILED: $*" >> results/calib_failed.txt; }

declare -A BUD=(
  [deit_tiny_patch16_224]="0.9 0.8 0.7"
  [deit_small_patch16_224]="3.6 3.0 2.6 2.4 2.2"
  [deit_base_patch16_224]="11.5 10.1 9.0"
  [deit3_small_patch16_224.fb_in1k]="3.0 2.6 2.2"
  [deit3_base_patch16_224.fb_in1k]="11.5 10.1 9.0"
)
for M in deit_small_patch16_224 deit_tiny_patch16_224 deit3_small_patch16_224.fb_in1k \
         deit_base_patch16_224 deit3_base_patch16_224.fb_in1k; do
  run --model $M --method none
  for G in ${BUD[$M]}; do
    S="--model $M --target-gflops $G"
    for T in 0 1 2 3; do run $S --tau $T --method tome; done
    for T in 1 2 3; do run $S --tau $T --method sadtm $CFG --prior uniform; done
    run $S --method calib $CFG --blend hard
    run $S --method calib $CFG --blend soft
    run $S --method calib $CFG --blend soft --gamma 4
    run $S --method calib $CFG --blend sim
  done
done
python compare_calib.py $OUT | tee results/calib_compare.txt
echo DONE
