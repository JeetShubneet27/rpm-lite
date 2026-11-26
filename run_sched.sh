#!/usr/bin/env bash
# Part 2, step 4: geometric (non-collapsing) schedule, tau = 1..3, three selection rules (~2 h).
#   IMAGENET=/path/to/imagenet bash run_sched.sh
set -uo pipefail
: "${IMAGENET:?set IMAGENET=/path/to/imagenet}"
W=${WORKERS:-16}
OUT=results/sched.jsonl
CFG="--weight-mode size --prop-attn on --schedule geo"
run() { python eval_imagenet.py --data $IMAGENET/val --workers $W --out $OUT --save-preds "$@" \
        || echo "FAILED: $*" >> results/sched_failed.txt; }
declare -A BUD=(
  [deit_tiny_patch16_224]="0.9 0.8 0.7"
  [deit_small_patch16_224]="3.6 3.0 2.6 2.4 2.2"
  [deit_base_patch16_224]="11.5 10.1 9.0"
  [deit3_small_patch16_224.fb_in1k]="3.0 2.6 2.2"
  [deit3_base_patch16_224.fb_in1k]="11.5 10.1 9.0"
)
for M in deit_small_patch16_224 deit_tiny_patch16_224 deit3_small_patch16_224.fb_in1k \
         deit_base_patch16_224 deit3_base_patch16_224.fb_in1k; do
  for G in ${BUD[$M]}; do
    for T in 1 2 3; do
      S="--model $M --target-gflops $G --tau $T"
      run $S --method calib $CFG --blend soft --gamma 4
      run $S --method sadtm $CFG --prior uniform
      run $S --method tome --schedule geo
    done
  done
done
python compare_sched.py results/calib.jsonl $OUT > results/sched_compare.txt
echo DONE
