#!/usr/bin/env bash
# DeiT-S prior study under the frozen configuration that reproduces the paper's SAD-TM-DM
# (79.68 @ 3.0G vs paper 79.71): norm=attn, size-weighted merging, proportional attention, tau=3.
# Usage:  IMAGENET=/path/to/imagenet bash run_deit_s.sh      (~20 min on one GPU)
set -euo pipefail
: "${IMAGENET:?set IMAGENET=/path/to/imagenet}"
M=deit_small_patch16_224
PF="--prior-file priors/$M.pt"
OUT=results/deit_s.jsonl
CFG="--weight-mode size --prop-attn on --norm attn --rho 0.25"
E="python eval_imagenet.py --model $M --data $IMAGENET/val --workers ${WORKERS:-16} --out $OUT"

# 1) every prior at two budgets (all control priors rescaled to the saliency prior's mean/std)
for G in 3.0 2.5; do
  $E --target-gflops $G --method tome
  $E --target-gflops $G --method sadtm $CFG --prior uniform
  $E --target-gflops $G --method sadtm $CFG --prior saliency $PF
  $E --target-gflops $G --method sadtm $CFG --prior center $PF
  $E --target-gflops $G --method sadtm $CFG --prior anticenter $PF
  for S in 0 1 2; do
    $E --target-gflops $G --method sadtm $CFG --prior random $PF --seed $S
    $E --target-gflops $G --method sadtm $CFG --prior saliency_shuffled $PF --seed $S
  done
  $E --target-gflops $G --method sadtm $CFG --prior oracle
done

# 2) gain sweep: amplify each prior's spatial variation (gain 0 == no prior)
for GAIN in 2 5 10 20; do
  $E --target-gflops 3.0 --method sadtm $CFG --prior saliency $PF --prior-gain $GAIN
  $E --target-gflops 3.0 --method sadtm $CFG --prior center $PF --prior-gain $GAIN
done

python summarize.py $OUT | tee results/deit_s_table.txt
