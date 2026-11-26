#!/usr/bin/env bash
# Part 1, full study (~1.5 h on one GPU).  Usage:  IMAGENET=/path/to/imagenet bash run_full.sh
#   A. DeiT-S prior comparison at 2.8G (tau=2) and 2.5G (tau=1)
#   B. DeiT-S accuracy-vs-GFLOPs curves: ToMe and SAD-TM (uniform / saliency) for tau = 0..3
#   C. DeiT-Tiny and DeiT-Base at the paper's Table 5 budgets, every tau, uniform / saliency / random
# All SAD-TM runs use the frozen configuration: norm=attn, size-weighted merging, proportional attention.
set -uo pipefail          # no -e: a single failed run must not stop the rest
: "${IMAGENET:?set IMAGENET=/path/to/imagenet}"
W=${WORKERS:-16}
OUT=results/full.jsonl
CFG="--weight-mode size --prop-attn on --norm attn"
run() { python eval_imagenet.py --data $IMAGENET/val --workers $W --out $OUT --save-preds "$@" \
        || echo "FAILED: $*" >> results/full_failed.txt; }

# ---------- priors for Tiny / Base (Small already exists) -------------------------------------
for M in deit_tiny_patch16_224 deit_base_patch16_224; do
  [ -f priors/$M.pt ] || python compute_prior.py --model $M --data $IMAGENET/train --num-images 10000 \
      --workers $W --out priors/$M.pt
done
python analyze_prior.py priors/deit_tiny_patch16_224.pt priors/deit_small_patch16_224.pt \
    priors/deit_base_patch16_224.pt --out figs | tee results/prior_analysis.txt

# ---------- A. DeiT-S prior comparison at tighter budgets --------------------------------------
M=deit_small_patch16_224; PF="--prior-file priors/$M.pt"
for GT in "3.0 3" "2.8 2" "2.5 1"; do
  set -- $GT; G=$1; T=$2
  S="--model $M --target-gflops $G --tau $T --method sadtm $CFG"
  run $S --prior uniform
  run $S --prior saliency $PF
  run $S --prior center $PF
  run $S --prior anticenter $PF
  for SEED in 0 1 2; do
    run $S --prior random $PF --seed $SEED
    run $S --prior saliency_shuffled $PF --seed $SEED
  done
  run $S --prior oracle
done

# ---------- B. DeiT-S accuracy-vs-GFLOPs curves ------------------------------------------------
for G in 3.6 3.3 3.0 2.8 2.6 2.4 2.2; do
  run --model $M --target-gflops $G --method tome
  for T in 0 1 2 3; do
    run --model $M --target-gflops $G --tau $T --method sadtm $CFG --prior uniform
    run --model $M --target-gflops $G --tau $T --method sadtm $CFG --prior saliency $PF
  done
done

# ---------- C. DeiT-Tiny and DeiT-Base --------------------------------------------------------
declare -A BUD=( [deit_tiny_patch16_224]="0.9 0.8" [deit_base_patch16_224]="11.5 10.1" )
for M in deit_tiny_patch16_224 deit_base_patch16_224; do
  PF="--prior-file priors/$M.pt"
  run --model $M --method none
  for G in ${BUD[$M]}; do
    run --model $M --target-gflops $G --method tome
    for T in 0 1 2 3; do
      S="--model $M --target-gflops $G --tau $T --method sadtm $CFG"
      run $S --prior uniform
      run $S --prior saliency $PF
      run $S --prior random $PF --seed 0
    done
  done
done

python summarize.py $OUT > results/full_table.txt
python significance.py $OUT --ref-prior uniform > results/full_significance.txt
echo "DONE"; cat results/full_failed.txt 2>/dev/null
