#!/usr/bin/env bash
# Part 1: is SAD-TM's saliency prior just a position prior?
# Usage:  IMAGENET=/data/imagenet bash run_part1.sh
#   $IMAGENET/train and $IMAGENET/val must both be ImageFolder layouts (1000 class sub-folders).
set -euo pipefail
: "${IMAGENET:?set IMAGENET=/path/to/imagenet}"
WORKERS=${WORKERS:-16}
OUT=${OUT:-results/part1.jsonl}
mkdir -p priors results figs

declare -A BUDGETS=(
  [deit_tiny_patch16_224]="0.9 0.8"      # paper Table 5 budgets
  [deit_small_patch16_224]="3.0 2.5"
  [deit_base_patch16_224]="11.5 10.1"
)

# ---- Step 1: offline saliency priors (from TRAIN images, never val) -------------------------
for M in "${!BUDGETS[@]}"; do
  [ -f priors/$M.pt ] || python compute_prior.py --model $M --data $IMAGENET/train \
      --num-images 10000 --workers $WORKERS --out priors/$M.pt
done

# ---- Step 2: diagnostics (center correlation, cross-model agreement, heatmaps) --------------
python analyze_prior.py priors/deit_tiny_patch16_224.pt priors/deit_small_patch16_224.pt \
    priors/deit_base_patch16_224.pt --out figs | tee results/prior_analysis.txt

# ---- Step 3: accuracy at matched FLOPs -------------------------------------------------------
E="python eval_imagenet.py --data $IMAGENET/val --workers $WORKERS --out $OUT --throughput"
for M in "${!BUDGETS[@]}"; do
  $E --model $M --method none
  for G in ${BUDGETS[$M]}; do
    S="--model $M --target-gflops $G"
    $E $S --method tome
    $E $S --method sadtm --prior saliency   --prior-file priors/$M.pt      # paper's method
    $E $S --method sadtm --prior uniform                                   # = attention only (alpha=1)
    $E $S --method sadtm --prior center
    $E $S --method sadtm --prior anticenter
    for SEED in 0 1 2; do
      $E $S --method sadtm --prior random --seed $SEED
      $E $S --method sadtm --prior saliency_shuffled --prior-file priors/$M.pt --seed $SEED
    done
    $E $S --method sadtm --prior saliency --prior-file priors/$M.pt --rho 0.25 --tag DM
    $E $S --method sadtm --prior center --rho 0.25 --tag DM
  done
done

# per-image (online, needs a backward pass) saliency: is per-image info any better than the mean?
$E --model deit_small_patch16_224 --target-gflops 2.5 --method sadtm --prior oracle

python summarize.py $OUT | tee results/part1_table.txt
