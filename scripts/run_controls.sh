#!/usr/bin/env bash
# Part 1 controls (~40 min).  Usage:  IMAGENET=/path/to/imagenet bash run_controls.sh
#   A. ToMe WITH deferred merging (tau = 1..3): does SAD-TM's CLS-attention selection add anything
#      over plain bipartite matching once both defer merging?
#   B. anti-center gain sweep: the saliency prior correlates -0.87 with a center Gaussian; does an
#      amplified anti-center prior hurt like an amplified saliency prior?
#   C. throughput (only meaningful when no other job is using the GPU: set THROUGHPUT=1)
set -uo pipefail
: "${IMAGENET:?set IMAGENET=/path/to/imagenet}"
W=${WORKERS:-16}
CFG="--weight-mode size --prop-attn on --norm attn"
run() { python eval_imagenet.py --data $IMAGENET/val --workers $W --save-preds "$@" \
        || echo "FAILED: $*" >> results/controls_failed.txt; }

# ---------- A. ToMe + deferral, all three models -----------------------------------------------
for MG in "deit_small_patch16_224:3.6 3.3 3.0 2.8 2.6 2.4 2.2" "deit_tiny_patch16_224:0.9 0.8" \
          "deit_base_patch16_224:11.5 10.1"; do
  M=${MG%%:*}; GS=${MG#*:}
  for G in $GS; do for T in 1 2 3; do
    run --out results/full.jsonl --model $M --target-gflops $G --tau $T --method tome
  done; done
done

# ---------- B. gain sweeps at 3.0G, tau=3 -------------------------------------------------------
M=deit_small_patch16_224; PF="--prior-file priors/$M.pt"
for GAIN in 1 2 5 10 20; do
  run --out results/deit_s.jsonl --model $M --target-gflops 3.0 --tau 3 --method sadtm $CFG \
      --prior anticenter $PF --prior-gain $GAIN
done

# ---------- C. throughput (bs 256, fp32) ---------------------------------------------------------
if [ "${THROUGHPUT:-0}" = "1" ]; then
  for G in 3.0 2.5; do
    run --out results/throughput.jsonl --model $M --target-gflops $G --method tome --throughput --limit 256
    run --out results/throughput.jsonl --model $M --target-gflops $G --tau 1 --method sadtm $CFG --prior uniform --throughput --limit 256
    run --out results/throughput.jsonl --model $M --target-gflops $G --tau 1 --method sadtm $CFG --prior saliency $PF --throughput --limit 256
  done
  run --out results/throughput.jsonl --model $M --method none --throughput --limit 256
fi

python summarize.py results/full.jsonl > results/full_table.txt
python significance.py results/full.jsonl --ref-prior uniform > results/full_significance.txt
python summarize.py results/deit_s.jsonl > results/deit_s_table.txt
python plot_results.py
echo "DONE"; cat results/controls_failed.txt 2>/dev/null
