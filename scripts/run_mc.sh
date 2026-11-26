#!/usr/bin/env bash
# Our method calibrated against MERGE COST at the layer instead of leave-one-out (same protocol:
# selection on 10 train images/class, one val pass). Waits for run_review.sh. ~5 h, resumable.
#   cd ~/S_VIT/tokenmerge_study && (nohup bash run_mc.sh > results/mc.log 2>&1 &)
set -uo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/VIT_5/data/imagenet}
W=${WORKERS:-12}
FAIL=results/mc_failed.txt
while pgrep -f "run_extra.sh|start_wsl.sh|run_review.sh" >/dev/null 2>&1; do echo "waiting  $(date +%H:%M)"; sleep 300; done
echo "== start $(date)"
python make_reliability_mc.py results/probe_validation.jsonl results/reliability_mc.jsonl || exit 1
REL="--reliability results/reliability_mc.jsonl --tag mc"
CFG="--weight-mode size --prop-attn on --blend soft --gamma 4"
settings() {
  for G in 3.6 3.0 2.6 2.4 2.2; do echo "--model deit_small_patch16_224|--target-gflops $G"; done
  for G in 0.9 0.8 0.7; do echo "--model deit_tiny_patch16_224|--target-gflops $G"; done
  for G in 3.0 2.6 2.2; do echo "--model deit3_small_patch16_224.fb_in1k|--target-gflops $G"; done
  for G in 11.5 10.1 9.0; do echo "--model deit_base_patch16_224|--target-gflops $G"; done
  for G in 11.5 10.1 9.0; do echo "--model deit3_base_patch16_224.fb_in1k|--target-gflops $G"; done
  for M in vit_small_patch14_dinov2.lvd142m vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m \
           vit_base_patch14_reg4_dinov2.lvd142m vit_base_patch16_clip_224.laion2b \
           vit_large_patch14_dinov2.lvd142m vit_large_patch14_reg4_dinov2.lvd142m; do
    for R in 0.75 0.6 0.45; do echo "--model $M --probe probes/$M.pt|--target-ratio $R"; done
  done
}
sel() { python eval_imagenet.py --data $IMAGENET/train --per-class 10 --workers $W --out results/mc_sel.jsonl \
        --skip-existing $REL "$@" >/dev/null 2>&1 || echo "FAILED sel: $*" >> $FAIL; }
ev()  { python eval_imagenet.py --data $IMAGENET/val --workers $W --out results/mc_val.jsonl --save-preds \
        --skip-existing $REL "$@" >/dev/null 2>&1 || echo "FAILED val: $*" >> $FAIL; }
settings | while IFS='|' read -r m b; do
  for T in 0 1 2 3; do sel $m $b --tau $T --method hybrid $CFG; done
  for T in 1 2 3; do sel $m $b --tau $T --method hybrid $CFG --schedule geo; done
  for K in 1 2 4; do sel $m $b --method hybrid $CFG --schedule rel --rel-kappa $K; done
done
settings | cut -d'|' -f1 | sort -u | while read -r m; do
  python pick_selected.py results/mc_sel.jsonl --family hybrid --model $(echo $m | awk '{print $2}') \
    | while read -r c; do ev $m $c; done
done
python review_report.py results/*.jsonl > results/review_report.txt 2>&1
grep -A6 "^H\." results/review_report.txt
echo "== MC DONE $(date)"
