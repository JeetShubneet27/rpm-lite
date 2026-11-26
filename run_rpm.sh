#!/usr/bin/env bash
# NEW METHOD: readout-preserving merging (rpm: CLS query, cheap; rpm_all: every query).
# Same fair protocol as everything else: tau x schedule (+ probe allocation) chosen on 10 train
# images/class, one val pass. Runs NOW (shares the GPU with the review queue). Resumable.
#   cd ~/S_VIT/tokenmerge_study && (nohup bash run_rpm.sh > results/rpm.log 2>&1 &)
#   progress: grep "==" results/rpm.log; tail -n 12 results/rpm_report.txt
set -uo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/VIT_5/data/imagenet}
W=${WORKERS:-8}
FAIL=results/rpm_failed.txt
echo "== start $(date)"
sel() { python eval_imagenet.py --data $IMAGENET/train --per-class 10 --workers $W --out results/rpm_sel.jsonl \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED sel: $*" >> $FAIL; }
ev()  { python eval_imagenet.py --data $IMAGENET/val --workers $W --out results/rpm_val.jsonl --save-preds \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED val: $*" >> $FAIL; }
report() { python select_report.py results/{calib,sched,probe,fix_val,fix_sel,hires_val,hires_sel,rel_val,rel_sel,review_val,review_sel,rpm_val,rpm_sel}.jsonl \
             > results/rpm_report.txt 2>&1; echo "   report $(date +%H:%M)"; }
model_budgets() {   # "<model args>|<budget list flag>|<budgets>"  foundation models first
  for M in vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m vit_small_patch14_dinov2.lvd142m \
           vit_base_patch14_reg4_dinov2.lvd142m vit_base_patch16_clip_224.laion2b vit_large_patch14_dinov2.lvd142m \
           vit_large_patch14_reg4_dinov2.lvd142m; do
    echo "--model $M --probe probes/$M.pt|--target-ratio|0.45 0.6 0.75"; done
  echo "--model deit_base_patch16_224|--target-gflops|9.0 10.1 11.5"
  echo "--model deit3_base_patch16_224.fb_in1k|--target-gflops|9.0 10.1 11.5"
  echo "--model deit_small_patch16_224|--target-gflops|2.2 2.4 2.6 3.0 3.6"
  echo "--model deit3_small_patch16_224.fb_in1k|--target-gflops|2.2 2.6 3.0"
  echo "--model deit_tiny_patch16_224|--target-gflops|0.7 0.8 0.9"
}
model_budgets | while IFS='|' read -r m flag budgets; do
  name=$(echo $m | awk '{print $2}')
  echo "== $name $(date +%H:%M)"
  for B in $budgets; do
    for F in rpm rpm_all; do
      for T in 0 1 2 3; do sel $m $flag $B --tau $T --method $F; done
      for T in 1 2 3; do sel $m $flag $B --tau $T --method $F --schedule geo; done
      for K in 1 2 4; do sel $m $flag $B --method $F --schedule rel --rel-kappa $K; done
    done
  done
  for F in rpm rpm_all; do
    python pick_selected.py results/rpm_sel.jsonl --family $F --model $name | while read -r c; do ev $m $c; done
  done
  report; tail -n 4 results/rpm_report.txt
done
echo "== RPM DONE $(date)"
