#!/usr/bin/env bash
# Reliability-aware budget allocation: the probe decides how much each block merges (no tau, no
# schedule choice). Waits until run_hires.sh has finished, then runs on its own (~10 h, resumable).
#   IMAGENET=/path/to/imagenet nohup bash run_rel.sh > results/rel.log 2>&1 &
#   progress: tail -n 3 results/rel.log; cat results/rel_report.txt
set -uo pipefail
: "${IMAGENET:?set IMAGENET=/path/to/imagenet}"
W=${WORKERS:-8}
VAL=results/rel_val.jsonl
SEL=results/rel_sel.jsonl
FAIL=results/rel_failed.txt
CFG="--weight-mode size --prop-attn on"
HYB="--method hybrid $CFG --blend soft --gamma 4"
KAPPAS=${KAPPAS:-"2 1 4"}            # kappa=2 first: a complete result after the first pass
HIRES=${HIRES:-"vit_base_patch14_dinov2.lvd142m"}   # 448 px models (Large ones are slow off an NVIDIA GPU)
export PYTORCH_ENABLE_MPS_FALLBACK=1

while pgrep -f "bash run_hires.sh" >/dev/null 2>&1; do echo "waiting for run_hires.sh  $(date +%H:%M)"; sleep 300; done
echo "start $(date)"

ev()  { python eval_imagenet.py --data $IMAGENET/val --workers $W --out $VAL --save-preds --skip-existing "$@" \
        >/dev/null 2>&1 || echo "FAILED val: $*" >> $FAIL; }
sel() { python eval_imagenet.py --data $IMAGENET/train --per-class 10 --workers $W --out $SEL --skip-existing "$@" \
        >/dev/null 2>&1 || echo "FAILED sel: $*" >> $FAIL; }
report() { python compare_rel.py $(ls results/{calib,sched,probe,fix_val,fix_sel,hires_val,hires_sel}.jsonl 2>/dev/null) \
           $VAL $SEL > results/rel_report.txt 2>&1; echo "report updated $(date +%H:%M)"; }

settings() {   # one line per setting: "<model/probe/resolution args> <budget args>"
  for G in 3.6 3.0 2.6 2.4 2.2; do echo "--model deit_small_patch16_224 --target-gflops $G"; done
  for G in 0.9 0.8 0.7; do echo "--model deit_tiny_patch16_224 --target-gflops $G"; done
  for G in 3.0 2.6 2.2; do echo "--model deit3_small_patch16_224.fb_in1k --target-gflops $G"; done
  for G in 11.5 10.1 9.0; do echo "--model deit_base_patch16_224 --target-gflops $G"; done
  for G in 11.5 10.1 9.0; do echo "--model deit3_base_patch16_224.fb_in1k --target-gflops $G"; done
  for M in vit_small_patch14_dinov2.lvd142m vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m \
           vit_base_patch14_reg4_dinov2.lvd142m vit_base_patch16_clip_224.laion2b \
           vit_large_patch14_dinov2.lvd142m vit_large_patch14_reg4_dinov2.lvd142m; do
    [ -f probes/$M.pt ] || continue          # probes come from the old machine (flash drive)
    for R in 0.75 0.6 0.45; do echo "--model $M --probe probes/$M.pt --target-ratio $R"; done
  done
  for M in $HIRES; do
    [ -f probes/${M}_448.pt ] || continue
    for R in 0.5 0.35; do echo "--model $M --probe probes/${M}_448.pt --img-size 448 --target-ratio $R"; done
  done
}

# 1. tuning-free hybrid on val, every kappa (the main result)
for K in $KAPPAS; do settings | while read -r s; do ev $s $HYB --schedule rel --rel-kappa $K; done; report; done
# 2. the same on the training subset: picks ONE kappa for all models
for K in $KAPPAS; do settings | while read -r s; do sel $s $HYB --schedule rel --rel-kappa $K; done; done
report
# 3. control: ToMe with the same reliability-aware schedule (is the gain from the schedule alone?)
for K in $KAPPAS; do settings | while read -r s; do ev $s --method tome --schedule rel --rel-kappa $K; done; done
report
cat results/rel_report.txt
echo "DONE $(date)"
