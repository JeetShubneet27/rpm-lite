#!/usr/bin/env bash
# E1: DINOv2 at 448 px (4x more tokens than 224 px), merged down to 50%, 35% and to the cost of the
# 224 px model. Same fair protocol as run_fix.sh: every method picks tau + schedule on 10 TRAIN
# images per class, then ONLY the selected configuration is evaluated once on ImageNet val.
# Accuracy only, so it can share the GPU with other jobs (~10 h, resumable: re-run to continue).
#   IMAGENET=/path/to/imagenet nohup bash run_hires.sh > results/hires.log 2>&1 &
set -uo pipefail
: "${IMAGENET:?set IMAGENET=/path/to/imagenet}"
W=${WORKERS:-16}
PX=${PX:-448}
MODELS=${MODELS:-"vit_base_patch14_dinov2.lvd142m vit_large_patch14_reg4_dinov2.lvd142m vit_large_patch14_dinov2.lvd142m"}
VAL=results/hires_val.jsonl
SEL=results/hires_sel.jsonl
FAIL=results/hires_failed.txt
CFG="--weight-mode size --prop-attn on"
ev()  { python eval_imagenet.py ${EXTRA:-} --data $IMAGENET/val --img-size $PX --workers $W --out $VAL --save-preds \
        --skip-existing "$@" || echo "FAILED val: $*" >> $FAIL; }
sel() { python eval_imagenet.py ${EXTRA:-} --data $IMAGENET/train --img-size $PX --per-class 10 --workers $W --out $SEL \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED sel (budget unreachable is expected): $*" >> $FAIL; }

fm_candidates() {   # identical grid to run_fix.sh
  for T in 0 1 2 3; do echo "$1 --tau $T --method tome"; done
  for T in 1 2; do echo "$1 --tau $T --method tome --schedule geo"; done
  for T in 1 2 3; do echo "$1 --tau $T --method sadtm $CFG --prior uniform"; done
  echo "$1 --tau 2 --method sadtm $CFG --prior uniform --schedule geo"
  for T in 0 1 2 3; do echo "$1 --tau $T --method calib $CFG --blend soft --gamma 4"; done
  for T in 1 2; do echo "$1 --tau $T --method calib $CFG --blend soft --gamma 4 --schedule geo"; done
  for T in 0 1 2 3; do echo "$1 --tau $T --method hybrid $CFG --blend soft --gamma 4"; done
  for T in 1 2 3; do echo "$1 --tau $T --method hybrid $CFG --blend soft --gamma 4 --schedule geo"; done
}

for M in $MODELS; do
  echo "=== $M @ ${PX}px  $(date)"
  PROBE=probes/${M}_${PX}.pt
  # 1. linear probe at this resolution (same recipe as the 224 px probes)
  [ -f $PROBE ] || python train_probe.py ${EXTRA:-} --model $M --img-size $PX --data $IMAGENET/train --per-class 100 \
                     --workers $W --out $PROBE || { echo "FAILED probe $M" >> $FAIL; continue; }
  # 2. reliability probe at this resolution (label-free; 64 LOO images, 1024 patches each)
  python - "$M" "$PX" <<'PY' || python measure_reliability.py ${EXTRA:-} ${REL_ARGS:-} --data $IMAGENET/train --models $M --img-size $PX \
                                  --loo-images 64 --batch-size 16 --workers $W --out results/reliability.jsonl
import json, sys, os
m, px = sys.argv[1], int(sys.argv[2])
f = "results/reliability.jsonl"
ok = os.path.exists(f) and any(json.loads(l)["model"] == m and json.loads(l).get("img_size", 224) == px
                               for l in open(f) if l.strip())
sys.exit(0 if ok else 1)
PY
  P="--model $M --probe $PROBE"
  # 3. unmerged accuracy at this resolution
  ev $P --method none
  # 4. budgets: 50%, 35% and the GFLOPs of the same model at 224 px
  G224=$(python -c "
import timm; from tm.patch import apply_patch, simulate_counts; from tm.flops import vit_gflops
m = apply_patch(timm.create_model('$M', pretrained=False, img_size=224, num_classes=0))
print(round(vit_gflops(m, simulate_counts(m, 'none', [0] * len(m.blocks))), 2))")
  for B in "--target-ratio 0.5" "--target-ratio 0.35" "--target-gflops $G224"; do
    fm_candidates "$P $B" | while read -r c; do sel $c; done
  done
  # 5. evaluate on val only what each method selected on the training subset
  python pick_selected.py $SEL --model $M --img-size $PX | while read -r c; do ev $P $c; done
  python select_report.py $VAL $SEL | tee results/hires_report.txt
done
echo DONE $(date)
