#!/usr/bin/env bash
# Evaluate the selected configurations of the GAP model (256 px), which run_rev3.sh missed. ~20 min, accuracy only.
#   cd ~/S_VIT/tokenmerge_study && systemd-run --user --unit=tmgfix -p WorkingDirectory=$PWD bash -c 'bash run_gfix.sh >> results/gfix.log 2>&1'
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/data/imagenet}
M=vit_medium_patch16_gap_256.sw_in12k_ft_in1k
for F in tome sadtm pitome mctf adamerge rpmlnw evit; do
  python pick_selected.py results/rpm_sel.jsonl --split sel10s1 --img-size 256 --family $F --model $M | while read -r c; do
    echo "eval $F $c"
    python eval_imagenet.py --data $IMAGENET/val --workers 4 --out results/rpm_val.jsonl --save-preds --skip-existing --model $M $c \
      >/dev/null 2>&1 || echo "FAILED val: $F $c" >> results/gfix_failed.txt
  done
done
echo "== GFIX DONE $(date)"
