#!/usr/bin/env bash
# Final batch before the first draft. Waits for segmentation (unit tmseg). Resumable.
#   cd ~/S_VIT/tokenmerge_study && systemd-run --user --unit=tmfinal -p WorkingDirectory=$PWD bash -c 'bash run_final.sh >> results/final.log 2>&1'
#  F1 merge visualisations (ToMe, MCTF, RPM-lite)                                                   ~5 min
#  F2 cheapest-pairs precision: why the product beats a weighted sum (6 models, 8 images)            ~1.5 h
#  F3 two more supervised ViTs (AugReg ViT-S/B), 6 methods, same 7-candidate grid (no probe needed)   ~5 h
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/data/imagenet}
W=${WORKERS:-6}
FAIL=results/final_failed.txt
while systemctl --user is-active --quiet tmseg; do echo "waiting for segmentation  $(date +%H:%M)"; sleep 300; done
echo "== start $(date)"

echo "== F1 visualisations $(date)"
python vis_merge.py --data $IMAGENET/val --out results/vis 2>&1 | grep -v -i warn || echo "FAILED vis" >> $FAIL

echo "== F2 cheapest-pairs precision $(date)"
python pair_cost_v4.py --data $IMAGENET/train --images 8 --out results/pair_cost_v4.jsonl 2>&1 | grep -v -i warn

echo "== F3 AugReg ViT-S/B $(date)"
sel() { python eval_imagenet.py --data $IMAGENET/train --per-class 10 --workers $W --out results/rpm_sel.jsonl \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED sel: $*" >> $FAIL; }
ev()  { python eval_imagenet.py --data $IMAGENET/val --workers $W --out results/rpm_val.jsonl --save-preds \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED val: $*" >> $FAIL; }
for M in vit_small_patch16_224.augreg_in21k_ft_in1k vit_base_patch16_224.augreg_in21k_ft_in1k; do
  echo "== augreg $M $(date +%H:%M)"
  ev --model $M --method none
  for B in 0.45 0.6 0.75; do
    for METH in "--method tome" "--method sadtm --weight-mode size --prop-attn on --prior uniform" "--method pitome" \
                "--method mctf" "--method adamerge" "--method rpmlnw"; do
      for T in 0 1 2 3; do sel --model $M --target-ratio $B --tau $T $METH; done
      for T in 1 2 3; do sel --model $M --target-ratio $B --tau $T --schedule geo $METH; done
    done
  done
  for F in tome sadtm pitome mctf adamerge rpmlnw; do
    python pick_selected.py results/rpm_sel.jsonl --family $F --model $M | while read -r c; do ev --model $M $c; done
  done
done
echo "== FINAL DONE $(date)"
