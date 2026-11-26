#!/usr/bin/env bash
# AdaMerge with size-weighted averaging (its pair score kept), full protocol on the 38 settings + AugReg ViT-S/B.
# Waits for the final batch (unit tmfinal). Resumable.
#   cd ~/S_VIT/tokenmerge_study && systemd-run --user --unit=tmfinal2 -p WorkingDirectory=$PWD bash -c 'bash run_final2.sh >> results/final2.log 2>&1'
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/data/imagenet}
W=${WORKERS:-6}
FAIL=results/final2_failed.txt
while systemctl --user is-active --quiet tmfinal; do echo "waiting for final batch  $(date +%H:%M)"; sleep 300; done
echo "== start $(date)"
margs() { case $1 in vit_*dinov2*|vit_*clip*) echo "--model $1 --probe probes/$1.pt";; *) echo "--model $1";; esac; }
sel() { python eval_imagenet.py --data $IMAGENET/train --per-class 10 --workers $W --out results/rpm_sel.jsonl \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED sel: $*" >> $FAIL; }
ev()  { python eval_imagenet.py --data $IMAGENET/val --workers $W --out results/rpm_val.jsonl --save-preds \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED val: $*" >> $FAIL; }
{
  for M in vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m vit_base_patch16_clip_224.laion2b \
           vit_large_patch14_dinov2.lvd142m vit_small_patch14_dinov2.lvd142m vit_base_patch14_reg4_dinov2.lvd142m \
           vit_large_patch14_reg4_dinov2.lvd142m; do echo "$M|--target-ratio|0.45 0.6 0.75|rel"; done
  echo "deit3_small_patch16_224.fb_in1k|--target-gflops|2.2 2.6 3.0|rel"
  echo "deit_base_patch16_224|--target-gflops|9.0 10.1 11.5|rel"
  echo "deit3_base_patch16_224.fb_in1k|--target-gflops|9.0 10.1 11.5|rel"
  echo "deit_small_patch16_224|--target-gflops|2.2 2.4 2.6 3.0 3.6|rel"
  echo "deit_tiny_patch16_224|--target-gflops|0.7 0.8 0.9|rel"
  echo "vit_small_patch16_224.augreg_in21k_ft_in1k|--target-ratio|0.45 0.6 0.75|norel"
  echo "vit_base_patch16_224.augreg_in21k_ft_in1k|--target-ratio|0.45 0.6 0.75|norel"
} | while IFS='|' read -r M flag budgets rel; do
  m=$(margs $M)
  echo "== adamerge2 $M $(date +%H:%M)"
  for B in $budgets; do
    for T in 0 1 2 3; do sel $m $flag $B --tau $T --method adamerge2; done
    for T in 1 2 3; do sel $m $flag $B --tau $T --schedule geo --method adamerge2; done
    if [ "$rel" = "rel" ]; then for R in 1 2 4; do sel $m $flag $B --schedule rel --rel-kappa $R --method adamerge2; done; fi
  done
  python pick_selected.py results/rpm_sel.jsonl --family adamerge2 --model $M | while read -r c; do ev $m $c; done
done
echo "== FINAL2 DONE $(date)"
