#!/usr/bin/env bash
# Dense-aware cost (rpmd = RPM-lite + replacement error). Waits for the AdaMerge variant (unit tmfinal2). Resumable.
#   cd ~/S_VIT/tokenmerge_study && systemd-run --user --unit=tmdense -p WorkingDirectory=$PWD bash -c 'bash run_dense.sh >> results/dense.log 2>&1'
#  D1 ADE20K segmentation with rpmd (heads already trained), DINOv2-S/B/L, 75/60/45%               ~40 min
#  D2 ImageNet: rpmd at RPM-lite's selected configuration, all 38 settings                          ~1.5 h
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/data/imagenet}
W=${WORKERS:-6}
FAIL=results/dense_failed.txt
while systemctl --user is-active --quiet tmfinal2; do echo "waiting for AdaMerge variant  $(date +%H:%M)"; sleep 300; done
echo "== start $(date)"
echo "== D1 segmentation $(date)"
for M in vit_small_patch14_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m vit_large_patch14_dinov2.lvd142m; do
  python seg_ade20k.py --model $M --data $HOME/data/ade20k/ADEChallengeData2016 --img 448 --methods rpmd 2>&1 | grep -v -i warn \
    || echo "FAILED seg $M" >> $FAIL
done
echo "== D2 ImageNet $(date)"
margs() { case $1 in vit_*dinov2*|vit_*clip*) echo "--model $1 --probe probes/$1.pt";; *) echo "--model $1";; esac; }
for M in vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m vit_base_patch16_clip_224.laion2b \
         vit_large_patch14_dinov2.lvd142m vit_small_patch14_dinov2.lvd142m vit_base_patch14_reg4_dinov2.lvd142m \
         vit_large_patch14_reg4_dinov2.lvd142m deit3_small_patch16_224.fb_in1k deit_base_patch16_224 \
         deit3_base_patch16_224.fb_in1k deit_small_patch16_224 deit_tiny_patch16_224; do
  python pick_selected.py results/rpm_sel.jsonl --family rpmlnw --model $M | while read -r c; do
    python eval_imagenet.py --data $IMAGENET/val --workers $W --out results/rpm_val.jsonl --save-preds --skip-existing \
      $(margs $M) ${c/--method rpmlnw/--method rpmd} >/dev/null 2>&1 || echo "FAILED val $M $c" >> $FAIL
  done
done
echo "== DENSE DONE $(date)"
