#!/usr/bin/env bash
# ADE20K segmentation with test-time merging (DINOv2-S/B/L, 448 px). Waits for batch 2 (unit tmm2). Resumable.
#   cd ~/S_VIT/tokenmerge_study && systemd-run --user --unit=tmseg -p WorkingDirectory=$PWD bash -c 'bash run_seg.sh >> results/seg.log 2>&1'
cd "$(dirname "$0")"
source venv/bin/activate
ADE=$HOME/data/ade20k/ADEChallengeData2016
while systemctl --user is-active --quiet tmm2; do echo "waiting for batch 2  $(date +%H:%M)"; sleep 300; done
echo "== start $(date)"
[ -d $ADE/images/validation ] || { echo "ADE20K not found at $ADE"; exit 1; }
for M in vit_small_patch14_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m vit_large_patch14_dinov2.lvd142m; do
  echo "== seg $M $(date +%H:%M)"
  python seg_ade20k.py --model $M --data $ADE --img 448 2>&1 | grep -v -i warn || echo "FAILED seg $M" >> results/seg_failed.txt
done
echo "== SEG DONE $(date)"
