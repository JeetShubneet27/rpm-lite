#!/usr/bin/env bash
# KL / flip / ECE for ToMe (missed by run_rev3.sh because ToMe's selection rows live in older files). Waits for batch 3.
#   cd ~/S_VIT/tokenmerge_study && systemd-run --user --unit=tmfid -p WorkingDirectory=$PWD bash -c 'bash run_fidtome.sh >> results/fidtome.log 2>&1'
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/data/imagenet}
while systemctl --user is-active --quiet tmrev3; do echo "waiting for batch 3  $(date +%H:%M)"; sleep 300; done
margs() { case $1 in vit_*dinov2*|vit_*clip*) echo "--model $1 --probe probes/$1.pt";; *) echo "--model $1";; esac; }
L=results/cfgs/list_fid_tome.txt; : > $L
while IFS='|' read -r M C; do
  P=results/cfgs/fid_${M}_tome_$(echo $C | awk '{print $2}').pt
  [ -f $P ] || python eval_imagenet.py --data $IMAGENET/val $(margs $M) $C --save-config $P >/dev/null 2>&1 || { echo "FAILED cfg $P" >> results/fidtome_failed.txt; continue; }
  echo "tome|$P" >> $L
done <<'LIST'
deit3_base_patch16_224.fb_in1k|--target-gflops 9.0 --tau 0 --method tome
deit3_base_patch16_224.fb_in1k|--target-gflops 10.1 --tau 1 --method tome
deit3_base_patch16_224.fb_in1k|--target-gflops 11.5 --tau 3 --method tome
deit3_small_patch16_224.fb_in1k|--target-gflops 2.2 --tau 0 --method tome
deit3_small_patch16_224.fb_in1k|--target-gflops 2.6 --tau 1 --method tome
deit3_small_patch16_224.fb_in1k|--target-gflops 3.0 --tau 3 --method tome
deit_base_patch16_224|--target-gflops 9.0 --tau 0 --method tome
deit_base_patch16_224|--target-gflops 10.1 --tau 1 --method tome
deit_base_patch16_224|--target-gflops 11.5 --tau 2 --method tome
deit_small_patch16_224|--target-gflops 2.2 --tau 2 --method tome --schedule geo
deit_small_patch16_224|--target-gflops 2.4 --tau 3 --method tome --schedule geo
deit_small_patch16_224|--target-gflops 2.6 --tau 3 --method tome --schedule geo
deit_small_patch16_224|--target-gflops 3.0 --tau 3 --method tome
deit_small_patch16_224|--target-gflops 3.6 --tau 2 --method tome
deit_tiny_patch16_224|--target-gflops 0.7 --tau 3 --method tome --schedule geo
deit_tiny_patch16_224|--target-gflops 0.8 --tau 3 --method tome --schedule geo
deit_tiny_patch16_224|--target-gflops 0.9 --tau 1 --method tome
vit_base_patch14_dinov2.lvd142m|--target-ratio 0.45 --tau 1 --method tome --schedule geo
vit_base_patch14_dinov2.lvd142m|--target-ratio 0.6 --tau 1 --method tome --schedule geo
vit_base_patch14_dinov2.lvd142m|--target-ratio 0.75 --tau 2 --method tome
vit_base_patch14_reg4_dinov2.lvd142m|--target-ratio 0.45 --tau 1 --method tome --schedule geo
vit_base_patch14_reg4_dinov2.lvd142m|--target-ratio 0.6 --tau 1 --method tome
vit_base_patch14_reg4_dinov2.lvd142m|--target-ratio 0.75 --tau 3 --method tome
vit_base_patch16_clip_224.laion2b|--target-ratio 0.45 --tau 0 --method tome
vit_base_patch16_clip_224.laion2b|--target-ratio 0.6 --tau 3 --method tome
vit_base_patch16_clip_224.laion2b|--target-ratio 0.75 --tau 3 --method tome
vit_large_patch14_dinov2.lvd142m|--target-ratio 0.45 --tau 1 --method tome --schedule geo
vit_large_patch14_dinov2.lvd142m|--target-ratio 0.6 --tau 1 --method tome --schedule geo
vit_large_patch14_dinov2.lvd142m|--target-ratio 0.75 --tau 3 --method tome
vit_large_patch14_reg4_dinov2.lvd142m|--target-ratio 0.45 --tau 1 --method tome --schedule geo
vit_large_patch14_reg4_dinov2.lvd142m|--target-ratio 0.6 --tau 1 --method tome --schedule geo
vit_large_patch14_reg4_dinov2.lvd142m|--target-ratio 0.75 --tau 1 --method tome
vit_small_patch14_dinov2.lvd142m|--target-ratio 0.45 --tau 1 --method tome --schedule geo
vit_small_patch14_dinov2.lvd142m|--target-ratio 0.6 --tau 0 --method tome
vit_small_patch14_dinov2.lvd142m|--target-ratio 0.75 --tau 0 --method tome
vit_small_patch14_reg4_dinov2.lvd142m|--target-ratio 0.45 --tau 1 --method tome --schedule geo
vit_small_patch14_reg4_dinov2.lvd142m|--target-ratio 0.6 --tau 1 --method tome
vit_small_patch14_reg4_dinov2.lvd142m|--target-ratio 0.75 --tau 1 --method tome --schedule geo
LIST
python fidelity.py --list $L --data $IMAGENET/val --out results/fidelity.jsonl --workers 6 2>&1 | grep -v -i warn
echo "== FIDTOME DONE $(date)"
