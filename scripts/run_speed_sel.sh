#!/usr/bin/env bash
# Fair speed of every selected configuration (ToMe, MCTF, probe hybrid, RPM-lite), eager and compiled, on this GPU.
# Waits for run_rpmlnw.sh.  Resumable.
#   cd ~/S_VIT/tokenmerge_study && (nohup bash run_speed_sel.sh > results/speed_sel.log 2>&1 &)
set -uo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/data/imagenet}
while pgrep -f "^bash run_rpmlnw.sh" >/dev/null 2>&1; do echo "waiting for run_rpmlnw.sh  $(date +%H:%M)"; sleep 120; done
echo "== start $(date)"
mkdir -p results/cfgs
LIST=results/cfgs/list.txt
: > $LIST
for M in vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m vit_base_patch16_clip_224.laion2b \
         vit_large_patch14_dinov2.lvd142m vit_small_patch14_dinov2.lvd142m vit_base_patch14_reg4_dinov2.lvd142m \
         vit_large_patch14_reg4_dinov2.lvd142m deit3_small_patch16_224.fb_in1k deit_base_patch16_224 \
         deit3_base_patch16_224.fb_in1k deit_small_patch16_224 deit_tiny_patch16_224; do
  case $M in vit_*) m="--model $M --probe probes/$M.pt";; *) m="--model $M";; esac
  for F in tome mctf hybrid rpmlnw; do
    python pick_selected.py results/*sel*.jsonl --split sel10s1 --family $F --model $M | while read -r c; do
      B=$(echo "$c" | awk '{print $2}')
      P=results/cfgs/${M}_${F}_${B}.pt
      [ -f "$P" ] || python eval_imagenet.py --data $IMAGENET/val --save-config $P $m $c >/dev/null 2>&1 \
        || { echo "FAILED config $M $F $c"; continue; }
      echo "$F|$P" >> $LIST
    done
  done
done
echo "== configs: $(wc -l < $LIST)  $(date)"
python bench_selected.py $LIST --out results/speed_sel.jsonl 2>&1 | grep -v -i warn
echo "== SPEED_SEL DONE $(date)"
