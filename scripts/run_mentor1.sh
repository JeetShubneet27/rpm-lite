#!/usr/bin/env bash
# Mentor feedback, batch 1.  Resumable.
#   cd ~/S_VIT/tokenmerge_study && systemd-run --user --unit=tmm1 -p WorkingDirectory=$PWD bash -c 'bash run_mentor1.sh >> results/mentor1.log 2>&1'
#  M1 latency, throughput, memory and scoring overhead at identical token counts (5 settings)       ~1 h
#  M2 predicted cost vs actual damage of a merge; product vs weighted sum; term ablations (6 models)  ~3 h
#  M3 weighted-sum merge rule (alpha chosen on training images, RPM-lite's schedule), 38 settings     ~5 h
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/data/imagenet}
export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor TRITON_CACHE_DIR=$HOME/.cache/triton TMPDIR=$HOME/tmp
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $TMPDIR
W=${WORKERS:-6}
FAIL=results/mentor1_failed.txt
margs() { case $1 in vit_*) echo "--model $1 --probe probes/$1.pt";; *) echo "--model $1";; esac; }
echo "== start $(date)"

echo "== M1 latency and memory $(date)"
for i in 0 1 2 3 4; do
  python -c "import json,sys; d=[json.loads(l) for l in open('results/latency.jsonl')] if __import__('os').path.exists('results/latency.jsonl') else []; sys.exit(0 if len(d)>$i else 1)" && continue
  python bench_latency.py --setting $i --out results/latency.jsonl 2>&1 | grep -v -i warn || echo "FAILED latency $i" >> $FAIL
done

echo "== M2 predicted vs actual merge damage $(date)"
python pair_cost_v3.py --data $IMAGENET/train --images 16 --out results/pair_cost_v3.jsonl 2>&1 | grep -v -i warn

echo "== M3 weighted-sum control, full protocol $(date)"
sel() { python eval_imagenet.py --data $IMAGENET/train --per-class 10 --workers $W --out results/rpm_sel.jsonl \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED sel: $*" >> $FAIL; }
ev()  { python eval_imagenet.py --data $IMAGENET/val --workers $W --out results/rpm_val.jsonl --save-preds \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED val: $*" >> $FAIL; }
for M in vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m vit_base_patch16_clip_224.laion2b \
         vit_large_patch14_dinov2.lvd142m vit_small_patch14_dinov2.lvd142m vit_base_patch14_reg4_dinov2.lvd142m \
         vit_large_patch14_reg4_dinov2.lvd142m deit3_small_patch16_224.fb_in1k deit_base_patch16_224 \
         deit3_base_patch16_224.fb_in1k deit_small_patch16_224 deit_tiny_patch16_224; do
  m=$(margs $M)
  echo "== rpmsum $M $(date +%H:%M)"
  python pick_selected.py results/rpm_sel.jsonl --family rpmlnw --model $M | while read -r c; do
    for A in 0.1 0.25 0.5 0.75 0.9; do sel $m ${c/--method rpmlnw/--method rpmsum --sum-alpha $A}; done
  done
  python pick_selected.py results/rpm_sel.jsonl --family rpmsum --model $M | while read -r c; do ev $m $c; done
done
echo "== MENTOR1 DONE $(date)"
