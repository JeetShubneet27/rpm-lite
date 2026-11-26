#!/usr/bin/env bash
# Revision batch (mentor's essential revisions). Resumable: re-running skips finished parts.
#   cd ~/S_VIT/tokenmerge_study && systemd-run --user --unit=tmrev -p WorkingDirectory=$PWD bash -c 'bash run_rev.sh >> results/rev.log 2>&1'
#  R0 environment details                                                        (items 10, 13)    1 min
#  R1 real-network check of the theory + assumptions + larger damage study       (items 1, 2, 5, 6) ~3-4 h
#  R2 factor scales during actual merging                                         (item 4)          ~20 min
#  R3 full latency / throughput / memory with spread, all 14 models               (item 10)         ~2 h
#  R4 merge visualisations at middle blocks                                       (figures)         ~5 min
#  R5 stronger weighted-sum controls on 12 settings (layer-normalised sum,
#     geometric mean, normalised product), selected and evaluated like all methods (item 4)          ~7 h
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/data/imagenet}
export TORCHINDUCTOR_CACHE_DIR=$HOME/tmp/inductor TRITON_CACHE_DIR=$HOME/tmp/triton TMPDIR=$HOME/tmp
clean_cache() { rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"; mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"; }
diskcheck() { local a=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9); if [ "${a:-0}" -lt 20 ]; then clean_cache; a=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9); [ "${a:-0}" -lt 20 ] && { echo "STOPPED: less than 20 GB free on disk $(date)"; exit 2; }; fi; }
clean_cache
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $TMPDIR
W=${WORKERS:-6}
FAIL=results/rev_failed.txt
echo "== start $(date)"

echo "== R0 environment $(date)"
{ nvidia-smi; python -c "import torch, timm, platform; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), 'timm', timm.__version__, 'python', platform.python_version())"; lscpu | head -20; free -g; } > results/rev_env.txt 2>&1

echo "== R1 theory check + damage study $(date)"
python rev_diag.py --data $IMAGENET/train --images 64 --out results/rev_diag 2>&1 | grep -v -i warn; [ "${PIPESTATUS[0]}" = 0 ] || echo "FAILED R1" >> $FAIL

echo "== R2 factor scales $(date)"
python rev_factors.py --data $IMAGENET/val --out results/rev_factors.jsonl 2>&1 | grep -v -i warn; [ "${PIPESTATUS[0]}" = 0 ] || echo "FAILED R2" >> $FAIL

echo "== R3 latency $(date)"
for i in $(seq 0 13); do
  NAME=$(python -c "import bench_latency as b; print(b.SETTINGS[$i][0])")
  grep -q "\"model\": \"$NAME\"" results/latency_rev.jsonl 2>/dev/null && continue
  diskcheck
  python bench_latency.py --setting $i --out results/latency_rev.jsonl 2>&1 | grep -v -i warn
  [ "${PIPESTATUS[0]}" = 0 ] || echo "FAILED R3 $i" >> $FAIL
  clean_cache
done

echo "== R4 visualisations $(date)"
python vis_merge.py --data $IMAGENET/val --out results/vis_rev 2>&1 | grep -v -i warn; [ "${PIPESTATUS[0]}" = 0 ] || echo "FAILED R4" >> $FAIL

echo "== R5 weighted-sum controls $(date)"
margs() { case $1 in vit_*dinov2*|vit_*clip*) echo "--model $1 --probe probes/$1.pt";; *) echo "--model $1";; esac; }
sel() { python eval_imagenet.py --data $IMAGENET/train --per-class 10 --workers $W --out results/rpm_sel.jsonl \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED sel: $*" >> $FAIL; }
ev()  { python eval_imagenet.py --data $IMAGENET/val --workers $W --out results/rpm_val.jsonl --save-preds \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED val: $*" >> $FAIL; }
grid10() { for T in 0 1 2 3; do sel "$@" --tau $T; done; for T in 1 2 3; do sel "$@" --tau $T --schedule geo; done; for R in 1 2 4; do sel "$@" --schedule rel --rel-kappa $R; done; }
for MB in "vit_small_patch14_reg4_dinov2.lvd142m|--target-ratio 0.6" "vit_base_patch14_dinov2.lvd142m|--target-ratio 0.6" \
          "vit_base_patch16_clip_224.laion2b|--target-ratio 0.6" "vit_large_patch14_dinov2.lvd142m|--target-ratio 0.6" \
          "vit_small_patch14_dinov2.lvd142m|--target-ratio 0.45" "vit_base_patch14_reg4_dinov2.lvd142m|--target-ratio 0.45" \
          "vit_large_patch14_reg4_dinov2.lvd142m|--target-ratio 0.45" "deit3_small_patch16_224.fb_in1k|--target-gflops 2.6" \
          "deit_base_patch16_224|--target-gflops 10.1" "deit3_base_patch16_224.fb_in1k|--target-gflops 9.0" \
          "deit_small_patch16_224|--target-gflops 2.6" "deit_tiny_patch16_224|--target-gflops 0.8"; do
  M=${MB%%|*}; m=$(margs $M); b=${MB#*|}
  echo "== controls $M $b $(date +%H:%M)"; diskcheck
  for A in 0.25 0.5 0.75; do grid10 $m $b --method rpmsum --sum-mode layer --sum-alpha $A; done
  grid10 $m $b --method rpmsum --sum-mode geo
  grid10 $m $b --method rpmsum --sum-mode nprod
  for F in rpmsuml rpmgeo rpmnprod; do
    python pick_selected.py results/rpm_sel.jsonl --split sel10s1 --family $F --model $M | while read -r c; do
      case "$c" in *"$b "*|*"$b") ev $m $c;; esac
    done
  done
done
echo "== REV DONE $(date)"
