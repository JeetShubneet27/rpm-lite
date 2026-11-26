#!/usr/bin/env bash
# Revision batch 3 (mentor's roadmap). Waits for batch 2 (unit tmrev2). Resumable: re-running skips finished parts.
#   cd ~/S_VIT/tokenmerge_study && systemd-run --user --unit=tmrev3 -p WorkingDirectory=$PWD bash -c 'bash run_rev3.sh >> results/rev3.log 2>&1'
#  C  CLIP-B diagnosis: attention-level capping in RPM-lite; MCTF without informativeness / size        ~1.5 h
#  G  a ViT without [CLS] token (global average pooling): ViT-M/16 GAP, 3 budgets, all methods          ~2 h
#  E  training-free EViT (pruning + fusion) as an extra baseline on the 38 main settings                 ~5 h
#  F  output fidelity on the 38 settings: KL, JS, flip rate, ECE                                         ~4 h
#  H  higher resolutions: DINOv2-B at 392/518 px, DeiT-III-B at 384 px (accuracy); latency/memory at
#     224-518 px incl. exact RPM; bf16 timing                                                           ~4 h
#  S  selection stability: two more selection sets (seeds 3, 4) on 12 settings                          ~10 h
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/data/imagenet}
export TORCHINDUCTOR_CACHE_DIR=$HOME/tmp/inductor TRITON_CACHE_DIR=$HOME/tmp/triton TMPDIR=$HOME/tmp
clean_cache() { rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"; mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"; }
diskcheck() { local a=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9); if [ "${a:-0}" -lt 20 ]; then clean_cache; a=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9); [ "${a:-0}" -lt 20 ] && { echo "STOPPED: less than 20 GB free on disk $(date)"; exit 2; }; fi; }
mkdir -p results/cfgs
gpu_alone() {   # timing steps need an otherwise idle GPU: wait until no other process computes on it
  # (programs under 500 MiB, such as RustDesk or the desktop, are ignored)
  while [ -n "$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F, '$2+0 > 500')" ]; do
    echo "timing step waiting for the GPU to be free  $(date +%H:%M)"; sleep 300; done; }
W=${WORKERS:-6}
FAIL=results/rev3_failed.txt
while systemctl --user is-active --quiet tmrev2 || systemctl --user is-active --quiet tmrev; do echo "waiting for batches 1-2  $(date +%H:%M)"; sleep 300; done
clean_cache
echo "== start $(date)"
margs() { case $1 in vit_*dinov2*|vit_*clip*) echo "--model $1 --probe probes/$1.pt";; *) echo "--model $1";; esac; }
sel() { python eval_imagenet.py --data $IMAGENET/train --per-class 10 --workers $W --out results/rpm_sel.jsonl \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED sel: $*" >> $FAIL; }
ev()  { python eval_imagenet.py --data $IMAGENET/val --workers $W --out results/rpm_val.jsonl --save-preds \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED val: $*" >> $FAIL; }
grid7()  { for T in 0 1 2 3; do sel "$@" --tau $T; done; for T in 1 2 3; do sel "$@" --tau $T --schedule geo; done; }
grid10() { grid7 "$@"; for R in 1 2 4; do sel "$@" --schedule rel --rel-kappa $R; done; }
pickev() {  # family model budget-args margs...
  local F=$1 M=$2 b=$3; shift 3
  python pick_selected.py results/rpm_sel.jsonl --split sel10s1 --family $F --model $M | while read -r c; do
    case "$c" in *"$b "*|*"$b") ev "$@" $c;; esac
  done
}
HL="--method hybrid --weight-mode size --prop-attn on --blend soft --gamma 4 --attn-op rpmlnw"
CORE=("deit3_base_patch16_224.fb_in1k|--target-gflops 9.0" "deit3_base_patch16_224.fb_in1k|--target-gflops 10.1" "deit3_base_patch16_224.fb_in1k|--target-gflops 11.5" "deit3_small_patch16_224.fb_in1k|--target-gflops 2.2" "deit3_small_patch16_224.fb_in1k|--target-gflops 2.6" "deit3_small_patch16_224.fb_in1k|--target-gflops 3.0" "deit_base_patch16_224|--target-gflops 9.0" "deit_base_patch16_224|--target-gflops 10.1" "deit_base_patch16_224|--target-gflops 11.5" "deit_small_patch16_224|--target-gflops 2.2" "deit_small_patch16_224|--target-gflops 2.4" "deit_small_patch16_224|--target-gflops 2.6" "deit_small_patch16_224|--target-gflops 3.0" "deit_small_patch16_224|--target-gflops 3.6" "deit_tiny_patch16_224|--target-gflops 0.7" "deit_tiny_patch16_224|--target-gflops 0.8" "deit_tiny_patch16_224|--target-gflops 0.9" "vit_base_patch14_dinov2.lvd142m|--target-ratio 0.45" "vit_base_patch14_dinov2.lvd142m|--target-ratio 0.6" "vit_base_patch14_dinov2.lvd142m|--target-ratio 0.75" "vit_base_patch14_reg4_dinov2.lvd142m|--target-ratio 0.45" "vit_base_patch14_reg4_dinov2.lvd142m|--target-ratio 0.6" "vit_base_patch14_reg4_dinov2.lvd142m|--target-ratio 0.75" "vit_base_patch16_clip_224.laion2b|--target-ratio 0.45" "vit_base_patch16_clip_224.laion2b|--target-ratio 0.6" "vit_base_patch16_clip_224.laion2b|--target-ratio 0.75" "vit_large_patch14_dinov2.lvd142m|--target-ratio 0.45" "vit_large_patch14_dinov2.lvd142m|--target-ratio 0.6" "vit_large_patch14_dinov2.lvd142m|--target-ratio 0.75" "vit_large_patch14_reg4_dinov2.lvd142m|--target-ratio 0.45" "vit_large_patch14_reg4_dinov2.lvd142m|--target-ratio 0.6" "vit_large_patch14_reg4_dinov2.lvd142m|--target-ratio 0.75" "vit_small_patch14_dinov2.lvd142m|--target-ratio 0.45" "vit_small_patch14_dinov2.lvd142m|--target-ratio 0.6" "vit_small_patch14_dinov2.lvd142m|--target-ratio 0.75" "vit_small_patch14_reg4_dinov2.lvd142m|--target-ratio 0.45" "vit_small_patch14_reg4_dinov2.lvd142m|--target-ratio 0.6" "vit_small_patch14_reg4_dinov2.lvd142m|--target-ratio 0.75")

diskcheck; echo "== C CLIP diagnosis $(date)"
M=vit_base_patch16_clip_224.laion2b; m=$(margs $M)
for b in "--target-ratio 0.45" "--target-ratio 0.6"; do
  for Q in 0.98 0.95; do grid10 $m $b --method rpmlnw --u0-clip $Q; done
  grid10 $m $b --method mctf --mctf-drop info
  grid10 $m $b --method mctf --mctf-drop size
  for F in rpmlnw_u0c mctf_noinfo mctf_nosize; do pickev $F $M "$b" $m; done
done

diskcheck; echo "== G ViT without [CLS] token $(date)"
M=vit_medium_patch16_gap_256.sw_in12k_ft_in1k
ev --model $M --method none
for B in 0.75 0.6 0.45; do
  for METH in "--method tome" "--method sadtm --weight-mode size --prop-attn on --prior uniform" "--method pitome" \
              "--method mctf" "--method adamerge" "--method rpmlnw" "--method evit"; do
    grid7 --model $M --target-ratio $B $METH
  done
  for F in tome sadtm pitome mctf adamerge rpmlnw evit; do python pick_selected.py results/rpm_sel.jsonl --split sel10s1 --img-size 256 --family $F --model $M | while read -r c; do case "$c" in *"--target-ratio $B "*|*"--target-ratio $B") ev --model $M $c;; esac; done; done
done

diskcheck; echo "== E EViT on the 38 main settings $(date)"
for MB in "${CORE[@]}"; do
  M=${MB%%|*}; m=$(margs $M); b=${MB#*|}
  grid10 $m $b --method evit
  pickev evit $M "$b" $m
done

diskcheck; echo "== F output fidelity $(date)"
L=results/cfgs/list_fid.txt; touch $L
for MB in "${CORE[@]}"; do
  M=${MB%%|*}; m=$(margs $M); b=${MB#*|}
  for F in tome mctf evit rpmlnw hybrid_lite; do
    python pick_selected.py results/rpm_sel.jsonl --split sel10s1 --family $F --model $M | while read -r c; do
      case "$c" in *"$b "*|*"$b") ;; *) continue;; esac
      P=results/cfgs/fid_${M}_${F}_$(echo $b | tr -cd '0-9.').pt
      [ -f $P ] || python eval_imagenet.py --data $IMAGENET/val $m $c --save-config $P >/dev/null 2>&1 || { echo "FAILED cfg $P" >> $FAIL; continue; }
      grep -q "|$P$" $L || echo "$F|$P" >> $L
    done
  done
done
python fidelity.py --list $L --data $IMAGENET/val --out results/fidelity.jsonl --workers $W 2>&1 | grep -v -i warn
[ "${PIPESTATUS[0]}" = 0 ] || echo "FAILED fidelity" >> $FAIL

diskcheck; echo "== H higher resolutions $(date)"
M=vit_base_patch14_dinov2.lvd142m
for PX in 392 518; do
  PR=probes/${M}_${PX}.pt
  [ -f $PR ] || python train_probe.py --model $M --img-size $PX --data $IMAGENET/train --per-class 100 --workers $W --out $PR 2>&1 | grep -v -i warn | tail -2
  ev --model $M --probe $PR --img-size $PX --method none
  for B in 0.6 0.45; do for METH in tome mctf rpmlnw; do ev --model $M --probe $PR --img-size $PX --target-ratio $B --tau 1 --method $METH; done; done
done
M=deit3_base_patch16_384.fb_in1k
ev --model $M --method none
for B in 0.6 0.45; do for METH in tome mctf rpmlnw; do ev --model $M --target-ratio $B --tau 1 --method $METH; done; done
for MP in "vit_base_patch14_dinov2.lvd142m 224 256" "vit_base_patch14_dinov2.lvd142m 392 64" "vit_base_patch14_dinov2.lvd142m 518 32" \
          "vit_large_patch14_dinov2.lvd142m 224 128" "vit_large_patch14_dinov2.lvd142m 392 32" "vit_large_patch14_dinov2.lvd142m 518 16" \
          "deit3_base_patch16_224.fb_in1k 224 256" "deit3_base_patch16_384.fb_in1k 384 64"; do
  set -- $MP
  grep -q "\"model\": \"$1\", \"img_size\": $2, \"dtype\": \"fp32\"" results/latency_res.jsonl 2>/dev/null || \
    { diskcheck; gpu_alone; python bench_latency.py --model $1 --img-size $2 --batch $3 --ratio 0.6 --out results/latency_res.jsonl 2>&1 | grep -v -i warn | tail -8; clean_cache; }
done
for MP in "deit_base_patch16_224 224 256" "vit_base_patch14_dinov2.lvd142m 224 256" "vit_large_patch14_dinov2.lvd142m 224 128"; do
  set -- $MP
  grep -q "\"model\": \"$1\", \"img_size\": $2, \"dtype\": \"bf16\"" results/latency_res.jsonl 2>/dev/null || \
    { diskcheck; gpu_alone; python bench_latency.py --model $1 --img-size $2 --batch $3 --ratio 0.6 --dtype bf16 --out results/latency_res.jsonl 2>&1 | grep -v -i warn | tail -8; clean_cache; }
done

diskcheck; echo "== S selection stability (seeds 3, 4) $(date)"
for SEED in 3 4; do
  for MB in "vit_small_patch14_reg4_dinov2.lvd142m|--target-ratio 0.6" "vit_base_patch14_dinov2.lvd142m|--target-ratio 0.6" \
            "vit_base_patch16_clip_224.laion2b|--target-ratio 0.6" "vit_large_patch14_dinov2.lvd142m|--target-ratio 0.6" \
            "vit_small_patch14_dinov2.lvd142m|--target-ratio 0.45" "vit_base_patch14_reg4_dinov2.lvd142m|--target-ratio 0.45" \
            "vit_large_patch14_reg4_dinov2.lvd142m|--target-ratio 0.45" "deit3_small_patch16_224.fb_in1k|--target-gflops 2.6" \
            "deit_base_patch16_224|--target-gflops 10.1" "deit3_base_patch16_224.fb_in1k|--target-gflops 9.0" \
            "deit_small_patch16_224|--target-gflops 2.6" "deit_tiny_patch16_224|--target-gflops 0.8"; do
    M=${MB%%|*}; m=$(margs $M); b=${MB#*|}
    echo "== seed $SEED $M $(date +%H:%M)"
    for METH in "--method tome" "--method mctf" "--method rpmlnw" "$HL"; do grid10 $m $b --split-seed $SEED $METH; done
    for F in tome mctf rpmlnw hybrid_lite; do
      python pick_selected.py results/rpm_sel.jsonl --split sel10s$SEED --family $F --model $M | while read -r c; do
        case "$c" in *"$b "*|*"$b") ev $m $c;; esac
      done
    done
  done
done
echo "== REV3 DONE $(date)"
