#!/usr/bin/env bash
# Saturday batch. Waits for the dense test (unit tmdense). Resumable.
#   cd ~/S_VIT/tokenmerge_study && systemd-run --user --unit=tmsat -p WorkingDirectory=$PWD bash -c 'bash run_sat.sh >> results/sat.log 2>&1'
#  S1 compiled speed: AdaMerge variant, dense cost, AugReg models (all methods)                      ~3 h
#  S2 third accuracy-vs-speed curve: CLIP-B, 4 extra budgets, 5 methods, plus their speed             ~4 h
#  S3 selection robustness: re-select on a different 10-images-per-class split, 12 settings          ~5 h
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/data/imagenet}
export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor TRITON_CACHE_DIR=$HOME/.cache/triton TMPDIR=$HOME/tmp
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $TMPDIR results/cfgs
W=${WORKERS:-6}
FAIL=results/sat_failed.txt
while systemctl --user is-active --quiet tmdense; do echo "waiting for dense test  $(date +%H:%M)"; sleep 300; done
echo "== start $(date)"
margs() { case $1 in vit_*dinov2*|vit_*clip*) echo "--model $1 --probe probes/$1.pt";; *) echo "--model $1";; esac; }
sel() { python eval_imagenet.py --data $IMAGENET/train --per-class 10 --workers $W --out results/rpm_sel.jsonl \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED sel: $*" >> $FAIL; }
ev()  { python eval_imagenet.py --data $IMAGENET/val --workers $W --out results/rpm_val.jsonl --save-preds \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED val: $*" >> $FAIL; }
cfg() {   # name family args...
  local P=results/cfgs/$1.pt; shift; local F=$1; shift
  [ -f "$P" ] || python eval_imagenet.py --data $IMAGENET/val --save-config $P "$@" >/dev/null 2>&1 || { echo "FAILED cfg $P" >> $FAIL; return; }
  echo "$F|$P" >> $L
}
bench_loop() {
  local fails=0
  while true; do
    python bench_selected.py "$1" --out "$2" --max-groups 1 --compiled-only 2>&1 | grep -v -i warn
    rc=${PIPESTATUS[0]}; [ "$rc" = "3" ] && break
    if [ "$rc" != "0" ]; then fails=$((fails + 1)); [ $fails -ge 3 ] && break; else fails=0; fi
  done
}
MODELS="vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m vit_base_patch16_clip_224.laion2b
        vit_large_patch14_dinov2.lvd142m vit_small_patch14_dinov2.lvd142m vit_base_patch14_reg4_dinov2.lvd142m
        vit_large_patch14_reg4_dinov2.lvd142m deit3_small_patch16_224.fb_in1k deit_base_patch16_224
        deit3_base_patch16_224.fb_in1k deit_small_patch16_224 deit_tiny_patch16_224"
AUG="vit_small_patch16_224.augreg_in21k_ft_in1k vit_base_patch16_224.augreg_in21k_ft_in1k"

echo "== S1 speed of the remaining configurations $(date)"
L=results/cfgs/list_sat1.txt; : > $L
for M in $MODELS $AUG; do
  for F in adamerge2 rpmlnw tome mctf sadtm pitome adamerge; do
    case " $AUG " in *" $M "*) ;; *) case $F in adamerge2|rpmlnw|tome|mctf) ;; *) continue;; esac;; esac
    python pick_selected.py results/rpm_sel.jsonl --split sel10s1 --family $F --model $M | while read -r c; do
      B=$(echo "$c" | awk '{print $2}')
      cfg ${M}_${F}_${B} $F $(margs $M) $c
      [ "$F" = "rpmlnw" ] && cfg ${M}_rpmd_${B} rpmd $(margs $M) ${c/--method rpmlnw/--method rpmd}
    done
  done
done
sort -u $L -o $L
bench_loop $L results/speed_sel_sat.jsonl

echo "== S2 CLIP-B accuracy-vs-speed curve $(date)"
M=vit_base_patch16_clip_224.laion2b; m=$(margs $M)
HL="--method hybrid --weight-mode size --prop-attn on --blend soft --gamma 4 --attn-op rpmlnw"
grid10() { for T in 0 1 2 3; do sel "$@" --tau $T; done; for T in 1 2 3; do sel "$@" --tau $T --schedule geo; done; for R in 1 2 4; do sel "$@" --schedule rel --rel-kappa $R; done; }
for B in 0.4 0.52 0.68 0.85; do
  for METH in "--method tome" "--method mctf" "--method adamerge2" "--method rpmlnw" "$HL"; do grid10 $m --target-ratio $B $METH; done
done
L=results/cfgs/list_sat2.txt; : > $L
for F in tome mctf adamerge2 rpmlnw hybrid_lite; do
  python pick_selected.py results/rpm_sel.jsonl --family $F --model $M | while read -r c; do
    ev $m $c
    B=$(echo "$c" | awk '{print $2}'); case $B in 0.4|0.52|0.68|0.85) cfg ${M}_${F}_${B} $F $m $c;; esac
  done
done
sort -u $L -o $L
bench_loop $L results/speed_sel_sat.jsonl

echo "== S3 selection robustness (split seed 2) $(date)"
for MB in "vit_small_patch14_reg4_dinov2.lvd142m|--target-ratio 0.6" "vit_base_patch14_dinov2.lvd142m|--target-ratio 0.6" \
          "vit_base_patch16_clip_224.laion2b|--target-ratio 0.6" "vit_large_patch14_dinov2.lvd142m|--target-ratio 0.6" \
          "vit_small_patch14_dinov2.lvd142m|--target-ratio 0.45" "vit_base_patch14_reg4_dinov2.lvd142m|--target-ratio 0.45" \
          "vit_large_patch14_reg4_dinov2.lvd142m|--target-ratio 0.45" "deit3_small_patch16_224.fb_in1k|--target-gflops 2.6" \
          "deit_base_patch16_224|--target-gflops 10.1" "deit3_base_patch16_224.fb_in1k|--target-gflops 9.0" \
          "deit_small_patch16_224|--target-gflops 2.6" "deit_tiny_patch16_224|--target-gflops 0.8"; do
  M=${MB%%|*}; m=$(margs $M); b=${MB#*|}
  echo "== robust $M $(date +%H:%M)"
  for METH in "--method tome" "--method mctf" "--method rpmlnw" "$HL"; do
    for T in 0 1 2 3; do sel $m $b --split-seed 2 --tau $T $METH; done
    for T in 1 2 3; do sel $m $b --split-seed 2 --tau $T --schedule geo $METH; done
    for R in 1 2 4; do sel $m $b --split-seed 2 --schedule rel --rel-kappa $R $METH; done
  done
  for F in tome mctf rpmlnw hybrid_lite; do
    python pick_selected.py results/rpm_sel.jsonl --split sel10s2 --family $F --model $M | while read -r c; do
      case "$c" in *"$b "*|*"$b") ev $m $c;; esac
    done
  done
done
echo "== SAT DONE $(date)"
