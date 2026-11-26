#!/usr/bin/env bash
# Mentor feedback, batch 2. Waits for batch 1 (unit tmm1). Resumable.
#   cd ~/S_VIT/tokenmerge_study && systemd-run --user --unit=tmm2 -p WorkingDirectory=$PWD bash -c 'bash run_mentor2.sh >> results/mentor2.log 2>&1'
#  B1 AdaMerge (2026) as a 5th baseline, full protocol, 38 settings                                 ~8 h
#  B2 SAD-TM exactly as published: (a) its own DeiT-T/S budgets vs its paper's tables;
#     (b) under our protocol on the 17 supervised settings                                           ~4.5 h
#  B3 CLIP diagnostic: RPM-lite with attention level from all queries instead of [CLS]              ~15 min
#  B4 accuracy-vs-speed curves: 4 extra budgets on DeiT-B and DINOv2-B for 5 methods                ~4 h
#  B5 compiled speed of every new selected configuration (AdaMerge, curve points)                    ~5 h
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/data/imagenet}
export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor TRITON_CACHE_DIR=$HOME/.cache/triton TMPDIR=$HOME/tmp
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $TMPDIR results/cfgs priors
W=${WORKERS:-6}
FAIL=results/mentor2_failed.txt
while systemctl --user is-active --quiet tmm1; do echo "waiting for batch 1  $(date +%H:%M)"; sleep 300; done
echo "== start $(date)"
margs() { case $1 in vit_*) echo "--model $1 --probe probes/$1.pt";; *) echo "--model $1";; esac; }
sel() { python eval_imagenet.py --data $IMAGENET/train --per-class 10 --workers $W --out results/rpm_sel.jsonl \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED sel: $*" >> $FAIL; }
ev()  { python eval_imagenet.py --data $IMAGENET/val --workers $W --out results/rpm_val.jsonl --save-preds \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED val: $*" >> $FAIL; }
grid10() {   # args...: the standard 10 candidates (deferral, geometric, probe allocation)
  for T in 0 1 2 3; do sel "$@" --tau $T; done
  for T in 1 2 3; do sel "$@" --tau $T --schedule geo; done
  for R in 1 2 4; do sel "$@" --schedule rel --rel-kappa $R; done
}
bench_loop() {
  local fails=0
  while true; do
    python bench_selected.py "$1" --out "$2" --max-groups 1 --compiled-only 2>&1 | grep -v -i warn
    rc=${PIPESTATUS[0]}; [ "$rc" = "3" ] && break
    if [ "$rc" != "0" ]; then fails=$((fails + 1)); [ $fails -ge 3 ] && break; else fails=0; fi
  done
}
SETTINGS() {
  for M in vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m vit_base_patch16_clip_224.laion2b \
           vit_large_patch14_dinov2.lvd142m vit_small_patch14_dinov2.lvd142m vit_base_patch14_reg4_dinov2.lvd142m \
           vit_large_patch14_reg4_dinov2.lvd142m; do echo "$M|--target-ratio|0.45 0.6 0.75"; done
  echo "deit3_small_patch16_224.fb_in1k|--target-gflops|2.2 2.6 3.0"
  echo "deit_base_patch16_224|--target-gflops|9.0 10.1 11.5"
  echo "deit3_base_patch16_224.fb_in1k|--target-gflops|9.0 10.1 11.5"
  echo "deit_small_patch16_224|--target-gflops|2.2 2.4 2.6 3.0 3.6"
  echo "deit_tiny_patch16_224|--target-gflops|0.7 0.8 0.9"
}

echo "== B1 AdaMerge full protocol $(date)"
SETTINGS | while IFS='|' read -r M flag budgets; do
  echo "== adamerge $M $(date +%H:%M)"
  for B in $budgets; do grid10 $(margs $M) $flag $B --method adamerge; done
  python pick_selected.py results/rpm_sel.jsonl --family adamerge --model $M | while read -r c; do ev $(margs $M) $c; done
done

echo "== B2 SAD-TM as published $(date)"
for M in deit_tiny_patch16_224 deit_small_patch16_224 deit_base_patch16_224 deit3_small_patch16_224.fb_in1k deit3_base_patch16_224.fb_in1k; do
  [ -f priors/$M.pt ] || python compute_prior.py --model $M --data $IMAGENET/train --num-images 10000 --out priors/$M.pt \
    >/dev/null 2>&1 || echo "FAILED prior $M" >> $FAIL
done
PUB="--method sadtm --prior saliency --alpha 0.5"
for MB in "deit_small_patch16_224|3.1 3.0 2.9 2.8 2.6 2.5" "deit_tiny_patch16_224|1.0 0.9 0.8 0.7"; do
  M=${MB%%|*}
  for B in ${MB#*|}; do for T in 0 1 2 3; do
    python eval_imagenet.py --data $IMAGENET/val --workers $W --out results/sadtm_repro.jsonl --skip-existing \
      --model $M --prior-file priors/$M.pt $PUB --target-gflops $B --tau $T >/dev/null 2>&1 || echo "FAILED repro $M $B $T" >> $FAIL
  done; done
done
SETTINGS | grep -v "^vit_" | while IFS='|' read -r M flag budgets; do
  echo "== sadtm_pub $M $(date +%H:%M)"
  for B in $budgets; do grid10 --model $M --prior-file priors/$M.pt $flag $B $PUB; done
  python pick_selected.py results/rpm_sel.jsonl --family sadtm_pub --model $M | while read -r c; do ev --model $M --prior-file priors/$M.pt $c; done
done

echo "== B3 CLIP diagnostic $(date)"
for MB in "vit_base_patch16_clip_224.laion2b|0.45 0.6 0.75" "vit_small_patch14_reg4_dinov2.lvd142m|0.45" "vit_base_patch14_dinov2.lvd142m|0.45" "deit_base_patch16_224|9.0"; do
  M=${MB%%|*}
  python pick_selected.py results/rpm_sel.jsonl --family rpmlnw --model $M | while read -r c; do
    B=$(echo "$c" | awk '{print $2}'); case " ${MB#*|} " in *" $B "*) ev $(margs $M) ${c/--method rpmlnw/--method rpmlnwq};; esac
  done
done

echo "== B4 accuracy-vs-speed curves $(date)"
HL="--method hybrid --weight-mode size --prop-attn on --blend soft --gamma 4 --attn-op rpmlnw"
for MB in "deit_base_patch16_224|--target-gflops|8.0 8.5 9.5 12.5" "vit_base_patch14_dinov2.lvd142m|--target-ratio|0.4 0.52 0.68 0.85"; do
  IFS='|' read -r M flag budgets <<< "$MB"
  echo "== curves $M $(date +%H:%M)"
  for B in $budgets; do
    for METH in "--method tome" "--method mctf" "--method adamerge" "--method rpmlnw" "$HL"; do grid10 $(margs $M) $flag $B $METH; done
  done
  for F in tome mctf adamerge rpmlnw hybrid_lite; do
    python pick_selected.py results/rpm_sel.jsonl --family $F --model $M | while read -r c; do ev $(margs $M) $c; done
  done
done

echo "== B5 compiled speed of the new configurations $(date)"
L=results/cfgs/list_m2.txt; : > $L
SETTINGS | while IFS='|' read -r M flag budgets; do
  for F in adamerge tome mctf rpmlnw hybrid_lite; do
    python pick_selected.py results/rpm_sel.jsonl --split sel10s1 --family $F --model $M | while read -r c; do
      B=$(echo "$c" | awk '{print $2}')
      case "$M $B" in deit_base_patch16_224\ 8.0|deit_base_patch16_224\ 8.5|deit_base_patch16_224\ 9.5|deit_base_patch16_224\ 12.5|vit_base_patch14_dinov2.lvd142m\ 0.4|vit_base_patch14_dinov2.lvd142m\ 0.52|vit_base_patch14_dinov2.lvd142m\ 0.68|vit_base_patch14_dinov2.lvd142m\ 0.85) ;;
        *) [ "$F" != "adamerge" ] && continue;; esac
      P=results/cfgs/${M}_${F}_${B}.pt
      [ -f "$P" ] || python eval_imagenet.py --data $IMAGENET/val --save-config $P $(margs $M) $c >/dev/null 2>&1 || { echo "FAILED cfg $M $F $B" >> $FAIL; continue; }
      echo "$F|$P" >> $L
      if [ "$F" = "adamerge" ]; then for G in tome mctf; do [ -f results/cfgs/${M}_${G}_${B}.pt ] && echo "$G|results/cfgs/${M}_${G}_${B}.pt" >> $L; done; fi
    done
  done
done
sort -u $L -o $L
bench_loop $L results/speed_sel_m2.jsonl
echo "== MENTOR2 DONE $(date)"
