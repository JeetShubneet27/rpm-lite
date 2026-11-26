#!/usr/bin/env bash
# Overnight: (1) re-time the 6 hybrid settings with the configuration the accuracy analysis picks;
# (2) full protocol for the probe hybrid with RPM-lite as its accurate branch (hybrid_lite);
# (3) compiled speed of every selected hybrid_lite configuration next to ToMe and MCTF.  Resumable.
#   cd ~/S_VIT/tokenmerge_study && systemd-run --user --unit=tmnight -p WorkingDirectory=$PWD bash -c 'bash run_night.sh >> results/night.log 2>&1'
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/data/imagenet}
export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor TRITON_CACHE_DIR=$HOME/.cache/triton TMPDIR=$HOME/tmp
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $TMPDIR results/cfgs
W=${WORKERS:-6}
FAIL=results/night_failed.txt
margs() { case $1 in vit_*) echo "--model $1 --probe probes/$1.pt";; *) echo "--model $1";; esac; }
bench_loop() {   # list out [extra args]
  local fails=0
  while true; do
    python bench_selected.py "$1" --out "$2" --max-groups 1 ${3:-} 2>&1 | grep -v -i warn
    rc=${PIPESTATUS[0]}
    [ "$rc" = "3" ] && break
    if [ "$rc" != "0" ]; then fails=$((fails + 1)); echo "  (exit $rc, failure $fails)"; [ $fails -ge 3 ] && break; else fails=0; fi
  done
}
echo "== start $(date)"

echo "== N1 re-time 6 hybrid settings $(date)"
L1=results/cfgs/list_fix.txt; : > $L1
while IFS='|' read -r M FLAG B ARGS; do
  P=results/cfgs/fix_${M}_hybrid_${B}.pt
  [ -f "$P" ] || python eval_imagenet.py --data $IMAGENET/val --save-config $P $(margs $M) $FLAG $ARGS >/dev/null 2>&1 || echo "FAILED fixcfg $M $B" >> $FAIL
  echo "hybrid|$P" >> $L1
  for F in tome mctf rpmlnw; do [ -f results/cfgs/${M}_${F}_${B}.pt ] && echo "$F|results/cfgs/${M}_${F}_${B}.pt" >> $L1; done
done <<'FIX'
deit3_small_patch16_224.fb_in1k|--target-gflops 2.2|2.2|--tau 2 --method hybrid --weight-mode size --prop-attn on --blend soft --gamma 4 --schedule geo
deit3_small_patch16_224.fb_in1k|--target-gflops 2.6|2.6|--tau 0 --method hybrid --weight-mode size --prop-attn on --blend soft --gamma 4 --schedule rel --rel-kappa 2
vit_base_patch14_reg4_dinov2.lvd142m|--target-ratio 0.45|0.45|--tau 0 --method hybrid --weight-mode size --prop-attn on --blend soft --gamma 4 --schedule rel --rel-kappa 2
vit_base_patch16_clip_224.laion2b|--target-ratio 0.6|0.6|--tau 3 --method hybrid --weight-mode size --prop-attn on --blend soft --gamma 4 --schedule geo
vit_small_patch14_dinov2.lvd142m|--target-ratio 0.75|0.75|--tau 3 --method hybrid --weight-mode size --prop-attn on --blend soft --gamma 4
vit_small_patch14_reg4_dinov2.lvd142m|--target-ratio 0.75|0.75|--tau 2 --method hybrid --weight-mode size --prop-attn on --blend soft --gamma 4
FIX
bench_loop $L1 results/speed_sel_fix.jsonl

echo "== N2 hybrid_lite full protocol $(date)"
HL="--method hybrid --weight-mode size --prop-attn on --blend soft --gamma 4 --attn-op rpmlnw"
sel() { python eval_imagenet.py --data $IMAGENET/train --per-class 10 --workers $W --out results/rpm_sel.jsonl \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED sel: $*" >> $FAIL; }
ev()  { python eval_imagenet.py --data $IMAGENET/val --workers $W --out results/rpm_val.jsonl --save-preds \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED val: $*" >> $FAIL; }
{
  for M in vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m vit_base_patch16_clip_224.laion2b \
           vit_large_patch14_dinov2.lvd142m vit_small_patch14_dinov2.lvd142m vit_base_patch14_reg4_dinov2.lvd142m \
           vit_large_patch14_reg4_dinov2.lvd142m; do echo "$M|--target-ratio|0.45 0.6 0.75"; done
  echo "deit3_small_patch16_224.fb_in1k|--target-gflops|2.2 2.6 3.0"
  echo "deit_base_patch16_224|--target-gflops|9.0 10.1 11.5"
  echo "deit3_base_patch16_224.fb_in1k|--target-gflops|9.0 10.1 11.5"
  echo "deit_small_patch16_224|--target-gflops|2.2 2.4 2.6 3.0 3.6"
  echo "deit_tiny_patch16_224|--target-gflops|0.7 0.8 0.9"
} | while IFS='|' read -r M flag budgets; do
  m=$(margs $M)
  echo "== hybrid_lite $M $(date +%H:%M)"
  for B in $budgets; do
    for T in 0 1 2 3; do sel $m $flag $B --tau $T $HL; done
    for T in 1 2 3; do sel $m $flag $B --tau $T $HL --schedule geo; done
    for R in 1 2 4; do sel $m $flag $B $HL --schedule rel --rel-kappa $R; done
  done
  python pick_selected.py results/rpm_sel.jsonl --family hybrid_lite --model $M | while read -r c; do ev $m $c; done
done

echo "== N3 hybrid_lite compiled speed $(date)"
L3=results/cfgs/list_hl.txt; : > $L3
for M in vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m vit_base_patch16_clip_224.laion2b \
         vit_large_patch14_dinov2.lvd142m vit_small_patch14_dinov2.lvd142m vit_base_patch14_reg4_dinov2.lvd142m \
         vit_large_patch14_reg4_dinov2.lvd142m deit3_small_patch16_224.fb_in1k deit_base_patch16_224 \
         deit3_base_patch16_224.fb_in1k deit_small_patch16_224 deit_tiny_patch16_224; do
  python pick_selected.py results/rpm_sel.jsonl --split sel10s1 --family hybrid_lite --model $M | while read -r c; do
    B=$(echo "$c" | awk '{print $2}')
    P=results/cfgs/${M}_hybrid_lite_${B}.pt
    [ -f "$P" ] || python eval_imagenet.py --data $IMAGENET/val --save-config $P $(margs $M) $c >/dev/null 2>&1 \
      || { echo "FAILED hlcfg $M $c" >> $FAIL; continue; }
    echo "hybrid_lite|$P" >> $L3
    for F in tome mctf; do [ -f results/cfgs/${M}_${F}_${B}.pt ] && echo "$F|results/cfgs/${M}_${F}_${B}.pt" >> $L3; done
  done
done
bench_loop $L3 results/speed_sel_hl.jsonl --compiled-only
echo "== NIGHT DONE $(date)"
