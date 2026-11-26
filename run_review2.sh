#!/usr/bin/env bash
# Second review: experiments for M2/M3/M5/M6. Starts after run_review.sh finishes. Resumable.
#   cd ~/S_VIT/tokenmerge_study && (nohup bash run_review2.sh > results/review2.log 2>&1 &)
#  Q1 pair validation on ToMe / RPM / random partners with CIs (M5)                ~2 h
#  Q2 PiToMe without proportional attention (M6)                                   ~1 h
#  Q3 fast RPM with sqrt(size) query weights, full protocol, 12 models (M2)        ~12 h
#  Q4 measured speed of every selected RPM-f / RPM-f2 / MCTF configuration (M3)    ~1 h
set -uo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/VIT_5/data/imagenet}
W=${WORKERS:-6}
FAIL=results/review2_failed.txt
while pgrep -f "^bash run_review.sh" >/dev/null 2>&1; do echo "waiting for run_review.sh  $(date +%H:%M)"; sleep 300; done
echo "== start $(date)"
sel() { python eval_imagenet.py --data $IMAGENET/train --per-class 10 --workers $W --out results/rpm_sel.jsonl \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED sel: $*" >> $FAIL; }
ev()  { python eval_imagenet.py --data $IMAGENET/val --workers $W --out results/rpm_val.jsonl --save-preds \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED val: $*" >> $FAIL; }
report() { python select_report.py results/{calib,sched,probe,fix_val,fix_sel,hires_val,hires_sel,rel_val,rel_sel,review_val,review_sel,rpm_val,rpm_sel}.jsonl \
             > results/rpm_report.txt 2>&1; }

echo "== Q1 pair validation $(date)"
python pair_cost_v2.py --data $IMAGENET/train --images 16 --out results/pair_cost_v2.jsonl > results/pair_cost_v2.log 2>&1
grep -E "^\s+\[" results/pair_cost_v2.log | tail -n 18

echo "== Q2 PiToMe without proportional attention $(date)"
for MB in "deit_base_patch16_224|--target-gflops 9.0" "deit_small_patch16_224|--target-gflops 2.6" \
          "vit_small_patch14_dinov2.lvd142m --probe probes/vit_small_patch14_dinov2.lvd142m.pt|--target-ratio 0.45" \
          "vit_base_patch14_dinov2.lvd142m --probe probes/vit_base_patch14_dinov2.lvd142m.pt|--target-ratio 0.6"; do
  m="--model ${MB%%|*}"; b=${MB#*|}; name=${MB%%|*}; name=${name%% *}
  for T in 0 1 2 3; do sel $m $b --tau $T --method pitome --prop-attn off; done
  for T in 1 2; do sel $m $b --tau $T --method pitome --prop-attn off --schedule geo; done
  python pick_selected.py results/rpm_sel.jsonl --family pitome_np --model $name | while read -r c; do ev $m $c; done
done
python - <<'PY'
import json
rows=[json.loads(l) for l in open("results/rpm_val.jsonl") if l.strip()]
for r in rows:
    if r["method"]=="pitome" and r.get("prior")=="official-noprop":
        print("pitome no-prop", r["model"], r.get("target_ratio") or r["target_gflops"], "tau", r["tau"], r["schedule"], r["acc1"])
PY

echo "== Q3 RPM-f2 protocol $(date)"
{
  for M in vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m vit_base_patch16_clip_224.laion2b \
           vit_large_patch14_dinov2.lvd142m vit_small_patch14_dinov2.lvd142m vit_base_patch14_reg4_dinov2.lvd142m \
           vit_large_patch14_reg4_dinov2.lvd142m; do echo "--model $M --probe probes/$M.pt|--target-ratio|0.45 0.6 0.75"; done
  echo "--model deit3_small_patch16_224.fb_in1k|--target-gflops|2.2 2.6 3.0"
  echo "--model deit_base_patch16_224|--target-gflops|9.0 10.1 11.5"
  echo "--model deit3_base_patch16_224.fb_in1k|--target-gflops|9.0 10.1 11.5"
  echo "--model deit_small_patch16_224|--target-gflops|2.2 2.4 2.6 3.0 3.6"
  echo "--model deit_tiny_patch16_224|--target-gflops|0.7 0.8 0.9"
} | while IFS='|' read -r m flag budgets; do
  name=$(echo $m | awk '{print $2}')
  echo "== rpmf2 $name $(date +%H:%M)"
  for B in $budgets; do
    for T in 0 1 2 3; do sel $m $flag $B --tau $T --method rpmf2; done
    for T in 1 2 3; do sel $m $flag $B --tau $T --method rpmf2 --schedule geo; done
    for K in 1 2 4; do sel $m $flag $B --method rpmf2 --schedule rel --rel-kappa $K; done
  done
  python pick_selected.py results/rpm_sel.jsonl --family rpmf2 --model $name | while read -r c; do ev $m $c; done
  report; grep -E "rpmf2 +(found|super)" results/rpm_report.txt
done

echo "== Q5 RPM without the probe allocation (tau x const/geo only) $(date)"
for M in vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m vit_base_patch16_clip_224.laion2b \
         vit_large_patch14_dinov2.lvd142m vit_small_patch14_dinov2.lvd142m vit_base_patch14_reg4_dinov2.lvd142m \
         vit_large_patch14_reg4_dinov2.lvd142m deit3_small_patch16_224.fb_in1k deit_base_patch16_224 \
         deit3_base_patch16_224.fb_in1k deit_small_patch16_224 deit_tiny_patch16_224; do
  case $M in vit_*) m="--model $M --probe probes/$M.pt";; *) m="--model $M";; esac
  for F in rpmf rpmf2 rpm_all; do
    python pick_selected.py results/rpm_sel.jsonl --family $F --model $M --no-rel | while read -r c; do ev $m $c; done
  done
done

echo "== Q4 speed of selected configurations $(date)"
SP=results/speed_3090.jsonl
{ for M in vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m vit_base_patch16_clip_224.laion2b \
           vit_large_patch14_dinov2.lvd142m vit_small_patch14_dinov2.lvd142m vit_base_patch14_reg4_dinov2.lvd142m \
           vit_large_patch14_reg4_dinov2.lvd142m; do echo "--model $M --probe probes/$M.pt"; done
  for M in deit_tiny_patch16_224 deit_small_patch16_224 deit_base_patch16_224 deit3_small_patch16_224.fb_in1k deit3_base_patch16_224.fb_in1k; do echo "--model $M"; done
} | while read -r m; do
  name=$(echo $m | awk '{print $2}')
  python eval_imagenet.py --data $IMAGENET/val --speed-only --out $SP --skip-existing $m --method none >/dev/null 2>&1
  for F in tome mctf hybrid rpmf rpmf2; do
    python pick_selected.py results/*sel*.jsonl --split sel10s1 --family $F --model $name | while read -r c; do
      python eval_imagenet.py --data $IMAGENET/val --speed-only --out $SP --skip-existing $m $c >/dev/null 2>&1 || echo "FAILED speed $m $c" >> $FAIL
    done
  done
done
report; grep -E "wins" results/rpm_report.txt
echo "== REVIEW2 DONE $(date)"
