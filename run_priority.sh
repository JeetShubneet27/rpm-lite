#!/usr/bin/env bash
# Priority queue for Monday (starts when run_rpm.sh finishes; shares the GPU with the review queue):
#  P0 fill the RPM runs lost in the memory crash            (~1 h)
#  P1 RPM-2 (second-order term) on all 12 models            (~8 h)  foundation models first
#  P2 MCTF (CVPR 2024) as a 4th baseline, same protocol     (~4 h)
#  P3 wall-clock speed, review queue paused while measuring (~30 min)
#   cd ~/S_VIT/tokenmerge_study && (nohup bash run_priority.sh > results/priority.log 2>&1 &)
#   progress: grep "==" results/priority.log; cut -c1-190 results/rpm_report.txt | tail -n 60
set -uo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/VIT_5/data/imagenet}
export WORKERS=${WORKERS:-6}
W=$WORKERS
FAIL=results/priority_failed.txt
while pgrep -f "bash run_rpm.sh" >/dev/null 2>&1; do echo "waiting for run_rpm.sh  $(date +%H:%M)"; sleep 300; done
echo "== start $(date)"
sel() { python eval_imagenet.py --data $IMAGENET/train --per-class 10 --workers $W --out results/rpm_sel.jsonl \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED sel: $*" >> $FAIL; }
ev()  { python eval_imagenet.py --data $IMAGENET/val --workers $W --out results/rpm_val.jsonl --save-preds \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED val: $*" >> $FAIL; }
report() { python select_report.py results/{calib,sched,probe,fix_val,fix_sel,hires_val,hires_sel,rel_val,rel_sel,review_val,review_sel,rpm_val,rpm_sel}.jsonl \
             > results/rpm_report.txt 2>&1; echo "   report $(date +%H:%M)"; }
model_budgets() {
  for M in vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m vit_large_patch14_dinov2.lvd142m \
           vit_base_patch16_clip_224.laion2b vit_small_patch14_dinov2.lvd142m vit_base_patch14_reg4_dinov2.lvd142m \
           vit_large_patch14_reg4_dinov2.lvd142m; do
    echo "--model $M --probe probes/$M.pt|--target-ratio|0.45 0.6 0.75"; done
  echo "--model deit_base_patch16_224|--target-gflops|9.0 10.1 11.5"
  echo "--model deit3_small_patch16_224.fb_in1k|--target-gflops|2.2 2.6 3.0"
  echo "--model deit3_base_patch16_224.fb_in1k|--target-gflops|9.0 10.1 11.5"
  echo "--model deit_small_patch16_224|--target-gflops|2.2 2.4 2.6 3.0 3.6"
  echo "--model deit_tiny_patch16_224|--target-gflops|0.7 0.8 0.9"
}
run_family() {   # run_family <family> <with probe allocation candidates: 1/0>
  local F=$1 REL=$2
  model_budgets | while IFS='|' read -r m flag budgets; do
    name=$(echo $m | awk '{print $2}')
    echo "== $F $name $(date +%H:%M)"
    for B in $budgets; do
      for T in 0 1 2 3; do sel $m $flag $B --tau $T --method $F; done
      for T in 1 2 3; do sel $m $flag $B --tau $T --method $F --schedule geo; done
      [ "$REL" = 1 ] && for K in 1 2 4; do sel $m $flag $B --method $F --schedule rel --rel-kappa $K; done
    done
    python pick_selected.py results/rpm_sel.jsonl --family $F --model $name | while read -r c; do ev $m $c; done
    report; grep -E "$F +(found|super)" results/rpm_report.txt
  done
}

echo "== P0 fill RPM gaps $(date)"
bash run_rpm.sh > results/rpm_fill.log 2>&1; report
echo "== P1 RPM-2 $(date)"
run_family rpm2 1
echo "== P2 MCTF $(date)"
run_family mctf 0                 # baselines get the tau x {const, geo} grid, as ToMe / PiToMe

echo "== P3 speed $(date)"
pkill -STOP -f "eval_imagenet.py" ; pkill -STOP -f "run_review.sh|run_mc.sh|run_fb.sh"; sleep 20
SP=results/speed_priority.jsonl
for MB in "deit_base_patch16_224|--target-gflops 9.0" "deit_small_patch16_224|--target-gflops 2.6" \
          "vit_base_patch14_dinov2.lvd142m --probe probes/vit_base_patch14_dinov2.lvd142m.pt|--target-ratio 0.6" \
          "vit_large_patch14_reg4_dinov2.lvd142m --probe probes/vit_large_patch14_reg4_dinov2.lvd142m.pt|--target-ratio 0.6" \
          "vit_base_patch16_clip_224.laion2b --probe probes/vit_base_patch16_clip_224.laion2b.pt|--target-ratio 0.6"; do
  m="--model ${MB%%|*}"; b=${MB#*|}; name=${MB%%|*}; name=${name%% *}
  python eval_imagenet.py --data $IMAGENET/val --speed-only --out $SP --skip-existing $m --method none >/dev/null 2>&1
  for F in tome pitome mctf hybrid rpm_all rpm2; do
    python pick_selected.py results/*sel*.jsonl --split sel10s1 --family $F --model $name | grep -- "$b " | head -n 1 | while read -r c; do
      python eval_imagenet.py --data $IMAGENET/val --speed-only --out $SP --skip-existing $m $c >/dev/null 2>&1 \
        || echo "FAILED speed: $m $c" >> $FAIL; done
  done
done
pkill -CONT -f "run_review.sh|run_mc.sh|run_fb.sh"; pkill -CONT -f "eval_imagenet.py"
python - <<'PY' > results/speed_priority.txt
import json
rows = [json.loads(l) for l in open("results/speed_priority.jsonl") if l.strip()]
base = {r["model"]: r for r in rows if r["method"] == "none"}
for r in rows:
    b = base.get(r["model"])
    if not b: continue
    print(f"{r['model'][:34]:34s} {r['method']:8s} t{r['tau']} {r['schedule']:6s} {r['gflops']:6.2f}G  "
          f"fp32 {r['img_per_s_fp32']:7.0f} img/s ({r['img_per_s_fp32'] / b['img_per_s_fp32']:.2f}x)  "
          f"bf16 {r['img_per_s_bf16']:7.0f} img/s ({r['img_per_s_bf16'] / b['img_per_s_bf16']:.2f}x)")
PY
cat results/speed_priority.txt
report; tail -n 12 results/rpm_report.txt
echo "== PRIORITY DONE $(date)"
