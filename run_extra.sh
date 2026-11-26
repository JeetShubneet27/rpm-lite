#!/usr/bin/env bash
# Extra experiments for the paper (resumable; waits for start_wsl.sh to finish first):
#   A. PiToMe (NeurIPS 2024) as a third baseline, same selection protocol (tau + schedule on 10 train imgs/class)
#   B. Robustness of the calibration: reliabilities from 32 instead of 128 LOO images, and from other images
#   C. Wall-clock speed of the reliability-aware allocation (run last, alone on the GPU)
#   cd ~/S_VIT/tokenmerge_study && (nohup bash run_extra.sh > results/extra.log 2>&1 &)
set -uo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/VIT_5/data/imagenet}
W=${WORKERS:-12}
FAIL=results/extra_failed.txt
while pgrep -f "start_wsl.sh" >/dev/null 2>&1; do echo "waiting for start_wsl.sh  $(date +%H:%M)"; sleep 300; done
echo "== start $(date)"

settings() {   # "<model/probe/res args>|<budget args>"
  for G in 3.6 3.0 2.6 2.4 2.2; do echo "--model deit_small_patch16_224|--target-gflops $G"; done
  for G in 0.9 0.8 0.7; do echo "--model deit_tiny_patch16_224|--target-gflops $G"; done
  for G in 3.0 2.6 2.2; do echo "--model deit3_small_patch16_224.fb_in1k|--target-gflops $G"; done
  for G in 11.5 10.1 9.0; do echo "--model deit_base_patch16_224|--target-gflops $G"; done
  for G in 11.5 10.1 9.0; do echo "--model deit3_base_patch16_224.fb_in1k|--target-gflops $G"; done
  for M in vit_small_patch14_dinov2.lvd142m vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m \
           vit_base_patch14_reg4_dinov2.lvd142m vit_base_patch16_clip_224.laion2b \
           vit_large_patch14_dinov2.lvd142m vit_large_patch14_reg4_dinov2.lvd142m; do
    for R in 0.75 0.6 0.45; do echo "--model $M --probe probes/$M.pt|--target-ratio $R"; done
  done
  for M in vit_base_patch14_dinov2.lvd142m vit_large_patch14_reg4_dinov2.lvd142m vit_large_patch14_dinov2.lvd142m; do
    [ -f probes/${M}_448.pt ] || continue
    for R in 0.5 0.35; do echo "--model $M --probe probes/${M}_448.pt --img-size 448|--target-ratio $R"; done
  done
}
ev()  { local out=$1; shift; python eval_imagenet.py --data $IMAGENET/val --workers $W --out $out --save-preds \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED val: $*" >> $FAIL; }
sel() { python eval_imagenet.py --data $IMAGENET/train --per-class 10 --workers $W --out results/pitome_sel.jsonl \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED sel: $*" >> $FAIL; }
models() { settings | cut -d'|' -f1 | sort -u; }
res_of() { [[ "$1" == *"--img-size 448"* ]] && echo 448 || echo 224; }
name_of() { echo "$1" | awk '{print $2}'; }

# ---- A. PiToMe: selection runs, then val of the selected configuration --------------------------
echo "== A. PiToMe $(date)"
settings | while IFS='|' read -r m b; do
  for T in 0 1 2 3; do sel $m $b --tau $T --method pitome; done
  for T in 1 2; do sel $m $b --tau $T --method pitome --schedule geo; done
done
models | while read -r m; do
  python pick_selected.py results/pitome_sel.jsonl --family pitome --model $(name_of "$m") --img-size $(res_of "$m") \
    | while read -r c; do ev results/pitome_val.jsonl $m $c; done
done
python select_report.py results/{calib,sched,probe,fix_val,fix_sel,hires_val,hires_sel,rel_val,rel_sel,pitome_val,pitome_sel}.jsonl \
  > results/select_with_pitome.txt 2>&1; echo "== A done $(date)"; tail -n 8 results/select_with_pitome.txt

# ---- B. calibration robustness ------------------------------------------------------------------
echo "== B. robustness $(date)"
python - <<'PY' || python measure_reliability.py --data $IMAGENET/train --loo-images 32 --workers $W --out results/reliability_loo32.jsonl
import os, sys; sys.exit(0 if os.path.exists("results/reliability_loo32.jsonl") else 1)
PY
python - <<'PY' || python measure_reliability.py --data $IMAGENET/train --seed 1 --workers $W --out results/reliability_seed1.jsonl
import os, sys; sys.exit(0 if os.path.exists("results/reliability_seed1.jsonl") else 1)
PY
SELS=$(ls results/{calib,sched,probe,fix_sel,rel_sel}.jsonl 2>/dev/null)
for V in loo32 seed1; do
  models | grep -v "img-size 448" | while read -r m; do
    python pick_selected.py $SELS --family hybrid --model $(name_of "$m") --img-size 224 \
      | while read -r c; do ev results/robust_$V.jsonl $m $c --reliability results/reliability_$V.jsonl --tag robust-$V; done
  done
done
python compare_robust.py > results/robust_report.txt 2>&1; echo "== B done $(date)"; tail -n 12 results/robust_report.txt

# ---- C. wall-clock speed of the allocation (fp32 and bf16, batch 256) ---------------------------
echo "== C. speed $(date)"
TP=results/throughput_3090.jsonl
tp() { for AMP in "" "--amp"; do python eval_imagenet.py --data $IMAGENET/val --limit 256 --workers 4 --throughput \
         --out $TP $AMP "$@" >/dev/null 2>&1 || echo "FAILED tp: $*" >> $FAIL; done; }
CFG="--weight-mode size --prop-attn on --blend soft --gamma 4"
for MB in "deit_base_patch16_224|--target-gflops 11.5" "vit_base_patch14_dinov2.lvd142m --probe probes/vit_base_patch14_dinov2.lvd142m.pt|--target-ratio 0.6" \
          "vit_large_patch14_reg4_dinov2.lvd142m --probe probes/vit_large_patch14_reg4_dinov2.lvd142m.pt|--target-ratio 0.6"; do
  m="--model ${MB%%|*}"; b=${MB#*|}
  tp $m --method none
  tp $m $b --tau 1 --method tome
  tp $m $b --tau 1 --method pitome
  tp $m $b --tau 1 --method hybrid $CFG --schedule geo
  for K in 1 2 4; do tp $m $b --method hybrid $CFG --schedule rel --rel-kappa $K; done
done
python - <<'PY' > results/throughput_3090.txt
import json
for r in (json.loads(l) for l in open("results/throughput_3090.jsonl") if l.strip()):
    print(f"{r['model'][:34]:34s} {r['method']:7s} {r['schedule']:7s} t{r['tau']} {r['gflops']:7.2f}G amp={str(r['amp']):5s} {r.get('img_per_s',0):8.0f} img/s")
PY
cat results/throughput_3090.txt
echo "== EXTRA DONE $(date)"
