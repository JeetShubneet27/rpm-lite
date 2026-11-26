#!/usr/bin/env bash
# Fourth review: pay RPM's cost only where similarity fails.  Resumable.
#   cd ~/S_VIT/tokenmerge_study && (nohup bash run_rpmhyb.sh > results/rpmhyb.log 2>&1 &)
#  H1 speed: probe hybrid with sketched RPM, ToMe-then-RPM switch, sketch k = 4             ~30 min
#  H2 sketch k sweep (4, 8) and seed variance (seeds 1, 2 at k = 32), 6 settings             ~2 h
#  H3 full protocol: probe hybrid with sketched RPM (hybrid_rpm), 38 settings                ~7 h
#  H4 full protocol: ToMe in the first third / half, sketched RPM after (swrpm)              ~7 h
#  H5 measured speed of every selected hybrid_rpm / swrpm configuration                      ~40 min
# Do not train anything else on the GPU while this runs: H1 and H5 measure speed.
set -uo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/VIT_5/data/imagenet}
W=${WORKERS:-6}
FAIL=results/rpmhyb_failed.txt
SP=results/speed_sketch.jsonl
HY="--method hybrid --weight-mode size --prop-attn on --blend soft --gamma 4 --attn-op rpms --sketch-k 32"
SW="--method hybrid --weight-mode size --prop-attn on --attn-op rpms --sketch-k 32"
echo "== start $(date)"
sel() { python eval_imagenet.py --data $IMAGENET/train --per-class 10 --workers $W --out results/rpm_sel.jsonl \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED sel: $*" >> $FAIL; }
ev()  { python eval_imagenet.py --data $IMAGENET/val --workers $W --out results/rpm_val.jsonl --save-preds \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED val: $*" >> $FAIL; }
spd() { python eval_imagenet.py --data $IMAGENET/val --speed-only --out $SP --skip-existing "$@" >/dev/null 2>&1 \
        || echo "FAILED speed: $*" >> $FAIL; }
MODELS_FM="vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m vit_base_patch16_clip_224.laion2b \
           vit_large_patch14_dinov2.lvd142m vit_small_patch14_dinov2.lvd142m vit_base_patch14_reg4_dinov2.lvd142m \
           vit_large_patch14_reg4_dinov2.lvd142m"
grid() {   # model|flag|budgets
  for M in $MODELS_FM; do echo "--model $M --probe probes/$M.pt|--target-ratio|0.45 0.6 0.75"; done
  echo "--model deit3_small_patch16_224.fb_in1k|--target-gflops|2.2 2.6 3.0"
  echo "--model deit_base_patch16_224|--target-gflops|9.0 10.1 11.5"
  echo "--model deit3_base_patch16_224.fb_in1k|--target-gflops|9.0 10.1 11.5"
  echo "--model deit_small_patch16_224|--target-gflops|2.2 2.4 2.6 3.0 3.6"
  echo "--model deit_tiny_patch16_224|--target-gflops|0.7 0.8 0.9"
}

echo "== H1 speed $(date)"
for MB in "deit_base_patch16_224|--target-gflops 9.0" "deit_small_patch16_224|--target-gflops 2.6" \
          "vit_base_patch14_dinov2.lvd142m --probe probes/vit_base_patch14_dinov2.lvd142m.pt|--target-ratio 0.6" \
          "vit_base_patch16_clip_224.laion2b --probe probes/vit_base_patch16_clip_224.laion2b.pt|--target-ratio 0.6"; do
  m="--model ${MB%%|*}"; b=${MB#*|}
  spd $m $b --tau 1 $HY
  spd $m $b --tau 1 $SW --fixed-ops sw0.33
  spd $m $b --tau 1 $SW --fixed-ops sw0.5
  spd $m $b --tau 1 --method rpms --sketch-k 4
done
python - <<'PY'
import json
rows = [json.loads(l) for l in open("results/speed_sketch.jsonl") if l.strip()]
base = {r["model"]: r for r in rows if r["method"] == "none"}
tome = {r["model"]: r for r in rows if r["method"] == "tome" and r.get("prior") == "-"}
print(f"  {'model':26s} {'method':8s} {'variant':18s}  fp32 x  bf16 x  (vs unmerged)   fp32 vs ToMe")
for r in rows:
    if r["method"] == "none" or r["model"] not in base: continue
    b, t = base[r["model"]], tome.get(r["model"])
    print(f"  {r['model'][:26]:26s} {r['method']:8s} {str(r.get('prior')):18s}  {r['img_per_s_fp32']/b['img_per_s_fp32']:5.2f}   "
          f"{r['img_per_s_bf16']/b['img_per_s_bf16']:5.2f}          {r['img_per_s_fp32']/t['img_per_s_fp32'] if t else float('nan'):5.2f}")
PY

echo "== H2 sketch size and seed $(date)"
for MB in "vit_small_patch14_reg4_dinov2.lvd142m --probe probes/vit_small_patch14_reg4_dinov2.lvd142m.pt|vit_small_patch14_reg4_dinov2.lvd142m|0.45" \
          "vit_base_patch14_dinov2.lvd142m --probe probes/vit_base_patch14_dinov2.lvd142m.pt|vit_base_patch14_dinov2.lvd142m|0.45" \
          "vit_base_patch14_dinov2.lvd142m --probe probes/vit_base_patch14_dinov2.lvd142m.pt|vit_base_patch14_dinov2.lvd142m|0.6" \
          "vit_base_patch16_clip_224.laion2b --probe probes/vit_base_patch16_clip_224.laion2b.pt|vit_base_patch16_clip_224.laion2b|0.6" \
          "deit3_small_patch16_224.fb_in1k|deit3_small_patch16_224.fb_in1k|2.2" \
          "deit_base_patch16_224|deit_base_patch16_224|9.0"; do
  IFS='|' read -r m name bud <<< "$MB"; m="--model $m"
  python pick_selected.py results/rpm_sel.jsonl --family rpmf2 --model $name | while read -r c; do
    case "$c" in *"-ratio $bud"*|*"-gflops $bud"*) ;; *) continue;; esac
    for K in 4 8; do ev $m ${c/--method rpmf2/--method rpms --sketch-k $K}; done
    for S in 1 2; do ev $m ${c/--method rpmf2/--method rpms --sketch-k 32 --sketch-seed $S}; done
  done
done

echo "== H3 hybrid_rpm full protocol $(date)"
grid | while IFS='|' read -r m flag budgets; do
  name=$(echo $m | awk '{print $2}')
  echo "== hybrid_rpm $name $(date +%H:%M)"
  for B in $budgets; do
    for T in 0 1 2 3; do sel $m $flag $B --tau $T $HY; done
    for T in 1 2 3; do sel $m $flag $B --tau $T $HY --schedule geo; done
    for R in 1 2 4; do sel $m $flag $B $HY --schedule rel --rel-kappa $R; done
  done
  python pick_selected.py results/rpm_sel.jsonl --family hybrid_rpm --model $name | while read -r c; do ev $m $c; done
done

echo "== H4 swrpm full protocol $(date)"
grid | while IFS='|' read -r m flag budgets; do
  name=$(echo $m | awk '{print $2}')
  echo "== swrpm $name $(date +%H:%M)"
  for B in $budgets; do
    for F in sw0.33 sw0.5; do
      for T in 0 1 2; do sel $m $flag $B --tau $T $SW --fixed-ops $F; done
      for T in 1 2; do sel $m $flag $B --tau $T $SW --fixed-ops $F --schedule geo; done
    done
  done
  python pick_selected.py results/rpm_sel.jsonl --family swrpm --model $name | while read -r c; do ev $m $c; done
done

echo "== H5 speed of selected configurations $(date)"
{ for M in $MODELS_FM; do echo "--model $M --probe probes/$M.pt"; done
  for M in deit_tiny_patch16_224 deit_small_patch16_224 deit_base_patch16_224 deit3_small_patch16_224.fb_in1k deit3_base_patch16_224.fb_in1k; do echo "--model $M"; done
} | while read -r m; do
  name=$(echo $m | awk '{print $2}')
  for F in hybrid_rpm swrpm; do
    python pick_selected.py results/rpm_sel.jsonl --split sel10s1 --family $F --model $name | while read -r c; do
      python eval_imagenet.py --data $IMAGENET/val --speed-only --out results/speed_3090.jsonl --skip-existing $m $c >/dev/null 2>&1 \
        || echo "FAILED speed $m $c" >> $FAIL
    done
  done
done
echo "== RPMHYB DONE $(date)"
