#!/usr/bin/env bash
# Third review: can RPM be made cheap enough to win at equal measured throughput?  Resumable.
#   cd ~/S_VIT/tokenmerge_study && (nohup bash run_sketch.sh > results/sketch.log 2>&1 &)
#  S1 speed: ToMe, MCTF, RPM-f, sketched RPM (k = 8..64, per head / head-averaged), ToMe with
#     explicit attention (how much of RPM-f's overhead is just losing the fused kernel)        ~40 min
#  S2 accuracy vs k at RPM-f's selected configuration, 6 settings                               ~2 h
#  S3 full selection protocol for the best sketch (12 models, 38 settings)                      ~10 h
#  S4 measured speed of every selected sketch configuration                                     ~30 min
# Do not train anything else on the GPU while this runs: S1 and S4 measure speed.
set -uo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/VIT_5/data/imagenet}
W=${WORKERS:-6}
FAIL=results/sketch_failed.txt
SP=results/speed_sketch.jsonl
echo "== start $(date)"
sel() { python eval_imagenet.py --data $IMAGENET/train --per-class 10 --workers $W --out results/rpm_sel.jsonl \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED sel: $*" >> $FAIL; }
ev()  { python eval_imagenet.py --data $IMAGENET/val --workers $W --out results/rpm_val.jsonl --save-preds \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED val: $*" >> $FAIL; }
spd() { python eval_imagenet.py --data $IMAGENET/val --speed-only --out $SP --skip-existing "$@" >/dev/null 2>&1 \
        || echo "FAILED speed: $*" >> $FAIL; }

echo "== S1 speed $(date)"
for MB in "deit_base_patch16_224|--target-gflops 9.0" "deit_small_patch16_224|--target-gflops 2.6" \
          "vit_base_patch14_dinov2.lvd142m --probe probes/vit_base_patch14_dinov2.lvd142m.pt|--target-ratio 0.6" \
          "vit_base_patch16_clip_224.laion2b --probe probes/vit_base_patch16_clip_224.laion2b.pt|--target-ratio 0.6"; do
  m="--model ${MB%%|*}"; b=${MB#*|}
  spd $m --method none
  for M in tome mctf rpmf2; do spd $m $b --tau 1 --method $M; done
  spd $m $b --tau 1 --method tome --explicit-attn
  for K in 8 16 32 64; do spd $m $b --tau 1 --method rpms --sketch-k $K; done
  for K in 16 32; do spd $m $b --tau 1 --method rpmsh --sketch-k $K; done
done
python - <<'PY'
import json
rows = [json.loads(l) for l in open("results/speed_sketch.jsonl") if l.strip()]
base = {r["model"]: r for r in rows if r["method"] == "none"}
print(f"  {'model':26s} {'method':8s} {'variant':12s}  fp32 x   bf16 x   (vs unmerged)")
for r in rows:
    if r["method"] == "none" or r["model"] not in base: continue
    b = base[r["model"]]
    print(f"  {r['model'][:26]:26s} {r['method']:8s} {str(r.get('prior')):12s}  {r['img_per_s_fp32']/b['img_per_s_fp32']:5.2f}   {r['img_per_s_bf16']/b['img_per_s_bf16']:5.2f}")
PY

echo "== S2 accuracy vs sketch size $(date)"
for MB in "vit_small_patch14_reg4_dinov2.lvd142m --probe probes/vit_small_patch14_reg4_dinov2.lvd142m.pt|vit_small_patch14_reg4_dinov2.lvd142m|0.45" \
          "vit_base_patch14_dinov2.lvd142m --probe probes/vit_base_patch14_dinov2.lvd142m.pt|vit_base_patch14_dinov2.lvd142m|0.45" \
          "vit_base_patch14_dinov2.lvd142m --probe probes/vit_base_patch14_dinov2.lvd142m.pt|vit_base_patch14_dinov2.lvd142m|0.6" \
          "vit_base_patch16_clip_224.laion2b --probe probes/vit_base_patch16_clip_224.laion2b.pt|vit_base_patch16_clip_224.laion2b|0.6" \
          "deit3_small_patch16_224.fb_in1k|deit3_small_patch16_224.fb_in1k|2.2" \
          "deit_base_patch16_224|deit_base_patch16_224|9.0"; do
  IFS='|' read -r m name bud <<< "$MB"; m="--model $m"
  python pick_selected.py results/rpm_sel.jsonl --family rpmf2 --model $name | while read -r c; do
    case "$c" in *"-ratio $bud"*|*"-gflops $bud"*) ;; *) continue;; esac
    for K in 16 32 64; do ev $m ${c/--method rpmf2/--method rpms --sketch-k $K}; done
    ev $m ${c/--method rpmf2/--method rpmsh --sketch-k 16}
  done
done
python - <<'PY'
import json, statistics as st
rows = [json.loads(l) for l in open("results/rpm_val.jsonl") if l.strip()]
key = lambda r: (r["model"], r.get("target_ratio") or r["target_gflops"], r["tau"], r.get("schedule", "const"))
ref = {key(r): r["acc1"] for r in rows if r["method"] == "rpmf2"}
d = {}
for r in rows:
    if r["method"] in ("rpms", "rpmsh") and key(r) in ref:
        v = f"{r['method']}-{r['prior']}"
        d.setdefault(v, []).append(r["acc1"] - ref[key(r)])
        print(f"  {r['model'][:30]:30s} {key(r)[1]:>5}  {v:12s} {r['acc1']:6.2f}  (RPM-f {ref[key(r)]:6.2f})")
best = None
for v in ("rpms-k16", "rpms-k32", "rpms-k64"):
    if v in d:
        print(f"  {v}: mean change vs RPM-f {st.mean(d[v]):+.2f} pp over {len(d[v])} settings")
        if best is None and st.mean(d[v]) >= -0.2: best = v
for v in d:
    if v.startswith("rpmsh"): print(f"  {v}: mean change vs RPM-f {st.mean(d[v]):+.2f} pp over {len(d[v])} settings")
best = best or "rpms-k64"
open("results/sketch_k.txt", "w").write(best.split("-k")[1])
print("  chosen sketch size k =", best.split("-k")[1])
PY
K=$(cat results/sketch_k.txt 2>/dev/null || echo 32)

echo "== S3 full protocol, rpms k=$K $(date)"
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
  echo "== rpms k=$K $name $(date +%H:%M)"
  for B in $budgets; do
    for T in 0 1 2 3; do sel $m $flag $B --tau $T --method rpms --sketch-k $K; done
    for T in 1 2 3; do sel $m $flag $B --tau $T --method rpms --sketch-k $K --schedule geo; done
    for R in 1 2 4; do sel $m $flag $B --method rpms --sketch-k $K --schedule rel --rel-kappa $R; done
  done
  python pick_selected.py results/rpm_sel.jsonl --family rpms$K --model $name | while read -r c; do ev $m $c; done
done

echo "== S4 speed of selected sketch configurations $(date)"
{ for M in vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m vit_base_patch16_clip_224.laion2b \
           vit_large_patch14_dinov2.lvd142m vit_small_patch14_dinov2.lvd142m vit_base_patch14_reg4_dinov2.lvd142m \
           vit_large_patch14_reg4_dinov2.lvd142m; do echo "--model $M --probe probes/$M.pt"; done
  for M in deit_tiny_patch16_224 deit_small_patch16_224 deit_base_patch16_224 deit3_small_patch16_224.fb_in1k deit3_base_patch16_224.fb_in1k; do echo "--model $M"; done
} | while read -r m; do
  name=$(echo $m | awk '{print $2}')
  python pick_selected.py results/rpm_sel.jsonl --split sel10s1 --family rpms$K --model $name | while read -r c; do
    python eval_imagenet.py --data $IMAGENET/val --speed-only --out results/speed_3090.jsonl --skip-existing $m $c >/dev/null 2>&1 \
      || echo "FAILED speed $m $c" >> $FAIL
  done
done
python select_report.py results/{calib,sched,probe,fix_val,fix_sel,rel_val,rel_sel,review_val,review_sel,rpm_val,rpm_sel}.jsonl \
  > results/rpm_report.txt 2>&1; grep -E "wins" results/rpm_report.txt
echo "== SKETCH DONE $(date)"
