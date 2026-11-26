#!/usr/bin/env bash
# Wall-clock throughput (images/s, batch 256) -- run ONLY when nothing else uses the GPU (~15 min).
#   bash run_throughput.sh
set -uo pipefail
U=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
if [ "$U" -gt 10 ]; then echo "GPU is ${U}% busy -- throughput would be wrong. Stop other jobs first."; exit 1; fi
OUT=results/throughput.jsonl
CFG="--weight-mode size --prop-attn on"
D=${IMAGENET:-/home/user/桌面/Junaid/VIT_5/data/imagenet}/val
tp() { for AMP in "" "--amp"; do
  python eval_imagenet.py --data $D --limit 256 --workers 4 --throughput --out $OUT $AMP "$@" >/dev/null 2>&1 \
    || echo "FAILED: $*"; done; }
for MB in "deit_small_patch16_224:--target-gflops 3.0" "deit_base_patch16_224:--target-gflops 11.5" \
          "vit_base_patch14_dinov2.lvd142m:--target-ratio 0.6" "vit_large_patch14_reg4_dinov2.lvd142m:--target-ratio 0.6"; do
  M=${MB%%:*}; B=${MB#*:}
  P=""; [[ $M == *dinov2* ]] && P="--probe probes/$M.pt"
  tp --model $M $P --method none                       # timm, fused attention
  tp --model $M $P --method tome --r 0 --tag explicit  # same model through our explicit-attention blocks
  for T in 1 3; do
    tp --model $M $P $B --tau $T --method tome
    tp --model $M $P $B --tau $T --method sadtm $CFG --prior uniform
    tp --model $M $P $B --tau $T --method calib $CFG --blend soft --gamma 4
    tp --model $M $P $B --tau $T --method hybrid $CFG --blend soft --gamma 4
  done
done
python - <<'PY'
import json
rows=[json.loads(l) for l in open("results/throughput.jsonl") if l.strip()]
print(f"{'model':38s} {'method':8s} {'tau':>3s} {'GFLOPs':>7s} {'amp':>4s} {'img/s':>8s}")
for r in rows:
    print(f"{r['model']:38s} {r['method']+('*' if r.get('tag')=='explicit' else ''):8s} {r['tau']:3d} {r['gflops']:7.2f} {str(r['amp']):>4s} {r.get('img_per_s',0):8.0f}")
PY
