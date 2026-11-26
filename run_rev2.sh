#!/usr/bin/env bash
# Revision batch 2 (mentor item 11). Waits for the first revision batch (unit tmrev). Resumable.
#   cd ~/S_VIT/tokenmerge_study && systemd-run --user --unit=tmrev2 -p WorkingDirectory=$PWD bash -c 'bash run_rev2.sh >> results/rev2.log 2>&1'
#  P1 probe cost (time, GPU memory) on 5 models; probe entries for the two AugReg models        ~30 min
#  P2 AugReg: probe-based budget allocation (kappa 1,2,4) for every method, so all methods have
#     the same 10 candidates, plus probe-guided RPM-lite; selected and evaluated as everywhere    ~2.5 h
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/data/imagenet}
export TORCHINDUCTOR_CACHE_DIR=$HOME/tmp/inductor TRITON_CACHE_DIR=$HOME/tmp/triton TMPDIR=$HOME/tmp
clean_cache() { rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"; mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"; }
diskcheck() { local a=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9); if [ "${a:-0}" -lt 20 ]; then clean_cache; a=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9); [ "${a:-0}" -lt 20 ] && { echo "STOPPED: less than 20 GB free on disk $(date)"; exit 2; }; fi; }
gpu_alone() {   # timing steps need an otherwise idle GPU: wait until no other process computes on it
  # (programs under 500 MiB, such as RustDesk or the desktop, are ignored)
  while [ -n "$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F, '$2+0 > 500')" ]; do
    echo "timing step waiting for the GPU to be free  $(date +%H:%M)"; sleep 300; done; }
W=${WORKERS:-6}
FAIL=results/rev2_failed.txt
while systemctl --user is-active --quiet tmrev; do echo "waiting for revision batch 1  $(date +%H:%M)"; sleep 300; done
clean_cache
echo "== start $(date)"

echo "== P1 probe cost $(date)"
probe() {   # model out
  grep -q "\"model\": \"$1\"" results/probe_cost.jsonl 2>/dev/null && return
  gpu_alone
  python - "$1" "$2" <<'PY' || echo "FAILED probe $1" >> $FAIL
import json, runpy, sys, time, torch
m, out = sys.argv[1], sys.argv[2]
sys.argv = ["measure_reliability.py", "--data", __import__("os").environ["IMAGENET"] + "/train", "--models", m,
            "--workers", "6", "--out", out]
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
runpy.run_path("measure_reliability.py", run_name="__main__")
torch.cuda.synchronize()
rec = dict(model=m, seconds=round(time.time() - t, 1), peak_gpu_mb=round(torch.cuda.max_memory_allocated() / 2**20),
           gpu=torch.cuda.get_device_name(0), num_images=512, loo_images=128)
open("results/probe_cost.jsonl", "a").write(json.dumps(rec) + "\n"); print(rec)
PY
}
for M in deit_small_patch16_224 vit_base_patch14_dinov2.lvd142m vit_large_patch14_dinov2.lvd142m; do
  probe $M results/reliability_timing.jsonl
done
for M in vit_small_patch16_224.augreg_in21k_ft_in1k vit_base_patch16_224.augreg_in21k_ft_in1k; do
  probe $M results/reliability.jsonl
done

echo "== P2 AugReg with the probe $(date)"
sel() { python eval_imagenet.py --data $IMAGENET/train --per-class 10 --workers $W --out results/rpm_sel.jsonl \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED sel: $*" >> $FAIL; }
ev()  { python eval_imagenet.py --data $IMAGENET/val --workers $W --out results/rpm_val.jsonl --save-preds \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED val: $*" >> $FAIL; }
HL="--method hybrid --weight-mode size --prop-attn on --blend soft --gamma 4 --attn-op rpmlnw"
for M in vit_small_patch16_224.augreg_in21k_ft_in1k vit_base_patch16_224.augreg_in21k_ft_in1k; do
  echo "== augreg $M $(date +%H:%M)"
  for B in 0.45 0.6 0.75; do
    for METH in "--method tome" "--method sadtm --weight-mode size --prop-attn on --prior uniform" "--method pitome" \
                "--method mctf" "--method adamerge" "--method rpmlnw"; do
      for R in 1 2 4; do sel --model $M --target-ratio $B --schedule rel --rel-kappa $R $METH; done
    done
    for T in 0 1 2 3; do sel --model $M --target-ratio $B --tau $T $HL; done
    for T in 1 2 3; do sel --model $M --target-ratio $B --tau $T --schedule geo $HL; done
    for R in 1 2 4; do sel --model $M --target-ratio $B --schedule rel --rel-kappa $R $HL; done
  done
  for F in tome sadtm pitome mctf adamerge rpmlnw hybrid_lite; do
    python pick_selected.py results/rpm_sel.jsonl --split sel10s1 --family $F --model $M | while read -r c; do ev --model $M $c; done
  done
done
echo "== REV2 DONE $(date)"
