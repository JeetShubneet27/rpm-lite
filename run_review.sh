#!/usr/bin/env bash
# Experiments that answer the reviewer (resumable: rerun to continue; waits for the current queue).
#   cd ~/S_VIT/tokenmerge_study && (nohup bash run_review.sh > results/review.log 2>&1 &)
#   progress: grep "==" results/review.log; tail -n 40 results/review_report.txt
#  R0 PiToMe again, now matching the official code (energy ELU(cos - m), margin 0.75 / CLIP 0.9)       ~10 h
#  R1 probe validation: merge cost at the layer, block-wise operator cost, bootstrap CIs  (W1, W2, Q1, Q5)  ~1 h
#  R2 fixed-rule hybrids (ToMe first+last 2 blocks; RAPID-style early/late switch)       (W3, Q2)          ~8 h
#  R3 outlier-masked class attention for SAD-TM and for ours                             (W3, Q3)          ~3 h
#  R4 CLIP zero-shot                                                                     (W6)              ~1 h
#  R5 gamma ablation (selection split)                                                   (moderate)        ~1 h
#  R6 more budgets for the foundation models (85%, 70%, 52.5%) for Pareto curves         (W4, Q4)          ~7 h
#  R7 clean selection split without the linear-probe training images                    (W6)              ~5 h
#  R8 measured throughput of every selected configuration (fp32 + bf16), alone on the GPU (W5, Q4)         ~2 h
set -uo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
export IMAGENET=${IMAGENET:-$HOME/VIT_5/data/imagenet}
W=${WORKERS:-12}
FAIL=results/review_failed.txt
while pgrep -f "run_extra.sh|start_wsl.sh" >/dev/null 2>&1; do echo "waiting for the current queue  $(date +%H:%M)"; sleep 300; done
echo "== start $(date)"
CFG="--weight-mode size --prop-attn on"
FM="vit_small_patch14_dinov2.lvd142m vit_small_patch14_reg4_dinov2.lvd142m vit_base_patch14_dinov2.lvd142m
    vit_base_patch14_reg4_dinov2.lvd142m vit_base_patch16_clip_224.laion2b vit_large_patch14_dinov2.lvd142m
    vit_large_patch14_reg4_dinov2.lvd142m"
ZS=probes/vit_base_patch16_clip_224.laion2b_zeroshot.pt

sup_settings() {
  for G in 3.6 3.0 2.6 2.4 2.2; do echo "--model deit_small_patch16_224|--target-gflops $G"; done
  for G in 0.9 0.8 0.7; do echo "--model deit_tiny_patch16_224|--target-gflops $G"; done
  for G in 3.0 2.6 2.2; do echo "--model deit3_small_patch16_224.fb_in1k|--target-gflops $G"; done
  for G in 11.5 10.1 9.0; do echo "--model deit_base_patch16_224|--target-gflops $G"; done
  for G in 11.5 10.1 9.0; do echo "--model deit3_base_patch16_224.fb_in1k|--target-gflops $G"; done
}
fm_settings() { for M in $FM; do for R in ${RATIOS:-0.75 0.6 0.45}; do echo "--model $M --probe probes/$M.pt|--target-ratio $R"; done; done; }
settings() { sup_settings; fm_settings; }
VALS="results/calib.jsonl results/sched.jsonl results/probe.jsonl results/fix_val.jsonl results/rel_val.jsonl results/pitome_val.jsonl results/review_val.jsonl"
SELS="results/calib.jsonl results/sched.jsonl results/probe.jsonl results/fix_sel.jsonl results/rel_sel.jsonl results/pitome_sel.jsonl results/review_sel.jsonl"
ev()  { python eval_imagenet.py --data $IMAGENET/val --workers $W --out results/review_val.jsonl --save-preds \
        --skip-existing --also-check $VALS "$@" >/dev/null 2>&1 || echo "FAILED val: $*" >> $FAIL; }
sel() { python eval_imagenet.py --data $IMAGENET/train --per-class 10 --workers $W --out results/review_sel.jsonl \
        --skip-existing --also-check $SELS "$@" >/dev/null 2>&1 || echo "FAILED sel: $*" >> $FAIL; }
name_of() { echo "$1" | awk '{print $2}'; }
pick_val() {   # pick_val <family> <settings-function> [extra pick args]: evaluate on val what each family selected
  local fam=$1 fn=$2; shift 2
  $fn | cut -d'|' -f1 | sort -u | while read -r m; do
    python pick_selected.py $SELS --family $fam --model $(name_of "$m") "$@" | while read -r c; do ev $m $c; done
  done
}
report() { python review_report.py results/*.jsonl > results/review_report.txt 2>&1; echo "   report updated $(date +%H:%M)"; }
grid() {     # the tau x schedule grid every family uses (same as run_fix.sh)
  for T in 0 1 2 3; do echo "--tau $T"; done; for T in 1 2 3; do echo "--tau $T --schedule geo"; done; }

# ---- R1 ------------------------------------------------------------------------------------------
echo "== R1 probe validation $(date)"
python validate_probe.py --data $IMAGENET/train --workers 8 --out results/probe_validation.jsonl > results/probe_validation.log 2>&1
grep -E "probe op|mc op|cheaper|flip|FAILED" results/probe_validation.log | tail -n 48; report

# ---- R0: PiToMe re-run with the official energy (ELU(cos - m)) and margins (0.75, CLIP 0.9) -----------
echo "== R0 PiToMe (official energy + margins) $(date)"
hires_settings() {
  for M in vit_base_patch14_dinov2.lvd142m vit_large_patch14_reg4_dinov2.lvd142m vit_large_patch14_dinov2.lvd142m; do
    [ -f probes/${M}_448.pt ] && for R in 0.5 0.35; do echo "--model $M --probe probes/${M}_448.pt --img-size 448|--target-ratio $R"; done
  done; }
{ settings; hires_settings; } | while IFS='|' read -r m b; do
  for T in 0 1 2 3; do sel $m $b --tau $T --method pitome; done
  for T in 1 2; do sel $m $b --tau $T --method pitome --schedule geo; done
done
pick_val pitome settings
hires_settings | cut -d'|' -f1 | sort -u | while read -r m; do
  python pick_selected.py $SELS --family pitome --model $(name_of "$m") --img-size 448 | while read -r c; do ev $m $c; done
done
python select_report.py results/{calib,sched,probe,fix_val,fix_sel,hires_val,hires_sel,rel_val,rel_sel,review_val,review_sel}.jsonl \
  > results/select_with_pitome_official.txt 2>&1; tail -n 6 results/select_with_pitome_official.txt; report

# ---- R2 ------------------------------------------------------------------------------------------
echo "== R2 fixed-rule hybrids $(date)"
settings | while IFS='|' read -r m b; do
  grid | while read -r g; do
    sel $m $b $g --method hybrid $CFG --fixed-ops edge2
    for S in sw0.33 sw0.5; do sel $m $b $g --method hybrid $CFG --fixed-ops $S; done
  done
done
pick_val fixedge settings; pick_val fixsw settings; report

# ---- R3 ------------------------------------------------------------------------------------------
echo "== R3 outlier-masked attention $(date)"
fm_settings | while IFS='|' read -r m b; do
  for T in 1 2 3; do sel $m $b --tau $T --method sadtm $CFG --prior uniform --outlier-mask 0.02; done
  sel $m $b --tau 2 --schedule geo --method sadtm $CFG --prior uniform --outlier-mask 0.02
  grid | while read -r g; do sel $m $b $g --method hybrid $CFG --blend soft --gamma 4 --outlier-mask 0.02; done
done
pick_val sadtm_om fm_settings; pick_val hybrid_om fm_settings; report

# ---- R4 ------------------------------------------------------------------------------------------
echo "== R4 CLIP zero-shot $(date)"
if [ ! -f $ZS ]; then
  python -c "import open_clip" 2>/dev/null || python -m pip install -q open_clip_torch
  python make_zeroshot_head.py --out $ZS || echo "FAILED zero-shot head" >> $FAIL
fi
if [ -f $ZS ]; then
  zs_settings() { for R in 0.75 0.6 0.45; do echo "--model vit_base_patch16_clip_224.laion2b --probe $ZS --tag zs|--target-ratio $R"; done; }
  ev --model vit_base_patch16_clip_224.laion2b --probe $ZS --tag zs --method none
  zs_settings | while IFS='|' read -r m b; do
    grid | while read -r g; do
      sel $m $b $g --method tome; sel $m $b $g --method pitome
      sel $m $b $g --method hybrid $CFG --blend soft --gamma 4
    done
    for T in 1 2 3; do sel $m $b --tau $T --method sadtm $CFG --prior uniform; done
    sel $m $b --tau 2 --schedule geo --method sadtm $CFG --prior uniform
    for K in 1 2 4; do sel $m $b --method hybrid $CFG --blend soft --gamma 4 --schedule rel --rel-kappa $K; done
  done
  for F in tome sadtm pitome hybrid; do
    python pick_selected.py results/review_sel.jsonl --family $F --model vit_base_patch16_clip_224.laion2b --probe $ZS \
      | while read -r c; do ev --model vit_base_patch16_clip_224.laion2b --probe $ZS --tag zs $c; done
  done
fi
report

# ---- R5 ------------------------------------------------------------------------------------------
echo "== R5 gamma ablation $(date)"
settings | cut -d'|' -f1 | sort -u | while read -r m; do
  python pick_selected.py $SELS --family hybrid --model $(name_of "$m") | while read -r c; do
    sel $m $c                                                      # gamma 4 (normally already done -> skipped)
    for G in 1 2 8; do sel $m ${c/--gamma 4/--gamma $G}; done
    sel $m ${c/--blend soft --gamma 4/--blend hard}
  done
done
report

# ---- R6 ------------------------------------------------------------------------------------------
echo "== R6 extra budgets (Pareto) $(date)"
RATIOS="0.85 0.7 0.525" fm_settings | while IFS='|' read -r m b; do
  grid | while read -r g; do
    sel $m $b $g --method tome; sel $m $b $g --method pitome; sel $m $b $g --method hybrid $CFG --blend soft --gamma 4
  done
  for T in 1 2 3; do sel $m $b --tau $T --method sadtm $CFG --prior uniform; done
  sel $m $b --tau 2 --schedule geo --method sadtm $CFG --prior uniform
  for K in 1 2 4; do sel $m $b --method hybrid $CFG --blend soft --gamma 4 --schedule rel --rel-kappa $K; done
done
r6() { RATIOS="0.85 0.7 0.525" fm_settings; }
for F in tome sadtm pitome hybrid; do pick_val $F r6; done
report

# ---- R7 ------------------------------------------------------------------------------------------
echo "== R7 clean selection split $(date)"
csel() { python eval_imagenet.py --data $IMAGENET/train --per-class 10 --sel-exclude 110 --workers $W \
         --out results/clean_sel.jsonl --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED clean: $*" >> $FAIL; }
fm_settings | while IFS='|' read -r m b; do
  grid | while read -r g; do
    csel $m $b $g --method tome; csel $m $b $g --method pitome; csel $m $b $g --method hybrid $CFG --blend soft --gamma 4
  done
  for T in 1 2 3; do csel $m $b --tau $T --method sadtm $CFG --prior uniform; done
  csel $m $b --tau 2 --schedule geo --method sadtm $CFG --prior uniform
  for K in 1 2 4; do csel $m $b --method hybrid $CFG --blend soft --gamma 4 --schedule rel --rel-kappa $K; done
done
for F in tome sadtm pitome hybrid; do
  fm_settings | cut -d'|' -f1 | sort -u | while read -r m; do
    python pick_selected.py results/clean_sel.jsonl --family $F --model $(name_of "$m") | while read -r c; do ev $m $c; done
  done
done
report

# ---- R8 ------------------------------------------------------------------------------------------
echo "== R8 throughput of every selected configuration $(date)"
spd() { python eval_imagenet.py --data $IMAGENET/val --speed-only --workers 2 --out results/speed_3090.jsonl \
        --skip-existing "$@" >/dev/null 2>&1 || echo "FAILED speed: $*" >> $FAIL; }
all_settings() { sup_settings; RATIOS="0.85 0.75 0.7 0.6 0.525 0.45" fm_settings; }
all_settings | cut -d'|' -f1 | sort -u | while read -r m; do
  spd $m --method none
  for F in tome sadtm pitome hybrid; do
    python pick_selected.py $SELS --family $F --model $(name_of "$m") | while read -r c; do spd $m $c; done
  done
done
report
cat results/review_report.txt
echo "== REVIEW DONE $(date)"
