#!/usr/bin/env bash
# One (model, budget) group per Python process, until all are measured. Keeps RAM flat (compiled graphs are freed
# when each process exits). Resumable.
#   cd ~/S_VIT/tokenmerge_study && (nohup bash run_speed_loop.sh > results/speed_sel3.log 2>&1 &)
cd "$(dirname "$0")"
source venv/bin/activate
export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor TRITON_CACHE_DIR=$HOME/.cache/triton TMPDIR=$HOME/tmp
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $TMPDIR
echo "== start $(date)"
fails=0
while true; do
  python bench_selected.py results/cfgs/list38.txt --out results/speed_sel.jsonl --max-groups 1 2>&1 | grep -v -i warn
  rc=${PIPESTATUS[0]}
  [ "$rc" = "3" ] && break
  if [ "$rc" != "0" ]; then fails=$((fails + 1)); echo "  (exit code $rc, failure $fails)"; [ $fails -ge 3 ] && break; else fails=0; fi
done
echo "== SPEED_SEL DONE $(date)"
