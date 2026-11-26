#!/usr/bin/env bash
# Fallback variant (ToMe where the probe is unreliable or uncertain). Waits for the other queues.
#   cd ~/S_VIT/tokenmerge_study && (nohup bash run_fb.sh > results/fb.log 2>&1 &)
cd "$(dirname "$0")"
source venv/bin/activate
python make_reliability_fallback.py results/probe_validation.jsonl results/reliability_fb.jsonl || exit 1
bash run_variant.sh fb results/reliability_fb.jsonl
