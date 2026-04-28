#!/bin/bash
# ============================================================
# MediLink RAG — Re-run failed steps (generation eval + annotation)
# GPUs: physical 1 and 6 → logical 0 and 1
# ============================================================
set -e

export CUDA_VISIBLE_DEVICES=1,6
cd /home/medilink/RAG
source .venv/bin/activate

LOG="results/full_test_rerun_$(date +%Y%m%d_%H%M%S).log"
mkdir -p results/plots

echo "============================================================" | tee "$LOG"
echo "MediLink RAG — Re-run (Gen Eval + Annotation) — $(date)"     | tee -a "$LOG"
echo "GPUs: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"            | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# Verify GPUs are visible
python -c "
import torch
n = torch.cuda.device_count()
print(f'  Visible GPUs: {n}')
for i in range(n):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    print(f'    [{i}] {name} ({mem:.1f} GB)')
" 2>&1 | tee -a "$LOG"

# ── 1. Full Evaluation + Plots (Qwen2.5-32B) ────────────────
echo ""                                                             | tee -a "$LOG"
echo "▶ [1/2] FULL EVALUATION + PLOTS (Qwen2.5-32B)"              | tee -a "$LOG"
echo "------------------------------------------------------------" | tee -a "$LOG"
python evaluate_plots.py 2>&1 | tee -a "$LOG"
echo ""                                                             | tee -a "$LOG"

# ── 2. Ground Truth Annotation (dry run) ─────────────────────
echo "▶ [2/2] GROUND TRUTH ANNOTATION (dry run — 2 queries)"       | tee -a "$LOG"
echo "------------------------------------------------------------" | tee -a "$LOG"
python annotate_ground_truth.py --dry-run 2>&1 | tee -a "$LOG"
echo ""                                                             | tee -a "$LOG"

# ── Summary ──────────────────────────────────────────────────
echo "============================================================" | tee -a "$LOG"
echo "RE-RUN COMPLETE — $(date)"                                    | tee -a "$LOG"
echo "Log: $LOG"                                                    | tee -a "$LOG"
echo "Results:"                                                     | tee -a "$LOG"
ls -la results/*.csv results/plots/*.png 2>/dev/null | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
