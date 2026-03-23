#!/bin/bash
# Full evaluation pipeline — runs on GPUs 3,4 via CUDA_VISIBLE_DEVICES
# Survives laptop close / SSH disconnect when launched with nohup

set -e
export CUDA_VISIBLE_DEVICES=3,4
cd /home/medilink/RAG
source .venv/bin/activate

LOG="results/eval_run.log"
echo "========================================" >> "$LOG"
echo "EVAL START: $(date)" >> "$LOG"
echo "GPUs: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >> "$LOG"
echo "========================================" >> "$LOG"

# Step 1: Retrieval evaluation
echo "[$(date)] Step 1/3: Running retrieval evaluation..." >> "$LOG"
python3 evaluate_retrieval.py >> "$LOG" 2>&1
echo "[$(date)] Step 1/3: Retrieval evaluation DONE" >> "$LOG"

# Step 2: Generation evaluation + all 7 plots
echo "[$(date)] Step 2/3: Running generation evaluation + plots..." >> "$LOG"
python3 evaluate_plots.py >> "$LOG" 2>&1
echo "[$(date)] Step 2/3: Generation evaluation + plots DONE" >> "$LOG"

# Step 3: Print final summary
echo "[$(date)] Step 3/3: Final summary" >> "$LOG"
echo "" >> "$LOG"
echo "=== RETRIEVAL METRICS ===" >> "$LOG"
cat results/retrieval_metrics.csv >> "$LOG"
echo "" >> "$LOG"
echo "=== GENERATION EVAL (first 5 rows) ===" >> "$LOG"
head -6 results/generation_eval.csv >> "$LOG"
echo "" >> "$LOG"
echo "=== PLOTS GENERATED ===" >> "$LOG"
ls -la results/plots/ >> "$LOG" 2>&1
echo "" >> "$LOG"
echo "========================================" >> "$LOG"
echo "EVAL COMPLETE: $(date)" >> "$LOG"
echo "========================================" >> "$LOG"
