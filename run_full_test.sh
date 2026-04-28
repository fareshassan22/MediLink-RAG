#!/bin/bash
# ============================================================
# MediLink RAG — Full Test & Evaluation Suite
# GPUs: 1 and 6 | Run inside tmux for persistence
# ============================================================
set -e

export CUDA_VISIBLE_DEVICES=1,6
cd /home/medilink/RAG
source .venv/bin/activate

LOG="results/full_test_$(date +%Y%m%d_%H%M%S).log"
mkdir -p results/plots

echo "============================================================" | tee "$LOG"
echo "MediLink RAG — Full Test Suite — $(date)"                     | tee -a "$LOG"
echo "GPUs: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"            | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# ── 1. Unit Tests ────────────────────────────────────────────
echo ""                                                             | tee -a "$LOG"
echo "▶ [1/6] UNIT TESTS"                                          | tee -a "$LOG"
echo "------------------------------------------------------------" | tee -a "$LOG"
python -m pytest tests/ -v --tb=short 2>&1 | tee -a "$LOG"
echo ""                                                             | tee -a "$LOG"

# ── 2. Safety Module — Targeted Tests ────────────────────────
echo "▶ [2/6] SAFETY MODULE — TARGETED TESTS"                      | tee -a "$LOG"
echo "------------------------------------------------------------" | tee -a "$LOG"
python -c "
from app.safety.emergency_detector import detect_emergency
from app.safety.content_filter import contains_sensitive_content

# Emergency detection accuracy
emergency_cases = [
    ('ألم في الصدر', True),
    ('chest pain', True),
    ('نزيف حاد', True),
    ('severe bleeding', True),
    ('stroke', True),
    ('فقدان الوعي', True),
    ('cannot breathe', True),
    ('ما هو علاج السكري', False),
    ('how to treat headache', False),
    ('أعراض البرد', False),
    ('hello', False),
    ('what is diabetes', False),
]

correct = 0
for query, expected in emergency_cases:
    result = detect_emergency(query)
    status = '✓' if result == expected else '✗'
    if result == expected:
        correct += 1
    print(f'  {status} detect_emergency(\"{query[:40]}\") = {result} (expected {expected})')

print(f'\n  Emergency accuracy: {correct}/{len(emergency_cases)} ({correct/len(emergency_cases)*100:.0f}%)')

# Content filter
filter_cases = [
    'some normal medical text',
    'how to make drugs',
]
print()
for q in filter_cases:
    r = contains_sensitive_content(q)
    print(f'  contains_sensitive_content(\"{q}\") = {r}')
print()
" 2>&1 | tee -a "$LOG"

# ── 3. Calibration Pipeline ─────────────────────────────────
echo "▶ [3/6] CALIBRATION PIPELINE"                                | tee -a "$LOG"
echo "------------------------------------------------------------" | tee -a "$LOG"
python -c "
import numpy as np
from app.calibration.calibrator import (
    train_calibrator, predict_confidence,
    generate_synthetic_training_data, _expected_calibration_error,
    train_with_synthetic_data,
)
import os
os.makedirs('models', exist_ok=True)

print('  Testing synthetic data generation...')
X, y = generate_synthetic_training_data(200)
print(f'    Generated: X={X.shape}, y={y.shape}, pos_rate={y.mean():.2f}')

print('  Training calibrator with cross-validation...')
result = train_with_synthetic_data()
print(f'    ECE:      {result.ece:.4f}')
print(f'    Brier:    {result.brier:.4f}')
print(f'    Accuracy: {result.accuracy:.4f}')
print(f'    Weights:  {result.weights}')

print('  Testing predict_confidence...')
score = predict_confidence(
    grounding_score=0.85,
    retrieval_score=0.90,
    rerank_score=0.75,
    context_length=500,
    answer_length=120,
    top_similarity=0.88,
)
print(f'    High-quality input -> confidence: {score:.3f}')

score_low = predict_confidence(
    grounding_score=0.1,
    retrieval_score=0.2,
    rerank_score=0.05,
    context_length=50,
    answer_length=10,
    top_similarity=0.15,
)
print(f'    Low-quality input  -> confidence: {score_low:.3f}')
print(f'    Separation: {score - score_low:.3f} (higher is better)')
print()
" 2>&1 | tee -a "$LOG"

# ── 4. Retrieval Evaluation (4 modes, 99 queries) ───────────
echo "▶ [4/6] RETRIEVAL EVALUATION (4 modes × 99 queries)"         | tee -a "$LOG"
echo "------------------------------------------------------------" | tee -a "$LOG"
python evaluate_retrieval.py 2>&1 | tee -a "$LOG"
echo ""                                                             | tee -a "$LOG"

# ── 5. Full Evaluation + Plots (retrieval + generation + judge)
echo "▶ [5/6] FULL EVALUATION + PLOTS (Qwen2.5-32B on GPUs 1+6)"  | tee -a "$LOG"
echo "------------------------------------------------------------" | tee -a "$LOG"
python evaluate_plots.py 2>&1 | tee -a "$LOG"
echo ""                                                             | tee -a "$LOG"

# ── 6. Ground Truth Annotation (dry run) ─────────────────────
echo "▶ [6/6] GROUND TRUTH ANNOTATION (dry run — 2 queries)"       | tee -a "$LOG"
echo "------------------------------------------------------------" | tee -a "$LOG"
python annotate_ground_truth.py --dry-run 2>&1 | tee -a "$LOG"
echo ""                                                             | tee -a "$LOG"

# ── Summary ──────────────────────────────────────────────────
echo "============================================================" | tee -a "$LOG"
echo "ALL TESTS COMPLETE — $(date)"                                 | tee -a "$LOG"
echo "Log: $LOG"                                                    | tee -a "$LOG"
echo "Results:"                                                     | tee -a "$LOG"
ls -la results/*.csv results/plots/*.png 2>/dev/null | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
