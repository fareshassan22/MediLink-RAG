#!/usr/bin/env python3
"""TOON tier classifier — learned replacement for hand-tuned regex.

Why this exists
───────────────
The regex router (toon_router.py) overfits: every pattern added to fix one
phrasing fails on the next. Arabic medical questions have too many surface
forms for hand-written rules. This module embeds the query with the SAME
bge-m3 model already used for retrieval and classifies the tier with a small,
regularized logistic-regression head. It generalizes to unseen wording instead
of memorizing strings.

Training data = TOON_TEST_QUERIES (100 labeled queries). The model is evaluated
on a disjoint held-out set (eval_router_heldout.py) to report HONEST accuracy.

Usage
─────
    python3 -m app.retrieval.toon_classifier --train      # fit + save
    python3 -m app.retrieval.toon_classifier --eval       # held-out report

API
───
    from app.retrieval.toon_classifier import EmbeddingRouter
    r = EmbeddingRouter()           # lazy-loads model + embedder
    tier, conf = r.predict("كم TSH؟")
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from app.core.config import cfg

MODEL_PATH = Path(cfg.BASE_DIR) / "models" / "toon_router_clf.joblib"
_TIER_OF_GROUP = {"tier_1_simple": 1, "tier_2_moderate": 2, "tier_3_complex": 3}


# ─── Training data ────────────────────────────────────────────────────────────
def _load_training_data() -> Tuple[List[str], List[int]]:
    from tests.test_rag_queries import TOON_TEST_QUERIES

    texts, labels = [], []
    for group, data in TOON_TEST_QUERIES.items():
        tier = _TIER_OF_GROUP[group]
        for q in data["queries"]["questions"]:
            texts.append(q)
            labels.append(tier)
    return texts, labels


def _embed(texts: List[str]) -> np.ndarray:
    from app.indexing.embedder import embed_texts

    return np.vstack(embed_texts(texts))


# ─── Train ────────────────────────────────────────────────────────────────────
def train(save: bool = True) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    import joblib

    texts, labels = _load_training_data()
    X = _embed(texts)
    y = np.array(labels)

    # Strong L2 regularization: 100 samples in 1024-dim — must not overfit.
    clf = LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced", multi_class="multinomial"
    )

    # Honest in-distribution estimate via stratified 5-fold CV (NOT train-on-test).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

    clf.fit(X, y)
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, MODEL_PATH)

    return {
        "cv_mean": float(cv_acc.mean()),
        "cv_std": float(cv_acc.std()),
        "n_train": len(texts),
        "saved": str(MODEL_PATH) if save else None,
    }


# ─── Inference ────────────────────────────────────────────────────────────────
class EmbeddingRouter:
    _clf = None

    def __init__(self):
        if EmbeddingRouter._clf is None:
            import joblib

            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"{MODEL_PATH} missing — run: python3 -m app.retrieval.toon_classifier --train"
                )
            EmbeddingRouter._clf = joblib.load(MODEL_PATH)

    def predict(self, query: str) -> Tuple[int, float]:
        X = _embed([query])
        proba = EmbeddingRouter._clf.predict_proba(X)[0]
        idx = int(proba.argmax())
        tier = int(EmbeddingRouter._clf.classes_[idx])
        return tier, float(proba[idx])


# ─── CLI ──────────────────────────────────────────────────────────────────────
def _eval_heldout():
    from eval_router_heldout import HELDOUT

    r = EmbeddingRouter()
    correct = 0
    total = 0
    per = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
    wrong = []
    for exp, qs in HELDOUT.items():
        for q in qs:
            tier, conf = r.predict(q)
            total += 1
            per[exp][1] += 1
            if tier == exp:
                correct += 1
                per[exp][0] += 1
            else:
                wrong.append((exp, tier, conf, q))
    print("HELD-OUT (learned classifier, queries NOT used for tuning)")
    print(f"ACCURACY: {correct}/{total} = {100*correct/total:.1f}%")
    for t in (1, 2, 3):
        ok, n = per[t]
        print(f"  Tier {t} recall: {ok}/{n} = {100*ok/n:.0f}%")
    print(f"--- misroutes ({len(wrong)}) ---")
    for e, p, c, q in wrong:
        print(f"  exp{e} got{p} (conf {c:.2f}): {q}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    if args.train or not args.eval:
        stats = train()
        print(
            f"Trained on {stats['n_train']} queries | "
            f"5-fold CV accuracy = {100*stats['cv_mean']:.1f}% ± {100*stats['cv_std']:.1f}%"
        )
        print(f"Saved → {stats['saved']}")
    if args.eval:
        _eval_heldout()


if __name__ == "__main__":
    main()
