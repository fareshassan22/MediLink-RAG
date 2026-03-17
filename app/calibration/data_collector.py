"""
Calibration Data Collection System for MediLink RAG.

This module provides functionality to collect real inference data from production
for training and improving the confidence calibration model.

Data Collection Pipeline:
1. Log inference data (features + predictions) to calibration_data.jsonl
2. Human evaluation interface for labeling correctness
3. Retrain calibrator when enough labeled data is collected
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import numpy as np

from app.core.config import cfg


@dataclass
class InferenceRecord:
    """Single inference record for calibration training."""

    timestamp: str
    query: str
    answer: str

    grounding_score: float
    retrieval_score: float
    rerank_score: float
    context_length: int
    answer_length: int
    top_similarity: float

    confidence: float

    is_emergency: bool
    language: str

    is_correct: Optional[bool] = None
    evaluator_note: Optional[str] = None


class CalibrationDataCollector:
    """Collects inference data for calibration model training."""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or (cfg.DATA_DIR / "calibration_data.jsonl")
        self.data_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.data_path.exists():
            self.data_path.touch()

    def log_inference(
        self,
        query: str,
        answer: str,
        grounding_score: float,
        retrieval_score: float,
        rerank_score: float = 0.0,
        context_length: int = 0,
        answer_length: int = 0,
        top_similarity: float = 0.0,
        confidence: float = 0.0,
        is_emergency: bool = False,
        language: str = "arabic",
    ) -> str:
        """Log a single inference for later labeling."""

        record = InferenceRecord(
            timestamp=datetime.now().isoformat(),
            query=query,
            answer=answer,
            grounding_score=grounding_score,
            retrieval_score=retrieval_score,
            rerank_score=rerank_score,
            context_length=context_length,
            answer_length=answer_length,
            top_similarity=top_similarity,
            confidence=confidence,
            is_emergency=is_emergency,
            language=language,
        )

        record_id = f"inf_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        with open(self.data_path, "a", encoding="utf-8") as f:
            line = json.dumps({"id": record_id, **asdict(record)})
            f.write(line + "\n")

        return record_id

    def get_unlabeled_records(self) -> List[Dict]:
        """Get all records that haven't been labeled yet."""
        records = []

        if not self.data_path.exists():
            return records

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("is_correct") is None:
                    records.append(record)

        return records

    def get_labeled_records(self) -> List[Dict]:
        """Get all records that have been labeled."""
        records = []

        if not self.data_path.exists():
            return records

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("is_correct") is not None:
                    records.append(record)

        return records

    def label_record(
        self,
        record_id: str,
        is_correct: bool,
        note: Optional[str] = None,
    ) -> bool:
        """Label a specific record with correctness and optional note."""

        if not self.data_path.exists():
            return False

        updated_records = []
        found = False

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("id") == record_id:
                    record["is_correct"] = is_correct
                    record["evaluator_note"] = note
                    record["labeled_at"] = datetime.now().isoformat()
                    found = True
                updated_records.append(record)

        if found:
            with open(self.data_path, "w", encoding="utf-8") as f:
                for record in updated_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return found

    def get_training_data(self) -> tuple:
        """Get training data (features, labels) for calibrator."""

        labeled = self.get_labeled_records()

        if len(labeled) < cfg.CALIBRATION_MIN_SAMPLES:
            return None, None

        features = []
        labels = []

        for record in labeled:
            if record.get("is_correct") is None:
                continue

            feature_vec = [
                record.get("grounding_score", 0.0),
                record.get("retrieval_score", 0.0),
                record.get("rerank_score", 0.0),
                float(record.get("context_length", 0)),
                float(record.get("answer_length", 0)),
                record.get("top_similarity", 0.0),
            ]

            features.append(feature_vec)
            labels.append(1 if record["is_correct"] else 0)

        if len(features) < cfg.CALIBRATION_MIN_SAMPLES:
            return None, None

        return np.array(features), np.array(labels)

    def get_stats(self) -> Dict:
        """Get statistics about collected data."""

        total = 0
        labeled = 0
        unlabeled = 0
        correct = 0
        incorrect = 0

        if self.data_path.exists():
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    total += 1
                    record = json.loads(line)

                    if record.get("is_correct") is not None:
                        labeled += 1
                        if record["is_correct"]:
                            correct += 1
                        else:
                            incorrect += 1
                    else:
                        unlabeled += 1

        return {
            "total_inferences": total,
            "labeled": labeled,
            "unlabeled": unlabeled,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": correct / labeled if labeled > 0 else 0.0,
            "min_samples_required": cfg.CALIBRATION_MIN_SAMPLES,
            "ready_for_training": labeled >= cfg.CALIBRATION_MIN_SAMPLES,
        }

    def export_for_human_review(self, output_path: Optional[Path] = None) -> Path:
        """Export unlabeled records for human evaluation."""

        output_path = output_path or (cfg.RESULTS_DIR / "human_review.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        unlabeled = self.get_unlabeled_records()

        export_data = {
            "exported_at": datetime.now().isoformat(),
            "count": len(unlabeled),
            "records": [
                {
                    "id": r["id"],
                    "query": r["query"],
                    "answer": r["answer"],
                    "grounding_score": r["grounding_score"],
                    "retrieval_score": r["retrieval_score"],
                    "confidence": r["confidence"],
                    "language": r["language"],
                }
                for r in unlabeled
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        return output_path

    def import_labels(self, labeled_data: List[Dict]) -> int:
        """Import labels from human review file."""

        count = 0
        for item in labeled_data:
            record_id = item.get("id")
            is_correct = item.get("is_correct")
            note = item.get("note")

            if record_id and is_correct is not None:
                if self.label_record(record_id, is_correct, note):
                    count += 1

        return count


def log_inference_for_calibration(
    query: str,
    answer: str,
    grounding_score: float,
    retrieval_score: float,
    rerank_score: float = 0.0,
    context_length: int = 0,
    answer_length: int = 0,
    top_similarity: float = 0.0,
    confidence: float = 0.0,
    is_emergency: bool = False,
    language: str = "arabic",
) -> str:
    """Convenience function to log inference data."""

    collector = CalibrationDataCollector()
    return collector.log_inference(
        query=query,
        answer=answer,
        grounding_score=grounding_score,
        retrieval_score=retrieval_score,
        rerank_score=rerank_score,
        context_length=context_length,
        answer_length=answer_length,
        top_similarity=top_similarity,
        confidence=confidence,
        is_emergency=is_emergency,
        language=language,
    )


def get_calibration_stats() -> Dict:
    """Get current calibration data statistics."""
    collector = CalibrationDataCollector()
    return collector.get_stats()


def prepare_calibration_training() -> Optional[tuple]:
    """Prepare training data for calibration model retraining."""
    collector = CalibrationDataCollector()
    features, labels = collector.get_training_data()
    return (features, labels) if features is not None else None


if __name__ == "__main__":
    stats = get_calibration_stats()
    print("Calibration Data Statistics:")
    print(json.dumps(stats, indent=2))
