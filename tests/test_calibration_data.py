import pytest
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCalibrationDataCollector:
    """Tests for calibration data collection system."""

    @pytest.fixture
    def temp_data_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_log_inference_creates_record(self, temp_data_file):
        from app.calibration.data_collector import (
            CalibrationDataCollector,
            InferenceRecord,
        )

        collector = CalibrationDataCollector(data_path=Path(temp_data_file))

        record_id = collector.log_inference(
            query="test query",
            answer="test answer",
            grounding_score=0.8,
            retrieval_score=0.7,
            rerank_score=0.6,
            context_length=500,
            answer_length=100,
            top_similarity=0.75,
            confidence=0.7,
            is_emergency=False,
            language="arabic",
        )

        assert record_id.startswith("inf_")

        with open(temp_data_file, "r") as f:
            line = f.readline()
            data = json.loads(line)
            assert data["query"] == "test query"
            assert data["grounding_score"] == 0.8
            assert data["is_correct"] is None

    def test_get_unlabeled_records(self, temp_data_file):
        from app.calibration.data_collector import CalibrationDataCollector

        collector = CalibrationDataCollector(data_path=Path(temp_data_file))

        collector.log_inference(
            query="query1",
            answer="answer1",
            grounding_score=0.8,
            retrieval_score=0.7,
        )

        collector.log_inference(
            query="query2",
            answer="answer2",
            grounding_score=0.6,
            retrieval_score=0.5,
        )

        unlabeled = collector.get_unlabeled_records()
        assert len(unlabeled) == 2

    def test_label_record(self, temp_data_file):
        from app.calibration.data_collector import CalibrationDataCollector

        collector = CalibrationDataCollector(data_path=Path(temp_data_file))

        record_id = collector.log_inference(
            query="test",
            answer="test",
            grounding_score=0.8,
            retrieval_score=0.7,
        )

        success = collector.label_record(
            record_id, is_correct=True, note="correct answer"
        )

        assert success

        unlabeled = collector.get_unlabeled_records()
        assert len(unlabeled) == 0

        labeled = collector.get_labeled_records()
        assert len(labeled) == 1
        assert labeled[0]["is_correct"] == True

    def test_get_labeled_records(self, temp_data_file):
        from app.calibration.data_collector import CalibrationDataCollector

        collector = CalibrationDataCollector(data_path=Path(temp_data_file))

        id1 = collector.log_inference(
            query="q1", answer="a1", grounding_score=0.8, retrieval_score=0.7
        )
        id2 = collector.log_inference(
            query="q2", answer="a2", grounding_score=0.6, retrieval_score=0.5
        )

        collector.label_record(id1, True)

        labeled = collector.get_labeled_records()
        assert len(labeled) == 1
        assert labeled[0]["id"] == id1

    def test_get_stats(self, temp_data_file):
        from app.calibration.data_collector import CalibrationDataCollector

        collector = CalibrationDataCollector(data_path=Path(temp_data_file))

        id1 = collector.log_inference(
            query="q1", answer="a1", grounding_score=0.8, retrieval_score=0.7
        )
        id2 = collector.log_inference(
            query="q2", answer="a2", grounding_score=0.6, retrieval_score=0.5
        )

        collector.label_record(id1, True)

        stats = collector.get_stats()

        assert stats["total_inferences"] == 2
        assert stats["labeled"] == 1
        assert stats["unlabeled"] == 1
        assert stats["correct"] == 1
        assert stats["incorrect"] == 0

    def test_get_training_data_insufficient(self, temp_data_file):
        from app.calibration.data_collector import CalibrationDataCollector

        collector = CalibrationDataCollector(data_path=Path(temp_data_file))

        collector.log_inference(
            query="q1", answer="a1", grounding_score=0.8, retrieval_score=0.7
        )

        features, labels = collector.get_training_data()

        assert features is None
        assert labels is None

    def test_export_for_human_review(self, temp_data_file):
        from app.calibration.data_collector import CalibrationDataCollector

        collector = CalibrationDataCollector(data_path=Path(temp_data_file))

        collector.log_inference(
            query="q1", answer="a1", grounding_score=0.8, retrieval_score=0.7
        )
        collector.log_inference(
            query="q2", answer="a2", grounding_score=0.6, retrieval_score=0.5
        )

        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            export_path = f.name

        try:
            result_path = collector.export_for_human_review(Path(export_path))

            assert result_path.exists()

            with open(result_path, "r") as f:
                data = json.load(f)
                assert data["count"] == 2
                assert len(data["records"]) == 2
        finally:
            if os.path.exists(export_path):
                os.remove(export_path)

    def test_import_labels(self, temp_data_file):
        from app.calibration.data_collector import CalibrationDataCollector

        collector = CalibrationDataCollector(data_path=Path(temp_data_file))

        record_id = collector.log_inference(
            query="q1", answer="a1", grounding_score=0.8, retrieval_score=0.7
        )

        labeled_data = [{"id": record_id, "is_correct": True, "note": "Good answer"}]

        count = collector.import_labels(labeled_data)

        assert count == 1

        labeled = collector.get_labeled_records()
        assert labeled[0]["is_correct"] == True
        assert labeled[0]["evaluator_note"] == "Good answer"


class TestCalibrationIntegration:
    """Integration tests for calibration workflow."""

    def test_full_labeling_workflow(self):
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = f.name

        try:
            from app.calibration.data_collector import CalibrationDataCollector

            collector = CalibrationDataCollector(data_path=Path(temp_path))

            id1 = collector.log_inference(
                query="ما هي أعراض السكري؟",
                answer="أعراض السكري هي العطش والتبول",
                grounding_score=0.9,
                retrieval_score=0.85,
                confidence=0.8,
                language="arabic",
            )

            id2 = collector.log_inference(
                query="كيف يعالج ارتفاع الضغط؟",
                answer="علاج ارتفاع الضغط",
                grounding_score=0.7,
                retrieval_score=0.6,
                confidence=0.5,
                language="arabic",
            )

            collector.label_record(id1, True, "accurate")
            collector.label_record(id2, False, "incomplete")

            stats = collector.get_stats()

            assert stats["labeled"] == 2
            assert stats["correct"] == 1
            assert stats["incorrect"] == 1

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
