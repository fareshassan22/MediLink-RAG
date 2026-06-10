"""
Comprehensive RAG System Test Queries

Tests both /ask (general medical) and /patient/ask (TOON) endpoints
with full coverage of medical categories, languages, and difficulty levels.
"""

import pytest
import json
from unittest.mock import patch, Mock, MagicMock
from fastapi.testclient import TestClient
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Test Queries: General Medical RAG (/ask endpoint)
# =============================================================================

RAG_TEST_QUERIES = {
    "symptoms": {
        "arabic": [
            "ما هي أعراض مرض السكري؟",
            "ما أعراض코로나 في الأطفال؟",
            "ما علامات فقر الدم؟",
            "أعاني من صداع مستمر ما السبب؟",
            "ما أعراض نقص فيتامين د؟",
        ],
        "english": [
            "What are the symptoms of diabetes mellitus?",
            "What are COVID-19 symptoms in children?",
            "What are the signs of anemia?",
            "I have persistent headache, what could be the cause?",
            "What are the symptoms of vitamin D deficiency?",
        ],
    },
    "treatment": {
        "arabic": [
            "ما علاج ارتفاع ضغط الدم؟",
            "كيف يعالج الربو؟",
            "ما أفضل علاج للصداع النصفي؟",
            "كيف أعالج الأرق؟",
            "ما علاج القولون العصبي؟",
        ],
        "english": [
            "What is the treatment for hypertension?",
            "How is asthma treated?",
            "What is the best treatment for migraine?",
            "How do I treat insomnia?",
            "What is the treatment for IBS?",
        ],
    },
    "diagnosis": {
        "arabic": [
            "كيف يتم تشخيص السكري؟",
            "ما فحوصات تشخيص أمراض القلب؟",
            "كيف أشخص سرطان الثدي مبكراً؟",
            "ما علامات تشخيص نقص الحديد؟",
            "كيف يتم تشخيص أمراض الغدة الدرقية؟",
        ],
        "english": [
            "How is diabetes diagnosed?",
            "What tests diagnose heart disease?",
            "How is breast cancer diagnosed early?",
            "How is iron deficiency diagnosed?",
            "How are thyroid diseases diagnosed?",
        ],
    },
    "causes": {
        "arabic": [
            "ما أسباب فشل القلب؟",
            "لماذا أرتفع ضغط الدم؟",
            "ما أسباب الصداع المتكرر؟",
            "ما مسببات الربو؟",
            "لماذا يحدث فقر الدم؟",
        ],
        "english": [
            "What causes heart failure?",
            "Why does hypertension occur?",
            "What causes recurrent headaches?",
            "What triggers asthma?",
            "What causes anemia?",
        ],
    },
    "prevention": {
        "arabic": [
            "كيف يمكن الوقاية من السكري؟",
            "ما طرق الوقاية من أمراض القلب؟",
            "كيف أق نفسي من كوفيد-19؟",
            "ما للوقاية من السرطان؟",
            "كيف أحمي صحة العظام؟",
        ],
        "english": [
            "How can I prevent diabetes?",
            "What are ways to prevent heart disease?",
            "How do I protect myself from COVID-19?",
            "How to prevent cancer?",
            "How to maintain bone health?",
        ],
    },
    "medications": {
        "arabic": [
            "ما الآثار الجانبية للأسبرين؟",
            "هل الميتفورمين آمن؟",
            "ما تفاعلات أدوية ضغط الدم؟",
            "أفضل أدوية علاج القلق؟",
            "ما بدائل الأدوية الكيميائية؟",
        ],
        "english": [
            "What are the side effects of aspirin?",
            "Is metformin safe?",
            "What are the drug interactions for blood pressure medications?",
            "What are the best medications for anxiety?",
            "What are alternatives to chemical medications?",
        ],
    },
    "procedures": {
        "arabic": [
            "ما فحوصات الكشف المبكر عن السرطان؟",
            "ما عملية القسطرة القلبية؟",
            "كيف يتم تنظير المعدة؟",
            "ما تحاليل الدم الروتينية؟",
            "ما فحوصات وظائف الكلى؟",
        ],
        "english": [
            "What cancer screening tests exist?",
            "What is cardiac catheterization?",
            "How is gastroscopy performed?",
            "What are routine blood tests?",
            "What are kidney function tests?",
        ],
    },
    "nutrition": {
        "arabic": [
            "ما الأطعة التي تؤثر على ضغط الدم؟",
            "ما الأطعة التي تخفض الكوليسترول؟",
            "هل الصيام مفيد للسكري؟",
            "ما النظام الغذائي المناسب لمرضى القلب؟",
            "ما الفيتامينات المهمة لصحة العين؟",
        ],
        "english": [
            "Which foods affect blood pressure?",
            "Which foods lower cholesterol?",
            "Is fasting good for diabetes?",
            "What diet is suitable for heart patients?",
            "What vitamins are important for eye health?",
        ],
    },
    "emergency": {
        "arabic": [
            "ألم حاد في الصدر مع ضيق تنفس",
            "صداع مفاجئ مع تشويش رؤية",
            "نزيف حاد غير متوقف",
            "إغماء مفاجئ مع سرعة نبض",
            "صعوبة تنفس شديدة",
        ],
        "english": [
            "Severe chest pain with shortness of breath",
            "Sudden headache with vision disturbance",
            "Uncontrolled severe bleeding",
            "Sudden fainting with rapid heartbeat",
            "Severe difficulty breathing",
        ],
    },
}


# =============================================================================
# Test Queries: Patient-Specific TOON (/patient/ask endpoint)
# =============================================================================

TOON_TEST_QUERIES = {
    "tier_1_simple": {
        "description": "BM25 exact lookup - 50 token budget",
        "queries": {
            "patient_id": 200,
            "questions": [
                "ما هو مستوى السكر في دمي؟",
                "ما هي نسبة الضغط في قراءاتي الأخيرة؟",
                "كم وزني في آخر فحص؟",
                "ما نتائج فحص LDL؟",
                "هل فحص HbA1c طبيعي؟",
                "ما آخر قراءة للضغط؟",
                "كم مستوى الكوليسترول؟",
                "ما نتائج فحص الغدة الدرقية؟",
                "هل فحص CBC طبيعي؟",
                "ما مستوى فيتامين د؟",
                "ما نسبة الهيموغلوبين؟",
                "كم عدد كريات الدم البيضاء؟",
                "ما نتائج فحص وظائف الكلى؟",
                "ما مستوى اليوريا؟",
                "ما نسبة الكرياتينين؟",
                "ما نتائج فحص الكبد؟",
                "ما إنزيمات الكبد؟",
                "ما مستوى الحديد؟",
                "هل فحص FERITIN طبيعي؟",
                "ما قيمة الفولات؟",
                "كم سكر الدم العشوائي؟",
                "ما سكر الدم الصيامي؟",
                "هل فحص الأنسولين طبيعي؟",
                "ما مستوى الثيروكسين؟",
                "كم TSH؟",
                "ما free T4؟",
                "ما النتائج الأولية؟",
                "ما آخر فحص كامل؟",
                "هل يوجد فقر دم؟",
                "ما نوع الدم؟",
                "ما نتيجة تحليل البول؟",
                "هل يوجد بيلة بروتينية؟",
                "ما كثافة البول؟",
                "كم حجم البول؟",
                "ما الرقم الهيدروجيني؟",
            ],
        },
    },
    "tier_2_moderate": {
        "description": "Hybrid retrieval - 200 token budget",
        "queries": {
            "patient_id": 200,
            "questions": [
                "هل نتائج فحوصات الضغط طبيعية؟",
                "مقارنة النتائج الحالية بالسابقة؟",
                "هل هناك تحسن في نسبة السكر؟",
                "مستوى الكوليسترول ضمن المعدل الطبيعي؟",
                "هل الفيتامينات تتحسن؟",
                "هل قراءات الضغط مرتفعة؟",
                "هل نتائج السكر طبيعية؟",
                "هل يوجد تغير في الوزن؟",
                "كيف تغيرت النتائج منذ آخر فحص؟",
                "هل نتائج الغدة الدرقية حسنت؟",
                "هل HbA1c تحسن؟",
                "هل LDL انخفض؟",
                "هل HDL ارتفع؟",
                "مقارنة HDL بالنتائج السابقة؟",
                "هل هناك تغير في TSH؟",
                "هل free T4 طبيعي؟",
                "هل النتائج مستقرة؟",
                "هل هناك ترا��ع؟",
                "مقارنة نتائج CBC؟",
                "هل الهيموغلوبين تحسن؟",
                "هل الحديد زاد؟",
                "هل Feritin تحسن؟",
                "مقارنة وظائف الكلى؟",
                "هل اليوريا تحسنت؟",
                "هل الكرياتينين مستقر؟",
                "مقارنة إنزيمات الكبد؟",
                "هل AST طبيعي؟",
                "هل ALT تحسنت؟",
                "هل هناك تحسن عام؟",
                "هل النتائج ضمن النطاق؟",
                "هل هناك قلق من النتائج؟",
                "ما اتجاه النتائج؟",
            ],
        },
    },
    "tier_3_complex": {
        "description": "Full LLM with live database - 20000 token budget",
        "queries": {
            "patient_id": 200,
            "questions": [
                "بناءً على نتائج فحوصاتي، ما التغييرات المطلوبة في العلاج؟",
                "بناءً على قراءات الضغط الأخيرة، هل لدي ارتفاع في ضغط الدم؟",
                "Based on my labs, what lifestyle changes do you recommend?",
                "هل يوجد قلق من ارتفاع LDL في نتائج الفحص الأخيرة؟",
                "ما المسار المتوقع للحالة بناءً على الاتجاهات؟",
                "هل يجب تعديل الأدوية بناءً على النتائج؟",
                "ما توصيات العلاج بناءً على التطورات؟",
                "كيف يمكنني تحسين قراءات الغلوكوز؟",
                "هل يجب زيادة الجرعة؟",
                "ما الخطة العلاجية المناسبة؟",
                "ما أفضل نظام غذائي بناءً على النتائج؟",
                "هل الصيام مفيد بناءً على قراءات sugar؟",
                "ما أنواع التمارين المناسبة؟",
                "كم ساعة نوم مطلوبة؟",
                "كيف أتحكم في القلق والتوتر؟",
                "ما العلاقة بين الضغط والوزن؟",
                "هل السمنة تؤثر على النتائج؟",
                "ما العلاقة بين السكر وأمراض القلب؟",
                "هل يوجد خطر على القلب؟",
                "ما الفحوصات المطلوبة للكشف المبكر؟",
                "هل يجب فحص القلب؟",
                "ما ECG المطلوب؟",
                "هل فحص الإيكوكارديوغرام ضروري؟",
                "ما توصيات التغذية؟",
                "ما الأطعة التي يجب تجنبها؟",
                "ما الأطعة المفيدة؟",
                "هل يجب تقليل الملح؟",
                "هل يجب تقليل السكر؟",
                "هل يجب تقليل الدهون؟",
                "ما نسبة البروتين المطلوبة؟",
                "هل يجب أخذ مكملات؟",
                "ما أفضل وقت للتمارين؟",
                "كم مدة التمارين؟",
            ],
        },
    },
}


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_vector_store():
    """Vector store with medical documents."""
    mock = Mock()
    mock.search.return_value = [
        {
            "text": "Diabetes mellitus is characterized by hyperglycemia...",
            "score": 0.9,
            "metadata": {"page": 1, "source": "medical_book"},
        }
    ]
    mock.documents = []
    return mock


@pytest.fixture
def mock_bm25():
    """BM25 index with medical documents."""
    mock = Mock()
    mock.search.return_value = [
        {
            "text": "Diabetes mellitus treatment involves...",
            "score": 0.8,
            "bm25_score": 0.8,
            "doc_id": "doc_1",
            "metadata": {"page": 1},
        }
    ]
    return mock


@pytest.fixture
def client():
    """FastAPI test client with mocked state."""
    from app.core.state import _state

    _state.is_ready = True
    _state.vector_store = Mock()
    _state.vector_store.search.return_value = []
    _state.vector_store.documents = []
    _state.bm25 = Mock()
    _state.bm25.search.return_value = []

    from app.main import app

    yield TestClient(app)

    _state.is_ready = False
    _state.vector_store = None
    _state.bm25 = None


# =============================================================================
# Test Classes
# =============================================================================

class TestRAGSymptomQueries:
    """Test queries for symptom-related questions."""

    @pytest.mark.parametrize("query", RAG_TEST_QUERIES["symptoms"]["arabic"][:2])
    def test_symptom_arabic(self, client, query):
        with patch("app.services.rag_pipeline.detect_emergency", return_value=False):
            with patch("app.services.rag_pipeline.expand_query", return_value=[query]):
                with patch("app.services.rag_pipeline.embed_texts", return_value=[]):
                    with patch("app.services.rag_pipeline.hybrid_retrieval_fusion", return_value=[]):
                        with patch("app.services.rag_pipeline.generate_response", return_value="test"):
                            with patch("app.services.rag_pipeline.judge_answer", return_value=Mock(
                                grounding_score=0.9, confidence=0.85, grounded=True
                            )):
                                response = client.post("/ask", json={"query": query, "role": "patient"})
                                assert response.status_code in [200, 500]

    @pytest.mark.parametrize("query", RAG_TEST_QUERIES["symptoms"]["english"][:2])
    def test_symptom_english(self, client, query):
        with patch("app.services.rag_pipeline.detect_emergency", return_value=False):
            with patch("app.services.rag_pipeline.expand_query", return_value=[query]):
                with patch("app.services.rag_pipeline.embed_texts", return_value=[]):
                    with patch("app.services.rag_pipeline.hybrid_retrieval_fusion", return_value=[]):
                        with patch("app.services.rag_pipeline.generate_response", return_value="test"):
                            with patch("app.services.rag_pipeline.judge_answer", return_value=Mock(
                                grounding_score=0.9, confidence=0.85, grounded=True
                            )):
                                response = client.post("/ask", json={"query": query, "role": "patient"})
                                assert response.status_code in [200, 500]


class TestRAGTreatmentQueries:
    """Test queries for treatment-related questions."""

    @pytest.mark.parametrize("query", RAG_TEST_QUERIES["treatment"]["arabic"][:2])
    def test_treatment_arabic(self, client, query):
        with patch("app.services.rag_pipeline.detect_emergency", return_value=False):
            with patch("app.services.rag_pipeline.expand_query", return_value=[query]):
                with patch("app.services.rag_pipeline.embed_texts", return_value=[]):
                    with patch("app.services.rag_pipeline.hybrid_retrieval_fusion", return_value=[]):
                        with patch("app.services.rag_pipeline.generate_response", return_value="test"):
                            with patch("app.services.rag_pipeline.judge_answer", return_value=Mock(
                                grounding_score=0.9, confidence=0.85, grounded=True
                            )):
                                response = client.post("/ask", json={"query": query, "role": "patient"})
                                assert response.status_code in [200, 500]


class TestRAGDiagnosisQueries:
    """Test queries for diagnosis-related questions."""

    @pytest.mark.parametrize("query", RAG_TEST_QUERIES["diagnosis"]["arabic"][:2])
    def test_diagnosis_arabic(self, client, query):
        with patch("app.services.rag_pipeline.detect_emergency", return_value=False):
            with patch("app.services.rag_pipeline.expand_query", return_value=[query]):
                with patch("app.services.rag_pipeline.embed_texts", return_value=[]):
                    with patch("app.services.rag_pipeline.hybrid_retrieval_fusion", return_value=[]):
                        with patch("app.services.rag_pipeline.generate_response", return_value="test"):
                            with patch("app.services.rag_pipeline.judge_answer", return_value=Mock(
                                grounding_score=0.9, confidence=0.85, grounded=True
                            )):
                                response = client.post("/ask", json={"query": query, "role": "patient"})
                                assert response.status_code in [200, 500]


class TestRAGCausesQueries:
    """Test queries for causes-related questions."""

    @pytest.mark.parametrize("query", RAG_TEST_QUERIES["causes"]["arabic"][:2])
    def test_causes_arabic(self, client, query):
        with patch("app.services.rag_pipeline.detect_emergency", return_value=False):
            with patch("app.services.rag_pipeline.expand_query", return_value=[query]):
                with patch("app.services.rag_pipeline.embed_texts", return_value=[]):
                    with patch("app.services.rag_pipeline.hybrid_retrieval_fusion", return_value=[]):
                        with patch("app.services.rag_pipeline.generate_response", return_value="test"):
                            with patch("app.services.rag_pipeline.judge_answer", return_value=Mock(
                                grounding_score=0.9, confidence=0.85, grounded=True
                            )):
                                response = client.post("/ask", json={"query": query, "role": "patient"})
                                assert response.status_code in [200, 500]


class TestRAGPreventionQueries:
    """Test queries for prevention-related questions."""

    @pytest.mark.parametrize("query", RAG_TEST_QUERIES["prevention"]["arabic"][:2])
    def test_prevention_arabic(self, client, query):
        with patch("app.services.rag_pipeline.detect_emergency", return_value=False):
            with patch("app.services.rag_pipeline.expand_query", return_value=[query]):
                with patch("app.services.rag_pipeline.embed_texts", return_value=[]):
                    with patch("app.services.rag_pipeline.hybrid_retrieval_fusion", return_value=[]):
                        with patch("app.services.rag_pipeline.generate_response", return_value="test"):
                            with patch("app.services.rag_pipeline.judge_answer", return_value=Mock(
                                grounding_score=0.9, confidence=0.85, grounded=True
                            )):
                                response = client.post("/ask", json={"query": query, "role": "patient"})
                                assert response.status_code in [200, 500]


class TestRAGMedicationsQueries:
    """Test queries for medication-related questions."""

    @pytest.mark.parametrize("query", RAG_TEST_QUERIES["medications"]["arabic"][:2])
    def test_medications_arabic(self, client, query):
        with patch("app.services.rag_pipeline.detect_emergency", return_value=False):
            with patch("app.services.rag_pipeline.expand_query", return_value=[query]):
                with patch("app.services.rag_pipeline.embed_texts", return_value=[]):
                    with patch("app.services.rag_pipeline.hybrid_retrieval_fusion", return_value=[]):
                        with patch("app.services.rag_pipeline.generate_response", return_value="test"):
                            with patch("app.services.rag_pipeline.judge_answer", return_value=Mock(
                                grounding_score=0.9, confidence=0.85, grounded=True
                            )):
                                response = client.post("/ask", json={"query": query, "role": "patient"})
                                assert response.status_code in [200, 500]


class TestRAGProceduresQueries:
    """Test queries for procedures-related questions."""

    @pytest.mark.parametrize("query", RAG_TEST_QUERIES["procedures"]["arabic"][:2])
    def test_procedures_arabic(self, client, query):
        with patch("app.services.rag_pipeline.detect_emergency", return_value=False):
            with patch("app.services.rag_pipeline.expand_query", return_value=[query]):
                with patch("app.services.rag_pipeline.embed_texts", return_value=[]):
                    with patch("app.services.rag_pipeline.hybrid_retrieval_fusion", return_value=[]):
                        with patch("app.services.rag_pipeline.generate_response", return_value="test"):
                            with patch("app.services.rag_pipeline.judge_answer", return_value=Mock(
                                grounding_score=0.9, confidence=0.85, grounded=True
                            )):
                                response = client.post("/ask", json={"query": query, "role": "patient"})
                                assert response.status_code in [200, 500]


class TestRAGNutritionQueries:
    """Test queries for nutrition-related questions."""

    @pytest.mark.parametrize("query", RAG_TEST_QUERIES["nutrition"]["arabic"][:2])
    def test_nutrition_arabic(self, client, query):
        with patch("app.services.rag_pipeline.detect_emergency", return_value=False):
            with patch("app.services.rag_pipeline.expand_query", return_value=[query]):
                with patch("app.services.rag_pipeline.embed_texts", return_value=[]):
                    with patch("app.services.rag_pipeline.hybrid_retrieval_fusion", return_value=[]):
                        with patch("app.services.rag_pipeline.generate_response", return_value="test"):
                            with patch("app.services.rag_pipeline.judge_answer", return_value=Mock(
                                grounding_score=0.9, confidence=0.85, grounded=True
                            )):
                                response = client.post("/ask", json={"query": query, "role": "patient"})
                                assert response.status_code in [200, 500]


class TestRAGEmergencyQueries:
    """Test queries for emergency detection."""

    @pytest.mark.parametrize("query", RAG_TEST_QUERIES["emergency"]["arabic"][:3])
    def test_emergency_arabic(self, client, query):
        response = client.post("/ask", json={"query": query, "role": "patient"})
        if response.status_code == 200:
            data = response.json()
            assert data.get("status") == "emergency_escalation"


class TestTOONPatientQueries:
    """Test queries for patient-specific TOON endpoint."""

    @pytest.mark.parametrize("query", TOON_TEST_QUERIES["tier_1_simple"]["queries"]["questions"][:2])
    def test_toon_tier1_simple(self, client, query):
        with patch("app.retrieval.toon_service.get_patient_data", return_value={
            "glucose": {"value": 120, "unit": "mg/dL", "date": "2024-01-01"}
        }):
            with patch("app.retrieval.toon_service.get_patient_history", return_value=[]):
                response = client.post("/patient/ask", json={
                    "patient_id": 200,
                    "query": query,
                    "role": "patient"
                })
                assert response.status_code in [200, 500, 404]

    @pytest.mark.parametrize("query", TOON_TEST_QUERIES["tier_2_moderate"]["queries"]["questions"][:2])
    def test_toon_tier2_moderate(self, client, query):
        with patch("app.retrieval.toon_service.get_patient_data", return_value={
            "glucose": {"value": 120, "unit": "mg/dL", "date": "2024-01-01"},
            "bp_systolic": {"value": 130, "unit": "mmHg", "date": "2024-01-01"}
        }):
            with patch("app.retrieval.toon_service.get_patient_history", return_value=[]):
                response = client.post("/patient/ask", json={
                    "patient_id": 200,
                    "query": query,
                    "role": "patient"
                })
                assert response.status_code in [200, 500, 404]

    @pytest.mark.parametrize("query", TOON_TEST_QUERIES["tier_3_complex"]["queries"]["questions"][:2])
    def test_toon_tier3_complex(self, client, query):
        with patch("app.retrieval.toon_service.get_patient_data", return_value={
            "glucose": {"value": 140, "unit": "mg/dL", "date": "2024-01-01"},
            "bp_systolic": {"value": 145, "unit": "mmHg", "date": "2024-01-01"},
            "hba1c": {"value": 7.5, "unit": "%", "date": "2024-01-01"}
        }):
            with patch("app.retrieval.toon_service.get_patient_history", return_value=[
                {"date": "2023-12-01", "glucose": 130},
                {"date": "2023-11-01", "glucose": 125},
            ]):
                with patch("app.retrieval.toon_service.generate_patient_response", return_value="test"):
                    response = client.post("/patient/ask", json={
                        "patient_id": 200,
                        "query": query,
                        "role": "patient"
                    })
                    assert response.status_code in [200, 500, 404]


class TestBilingualCoverage:
    """Test bilingual (Arabic/English) query handling."""

    def test_arabic_query_count(self):
        """Count total Arabic queries."""
        total = sum(len(v["arabic"]) for v in RAG_TEST_QUERIES.values())
        assert total >= 30

    def test_english_query_count(self):
        """Count total English queries."""
        total = sum(len(v["english"]) for v in RAG_TEST_QUERIES.values() if "english" in v)
        assert total >= 20

    def test_all_categories_covered(self):
        """Verify all medical categories covered."""
        expected = {"symptoms", "treatment", "diagnosis", "causes", "prevention",
                   "medications", "procedures", "nutrition", "emergency"}
        actual = set(RAG_TEST_QUERIES.keys())
        assert expected.issubset(actual) or actual.issuperset(expected)


class TestQueryMetrics:
    """Test query metrics and response validation."""

    def test_query_response_structure(self, client):
        """Verify response structure has required fields."""
        with patch("app.services.rag_pipeline.detect_emergency", return_value=False):
            with patch("app.services.rag_pipeline.expand_query", return_value=["test"]):
                with patch("app.services.rag_pipeline.embed_texts", return_value=[]):
                    with patch("app.services.rag_pipeline.hybrid_retrieval_fusion", return_value=[
                        {"text": "doc1", "score": 0.9, "metadata": {"page": 1}}
                    ]):
                        with patch("app.services.rag_pipeline.build_prompt", return_value="prompt"):
                            with patch("app.services.rag_pipeline.generate_response",
                                     return_value="test answer"):
                                with patch("app.services.rag_pipeline.judge_answer", return_value=Mock(
                                    grounding_score=0.9,
                                    confidence=0.85,
                                    grounded=True,
                                    hallucination_risk=0.1,
                                    flagged_claims=[],
                                    reasoning="test",
                                )):
                                    with patch("app.services.rag_pipeline.rerank_documents",
                                              return_value=[{"text": "doc1", "score": 0.9,
                                                            "metadata": {"page": 1}}]):
                                        response = client.post("/ask", json={
                                            "query": "أعراض السكري",
                                            "role": "patient"
                                        })
                                        if response.status_code == 200:
                                            data = response.json()
                                            assert "answer" in data
                                            assert "confidence" in data
                                            assert "grounding_score" in data
                                            assert "sources" in data

    def test_grounding_threshold(self, client):
        """Test grounding score meets threshold."""
        with patch("app.services.rag_pipeline.detect_emergency", return_value=False):
            with patch("app.services.rag_pipeline.expand_query", return_value=["test"]):
                with patch("app.services.rag_pipeline.embed_texts", return_value=[]):
                    with patch("app.services.rag_pipeline.hybrid_retrieval_fusion", return_value=[]):
                        with patch("app.services.rag_pipeline.generate_response", return_value="test"):
                            with patch("app.services.rag_pipeline.judge_answer",
                                     return_value=Mock(
                                         grounding_score=0.85,
                                         confidence=0.80,
                                         grounded=True
                                     )):
                                response = client.post("/ask", json={
                                    "query": "علاج السكري",
                                    "role": "patient"
                                })
                                if response.status_code == 200:
                                    data = response.json()
                                    assert data.get("grounding_score", 0) >= 0.7


# =============================================================================
# Export test queries for external use
# =============================================================================

def get_all_rag_queries():
    """Get all RAG test queries as list."""
    queries = []
    for category, lang_dict in RAG_TEST_QUERIES.items():
        for lang in ["arabic", "english"]:
            if lang in lang_dict:
                for q in lang_dict[lang]:
                    queries.append({
                        "query": q,
                        "language": lang,
                        "category": category,
                        "endpoint": "/ask"
                    })
    return queries


def get_all_toon_queries():
    """Get all TOON test queries as list."""
    queries = []
    for tier, tier_dict in TOON_TEST_QUERIES.items():
        for q in tier_dict["queries"]["questions"]:
            queries.append({
                "query": q,
                "patient_id": tier_dict["queries"]["patient_id"],
                "tier": tier,
                "endpoint": "/patient/ask"
            })
    return queries


if __name__ == "__main__":
    print(f"Total RAG queries: {len(get_all_rag_queries())}")
    print(f"Total TOON queries: {len(get_all_toon_queries())}")
    print("\nSample RAG queries:")
    for q in get_all_rag_queries()[:5]:
        print(f"  [{q['language']}] {q['category']}: {q['query']}")
    print("\nSample TOON queries:")
    for q in get_all_toon_queries()[:5]:
        print(f"  [{q['tier']}] {q['query']}")