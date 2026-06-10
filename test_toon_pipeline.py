"""Test both Medical RAG pipeline and Patient TOON pipeline."""
import uuid
import sys

print("=" * 60)
print("TESTING MEDICAL RAG PIPELINE")
print("=" * 60)

try:
    from app.services.rag_pipeline import RAGPipeline
    print("[OK] RAGPipeline imported successfully")
except Exception as e:
    print(f"[FAIL] RAGPipeline import failed: {e}")
    sys.exit(1)

try:
    from app.indexing.vector_store import VectorStore
    from app.indexing.bm25_index import BM25Index
    print("[OK] Indexing modules imported successfully")
except Exception as e:
    print(f"[FAIL] Indexing modules import failed: {e}")

try:
    from app.retrieval.query_expansion import expand_query
    from app.retrieval.query_translator import translate_query
    print("[OK] Query expansion/translation imported successfully")
except Exception as e:
    print(f"[FAIL] Query expansion import failed: {e}")

try:
    from app.generation.groq_client import generate_response
    print("[OK] Groq client imported successfully")
except Exception as e:
    print(f"[FAIL] Groq client import failed: {e}")

try:
    from app.safety.emergency_detector import detect_emergency
    from app.safety.content_filter import contains_sensitive_content
    from app.safety.judge import judge_answer
    print("[OK] Safety modules imported successfully")
except Exception as e:
    print(f"[FAIL] Safety modules import failed: {e}")

print("\n" + "=" * 60)
print("TESTING PATIENT TOON PIPELINE")
print("=" * 60)

try:
    from app.retrieval.toon_router import route, detect_language, RouterDecision
    print("[OK] TOON router imported successfully")
    
    test_lang = detect_language("What is my blood type?")
    print(f"  - Language detection: 'What is my blood type?' → {test_lang}")
    
    test_lang_ar = detect_language("ما هو فصيلة دمي؟")
    print(f"  - Language detection: 'ما هو فصيلة دمي؟' → {test_lang_ar}")
    
    decision = route("what is my blood type", uuid.uuid4())
    print(f"  - Route 'what is my blood type' → Tier {decision.tier}")
except Exception as e:
    print(f"[FAIL] TOON router test failed: {e}")

try:
    from app.retrieval.toon import (
        classify, search_bm25, search_hybrid, fetch_live_context,
        load_patient_index, index_patient, TOKEN_BUDGETS, TierDecision
    )
    print("[OK] TOON core module imported successfully")
    print(f"  - Token budgets: {TOKEN_BUDGETS}")
except Exception as e:
    print(f"[FAIL] TOON core import failed: {e}")

try:
    from app.retrieval.toon_data import (
        fetch_profile, fetch_vital_signs, fetch_appointments,
        fetch_diagnoses, fetch_prescriptions, fetch_lab_orders,
        fetch_payments, fetch_all_patient_chunks, index_patient_data,
        load_patient_index as load_index_data
    )
    print("[OK] TOON data module imported successfully")
except Exception as e:
    print(f"[FAIL] TOON data import failed: {e}")

try:
    from app.retrieval.toon_service import PatientRAGService, PipelineResult, patient_rag_service
    print("[OK] PatientRAGService imported successfully")
except Exception as e:
    print(f"[FAIL] PatientRAGService import failed: {e}")

try:
    from app.generation.prompts import build_toon_prompt, build_prompt, COLUMN_WHITELIST, _TIER3_PATIENT, _DOCTOR_SYSTEM
    print("[OK] Prompt builder imported successfully")
    
    test_prompt = build_toon_prompt("What is my blood type?", "blood_type: A+", "patient")
    print(f"  - build_toon_prompt generated {len(test_prompt)} chars")
except Exception as e:
    print(f"[FAIL] Prompt builder test failed: {e}")

print("\n" + "=" * 60)
print("TESTING TIER CLASSIFICATION")
print("=" * 60)

test_queries = [
    ("what is my blood type", 1, "T1 exact lookup"),
    ("when is my next appointment", 1, "T1 exact lookup"),
    ("show me my recent prescriptions", 2, "T2 semantic request"),
    ("any changes in my blood pressure", 2, "T2 semantic request"),
    ("why do I feel tired after taking medication", 3, "T3 complex reasoning"),
    ("should I be worried about my test results", 3, "T3 complex reasoning"),
]

try:
    from app.retrieval.toon_router import route
    all_passed = True
    for query, expected_tier, desc in test_queries:
        decision = route(query, uuid.uuid4())
        status = "✓" if decision.tier == expected_tier else "✗"
        if decision.tier != expected_tier:
            all_passed = False
        print(f"  {status} '{query}' → Tier {decision.tier} (expected {expected_tier}) - {desc}")
    
    if all_passed:
        print("[OK] All tier classifications passed!")
    else:
        print("[FAIL] Some tier classifications failed")
except Exception as e:
    print(f"[FAIL] Tier classification test failed: {e}")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED")
print("=" * 60)