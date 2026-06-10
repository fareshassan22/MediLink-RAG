#!/usr/bin/env python3
"""Live end-to-end test: real doctor questions through the full TOON pipeline.

Runs realistic clinical questions a doctor would ask about a real patient,
straight through patient_rag_service.run_doctor() — full Supabase history +
Groq generation. Prints the actual generated answers, latency and sources so
you can judge real answer quality (not just retrieval metrics).

Usage:
    python3 demo_doctor_questions.py [patient_id]
"""
from __future__ import annotations

import sys
import time

from app.retrieval.toon_service import patient_rag_service

PATIENT_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 169

DOCTOR_QUESTIONS = [
    "ما ملخص الحالة الطبية الكاملة لهذا المريض؟",
    "ما التشخيصات المسجلة وهل هناك حالة مزمنة؟",
    "ما الأدوية الموصوفة حالياً وهل هناك تعارض محتمل بينها؟",
    "ما الفحوصات المخبرية المطلوبة وما حالتها؟",
    "بناءً على تاريخ المريض، ما خطة المتابعة التي توصي بها؟",
    "What are the key clinical concerns for this patient based on their history?",
]


def main():
    print(f"\n{'='*72}\nLIVE DOCTOR Q&A — patient {PATIENT_ID}\n{'='*72}")
    for i, q in enumerate(DOCTOR_QUESTIONS, 1):
        t0 = time.time()
        res = patient_rag_service.run_doctor(query=q, patient_id=PATIENT_ID, mode="ask")
        dt = time.time() - t0
        print(f"\n[{i}] Q: {q}")
        print(f"    status={res.status}  conf={res.confidence}  latency={dt:.1f}s")
        print(f"    A: {res.answer.strip()}")
        print("-" * 72)
        # Respect hosted-LLM tokens-per-minute window between questions.
        if i < len(DOCTOR_QUESTIONS):
            time.sleep(60)


if __name__ == "__main__":
    main()
