def build_prompt(query, context, role="patient"):

    if role == "doctor":
        instruction = (
            "You are responding to a medical professional.\n"
            "Use precise clinical terminology. Include relevant mechanisms, "
            "differential diagnoses, or treatment protocols when supported by the context.\n"
            "Cite which source each claim comes from (e.g. [Source 1]).\n"
            "If the context contains conflicting information, acknowledge both views "
            "and note the discrepancy."
        )
    else:
        instruction = (
            "You are responding to a patient.\n"
            "Use simple, clear Arabic. Avoid medical jargon unless you explain it. "
            "Be reassuring but honest.\n"
            "If you mention a medication, note that dosage must be confirmed by a doctor.\n"
            "If the context contains conflicting information, present the most common view "
            "and advise consulting a healthcare provider."
        )

    # Label each context source so the LLM can reference them
    labeled_parts = []
    for i, part in enumerate(context.split("\n\n"), 1):
        if part.strip():
            labeled_parts.append(f"[Source {i}]: {part.strip()}")
    labeled_context = "\n\n".join(labeled_parts) if labeled_parts else context

    prompt = f"""You are MediLink, a trusted medical AI assistant.

RULES — follow ALL strictly:
1. Answer ONLY from the provided sources below. Never use outside knowledge.
2. If the sources do not contain enough information, say:
   "لا تتوفر معلومات كافية في المصادر المتاحة للإجابة على هذا السؤال."
3. Do NOT repeat the same point more than once.
4. Structure your answer: use short paragraphs or a brief numbered list when listing items (symptoms, causes, steps).
5. Keep the answer concise — maximum 6 key points.
6. If the answer involves medications or treatments, add: "يجب استشارة الطبيب قبل تناول أي دواء."
7. If the answer involves symptoms that could indicate a serious condition, add: "في حالة استمرار الأعراض أو تفاقمها، يُرجى مراجعة الطبيب فوراً."
8. End with: "هذه المعلومات للتثقيف فقط ولا تغني عن استشارة طبيب مختص."

{instruction}

Sources:
{labeled_context}

Question: {query}

أجب باللغة العربية:"""

    return prompt


COLUMN_WHITELIST = """
AVAILABLE DATA SOURCES AND COLUMNS:
You MUST answer ONLY from the data provided below. Do NOT use outside knowledge.
Do NOT make up values. If the information is not in the data, say so.
Table: profiles — full_name, email, phone, gender, date_of_birth, city, country
Table: patients — blood_type, height_cm, weight_kg, allergies, chronic_diseases,
    emergency_contact_name, emergency_contact_phone, insurance_provider, insurance_number
Table: vital_signs — temperature_c, blood_pressure_systolic, blood_pressure_diastolic,
    heart_rate, respiratory_rate, oxygen_saturation, blood_glucose, bmi, weight_kg, height_cm
Table: appointments — scheduled_at, end_at, status, reason_for_visit, symptoms, notes, visit_fee, is_first_visit
Table: diagnoses — diagnosis_name, icd_code, is_primary, notes
Table: prescriptions — medicine_name, dosage, frequency, duration, route, instructions, quantity, notes
Table: prescription_items — medicine_name, dosage, frequency, duration, route, instructions, quantity
Table: lab_test_orders — test_name, test_code, status, instructions, ordered_at
Table: lab_test_results — result_summary, result_details, completed_at
Table: payments — amount, currency, payment_method, status, transaction_reference, paid_at
"""

_DOCTOR_SYSTEM = """You are MediLink Clinical Assistant, supporting a licensed medical professional.
Answer ONLY from the patient data provided below — never invent values or use outside knowledge.
Use precise clinical terminology; you may surface any field present in the data (vitals, labs,
diagnoses, ICD codes, medications, doctor notes, billing). Cite the source section for each claim
(e.g. [medical_record], [lab_test]). If the data contains conflicting information, note the discrepancy.
Do NOT add patient-facing safety disclaimers — the reader is a clinician.
If the data does not contain the answer, state that explicitly."""


_DOCTOR_SUMMARY_EN = """You are MediLink Clinical Assistant, preparing a structured case summary for a clinician.
Summarize ONLY from the patient data provided below — never invent values or use outside knowledge.
Use precise clinical terminology and cite source sections (e.g. [diagnosis], [lab_test]).
Produce the summary under these headings, omitting any heading with no data:
1. Demographics & key identifiers
2. Active problems / diagnoses (with ICD codes)
3. Current medications
4. Recent vitals
5. Recent lab results
6. Appointments (recent & upcoming)
7. Billing / payments
8. Clinical notes & follow-up
Do NOT add patient-facing safety disclaimers — the reader is a clinician."""


_DOCTOR_SUMMARY_AR = """You are MediLink Clinical Assistant, preparing a structured case summary for a clinician.
Summarize ONLY from the patient data provided below — never invent values or use outside knowledge.
Use precise clinical terminology and cite source sections (e.g. [diagnosis], [lab_test]).
Write the summary in Arabic under these headings, omitting any heading with no data:
1. البيانات الديموغرافية والمعرّفات الأساسية
2. المشاكل/التشخيصات النشطة (مع رموز ICD)
3. الأدوية الحالية
4. أحدث العلامات الحيوية
5. أحدث نتائج المختبر
6. المواعيد (الحديثة والقادمة)
7. الفوترة/المدفوعات
8. الملاحظات السريرية والمتابعة
Do NOT add patient-facing safety disclaimers — the reader is a clinician."""


def build_doctor_summary_prompt(context: str, language: str = "en") -> str:
    """Build a structured full-history clinical summary prompt for a doctor.

    Args:
        context: The patient's complete history (from fetch_full_context).
        language: 'ar' for Arabic, 'en' for English summary.
    """
    system = _DOCTOR_SUMMARY_AR if language == "ar" else _DOCTOR_SUMMARY_EN
    reply_lang = "اكتب الملخص باللغة العربية:" if language == "ar" else "Write the summary in English:"
    return f"{system}{COLUMN_WHITELIST}\n\nPATIENT DATA:\n{context}\n\n{reply_lang}"

_TIER3_PATIENT_EN = """You are MediLink Patient Assistant.
HARD RULE: Answer ONLY using facts that appear verbatim in the PATIENT DATA below.
If the answer is not in the data, reply exactly: "I could not find that information in your records."
NEVER invent, estimate, or infer values (dates, dosages, results, numbers) that are not explicitly present.
Every value you state must be copied from the data, not generated.

Use clear, simple language. Avoid jargon unless explained.
If you mention a medication, note dosage must be confirmed by a doctor.
If symptoms could indicate a serious condition, add: "If symptoms persist or worsen, please see your doctor immediately."
End with: "This information is for educational purposes only and does not replace professional medical advice." """

_TIER3_PATIENT_AR = """You are MediLink Patient Assistant.
HARD RULE — READ FIRST: Answer ONLY using facts that appear verbatim in the PATIENT DATA below.
Every value you state (dates, dosages, lab results, numbers, names) MUST be copied directly from the data — never invent, estimate, round, or infer.
If the answer is not present in the data, reply EXACTLY in Arabic: "لم أجد هذه المعلومات في سجلاتك." and stop.

Use clear Arabic. Avoid medical jargon unless explained.
If you mention a medication, note dosage must be confirmed by a doctor.
If symptoms could indicate a serious condition, add: "في حالة استمرار الأعراض، يُرجى مراجعة الطبيب فورا."
End with: "هذه المعلومات للتثقيف فقط ولا تغني عن استشارة طبيب مختص." """


def build_toon_prompt(query: str, context: str, role: str = "patient", language: str = "ar") -> str:
    """Build prompt for TOON Tier 3 patient queries.

    Args:
        query: The patient's question.
        context: Retrieved patient data context.
        role: 'patient' or 'doctor'.
        language: 'ar' for Arabic, 'en' for English response.
    """
    if role == "doctor":
        system = _DOCTOR_SYSTEM
    elif language == "ar":
        system = _TIER3_PATIENT_AR
    else:
        system = _TIER3_PATIENT_EN

    reply_lang = "أجب باللغة العربية:" if language == "ar" else "Answer in English:"
    return f"{system}{COLUMN_WHITELIST}\n\nPATIENT DATA:\n{context}\n\nQUESTION: {query}\n\n{reply_lang}"


_SIMPLE_FORMAT_EN = (
    "You are MediLink Patient Assistant. "
    "Extract and present ONLY the information relevant to the patient's question. "
    "Keep it brief and clear. Do NOT list everything — only what answers the query. "
    'If the data does not contain the answer, say "I could not find that information in your records." '
    'End with: "This information is for educational purposes only."'
)

_SIMPLE_FORMAT_AR = (
    "أنت مساعد MediLink الطبي. "
    "استخرج وقدم فقط المعلومات المتعلقة بسؤال المريض. "
    "اجعله简短 وواضح. لا تسرد كل شيء — فقط ما يجيب على السؤال. "
    'إذا كانت البيانات لا تحتوي على الإجابة، قل "لم أجد هذه المعلومات في سجلاتك." '
    'End with: "هذه المعلومات للتثقيف فقط."'
)


def build_simple_prompt(query: str, context: str, language: str = "ar") -> str:
    """Build a simple formatting prompt for T1/T2 tiers.
    
    Extracts relevant info from raw retrieved data and formats it as a clean answer.
    Faster than full T3 prompt since queries are simpler.
    """
    system = _SIMPLE_FORMAT_AR if language == "ar" else _SIMPLE_FORMAT_EN
    reply_lang = "أجب باللغة العربية:" if language == "ar" else "Answer in English:"
    return f"{system}\n\nPATIENT DATA:\n{context}\n\nQUESTION: {query}\n\n{reply_lang}"


# ── Grounding gate ────────────────────────────────────────────────────────────
# Deterministic, judge-independent check: every concrete VALUE the model emits
# (numbers, dates, decimals) must appear verbatim in the retrieved context.
# Values that don't are "ungrounded" — a strong hallucination signal for T3.
import re as _re

_VALUE_RE = _re.compile(
    r"\d{4}-\d{2}-\d{2}"          # ISO dates
    r"|\d{1,2}:\d{2}"            # times
    r"|\d+(?:[.,]\d+)?"          # integers / decimals
)


def _normalize_digits(text: str) -> str:
    """Map Arabic-Indic digits to ASCII so value matching is script-agnostic."""
    trans = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
    return text.translate(trans)


def ungrounded_values(answer: str, context: str) -> list[str]:
    """Return concrete values present in `answer` but absent from `context`.

    Only numeric/date tokens are checked — these are the values a model
    fabricates (wrong dates, made-up dosages, invented lab numbers). Prose is
    ignored. An empty list means every value the answer states is supported.
    """
    if not answer or not context:
        return []
    ctx = _normalize_digits(context)
    ans = _normalize_digits(answer)
    missing = []
    for val in _VALUE_RE.findall(ans):
        # Ignore trivially short numbers (e.g. "1", "2") that are list markers
        if len(val.replace(".", "").replace(",", "").replace(":", "").replace("-", "")) < 2:
            continue
        if val not in ctx:
            missing.append(val)
    return missing


_REGEN_SUFFIX_EN = (
    "\n\nIMPORTANT: Your previous answer contained values that are NOT in the patient data. "
    "Re-answer using ONLY values copied verbatim from the data. If a value is not present, "
    'say "I could not find that information in your records." Do not invent any number or date.'
)
_REGEN_SUFFIX_AR = (
    "\n\nمهم: إجابتك السابقة تضمنت قيمًا غير موجودة في بيانات المريض. "
    "أعد الإجابة باستخدام القيم المنسوخة حرفيًا من البيانات فقط. إذا لم تكن القيمة موجودة، "
    'قل "لم أجد هذه المعلومات في سجلاتك." لا تخترع أي رقم أو تاريخ.'
)


def grounding_regen_suffix(language: str = "ar") -> str:
    """Instruction appended to the prompt for a single stricter regeneration."""
    return _REGEN_SUFFIX_AR if language == "ar" else _REGEN_SUFFIX_EN