from typing import List
import re

_ARABIC_TO_ENGLISH = {
    # Symptoms
    "ألم": "pain",
    "صدر": "chest",
    "صداع": "headache",
    "سعال": "cough",
    "حمى": "fever",
    "حرارة": "fever",
    "دوخة": "dizziness",
    "غثيان": "nausea",
    "تقيؤ": "vomiting",
    "إسهال": "diarrhea",
    "إمساك": "constipation",
    "تعب": "fatigue",
    "إرهاق": "exhaustion",
    "ضعف": "weakness",
    "فقدان": "loss",
    "وزن": "weight",
    "شهية": "appetite",
    "نوم": "sleep",
    "قلق": "anxiety",
    "اكتئاب": "depression",
    # Body parts
    "قلب": "heart",
    "كبد": "liver",
    "كلية": "kidney",
    "رئة": "lung",
    "معدة": "stomach",
    "أمعاء": "intestine",
    "دم": "blood",
    "عظم": "bone",
    "مفصل": "joint",
    "عضلة": "muscle",
    "جلد": "skin",
    "عين": "eye",
    "أذن": "ear",
    "أنف": "nose",
    "حلق": "throat",
    "غدة": "gland",
    "غدة درقية": "thyroid",
    "بنكرياس": "pancreas",
    # Diseases
    "سكري": "diabetes",
    "ضغط": "blood pressure",
    "ضغط الدم": "blood pressure",
    "ارتفاع ضغط الدم": "hypertension",
    "قلب": "heart",
    "أمراض القلب": "heart disease",
    "علاقة": "relationship",
    "سرطان": "cancer",
    "ورم": "tumor",
    "عدوى": "infection",
    "التهاب": "inflammation",
    "ربو": "asthma",
    "فقر الدم": "anemia",
    "أنيميا": "anemia",
    "فييتأمين د": "vitamin D",
    "فييتامين د": "vitamin D",
    "فييامين د": "vitamin D",
    "فييتأمين": "vitamin",
    "فييتامين": "vitamin",
    "فييامين": "vitamin",
    "vitamin د": "vitamin D",
    "نقص": "deficiency",
    "أنفلونزا": "influenza",
    "برد": "cold",
    " tuberculosis": "TB",
    "إيدز": "HIV",
    "كلى": "kidney",
    "كبد": "liver",
    "صدر": "pulmonary",
    "جهاز هضمي": "digestive",
    "قولون": "colon",
    # Treatments
    "علاج": "treatment",
    "دواء": "medication",
    "أدوية": "medications",
    "جراحة": "surgery",
    "عملية": "operation",
    "علاج كيميائي": "chemotherapy",
    "إشعاع": "radiation",
    "علاج طبيعي": "physical therapy",
    "تمارين": "exercise",
    "حمية": "diet",
    "نظام غذائي": "diet",
    # Medications
    "أنسولين": "insulin",
    "باراسيتامول": "paracetamol",
    "أسيتامينوفين": "acetaminophen",
    "إيبوبروفين": "ibuprofen",
    "أسبرين": "aspirin",
    "مضاد حيوي": "antibiotic",
    "مدرات": "diuretic",
    "حاصرات بيتا": "beta blocker",
    "مثبطات ACE": "ACE inhibitor",
    # Diagnosis
    "تشخيص": "diagnosis",
    "فحص": "test",
    "تحليل": "analysis",
    "خزعة": "biopsy",
    "أشعة": "imaging",
    "رنين": "MRI",
    "CT": "CT scan",
    "سونار": "ultrasound",
    "electrocardiogram": "ECG",
    "EKG": "EKG",
    # Emergency
    "طارئ": "emergency",
    "اسعاف": "ambulance",
    "جلطة": "clot",
    "جلطة قلبية": "heart attack",
    "سكتة": "stroke",
    "نزيف": "bleeding",
    "فقدان وعي": "loss of consciousness",
    # General
    "مرض": "disease",
    "أعراض": "symptoms",
    "أسباب": "causes",
    "وقاية": "prevention",
    "مخاطر": "risk",
    "وراثي": "genetic",
    "معدي": "contagious",
    "مزمن": "chronic",
    "حاد": "acute",
}


_ARABIC_QUESTIONS = {
    "ما هو": "what is",
    "ما هي": "what are",
    "ما أسباب": "what causes",
    "كيف": "how",
    "لماذا": "why",
    "متى": "when",
    "أين": "where",
    "هل": "is it",
    "هل يمكن": "can",
}

_GENERAL_WORDS = {
    "symptoms",
    "causes",
    "treatment",
    "prevention",
    "diagnosis",
    "disease",
    "symptom",
    "medical",
    "health",
    "medicine",
    "medication",
    "medications",
    "pain",
    "fever",
    "healthcare",
    "blood",
    "pressure",
    "heart",
    "cardiac",
    "vascular",
}


def _expand_arabic_medical_terms(text: str) -> List[str]:
    """Expand Arabic medical terms to English."""
    expansions = []
    text_lower = text.lower()

    # Check for multi-word phrases first
    sorted_terms = sorted(_ARABIC_TO_ENGLISH.keys(), key=len, reverse=True)

    for arabic_term in sorted_terms:
        if arabic_term in text_lower:
            english_term = _ARABIC_TO_ENGLISH[arabic_term]
            expansions.append(english_term)

    return list(set(expansions))


def expand_query(query: str) -> List[str]:
    """Return 2-4 query variations for dense retrieval.

    Variations include:
    - Original preprocessed query (Arabic or English)
    - Full translated query (Arabic→English) for cross-lingual matching
    - English medical term expansions (for dense multilingual embeddings)
    """
    query = query.strip()
    if not query:
        return []

    variants: list = [query]

    # For Arabic queries, add a full English translation as a variant
    # so dense retrieval can match English corpus documents better.
    _has_arabic = any("\u0600" <= c <= "\u06FF" for c in query)
    if _has_arabic:
        try:
            from app.retrieval.query_translator import translate_query
            translated = translate_query(query)
            if translated and translated != query:
                variants.append(translated)
        except Exception:
            pass

    # English medical expansions for dense retrieval only
    english_terms = _expand_arabic_medical_terms(query)
    disease_terms = [t for t in english_terms if t.lower() not in _GENERAL_WORDS]

    if disease_terms:
        # Add one focused English expansion
        variants.append(" ".join(disease_terms[:4]))

    # Limit to 4 variations
    return variants[:4]
