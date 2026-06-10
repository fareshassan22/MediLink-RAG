#!/usr/bin/env python3
"""Held-out generalization test for the TOON router.

The patterns in toon_router.py were tuned on TOON_TEST_QUERIES, so accuracy on
that set is optimistic (train-on-test). This file contains DIFFERENTLY-WORDED
Arabic queries with the same per-tier intents — none of these exact strings
were used while writing the regex patterns. It measures whether the router
generalizes or merely memorized the test set.

Tier intent definitions:
  T1 = single current value lookup (one lab/vital/demographic field)
  T2 = comparison / trend / "within normal range" judgment over time
  T3 = clinical reasoning, recommendations, lifestyle/diet/treatment advice
"""
from app.retrieval.toon_router import classify

HELDOUT = {
    1: [  # single-value lookups, new phrasings
        "ما قيمة السكر التراكمي عندي؟",
        "اعطني رقم ضغط الدم الانقباضي",
        "ما هو معدل نبضي الحالي؟",
        "كم درجة حرارتي المسجلة؟",
        "ما تركيز الصوديوم في الدم؟",
        "ما نسبة البوتاسيوم لدي؟",
        "ما قيمة الكالسيوم في التحليل؟",
        "كم نسبة الأكسجين في الدم؟",
        "ما رقم معامل التخثر INR؟",
        "ما مستوى حمض اليوريك؟",
        "ما هو معدل الترشيح الكلوي GFR؟",
        "كم عدد الصفائح الدموية؟",
        "ما قيمة البروتين في البول؟",
        "ما نتيجة فحص فيتامين B12؟",
        "ما هي فصيلة دمي المسجلة؟",
    ],
    2: [  # comparison / trend / range judgment, new phrasings
        "هل تحسن مستوى السكر التراكمي عن الشهر الماضي؟",
        "كيف تطورت قراءات الضغط خلال الزيارات الأخيرة؟",
        "هل انخفض الكوليسترول الضار مقارنة بالسابق؟",
        "هل أصبح وزني ضمن الحدود الطبيعية؟",
        "قارن نتائج وظائف الكبد الحالية بالقديمة",
        "هل استقرت قراءات الغلوكوز مؤخراً؟",
        "هل ارتفعت إنزيمات الكبد عن آخر مرة؟",
        "هل نتائج الغدة الدرقية ضمن المعدل الطبيعي؟",
        "هل هناك تراجع في وظائف الكلى؟",
        "ما اتجاه نسبة الهيموغلوبين عبر الفحوصات؟",
        "هل تحسنت مستويات الحديد منذ آخر تحليل؟",
        "هل قراءات ضغطي مرتفعة بشكل عام؟",
        "هل تغيرت نتائج الدهون الثلاثية؟",
        "هل النتائج الحالية أفضل من السابقة؟",
        "هل هناك قلق من اتجاه نتائجي؟",
    ],
    3: [  # reasoning / recommendation / lifestyle, new phrasings
        "بناءً على تحاليلي، ما النظام الغذائي الأنسب لي؟",
        "هل يجب أن أعدّل جرعة دواء الضغط حسب النتائج؟",
        "ما الخطوات العلاجية الموصى بها لحالتي؟",
        "كيف أخفض نسبة الكوليسترول الضار طبيعياً؟",
        "ما توصياتك لتحسين مستوى السكر لدي؟",
        "هل أنا معرض لخطر أمراض القلب بناءً على بياناتي؟",
        "ما العلاقة بين ارتفاع ضغطي وزيادة وزني؟",
        "ما التمارين الرياضية المناسبة لحالتي الصحية؟",
        "هل يجب أن أتجنب الأطعمة المالحة؟",
        "ما الفحوصات الوقائية التي ينصح بها لي؟",
        "كيف يمكنني التحكم في التوتر لتحسين ضغطي؟",
        "هل أحتاج إلى مكملات غذائية بناءً على نقص الفيتامينات؟",
        "ما المسار المتوقع لحالتي إذا استمرت النتائج كذلك؟",
        "Based on my lab trends, what diet changes do you advise?",
        "Should I be worried about my kidney function results?",
    ],
}


def main():
    correct = 0
    total = 0
    per = {1: [0, 0], 2: [0, 0], 3: [0, 0]}  # [ok, n]
    wrong = []
    for exp, qs in HELDOUT.items():
        for q in qs:
            d = classify(q)
            total += 1
            per[exp][1] += 1
            if d.tier == exp:
                correct += 1
                per[exp][0] += 1
            else:
                wrong.append((exp, d.tier, q))
    print("HELD-OUT GENERALIZATION (queries NOT used for tuning)")
    print(f"ACCURACY: {correct}/{total} = {100*correct/total:.1f}%")
    for t in (1, 2, 3):
        ok, n = per[t]
        print(f"  Tier {t} recall: {ok}/{n} = {100*ok/n:.0f}%")
    print(f"--- misroutes ({len(wrong)}) ---")
    for e, p, q in wrong:
        print(f"  exp{e} got{p}: {q}")


if __name__ == "__main__":
    main()
