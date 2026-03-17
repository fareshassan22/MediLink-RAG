def filter_by_metadata(results, specialty=None, language=None):
    """
    Filters retrieved documents based on metadata.

    results: list of dicts from vector search
    specialty: e.g. "cardiology"
    language: e.g. "arabic"
    """
    if not results:
        return results

    filtered = results

    if specialty:
        specialty_filtered = [
            r for r in filtered
            if r.get("specialty") == specialty
            or r.get("metadata", {}).get("specialty") == specialty
        ]
        if specialty_filtered:
            filtered = specialty_filtered

    if language:
        lang_filtered = [
            r for r in filtered
            if r.get("language") == language
            or r.get("metadata", {}).get("language") == language
        ]
        if lang_filtered:
            filtered = lang_filtered

    return filtered