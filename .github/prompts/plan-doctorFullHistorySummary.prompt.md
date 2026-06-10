# Plan: Doctor full-history Q&A + patient summary

Scope (this round): DOCTOR ONLY. A doctor can (1) ask anything about a patient and get answers from the patient's FULL history, and (2) get a structured summary of any patient whose data they access. Patient slot-booking is DEFERRED (out of scope now).

## Key facts (from research)
- `/patient/ask` [app/main.py:222] -> `patient_rag_service.run(query, patient_id, role)`.
- Today `role` only swaps the prompt in `build_toon_prompt` (doctor -> `_DOCTOR_SYSTEM`).
- Tier 3 uses `fetch_live_context()` [toon.py:224] which LIMITS data: 10 records, 5 vitals, 5 appts, 5 diag, 5 rx, 5 lab, NO payments.
- `fetch_all_chunks()` [toon.py:366] already fetches FULL history (all records + payments) but is used for indexing, returns chunk dicts not a string.
- No auth/role enforcement (role trusted from request body). Deferred.

## Plan / Steps
1. **toon.py**: add `fetch_full_context(patient_id) -> str` — like `fetch_live_context` but NO per-table limits, includes payments. Reuse `_fmt()` + existing batch fetchers + `fetch_payments`. (full clinical history as one text block)
2. **prompts.py**: add `build_doctor_summary_prompt(context, language)` — structured clinical summary (demographics, active problems, meds, recent vitals/labs, appointments, billing). Also strengthen `_DOCTOR_SYSTEM` doctor Q&A prompt (precise clinical terminology, cite sources, no patient-safety disclaimers, may surface all fields).
3. **toon_service.py**: add `run_doctor(query, patient_id, mode)` (or branch in `run()` when `role=="doctor"`): bypass tier routing, always use `fetch_full_context()`, use doctor prompt. `mode="summary"` -> `build_doctor_summary_prompt`; `mode="ask"` -> `build_toon_prompt(role="doctor")`. Keep emergency gate? For doctor, skip patient-style emergency escalation (doctor needs raw info). Keep `PipelineResult` shape + `stage_latencies`.
4. **main.py**: add doctor endpoints:
   - `POST /doctor/ask`  {patient_id, query} -> `run_doctor(mode="ask")`
   - `POST /doctor/patient/{patient_id}/summary` -> `run_doctor(query="", mode="summary")`
   - Add `DoctorQueryRequest` model mirroring `PatientQueryRequest`.
5. (Optional) note in code that role is not yet enforced (deferred auth).

## Relevant files
- `app/retrieval/toon.py` — add `fetch_full_context` (reuse `_fmt`, fetchers, `fetch_payments`)
- `app/generation/prompts.py` — add `build_doctor_summary_prompt`, beef up `_DOCTOR_SYSTEM`
- `app/retrieval/toon_service.py` — add `run_doctor` / doctor branch
- `app/main.py` — add `/doctor/ask` + `/doctor/patient/{id}/summary` endpoints + model

## Verification
1. Start API (uvicorn) -> `POST /doctor/patient/{id}/summary` returns structured summary.
2. `POST /doctor/ask` with clinical question -> answer cites full-history data not limited to 10/5 windows (e.g. ask about an old record beyond the 10-record window).
3. Confirm response uses clinical (doctor) tone, no patient disclaimer text.
4. Existing `/patient/ask` still works unchanged (regression).

## Decisions / open items
- Auth/role enforcement DEFERRED (role trusted from body for now).
- Summary language: follow query/patient language (detect) — default en/ar via `_detect_lang`.
- Emergency gate skipped on doctor path (recommended) — confirm if unsure.
