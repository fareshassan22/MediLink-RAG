import os, requests, json
with open('/home/medilink/RAG/.env') as f:
    for line in f:
        k, v = line.strip().split('=', 1)
        if k == 'NEXT_PUBLIC_SUPABASE_URL':
            url = v
        elif k == 'SUPABASE_SERVICE_ROLE_KEY':
            key = v

h = {'apikey': key, 'Authorization': f'Bearer {key}'}
print('URL:', url)
print('Key present:', bool(key))

for pid in (169, 250):
    r = requests.get(url + f'/rest/vi/medical_records?patient_id=eq.{pid}', headers=h, timeout=10)
    print(f'P{pid} MRs: {r.status_code} -> {r.text[:300]}')