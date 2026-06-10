import os, requests
url = 'https://icntpbdznkfajnieyrjq.supabase.co'
key = 'sb_publishable_kl0hcI516psBUTbd_oWDXA_NXnfTpDL'
h = {'apikey': key, 'Authorization': 'Bearer ' + key}
for pid in (169, 250):
    r = requests.get(url + '/rest/vi/patients?id=eq.' + str(pid), headers=h, timeout=10)
    s1 = r.status_code
    t1 = r.text[:200]
    r2 = requests.get(url + '/rest/vi/medical_records?patient_id=eq.' + str(pid), headers=h, timeout=10)
    s2 = r2.status_code
    t2 = r2.text[:200]
    print('P' + str(pid) + ' patients: ' + str(s1) + ' ' + t1)
    print('P' + str(pid) + ' MRs: ' + str(s2) + ' ' + t2)