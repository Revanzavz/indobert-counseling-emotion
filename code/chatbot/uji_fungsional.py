"""Pengujian fungsional (skenario 3 di proposal): menjalankan pesan uji pada alur chatbot lalu
membandingkan jalur dan label emosi dengan harapan. Hasil: hasil/uji_fungsional.csv dan .md

  python uji_fungsional.py            # memakai model IndoBERT final + modul respons
"""
import csv, os, json
from pipeline import proses_pesan

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'hasil'); os.makedirs(OUT, exist_ok=True)
kasus = list(csv.DictReader(open(os.path.join(HERE, 'uji_kasus.csv'), encoding='utf-8')))
baris = []
for k in kasus:
    h = proses_pesan(k['pesan'])
    jalur_ok = h['jalur'] == k['jalur_harapan']
    emosi_ok = '' if not k['emosi_harapan'] or h['label'] is None else h['label'] == k['emosi_harapan']
    baris.append({'id': k['id'], 'kategori': k['kategori'], 'pesan': k['pesan'], 'jalur_harapan': k['jalur_harapan'],
                  'jalur_aktual': h['jalur'], 'jalur_sesuai': jalur_ok, 'status_krisis': h['status_krisis'] or '',
                  'p_krisis': h['p_krisis'], 'emosi_harapan': k['emosi_harapan'],
                  'emosi_aktual': h['label'] or '', 'prob': round(h['probs'][h['label']], 4) if h['label'] else '',
                  'emosi_sesuai': emosi_ok, 'sumber_respons': h['sumber_respons'], 'waktu_detik': h.get('waktu_detik', ''),
                  'respons': h['respons']})
    print(k['id'], 'jalur', 'OK ' if jalur_ok else 'BEDA', '| emosi', {True: 'OK', False: 'BEDA', '': '-'}[emosi_ok],
          '|', h['jalur'], h['label'] or '', flush=True)
with open(os.path.join(OUT, 'uji_fungsional.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(baris[0])); w.writeheader(); w.writerows(baris)

ringkas = {}
for b in baris:
    r = ringkas.setdefault(b['kategori'], {'n': 0, 'jalur_sesuai': 0, 'n_emosi': 0, 'emosi_sesuai': 0})
    r['n'] += 1; r['jalur_sesuai'] += b['jalur_sesuai']
    if b['emosi_sesuai'] != '':
        r['n_emosi'] += 1; r['emosi_sesuai'] += b['emosi_sesuai']
with open(os.path.join(OUT, 'uji_fungsional.md'), 'w', encoding='utf-8') as f:
    f.write('| Kategori | Jumlah kasus | Jalur sesuai | Emosi sesuai |\n|---|---|---|---|\n')
    for k, r in ringkas.items():
        f.write(f"| {k} | {r['n']} | {r['jalur_sesuai']}/{r['n']} | "
                + (f"{r['emosi_sesuai']}/{r['n_emosi']}" if r['n_emosi'] else '-') + ' |\n')
    f.write('\n| ID | Pesan | Jalur | Emosi (prob) | Sesuai |\n|---|---|---|---|---|\n')
    for b in baris:
        ok = b['jalur_sesuai'] and b['emosi_sesuai'] in ('', True)
        f.write(f"| {b['id']} | {b['pesan']} | {b['jalur_aktual']} | {b['emosi_aktual']} {b['prob']} | {'ya' if ok else 'tidak'} |\n")
print(json.dumps(ringkas, indent=1))
