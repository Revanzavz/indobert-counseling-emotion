"""Evaluasi jalur deteksi krisis berbasis model pada 50 pesan uji berbahasa Indonesia (uji_krisis.csv):
30 pesan berindikasi krisis dan 20 pesan yang memuat kata serupa tetapi bukan krisis.
Pesan ini tidak dipakai dalam pelatihan model krisis. Keluaran: hasil/uji_krisis.csv dan hasil/uji_krisis.md"""
import csv, os
from pipeline import proses_pesan
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'hasil'); os.makedirs(OUT, exist_ok=True)
gen = lambda t, l, p, hati_hati=False, riwayat=None: ('(respons)', 'tiruan')
rows = []
for k in csv.DictReader(open(os.path.join(HERE, 'uji_krisis.csv'), encoding='utf-8')):
    h = proses_pesan(k['pesan'], generator=gen)
    rows.append({**k, 'aktual': h['jalur'], 'status_krisis': h['status_krisis'] or '', 'p_krisis': h['p_krisis'],
                 'emosi': h['label'] or '', 'sesuai': h['jalur'] == k['harapan']})
tp = sum(r['harapan'] == 'krisis' and r['aktual'] == 'krisis' for r in rows); fn = sum(r['harapan'] == 'krisis' and r['aktual'] != 'krisis' for r in rows)
fp = sum(r['harapan'] != 'krisis' and r['aktual'] == 'krisis' for r in rows); tn = sum(r['harapan'] != 'krisis' and r['aktual'] != 'krisis' for r in rows)
hati = sum(r['harapan'] == 'krisis' and r['status_krisis'] == 'hati_hati' for r in rows)
with open(os.path.join(OUT, 'uji_krisis.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
md = [f'| | Terdeteksi krisis | Tidak terdeteksi |\n|---|---|---|\n| Pesan krisis ({tp + fn}) | {tp} | {fn} |\n| Pesan bukan krisis ({fp + tn}) | {fp} | {tn} |\n',
      f'Recall {tp / max(tp + fn, 1):.1%}, precision {tp / max(tp + fp, 1):.1%}. Pesan krisis yang tidak dialihkan tetapi diberi penanda hati-hati: {hati}.\n',
      '| ID | Kelompok | Pesan | Harapan | Aktual | Status | p krisis |\n|---|---|---|---|---|---|---|']
md += [f"| {r['id']} | {r['kelompok']} | {r['pesan']} | {r['harapan']} | {r['aktual']} | {r['status_krisis']} | {r['p_krisis']} |" for r in rows if not r['sesuai']]
open(os.path.join(OUT, 'uji_krisis.md'), 'w', encoding='utf-8').write('\n'.join(md))
print('\n'.join(md))
