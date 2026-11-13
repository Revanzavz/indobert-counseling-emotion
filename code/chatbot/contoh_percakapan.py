"""Menjalankan satu percakapan contoh (curhat umum anak muda) melalui alur chatbot yang sama dengan server,
lalu menyimpan label emosi, probabilitas, dan respons tiap giliran. Dipakai untuk tabel contoh percakapan di Bab 4.
Hasil: hasil/contoh_percakapan.md dan hasil/contoh_percakapan.csv"""
import csv, os
from pipeline import proses_pesan, tambah_riwayat

PESAN = [
    'akhir-akhir ini aku ngerasa sendirian banget, temen-temenku kayak udah punya circle masing-masing',
    'tiap mereka nongkrong aku jarang diajak, padahal dulu kita deket banget',
    'aku jadi takut kalau chat duluan malah dianggap ganggu',
    'aku harus mulai dari mana ya biar gak canggung lagi sama mereka',
    'makasih ya, aku jadi lebih lega bisa cerita',
]
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'hasil'); os.makedirs(OUT, exist_ok=True)
riwayat, baris = [], []
for i, p in enumerate(PESAN, 1):
    h = proses_pesan(p, riwayat=list(riwayat))
    probs = h['probs'] or {}
    urut = sorted(probs.items(), key=lambda x: -x[1])
    baris.append({'giliran': i, 'pesan': p, 'jenis_pesan': h.get('jenis_pesan', ''), 'p_krisis': h['p_krisis'],
                  'status_krisis': h['status_krisis'] or 'normal', 'jalur': h['jalur'], 'label': h['label'] or '',
                  'prob': round(probs.get(h['label'], 0), 4) if h['label'] else '',
                  'semua_prob': ', '.join(f'{k} {v:.2f}' for k, v in urut), 'sumber_respons': h['sumber_respons'],
                  'respons': h['respons']})
    tambah_riwayat(riwayat, h)
    print(f'[{i}] {p}\n    -> {h["jalur"]} | {h["label"]} | {baris[-1]["semua_prob"]} | {h["sumber_respons"]}\n    {h["respons"]}\n', flush=True)
with open(os.path.join(OUT, 'contoh_percakapan.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(baris[0])); w.writeheader(); w.writerows(baris)
with open(os.path.join(OUT, 'contoh_percakapan.md'), 'w', encoding='utf-8') as f:
    for b in baris:
        f.write(f"### Giliran {b['giliran']}\nPesan: {b['pesan']}\n\nJenis pesan: {b['jenis_pesan']} | p krisis: {b['p_krisis']} ({b['status_krisis']}) | Jalur: {b['jalur']} | Emosi: {b['label']} ({b['prob']}) | "
                f"Semua: {b['semua_prob']} | Sumber respons: {b['sumber_respons']}\n\nRespons:\n{b['respons']}\n\n")
print('Selesai. Lihat hasil/contoh_percakapan.md')
