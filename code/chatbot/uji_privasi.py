"""Uji privasi: memastikan teks pengguna tidak pernah masuk ke permintaan Gemini.

Percakapan contoh dijalankan melalui pipeline yang sama dengan bot. Permintaan ke Gemini disadap (tidak
dikirim ke internet), lalu seluruh isi permintaan, termasuk instruksi sistem dan riwayat, diperiksa: tidak boleh
ada tiga kata berurutan dari pesan pengguna, dan tidak boleh ada kata khas pesan pengguna (minimal 5 huruf)
yang tidak berasal dari instruksi atau balasan bot. Model krisis dan emosi diganti tiruan agar uji cepat;
pilihan --model memakai model asli. Keluaran: hasil/uji_privasi.md"""
import json, os, re, sys
import respons
from pipeline import proses_pesan, tambah_riwayat

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'hasil'); os.makedirs(OUT, exist_ok=True)
PESAN = ['akhir-akhir ini aku ngerasa sendirian banget, temen-temenku kayak udah punya circle masing-masing',
         'tiap mereka nongkrong di kafe Senopati aku jarang diajak, padahal dulu kita deket banget',
         'namaku Rina, aku kuliah di Bandung dan dosenku Pak Budi sering marahin aku',
         'aku jadi takut kalau chat duluan malah dianggap ganggu',
         'terus aku harus gimana ya']
asli = '--model' in sys.argv
pemeta = None if asli else (lambda t: ('sadness', {'sadness': 0.7, 'anger': 0.2, 'fear': 0.05, 'happy': 0.03, 'neutral': 0.02}))
penilai = None if asli else (lambda t: (None, 0.05))
sadapan = []


def penyadap(sistem, isi):
    sadapan.append({'systemInstruction': sistem, 'contents': isi})
    return f'Balasan tiruan nomor {len(sadapan)} yang menanyakan perasaanmu dengan lembut.'


respons.PENYADAP = penyadap
rw = []
for p in PESAN:
    h = proses_pesan(p, riwayat=rw, pemeta=pemeta, penilai_krisis=penilai)
    tambah_riwayat(rw, h)
bocor = []
for i, s in enumerate(sadapan):
    muatan = json.dumps(s, ensure_ascii=False).lower()
    for p in PESAN:
        kata = re.findall(r'\w+', p.lower())
        for j in range(len(kata) - 2):
            if ' '.join(kata[j:j + 3]) in muatan:
                bocor.append((i + 1, 'tiga kata', ' '.join(kata[j:j + 3])))
        for k in set(kata):
            if len(k) >= 5 and re.search(r'\b' + re.escape(k) + r'\b', muatan) and not re.search(r'\b' + re.escape(k) + r'\b', s['systemInstruction'].lower() + ' balasan tiruan nomor yang menanyakan perasaanmu dengan lembut'):
                bocor.append((i + 1, 'kata khas', k))
riwayat_ada_teks = any('teks' in g and g['peran'] == 'pengguna' for g in rw)
md = [f'# Uji privasi ({len(sadapan)} permintaan ke Gemini disadap, model {"asli" if asli else "tiruan"})\n',
      f'Hasil: **{"LULUS" if not bocor and not riwayat_ada_teks else "GAGAL"}**. Potongan teks pengguna yang ditemukan: {len(bocor)}. '
      f'Riwayat menyimpan teks pengguna: {"ya" if riwayat_ada_teks else "tidak"}.\n', '## Contoh isi permintaan terakhir ke Gemini (selain instruksi sistem)\n', '```',
      json.dumps(sadapan[-1]['contents'], ensure_ascii=False, indent=1), '```']
if bocor:
    md += ['', '## Temuan', *[f'- permintaan {a}: {b} "{c}"' for a, b, c in bocor]]
open(os.path.join(OUT, 'uji_privasi.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
print('\n'.join(md))
sys.exit(1 if bocor or riwayat_ada_teks else 0)
