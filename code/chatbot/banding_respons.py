"""Membandingkan balasan Gemini sebelum dan sesudah penambahan jenis pesan dan strategi konseling (OARS).

Percakapan contoh yang sama dijalankan dua kali melalui pipeline: sekali dengan modul respons lama
(respons_sebelum_jenis_pesan.py.bak) dan sekali dengan respons.py yang baru. Model emosi dan model krisis sama.
Hasil: hasil/banding_respons.md"""
import io, json, os, time, contextlib, importlib.util, importlib.machinery
from pipeline import proses_pesan, tambah_riwayat
import respons as baru

HERE = os.path.dirname(os.path.abspath(__file__))
_p = os.path.join(HERE, 'respons_sebelum_jenis_pesan.py.bak')
_loader = importlib.machinery.SourceFileLoader('respons_lama', _p)
_spec = importlib.util.spec_from_loader('respons_lama', _loader)
lama = importlib.util.module_from_spec(_spec); _loader.exec_module(lama)

PESAN = [
    'halo, aku lagi pengen cerita nih',
    'akhir akhir ini aku ngerasa sendirian banget, temen temen pada sibuk sendiri',
    'tadi aku liat mereka nongkrong bareng tapi aku gak diajak, kesel banget jujur',
    'aku jadi takut kalau sebenernya mereka emang gak suka sama aku',
    'menurutmu aku sebaiknya ngapain ya',
    'makasih ya udah dengerin, aku jadi agak lega sekarang',
]

JEDA = 7   # detik antarpanggilan
# Hasil versi lama disimpan sekali saja agar kuota harian Gemini tidak habis untuk versi lama
CACHE_LAMA = os.path.join(HERE, 'hasil', 'banding_respons_lama.json')
hasil = {}
if os.path.exists(CACHE_LAMA):
    hasil['lama'] = json.load(open(CACHE_LAMA, encoding='utf-8'))
    print('Versi lama diambil dari hasil/banding_respons_lama.json (tidak memanggil Gemini lagi)', flush=True)
for nama, modul in [('lama', lama), ('baru', baru)]:
    if nama in hasil:
        continue
    rw, hasil[nama] = [], []
    print(f'=== versi {nama} ===', flush=True)
    for p in PESAN:
        catatan = io.StringIO()
        with contextlib.redirect_stdout(catatan):      # menangkap pesan galat Gemini, bila ada
            h = proses_pesan(p, riwayat=rw, generator=modul.buat_respons)
        h['catatan'] = catatan.getvalue().strip()
        tambah_riwayat(rw, h); hasil[nama].append(h)
        if h['catatan']:
            print('  ' + h['catatan'], flush=True)
        time.sleep(JEDA)                               # menghindari batas permintaan per menit Gemini gratis
        print(f'> {p}\n  [{h.get("jenis_pesan")} | {h["label"]} | {h["sumber_respons"]}] {h["respons"]}\n', flush=True)

os.makedirs(os.path.join(HERE, 'hasil'), exist_ok=True)
if not os.path.exists(CACHE_LAMA) and all(h['sumber_respons'] in ('gemini', 'rujukan') for h in hasil['lama']):
    json.dump(hasil['lama'], open(CACHE_LAMA, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
with open(os.path.join(HERE, 'hasil', 'banding_respons.md'), 'w', encoding='utf-8') as f:
    f.write('# Perbandingan balasan sebelum dan sesudah jenis pesan + strategi OARS\n\n')
    for i, p in enumerate(PESAN):
        a, b = hasil['lama'][i], hasil['baru'][i]
        f.write(f"## Giliran {i + 1}\n\nPesan: {p}\n\nJenis pesan: {b.get('jenis_pesan')} | Emosi: {b['label']} | "
                f"p krisis: {b['p_krisis']} ({b['status_krisis'] or 'normal'})\n\n"
                f"Sebelum ({a['sumber_respons']}): {a['respons']}\n\nSesudah ({b['sumber_respons']}): {b['respons']}\n\n")
        for x, n in [(a, 'sebelum'), (b, 'sesudah')]:
            if x.get('catatan'):
                f.write(f"Catatan {n}: {x['catatan']}\n\n")
print('Selesai. Lihat hasil/banding_respons.md')
