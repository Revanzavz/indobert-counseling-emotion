"""Konfigurasi chatbot. Nilai dibaca dari file .env di folder ini (salin dari .env.contoh)."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _muat_env(path):
    if not os.path.exists(path):
        return
    for baris in open(path, encoding='utf-8'):
        baris = baris.strip()
        if not baris or baris.startswith('#') or '=' not in baris:
            continue
        k, v = baris.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_muat_env(os.path.join(HERE, '.env'))

# Folder kode eksperimen (berisi prediksi.py dan hasil/model_final)
DIR_EKSPERIMEN = os.environ.get('DIR_EKSPERIMEN', os.path.join(HERE, '..', 'Eksperimen', 'eksperimen'))

# Modul respons generatif
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')

# Deteksi krisis, mekanisme kedua: label emosi + probabilitas + indikator kata krisis
EMOSI_RENTAN = ('sadness', 'fear')
AMBANG_PROB_KRISIS = float(os.environ.get('AMBANG_PROB_KRISIS', '0.70'))

# Di bawah ambang ini label emosi dianggap belum pasti dan modul respons diberi tahu
AMBANG_YAKIN = float(os.environ.get('AMBANG_YAKIN', '0.60'))
