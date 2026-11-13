"""[VERSI LAMA, TIDAK DIPAKAI LAGI] Jalur deteksi krisis berbasis daftar frasa.
Disimpan untuk pembanding. Versi yang dipakai sekarang ada di krisis.py (berbasis model).

Jalur deteksi krisis (lapisan pengaman chatbot).

Sesuai rancangan proposal, deteksi krisis memakai dua mekanisme:
  1. Pemeriksaan kata kunci eksplisit tentang bunuh diri atau menyakiti diri. Dijalankan SEBELUM
     model klasifikasi. Bila cocok, pesan langsung dialihkan ke jalur darurat.
  2. Pemeriksaan kombinasi label emosi dan probabilitas dari model (sadness atau fear dengan
     probabilitas tinggi) yang disertai indikator kata krisis tidak langsung.

Daftar frasa disusun dengan mengacu pada kategori Columbia Suicide Severity Rating Scale (C-SSRS):
keinginan untuk mati, ide bunuh diri, rencana atau cara, dan perilaku menyakiti diri, serta
tanda peringatan pada panduan WHO LIVE LIFE (putus asa, merasa menjadi beban, ingin menghilang).
Daftar ini adalah pengaman prototipe riset, bukan alat skrining klinis.
"""
import re

from konfigurasi import EMOSI_RENTAN, AMBANG_PROB_KRISIS

# ---------------------------------------------------------------- normalisasi ringan
_GANTI = {
    'gue': 'aku', 'gw': 'aku', 'gua': 'aku', 'aq': 'aku', 'ak': 'aku', 'saya': 'aku', 'sy': 'aku', 'w': 'aku',
    'ga': 'tidak', 'gak': 'tidak', 'gk': 'tidak', 'g': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak', 'enggak': 'tidak',
    'engga': 'tidak', 'kagak': 'tidak', 'tdk': 'tidak', 'tak': 'tidak', 'ndak': 'tidak', 'gada': 'tidak ada',
    'pengen': 'ingin', 'pengin': 'ingin', 'pingin': 'ingin', 'pgn': 'ingin', 'kepengen': 'ingin', 'mau': 'ingin',
    'mo': 'ingin', 'pingen': 'ingin', 'kepingin': 'ingin',
    'udah': 'sudah', 'udh': 'sudah', 'dah': 'sudah', 'sdh': 'sudah',
    'aja': 'saja', 'aj': 'saja', 'idup': 'hidup', 'hdp': 'hidup', 'cape': 'capek', 'capai': 'capek',
    'bundir': 'bunuh diri', 'mending': 'lebih baik', 'mendingan': 'lebih baik', 'buat': 'untuk', 'bwt': 'untuk',
    'lg': 'lagi', 'lgi': 'lagi', 'bgt': 'banget', 'sm': 'sama', 'dr': 'dari', 'dri': 'dari', 'yg': 'yang',
    'gabisa': 'tidak bisa', 'gakuat': 'tidak kuat', 'gasanggup': 'tidak sanggup', 'nyerah': 'menyerah',
    'ilang': 'hilang', 'ngilang': 'menghilang', 'nyakitin': 'menyakiti', 'ngelukain': 'melukai',
    'ngakhirin': 'mengakhiri', 'akhirin': 'akhiri', 'cepet': 'cepat', 'cpt': 'cepat', 'lompat': 'loncat', 'melompat': 'loncat',
    'meloncat': 'loncat', 'nyari': 'mencari', 'cari': 'mencari', 'nyiapin': 'menyiapkan', 'siapin': 'menyiapkan',
    'udahan': 'sudahan', 'kalo': 'kalau', 'klo': 'kalau', 'bakal': 'akan', 'bakalan': 'akan', 'almarhumah': 'almarhum', 'alm': 'almarhum', 'selamanya': 'selamanya', 'selama-lamanya': 'selamanya',
}


def normalisasi(teks):
    t = str(teks).lower()
    t = re.sub(r'(.)\1{2,}', r'\1', t)              # "matiiii" -> "mati"
    t = re.sub(r'[^a-z0-9\s-]', ' ', t)
    kata = [_GANTI.get(k, k) for k in t.replace('-', ' ').split()]
    return ' '.join(kata)


# ---------------------------------------------------------------- mekanisme 1: kata kunci eksplisit
FRASA_EKSPLISIT = [
    # ide dan niat bunuh diri
    r'\bbunuh diri\b', r'\bmembunuh diri(ku)?\b', r'\b(meng)?akhiri hidup(ku)?\b', r'\b(meng)?akhiri nyawa(ku)?\b',
    r'\bmenghabisi (diri|nyawa)(ku)?\b', r'\bmencabut nyawa(ku)? sendiri\b', r'\bsuicid\w*', r'\bkill myself\b',
    # keinginan untuk mati
    r'\bingin mati\b', r'\blebih baik (aku )?mati\b', r'\btidak ingin hidup\b(?! (di|sama|dengan|bareng)\b)', r'\bingin (segera |cepat )mati\b',
    r'\bberharap (aku )?(cepat |segera )?mati\b', r'\bsemoga (aku )?(cepat |segera )?mati\b',
    r'\b(untuk|buat) apa (lagi )?(aku )?(masih )?hidup\b', r'\bingin (menyusul|nyusul) (almarhum|ibu|ayah|mama|papa|bapak|dia|mereka)\b',
    # keinginan pasif untuk mati (C-SSRS kategori 1: wish to be dead)
    r'\b(capek|lelah|muak|bosan|jenuh) (banget )?(sama |dengan )?hidup\b',
    r'\bingin (menghilang|hilang|lenyap|pergi) (saja )?(untuk )?(selamanya|dari dunia|dari muka bumi)\b',
    r'\b(menghilang|hilang|lenyap) (saja )?selamanya\b', r'\btidur (untuk )?selamanya\b',
    r'\btidak ingin bangun lagi\b', r'\blebih baik (aku )?tidak ada\b',
    r'\blebih baik (aku )?tidak (pernah )?(lahir|dilahirkan)\b',
    r'\btidak ada (guna|gunanya|artinya|alasan) (lagi )?(aku )?(untuk )?hidup\b',
    # rencana atau cara
    r'\bgantung diri\b', r'\bmenggantung diri(ku)?\b', r'\bminum racun\b', r'\bloncat dari (gedung|jembatan|lantai|atap|balkon|rooftop)\b',
    r'\bterjun dari (gedung|jembatan|lantai|atap)\b', r'\bcara (untuk |biar |supaya |agar )?(cepat |bisa )?mati\b', r'\bloncat (ke|di) (rel|jalan|sungai|laut|kali)\b',
    r'\bmenabrakkan diri\b', r'\b(potong|iris|sayat|menyayat|nyayat|memotong|mengiris) (urat )?(nadi|pergelangan)\b',
    # menyakiti diri
    r'\b(menyakiti|melukai|menyiksa) diri(ku)?( sendiri)?\b', r'\bself ?harm\b', r'\b(menyayat|nyayat|sayat|silet|nyilet) (tangan|lengan|kulit|paha)(ku)?\b',
]

# ---------------------------------------------------------------- mekanisme 2: indikator tidak langsung
FRASA_TIDAK_LANGSUNG = [
    r'\bhidup(ku)? (sudah )?tidak (ada )?(arti|artinya|guna|gunanya|berguna|berarti|bermakna)\b',
    r'\btidak ada (guna|gunanya|artinya|alasan) (lagi )?(aku )?(untuk )?(hidup|bertahan)\b',
    r'\bingin (menghilang|hilang|lenyap)\b',
    r'\b(sudah )?tidak (kuat|sanggup|tahan) lagi\b', r'\bsudah tidak (kuat|sanggup|tahan)\b',
    r'\bmenyerah (saja )?(sama |dengan |pada )?hidup\b',
    r'\bmenyesal (di)?lahir(kan)?\b',
    r'\b(dunia|semua orang|mereka|keluarga(ku)?) (akan )?lebih baik tanpa aku\b',
    r'\b(cuma|hanya|jadi) (jadi )?beban\b', r'\baku (ini )?beban\b',
    r'\btidak ada harapan\b', r'\bputus asa\b', r'\bakhiri (saja )?(semua|semuanya)\b', r'\bmengakhiri (semua|semuanya)\b',
    r'\bovers?dosis\b', r'\bcutting\b', r'\b(kalau|jika|andai|seandainya) aku (sudah )?(tidak ada|mati|pergi)\b',
    r'\bingin sudahan (saja )?(sama|dengan) (semua|semuanya|hidup)\b', r'\bmenyiapkan (obat|racun|tali|pisau|silet)\b', r'\btidak ada yang peduli (kalau|jika) aku (mati|tidak ada|hilang)\b',
]

_RE_EKSPLISIT = [re.compile(p) for p in FRASA_EKSPLISIT]
_RE_TIDAK_LANGSUNG = [re.compile(p) for p in FRASA_TIDAK_LANGSUNG]


def _cocok(pola, teks):
    for p in pola:
        m = p.search(teks)
        if m:
            return m.group(0)
    return None


def cek_kata_kunci(teks):
    """Mekanisme 1. Dipanggil sebelum model klasifikasi. Mengembalikan frasa pemicu atau None."""
    return _cocok(_RE_EKSPLISIT, normalisasi(teks))


def cek_emosi_dan_indikator(teks, label, probs):
    """Mekanisme 2. Dipanggil setelah model klasifikasi. Krisis bila label emosi termasuk emosi rentan,
    probabilitasnya melewati ambang, dan pesan memuat indikator kata krisis tidak langsung."""
    frasa = _cocok(_RE_TIDAK_LANGSUNG, normalisasi(teks))
    if frasa and label in EMOSI_RENTAN and probs.get(label, 0) >= AMBANG_PROB_KRISIS:
        return frasa
    return None


def punya_indikator(teks):
    """True bila pesan memuat indikator tidak langsung (dipakai untuk memberi tahu modul respons
    agar lebih berhati-hati walaupun syarat emosi belum terpenuhi)."""
    return _cocok(_RE_TIDAK_LANGSUNG, normalisasi(teks)) is not None


PESAN_KRISIS = (
    "Terima kasih sudah mau cerita. Aku khawatir dengan keselamatanmu, dan apa yang kamu rasakan "
    "sekarang penting untuk ditangani bersama orang yang terlatih.\n\n"
    "Kamu bisa menghubungi layanan berikut:\n"
    "- Layanan kesehatan jiwa Kementerian Kesehatan: telepon 119 lalu tekan 8, atau buka healing119.id\n"
    "- Keadaan darurat: telepon 112 atau datang ke IGD rumah sakit terdekat\n\n"
    "Kalau bisa, hubungi juga orang yang kamu percaya dan minta ditemani sekarang. "
    "Aku hanya program komputer dan tidak bisa menggantikan bantuan profesional, "
    "tapi kamu tidak harus menghadapi ini sendirian."
)
