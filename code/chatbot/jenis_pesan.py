"""Penggolong jenis pesan (dialogue act) sederhana yang berjalan di komputer lokal.

Hanya menentukan BENTUK balasan (menyapa, bercerita, minta saran, berterima kasih, jawaban singkat), tidak dipakai
untuk deteksi krisis maupun emosi. Yang dikirim ke Gemini hanya nama jenisnya, bukan teks pengguna."""
import re

JENIS = {
    'sapaan': 'menyapa atau membuka percakapan',
    'bercerita': 'bercerita atau mengungkapkan perasaan',
    'minta_saran': 'bertanya atau meminta saran',
    'penutup': 'berterima kasih atau menutup percakapan',
    'jawaban_singkat': 'jawaban singkat',
}

_PENUTUP = re.compile(r'\b(makasi+h?|makasi|terima ?kasi+h|trims|thanks?|thx|tengkyu|udah dulu|segitu dulu|sampai sini dulu|'
                      r'bye|dadah|sampai jumpa|see you|udah lega|lebih lega|agak lega|jadi lega)\b')
_SARAN = re.compile(r'\b(gimana|bagaimana|harus apa|harus ngapain|sebaiknya|saran|solusi|enaknya|tips|cara(nya)?|'
                    r'apa yang harus|aku harus|apa aku|perlu gak|perlu nggak|boleh gak|boleh nggak)\b')
_SAPAAN = re.compile(r'^(halo+|hal+o|hai+|hi+|hello|hey|pagi|siang|sore|malam|malem|assalamualaikum|permisi|p)\b')
_MAU_CERITA = re.compile(r'\b(mau|pengen|pingin|ingin|boleh) (cerita|curhat)\b')
_SINGKAT = re.compile(r'^(iya+|iy+a|ya+|yaa+|y|ok(e|ay)?|hm+|hmm+|entah|gak tau|nggak tau|ga tau|gatau|mungkin|bisa jadi|'
                      r'betul|bener|iya juga|gitu deh|ya gitu|belum|udah|sudah|gak|nggak|enggak|ga)\b')


def golongkan(teks):
    t = re.sub(r'\s+', ' ', (teks or '').lower()).strip()
    n = len(t.split())
    if n == 0:
        return 'jawaban_singkat'
    if _PENUTUP.search(t) and n <= 15:
        return 'penutup'
    if (_SAPAAN.match(t) and n <= 8) or (_MAU_CERITA.search(t) and n <= 10):
        return 'sapaan'
    if _SARAN.search(t) or t.endswith('?'):
        return 'minta_saran'
    if n <= 3 or (_SINGKAT.match(t) and n <= 5):
        return 'jawaban_singkat'
    return 'bercerita'


if __name__ == '__main__':
    for c in ['halo, aku lagi pengen cerita nih', 'akhir akhir ini aku ngerasa sendirian banget',
              'terus aku harus gimana ya', 'makasih ya udah dengerin, aku jadi agak lega sekarang', 'iya', 'gak tau juga']:
        print(golongkan(c), '|', c)
