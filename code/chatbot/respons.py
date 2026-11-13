"""Modul respons: Gemini API menyusun balasan empatik HANYA dari hasil pemetaan emosi.

Teks yang ditulis pengguna tidak pernah dikirim ke Gemini, baik pesan terbaru maupun riwayatnya.
Yang dikirim ke Gemini hanya:
  1. emosi terdeteksi, tingkat keyakinannya, dan emosi kedua bila menonjol (dari model IndoBERTweet),
  2. urutan emosi pada giliran-giliran sebelumnya (pemetaan emosi sepanjang percakapan),
  3. balasan chatbot sebelumnya (yang juga disusun tanpa teks pengguna), supaya balasan tidak berulang,
  4. jenis pesan terbaru (menyapa, bercerita, meminta saran, berterima kasih, jawaban singkat) dari penggolong lokal
     jenis_pesan.py, serta tahap percakapan (awal, tengah, lanjut),
  5. penanda kehati-hatian dari model deteksi krisis bila risikonya sedang. Nomor layanan 119 ekstensi 8 baru
     disampaikan bila penanda muncul berulang (dua pesan berturut-turut).
Sebelum dikirim, isi permintaan diperiksa oleh periksa_privasi() supaya tidak ada potongan teks pengguna
yang ikut terkirim. Bila API key kosong, pemeriksaan gagal, atau panggilan gagal, dipakai respons
cadangan berbasis templat sehingga chatbot tetap menjawab.

Pemanggilan memakai REST API Gemini langsung (tanpa paket tambahan)."""
import json, random, re, time, urllib.request, urllib.error
from konfigurasi import GEMINI_API_KEY, GEMINI_MODEL, AMBANG_YAKIN
from jenis_pesan import JENIS, golongkan

NAMA_EMOSI = {'anger': 'marah atau kesal', 'fear': 'takut atau cemas', 'happy': 'senang',
              'sadness': 'sedih', 'neutral': 'netral'}
MAKS_RIWAYAT = 8            # jumlah giliran terakhir yang dipakai sebagai konteks
SUHU = 0.8
MAKS_TOKEN = 1024
MAKS_TUNGGU_429 = 30

# ------------------------------------------------------------------ rancangan prompt (Bab 3)
INSTRUKSI_SISTEM = """Kamu adalah "Teman Cerita", teman curhat dalam prototipe chatbot konseling virtual berbahasa Indonesia. Kamu berperan seperti konselor sebaya yang hangat untuk anak muda. Kamu bukan psikolog dan tidak menggantikan bantuan profesional.

PENTING: demi menjaga privasi, kamu TIDAK menerima isi pesan pengguna. Kamu hanya menerima peta percakapan dari sistem: jenis pesan terbaru (misalnya bercerita atau meminta saran), emosi yang terdeteksi beserta tingkat keyakinannya, urutan emosi pada pesan-pesan sebelumnya, dan tahap percakapan. Kamu juga melihat balasanmu sendiri sebelumnya.

TEKNIK KONSELING SEBAYA (OARS)
Gunakan teknik dari motivational interviewing: pertanyaan terbuka (Open question), afirmasi (Affirmation), refleksi perasaan (Reflection), dan rangkuman (Summary). Jangan memakai keempatnya sekaligus; pilih satu atau dua yang paling pas.

STRATEGI MENURUT TAHAP PERCAKAPAN
- Tahap awal: dengarkan dan validasi. Refleksikan perasaannya, lalu undang ia bercerita dengan satu pertanyaan terbuka. Belum perlu memberi saran.
- Tahap tengah: gali lebih dalam. Refleksikan perubahan perasaannya antargiliran, beri afirmasi atas keberaniannya bercerita, dan tanyakan apa yang paling ia butuhkan.
- Tahap lanjut: bantu ia melangkah. Rangkum perjalanan perasaannya secara singkat, lalu tawarkan satu langkah kecil yang umum dan aman sebagai pilihan, misalnya menarik napas pelan, menuliskan perasaan, istirahat sejenak, atau menghubungi orang yang dipercaya.

STRATEGI MENURUT JENIS PESAN
- Menyapa atau membuka percakapan: sambut dengan hangat dan singkat, lalu persilakan ia bercerita. Jangan langsung menebak perasaannya.
- Bercerita atau mengungkapkan perasaan: ikuti strategi tahap percakapan.
- Bertanya atau meminta saran: jangan membalas dengan pertanyaan lagi. Berikan satu atau dua saran umum yang konkret dan aman sesuai emosinya, lalu tanyakan mana yang menurutnya bisa dicoba.
- Berterima kasih atau menutup percakapan: tutup dengan hangat, beri afirmasi bahwa ia sudah berani bercerita, dan sampaikan bahwa ia boleh kembali kapan saja. Jangan mengajukan pertanyaan baru.
- Jawaban singkat: jangan memaksa. Beri ruang, tawarkan pertanyaan yang lebih ringan atau sampaikan bahwa tidak apa-apa bila ia belum ingin bercerita.

MEMBACA PETA EMOSI
- Jangan menyalin nama emosi dari peta (misalnya jangan menulis "kamu sedang merasa takut atau cemas"). Ungkapkan dengan bahasa sehari-hari, misalnya "kedengarannya bikin deg-degan ya" atau "pasti rasanya nyesek".
- Bila keyakinan disebut rendah atau belum pasti, jangan menyimpulkan perasaannya. Tanyakan dengan lembut apa yang sedang ia rasakan.
- Bila ada emosi kedua yang menonjol, akui keduanya.
- Bila emosi negatif bertahan atau menguat, tunjukkan bahwa kamu memperhatikan. Bila emosi membaik, akui perubahan itu.
- Jangan pernah menyebut model, label, peta, catatan, atau angka probabilitas kepada pengguna.
- Jangan mengarang detail cerita yang tidak kamu ketahui. Jangan menyebut kejadian, orang, atau tempat tertentu.

MENYESUAIKAN DENGAN EMOSI
- Sedih: lembut dan pelan, beri ruang, jangan buru-buru menghibur.
- Takut atau cemas: menenangkan, bantu ia merasa lebih aman.
- Marah: akui kemarahannya tanpa menghakimi.
- Senang: ikut senang dengan tulus.
- Netral: ramah dan terbuka.

GAYA BAHASA
- Bahasa Indonesia sehari-hari anak muda yang hangat dan sopan, memakai "aku" dan "kamu". Boleh santai, tetapi tidak berlebihan.
- 2 sampai 4 kalimat dalam satu paragraf pendek.
- Tanpa daftar, tanpa format tebal atau miring, paling banyak satu emoji.
- Variasikan pembuka kalimat. Jangan membuka dengan pola yang sama seperti balasanmu sebelumnya, dan jangan mengulang kalimat atau pertanyaan yang sudah dipakai.

BATASAN
- Jangan memberi diagnosis, menyebut nama gangguan jiwa, atau menyarankan obat.
- Jangan menggurui atau menghakimi.
- Bila diberi penanda kehati-hatian, tanyakan keadaannya dengan sangat lembut dan beri ruang untuk bercerita. Jangan menyebut layanan atau nomor telepon apa pun.
- Hanya bila diberi penanda kehati-hatian berulang, tanyakan keadaannya dengan sangat lembut dan sampaikan bahwa layanan kesehatan jiwa Kemenkes bisa dihubungi di 119 ekstensi 8.

CONTOH (hanya untuk menunjukkan gaya, jangan disalin)
Peta: tahap awal; jenis pesan bercerita; emosi sedih dengan keyakinan tinggi.
Balasan: Kedengarannya hari-harimu lagi berat banget ya. Nggak apa-apa kok ngerasa kayak gitu, kamu nggak harus kelihatan kuat terus. Kalau mau, ceritain aja pelan-pelan, bagian mana yang paling bikin kepikiran?

Peta: tahap lanjut; jenis pesan meminta saran; emosi takut atau cemas, sebelumnya sedih lalu marah.
Balasan: Dari tadi kayaknya perasaanmu campur aduk, dari sedih, kesal, sampai sekarang khawatir, dan wajar kalau kamu bingung mulai dari mana. Satu hal kecil yang bisa dicoba: tulis dulu apa yang paling bikin khawatir, lalu pilih satu yang paling mungkin kamu lakukan minggu ini. Cerita ke satu orang yang kamu percaya juga bisa bikin bebannya terasa lebih ringan. Menurutmu, mana yang paling masuk akal buat dicoba duluan?

Peta: tahap lanjut; jenis pesan berterima kasih atau menutup percakapan; emosi senang.
Balasan: Sama-sama, aku ikut senang kamu merasa sedikit lebih lega. Makasih juga udah mau cerita, itu butuh keberanian lho. Kapan pun butuh tempat cerita lagi, aku ada di sini ya."""

CADANGAN_JENIS = {
    'sapaan': ["Hai, makasih udah mampir. Aku di sini buat dengerin, ceritain aja apa yang lagi kamu rasain ya.",
               "Halo! Senang kamu mau ngobrol. Pelan-pelan aja, apa yang lagi pengen kamu ceritain?"],
    'penutup': ["Sama-sama, makasih juga udah mau cerita. Kapan pun butuh tempat cerita lagi, aku ada di sini ya.",
                "Aku senang bisa nemenin kamu cerita. Jaga diri baik-baik ya, dan kembali aja kapan pun kamu butuh."],
}

CADANGAN = {
    'anger': ["Kedengarannya kamu sedang kesal sekali, dan perasaan itu wajar. Mau cerita bagian mana yang paling mengganggumu?",
              "Aku bisa menangkap rasa marahmu, dan itu valid. Apa yang paling kamu butuhkan sekarang?"],
    'fear': ["Rasa takut dan cemas memang berat untuk dibawa sendiri. Apa yang paling kamu khawatirkan saat ini?",
             "Wajar kalau kamu merasa cemas. Mau cerita apa yang sedang terlintas di pikiranmu?"],
    'happy': ["Senang sekali mendengarnya! Apa yang paling bikin kamu bahagia hari ini?",
              "Ikut senang untukmu. Ceritakan lagi dong, apa yang membuatnya terasa istimewa?"],
    'sadness': ["Aku ikut sedih mendengarnya. Tidak apa-apa merasa seperti itu. Mau cerita lebih banyak?",
                "Kedengarannya berat sekali yang sedang kamu rasakan. Aku di sini mendengarkan. Apa yang paling terasa menyesakkan?"],
    'neutral': ["Terima kasih sudah cerita. Bagaimana perasaanmu tentang hal itu?",
                "Aku mendengarkan. Ada yang ingin kamu ceritakan lebih lanjut?"],
}


def deskripsi_emosi(label, probs):
    """Kalimat ringkas tentang emosi satu pesan, dari label dan probabilitas terkalibrasi."""
    urut = sorted(probs.items(), key=lambda x: -x[1])
    p = probs.get(label, 0)
    if p < AMBANG_YAKIN:
        c = f'belum pasti, dugaan terkuat {NAMA_EMOSI[label]} (keyakinan rendah, {p:.0%})'
    else:
        c = f'{NAMA_EMOSI[label]} (keyakinan {p:.0%})'
    if len(urut) > 1 and urut[1][1] >= 0.25:
        c += f', disertai {NAMA_EMOSI[urut[1][0]]} ({urut[1][1]:.0%})'
    return c


def hati_berulang(riwayat, hati_hati):
    """True bila pesan ini dan pesan pengguna tepat sebelumnya sama-sama diberi penanda hati-hati."""
    if not hati_hati:
        return False
    sebelumnya = [g for g in (riwayat or []) if g.get('peran') == 'pengguna'][-1:]
    return any(g.get('hati_hati') for g in sebelumnya)


def tahap_percakapan(giliran):
    return 'awal' if giliran <= 2 else ('tengah' if giliran <= 4 else 'lanjut')


def susun_isi(label, probs, riwayat=None, hati_hati=False, jenis='bercerita'):
    """Menyusun isi permintaan ke Gemini. Tidak menerima teks pengguna sama sekali.
    riwayat: daftar giliran; giliran pengguna hanya berisi 'label' dan 'probs', giliran chatbot berisi 'teks'.
    Hasilnya satu giliran 'user' dengan dua bagian: (1) peta emosi yang disusun sistem dan
    (2) balasan chatbot sebelumnya (disusun Gemini sendiri tanpa teks pengguna) agar tidak berulang."""
    rw = (riwayat or [])[-MAKS_RIWAYAT:]
    peta = [deskripsi_emosi(g['label'], g['probs']) for g in rw if g.get('peran') == 'pengguna' and g.get('label')]
    giliran = sum(1 for g in (riwayat or []) if g.get('peran') == 'pengguna') + 1
    baris = [f'Giliran ke-{giliran} dalam percakapan ini (tahap {tahap_percakapan(giliran)}).',
             f'Jenis pesan terbaru: {JENIS.get(jenis, JENIS["bercerita"])}.',
             f'Emosi pada pesan terbaru: {deskripsi_emosi(label, probs)}.']
    if peta:
        baris.append('Urutan emosi pada pesan-pesan sebelumnya: ' + '; '.join(peta) + '.')
    if hati_berulang(rw, hati_hati):
        baris.append('Penanda kehati-hatian berulang: model deteksi krisis menilai beberapa pesan terakhir perlu ditanggapi dengan sangat hati-hati.')
    elif hati_hati:
        baris.append('Penanda kehati-hatian: model deteksi krisis menilai pesan ini perlu ditanggapi dengan sangat hati-hati.')
    baris.append('Susun balasan untuk pengguna berdasarkan peta percakapan ini.')
    bagian = [{'text': '[Peta percakapan, jangan disebut ke pengguna] ' + ' '.join(baris)}]
    sebelumnya = [g['teks'] for g in rw if g.get('peran') == 'chatbot' and g.get('teks')]
    if sebelumnya:
        bagian.append({'text': '[Balasanmu sebelumnya, jangan diulang] ' + ' || '.join(sebelumnya[-3:])})
    return [{'role': 'user', 'parts': bagian}]


def periksa_privasi(badan, teks_pengguna):
    """Mengembalikan True bila tidak ada potongan teks pengguna (3 kata berurutan atau kata unik
    sepanjang minimal 6 huruf yang tidak ada di instruksi) di dalam isi permintaan ke Gemini."""
    # yang diperiksa adalah bagian peta emosi yang disusun sistem; balasan chatbot sebelumnya disusun Gemini
    # tanpa pernah melihat teks pengguna sehingga kemiripan kata di sana bersifat kebetulan
    muatan = ' '.join(c['parts'][0]['text'] for c in badan['contents']).lower()
    kata = re.findall(r'\w+', (teks_pengguna or '').lower())
    for i in range(len(kata) - 2):
        if ' '.join(kata[i:i + 3]) in muatan:
            return False
    acuan = (INSTRUKSI_SISTEM + ' '.join(NAMA_EMOSI.values()) + ' giliran percakapan pesan terbaru sebelumnya keyakinan '
             'dugaan terkuat disertai urutan penanda kehati-hatian model deteksi krisis menilai perlu ditanggapi '
             'sangat beberapa terakhir berulang susun balasan untuk pengguna berdasarkan peta emosi jangan disebut belum pasti rendah '
             'jenis tahap awal tengah lanjut ' + ' '.join(JENIS.values())).lower()
    for k in set(kata):
        if len(k) >= 6 and k in muatan and k not in acuan:
            return False
    return True


def _panggil_gemini(isi, tanpa_berpikir=True, coba_lagi=True):
    gen = {'temperature': SUHU, 'maxOutputTokens': MAKS_TOKEN}
    if tanpa_berpikir:
        gen['thinkingConfig'] = {'thinkingBudget': 0}
    badan = {'systemInstruction': {'parts': [{'text': INSTRUKSI_SISTEM}]}, 'contents': isi, 'generationConfig': gen}
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent'
    req = urllib.request.Request(url, data=json.dumps(badan).encode('utf-8'),
                                 headers={'Content-Type': 'application/json', 'x-goog-api-key': GEMINI_API_KEY})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        pesan_penuh = e.read().decode('utf-8', 'ignore'); pesan = pesan_penuh[:300]
        if tanpa_berpikir and e.code == 400 and 'thinking' in pesan.lower():
            return _panggil_gemini(isi, tanpa_berpikir=False, coba_lagi=coba_lagi)
        if e.code == 429 and coba_lagi:
            m = re.search(r'retry in ([0-9.]+)s', pesan_penuh)
            tunggu = float(m.group(1)) + 1 if m else 15
            if tunggu <= MAKS_TUNGGU_429:
                time.sleep(tunggu)
                return _panggil_gemini(isi, tanpa_berpikir, coba_lagi=False)
        raise RuntimeError(f'HTTP {e.code}: {pesan}')
    kand = (d.get('candidates') or [{}])[0]
    teks = ''.join(p.get('text', '') for p in kand.get('content', {}).get('parts', [])).strip()
    if not teks:
        raise RuntimeError(f"jawaban kosong (finishReason={kand.get('finishReason')})")
    return teks


def respons_cadangan(label, hati_hati=False, seed=None, berulang=False, jenis=None):
    pilihan = CADANGAN_JENIS.get(jenis) if not hati_hati else None
    r = random.Random(seed).choice(pilihan or CADANGAN[label])
    if hati_hati and berulang:
        r += (' Kalau rasanya makin berat, kamu juga bisa bicara dengan tenaga profesional lewat layanan '
              'kesehatan jiwa Kemenkes di 119 ekstensi 8.')
    return r


# Untuk pengujian privasi: bila diisi fungsi, permintaan ke Gemini diteruskan ke fungsi ini (tidak ke internet)
PENYADAP = None


def buat_respons(teks, label, probs, hati_hati=False, riwayat=None):
    """Mengembalikan (teks_respons, sumber) dengan sumber 'gemini' atau 'cadangan'.
    Parameter teks hanya dipakai untuk memeriksa bahwa teks itu tidak ikut terkirim."""
    jenis = golongkan(teks)                     # dihitung di komputer lokal; hanya nama jenisnya yang dikirim
    isi = susun_isi(label, probs, riwayat, hati_hati, jenis)
    badan = {'contents': isi}
    if not periksa_privasi(badan, teks):
        print('[privasi] isi permintaan memuat potongan teks pengguna, Gemini tidak dipanggil', flush=True)
        return respons_cadangan(label, hati_hati, berulang=hati_berulang(riwayat, hati_hati), jenis=jenis), 'cadangan'
    if PENYADAP is not None:
        return PENYADAP(INSTRUKSI_SISTEM, isi), 'gemini'
    if GEMINI_API_KEY:
        try:
            return _panggil_gemini(isi), 'gemini'
        except Exception as e:
            print(f'[peringatan] Gemini gagal, memakai respons cadangan: {e}', flush=True)
    return respons_cadangan(label, hati_hati, berulang=hati_berulang(riwayat, hati_hati), jenis=jenis), 'cadangan'
