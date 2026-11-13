"""Webhook WhatsApp (Twilio) untuk chatbot konseling. Jalankan lewat jalankan_server.bat.

Twilio mengirim setiap pesan WhatsApp masuk sebagai HTTP POST ke alamat /whatsapp di server ini.
Server memproses pesan lewat pipeline.proses_pesan lalu membalas dalam format TwiML, dan Twilio
meneruskan balasan itu ke pengguna. Riwayat percakapan disimpan terpisah per nomor pengirim
(hanya di memori, hilang saat server dimatikan). Server memakai pustaka bawaan Python saja.
"""
import base64, hashlib, hmac, json, os, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode
from xml.sax.saxutils import escape

import konfigurasi  # noqa: memuat .env
import respons
from pipeline import proses_pesan

PORT = int(os.environ.get('PORT', '5000'))
AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', '').strip()
ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', '').strip()
HERE = os.path.dirname(os.path.abspath(__file__))
# Cara mengirim balasan ke pengguna:
#   teks  : jawaban webhook berupa teks polos (untuk mode percobaan "Try out WhatsApp" yang menolak TwiML dan API)
#   twiml : jawaban webhook berupa TwiML (WhatsApp Sandbox)
#   api   : dikirim lewat REST API Twilio (Sandbox atau akun berbayar; perlu TWILIO_ACCOUNT_SID)
CARA_BALAS = os.environ.get('CARA_BALAS', 'twiml').strip().lower()
# PANTAU=1 menampilkan isi pesan dan jawaban di layar serta menyimpannya ke hasil/percakapan_whatsapp.csv
# (untuk dokumentasi pengujian). Nyalakan hanya bila penguji sudah tahu percakapannya dicatat.
PANTAU = os.environ.get('PANTAU', '0').strip() == '1'
CSV_PANTAU = os.path.join(HERE, 'hasil', 'percakapan_whatsapp.csv')
LOG = os.path.join(HERE, 'hasil', 'log_server.txt')
PUBLIC_URL = os.environ.get('PUBLIC_URL', '').strip().rstrip('/')      # mis. https://nama-anda.ngrok-free.app
if PUBLIC_URL.endswith('/whatsapp'):
    PUBLIC_URL = PUBLIC_URL[:-len('/whatsapp')]
if PUBLIC_URL and not PUBLIC_URL.startswith('http'):
    PUBLIC_URL = 'https://' + PUBLIC_URL                               # toleransi bila https:// terlewat
MAKS_KARAKTER = 1500                                            # batas satu pesan WhatsApp di Twilio adalah 1600
MAKS_GILIRAN = 20
KATA_RESET = {'reset', 'mulai baru', 'mulai ulang', '/reset'}
respons.MAKS_TUNGGU_429 = 20 if CARA_BALAS == 'api' else 6   # balasan lewat webhook dibatasi waktu 15 detik

riwayat = {}                       # nomor pengirim -> daftar giliran
kunci = threading.Lock()


def catat(teks):
    """Mencetak ke layar dan ke hasil/log_server.txt. Isi pesan pengguna tidak pernah dicatat."""
    baris = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {teks}"
    print(baris, flush=True)
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(baris + '\n')
    except OSError:
        pass


URUTAN_EMOSI = ['anger', 'fear', 'happy', 'sadness', 'neutral']


def semua_emosi(h):
    """Persentase kelima kelas emosi, diurutkan dari yang tertinggi. Kosong bila model tidak dijalankan."""
    if not h.get('probs'):
        return ''
    return ' | '.join(f'{l} {p:.1%}' for l, p in sorted(h['probs'].items(), key=lambda x: -x[1]))


def pantau(dari, teks, h):
    """Menampilkan dan menyimpan satu giliran percakapan bila PANTAU=1."""
    if not PANTAU:
        return
    emosi = h.get('label') or ''
    prob = f"{h['probs'][emosi]:.4f}" if emosi else ''
    print(f"   Pengguna: {teks}\n   Chatbot : {h['respons']}\n", flush=True)
    try:
        import csv
        baru = not os.path.exists(CSV_PANTAU)
        os.makedirs(os.path.dirname(CSV_PANTAU), exist_ok=True)
        with open(CSV_PANTAU, 'a', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            if baru:
                w.writerow(['waktu', 'pengirim', 'pesan', 'jalur', 'mekanisme_krisis', 'emosi', 'prob'] + [f'p_{l}' for l in URUTAN_EMOSI] + ['sumber_respons', 'respons'])
            w.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), samarkan(dari), teks, h.get('jalur'), h.get('mekanisme_krisis') or '',
                        emosi, prob] + [f"{h['probs'][l]:.4f}" if h.get('probs') else '' for l in URUTAN_EMOSI]
                       + [h.get('sumber_respons'), h['respons']])
    except OSError:
        pass


def kirim_whatsapp(dari, ke, teks):
    """Mengirim balasan lewat REST API Twilio (Messages). Mengembalikan (berhasil, keterangan)."""
    url = f'https://api.twilio.com/2010-04-01/Accounts/{ACCOUNT_SID}/Messages.json'
    data = urlencode({'From': dari, 'To': ke, 'Body': teks[:MAKS_KARAKTER]}).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    req.add_header('Authorization', 'Basic ' + base64.b64encode(f'{ACCOUNT_SID}:{AUTH_TOKEN}'.encode()).decode())
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.load(r)
        return True, f"status {d.get('status')} sid ...{str(d.get('sid'))[-6:]}"
    except urllib.error.HTTPError as e:
        try:
            d = json.loads(e.read().decode('utf-8', 'ignore'))
            return False, f"HTTP {e.code} kode Twilio {d.get('code')}: {d.get('message')}"
        except Exception:
            return False, f'HTTP {e.code}'
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'


def proses_dan_balas(dari, ke, teks):
    """Dijalankan di thread terpisah supaya webhook langsung dijawab dan tidak terkena batas waktu Twilio."""
    t0 = time.time(); detail = ''
    try:
        with kunci:
            if teks.lower() in KATA_RESET:
                riwayat.pop(dari, None)
                balasan, ringkas = 'Percakapan dimulai dari awal. Silakan ceritakan apa yang sedang kamu rasakan.', 'reset'
            elif not teks:
                balasan, ringkas = 'Saat ini aku baru bisa membaca pesan teks. Ceritakan lewat tulisan ya.', 'bukan teks'
            else:
                h = proses_pesan(teks, riwayat=riwayat.get(dari))
                r = riwayat.setdefault(dari, [])
                r += [{'peran': 'pengguna', 'teks': teks}, {'peran': 'chatbot', 'teks': h['respons']}]
                del r[:-MAKS_GILIRAN]
                emosi = f"{h['label']} {h['probs'][h['label']]:.0%}" if h['label'] else '-'
                balasan = h['respons']
                ringkas = f"jalur {h['jalur']} | emosi {emosi} | respons {h['sumber_respons']}"
                detail = semua_emosi(h)
                pantau(dari, teks, h)
    except Exception as e:
        balasan, ringkas = 'Maaf, sedang ada gangguan pada sistem. Coba kirim pesanmu lagi sebentar lagi.', f'GALAT {type(e).__name__}: {e}'
    ok, ket = kirim_whatsapp(ke, dari, balasan)
    catat(f"{samarkan(dari)} | {ringkas} | {time.time() - t0:.1f} dtk | kirim {'OK' if ok else 'GAGAL'}: {ket}")
    if detail:
        catat(f'   probabilitas: {detail}')


def tanda_tangan_sah(url, params, tanda):
    """Memastikan permintaan benar-benar dari Twilio (X-Twilio-Signature, HMAC-SHA1)."""
    data = url + ''.join(k + params[k] for k in sorted(params))
    harap = base64.b64encode(hmac.new(AUTH_TOKEN.encode(), data.encode('utf-8'), hashlib.sha1).digest()).decode()
    return hmac.compare_digest(harap, tanda or '')


def twiml(teks):
    return f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{escape(teks[:MAKS_KARAKTER])}</Message></Response>'


def samarkan(nomor):
    return nomor[:-6] + '******' if len(nomor) > 8 else '***'


class Penangan(BaseHTTPRequestHandler):
    def _kirim(self, kode, isi, tipe='text/xml; charset=utf-8'):
        b = isi.encode('utf-8')
        self.send_response(kode); self.send_header('Content-Type', tipe)
        self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._kirim(200, 'Server chatbot konseling aktif. Webhook Twilio: POST /whatsapp', 'text/plain; charset=utf-8')

    def do_POST(self):
        if self.path.split('?')[0].rstrip('/') != '/whatsapp':
            return self._kirim(404, 'tidak ditemukan', 'text/plain; charset=utf-8')
        badan = self.rfile.read(int(self.headers.get('Content-Length', 0))).decode('utf-8')
        p = {k: v[0] for k, v in parse_qs(badan, keep_blank_values=True).items()}
        if AUTH_TOKEN and PUBLIC_URL and not tanda_tangan_sah(PUBLIC_URL + '/whatsapp', p, self.headers.get('X-Twilio-Signature')):
            catat(f'[ditolak] tanda tangan Twilio tidak cocok. Pastikan Webhook URL di Twilio adalah {PUBLIC_URL}/whatsapp '
                  'dan TWILIO_AUTH_TOKEN di .env benar.')
            return self._kirim(403, 'tanda tangan tidak sah', 'text/plain; charset=utf-8')
        dari, ke, teks = p.get('From', ''), p.get('To', ''), p.get('Body', '').strip()
        if CARA_BALAS == 'api' and ACCOUNT_SID and AUTH_TOKEN:
            # Balasan dikirim lewat REST API Twilio; webhook dijawab kosong supaya tidak ada balasan ganda.
            threading.Thread(target=proses_dan_balas, args=(dari, ke, teks), daemon=True).start()
            return self._kirim(200, '<?xml version="1.0" encoding="UTF-8"?><Response></Response>')
        # Mode teks / twiml: balasan dikembalikan langsung pada jawaban webhook.
        polos = CARA_BALAS == 'teks'
        twiml = (lambda t: t[:MAKS_KARAKTER]) if polos else globals()['twiml']
        tipe = 'text/plain; charset=utf-8' if polos else 'text/xml; charset=utf-8'
        if teks.lower() in KATA_RESET:
            riwayat.pop(dari, None)
            return self._kirim(200, twiml('Percakapan dimulai dari awal. Silakan ceritakan apa yang sedang kamu rasakan.'), tipe)
        if not teks:
            return self._kirim(200, twiml('Saat ini aku baru bisa membaca pesan teks. Ceritakan lewat tulisan ya.'), tipe)
        t0 = time.time()
        try:
            h = proses_pesan(teks, riwayat=riwayat.get(dari))
        except Exception as e:
            catat(f'[galat] {type(e).__name__}: {e}')
            return self._kirim(200, twiml('Maaf, sedang ada gangguan pada sistem. Coba kirim pesanmu lagi sebentar lagi.'), tipe)
        r = riwayat.setdefault(dari, [])
        r += [{'peran': 'pengguna', 'teks': teks}, {'peran': 'chatbot', 'teks': h['respons']}]
        del r[:-MAKS_GILIRAN]
        emosi = f"{h['label']} {h['probs'][h['label']]:.0%}" if h['label'] else '-'
        catat(f"{samarkan(dari)} | jalur {h['jalur']} | emosi {emosi} | respons {h['sumber_respons']} | {time.time() - t0:.1f} dtk | balas via {CARA_BALAS}")
        if semua_emosi(h):
            catat(f'   probabilitas: {semua_emosi(h)}')
        pantau(dari, teks, h)
        self._kirim(200, twiml(h['respons']), tipe)

    def log_message(self, *a):          # matikan log bawaan
        pass


if __name__ == '__main__':
    print('Memuat model emosi, tunggu sebentar...', flush=True)
    from emosi import petakan_emosi
    petakan_emosi('halo')              # dimuat di awal supaya pesan pertama tidak melewati batas waktu Twilio
    print(f'Modul respons : {"Gemini API" if konfigurasi.GEMINI_API_KEY else "respons cadangan (GEMINI_API_KEY kosong)"}')
    if PUBLIC_URL:
        print(f'Webhook yang harus diisi di Twilio: {PUBLIC_URL}/whatsapp')
    print(f'Verifikasi Twilio: {"aktif" if AUTH_TOKEN and PUBLIC_URL else "nonaktif (isi TWILIO_AUTH_TOKEN dan PUBLIC_URL di .env)"}')
    print(f'Cara membalas : {CARA_BALAS}')
    print(f'Pemantauan isi: {"AKTIF, percakapan ditampilkan dan disimpan ke hasil/percakapan_whatsapp.csv" if PANTAU else "nonaktif (isi PANTAU=1 di .env untuk menyalakan)"}')
    print(f'Server berjalan di http://localhost:{PORT}  (webhook: /whatsapp). Tekan Ctrl+C untuk berhenti.', flush=True)
    ThreadingHTTPServer(('0.0.0.0', PORT), Penangan).serve_forever()
