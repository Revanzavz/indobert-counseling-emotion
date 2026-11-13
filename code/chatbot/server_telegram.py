"""Bot Telegram untuk chatbot konseling. Jalankan lewat jalankan_telegram.bat.

Bot memakai metode long polling (getUpdates) dari Telegram Bot API, sehingga tidak memerlukan alamat publik,
ngrok, atau webhook. Laptop cukup terhubung ke internet. Setiap pesan diproses dengan pipeline.proses_pesan:
model deteksi krisis -> model emosi -> modul respons (Gemini hanya menerima peta emosi, bukan teks pengguna).

Riwayat per pengguna disimpan di memori tanpa teks pengguna (hanya label emosi dan balasan bot) dan hilang saat
bot dimatikan. Log tidak memuat isi pesan kecuali PANTAU=1 di .env (untuk dokumentasi pengujian, beri tahu penguji).
Token bot dibaca dari TELEGRAM_BOT_TOKEN di file .env."""
import hashlib, json, os, threading, time, urllib.error, urllib.parse, urllib.request

import konfigurasi  # noqa: memuat .env
from pipeline import proses_pesan, tambah_riwayat

TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
API = f'https://api.telegram.org/bot{TOKEN}/'
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, 'hasil', 'log_telegram.txt')
PANTAU = os.environ.get('PANTAU', '0').strip() == '1'
URUTAN = ['anger', 'fear', 'happy', 'sadness', 'neutral']
SAMBUTAN = ("Halo, aku Teman Cerita, prototipe chatbot konseling untuk penelitian tugas akhir. "
            "Kamu bisa menceritakan apa yang sedang kamu rasakan.\n\n"
            "Catatan: aku bukan psikolog. Isi pesanmu dibaca oleh model di komputer peneliti dan tidak dikirim ke layanan AI luar; "
            "layanan AI hanya menerima hasil pembacaan emosi. Ketik /reset untuk memulai percakapan baru.")

riwayat, kunci = {}, {}
kunci_global = threading.Lock()


def catat(teks):
    baris = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {teks}"
    print(baris, flush=True)
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(baris + '\n')
    except OSError:
        pass


def panggil(metode, **param):
    data = urllib.parse.urlencode(param).encode()
    with urllib.request.urlopen(urllib.request.Request(API + metode, data=data), timeout=param.get('timeout', 20) + 10) as r:
        return json.load(r)


def kirim(chat_id, teks):
    for i in range(0, len(teks), 4000):                     # batas satu pesan Telegram 4096 karakter
        panggil('sendMessage', chat_id=chat_id, text=teks[i:i + 4000])


def samaran(chat_id):
    return hashlib.sha256(str(chat_id).encode()).hexdigest()[:8]


def tangani(pesan):
    try:
        _tangani(pesan)
    except Exception as e:                                   # galat satu pesan tidak menghentikan bot
        catat(f'[galat] {type(e).__name__}: {e}')


def _tangani(pesan):
    chat_id = pesan['chat']['id']; teks = (pesan.get('text') or '').strip(); sid = samaran(chat_id)
    with kunci_global:
        k = kunci.setdefault(chat_id, threading.Lock())
    with k:
        if not teks:
            kirim(chat_id, 'Untuk saat ini aku hanya bisa membaca pesan teks.'); return
        if teks in ('/start', '/reset'):
            riwayat.pop(chat_id, None); kirim(chat_id, SAMBUTAN); catat(f'[{sid}] {teks}'); return
        try:
            panggil('sendChatAction', chat_id=chat_id, action='typing')
        except Exception:
            pass
        rw = riwayat.setdefault(chat_id, [])
        h = proses_pesan(teks, riwayat=rw)
        tambah_riwayat(rw, h)
        kirim(chat_id, h['respons'])
        if h['jalur'] == 'krisis':
            info = f"KRISIS p={h['p_krisis']:.2f} -> rujukan"
        else:
            emosi = ', '.join(f"{e} {h['probs'][e]:.0%}" for e in URUTAN if e in h['probs'])
            info = (f"{h.get('jenis_pesan')} | emosi {h['label']} | {emosi} | krisis p={h['p_krisis']:.2f}"
                    f"{' (hati-hati)' if h['status_krisis'] == 'hati_hati' else ''} | respons {h['sumber_respons']} | {h.get('waktu_detik')} dtk")
        catat(f'[{sid}] {info}')
        if PANTAU:
            print(f'    pengguna: {teks}\n    bot     : {h["respons"]}\n', flush=True)


def main():
    if not TOKEN:
        raise SystemExit('TELEGRAM_BOT_TOKEN belum diisi di file .env')
    print('Memuat model...', flush=True)
    import krisis
    from emosi import petakan_emosi
    krisis.periksa('halo'); petakan_emosi('halo')                      # muat kedua model di awal
    me = panggil('getMe')['result']
    catat(f"Bot @{me['username']} aktif. Buka Telegram dan kirim pesan ke bot. Tekan Ctrl+C untuk berhenti.")
    offset = 0
    while True:
        try:
            r = panggil('getUpdates', offset=offset, timeout=30, allowed_updates='["message"]')
            for u in r.get('result', []):
                offset = u['update_id'] + 1
                if 'message' in u:
                    threading.Thread(target=tangani, args=(u['message'],), daemon=True).start()
        except KeyboardInterrupt:
            break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            catat(f'[peringatan] koneksi ke Telegram terputus ({e}), mencoba lagi...'); time.sleep(5)


if __name__ == '__main__':
    main()
