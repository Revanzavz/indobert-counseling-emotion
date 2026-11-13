"""Coba chatbot lewat terminal (tanpa Telegram). Jalankan coba_chatbot.bat lalu ketik pesan."""
print('Memuat chatbot, tunggu sebentar...', flush=True)
from pipeline import proses_pesan, tambah_riwayat
from emosi import petakan_emosi
import krisis
from konfigurasi import GEMINI_API_KEY
krisis.periksa('halo'); petakan_emosi('halo')            # memuat kedua model di awal
print('Modul respons:', 'Gemini API (hanya menerima peta emosi)' if GEMINI_API_KEY else 'respons cadangan (GEMINI_API_KEY belum diisi di .env)')
print("Ketik pesan lalu Enter. Ketik 'keluar' untuk berhenti.\n")
riwayat = []
while True:
    try:
        t = input('Kamu   : ').strip()
    except (EOFError, KeyboardInterrupt):
        break
    if t.lower() in ('keluar', 'exit', 'q'):
        break
    h = proses_pesan(t, riwayat=riwayat)
    tambah_riwayat(riwayat, h)
    if h['jalur'] == 'krisis':
        print(f"[JALUR KRISIS | p krisis {h['p_krisis']:.2f}]")
    elif h['label']:
        print(f"[emosi: {h['label']} {h['probs'][h['label']]:.0%} | p krisis {h['p_krisis']:.2f}"
              f"{' hati-hati' if h['status_krisis'] == 'hati_hati' else ''} | respons: {h['sumber_respons']} | {h['waktu_detik']} dtk]")
    print('Chatbot:', h['respons'], '\n')
