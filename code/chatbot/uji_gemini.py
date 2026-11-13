"""Uji modul respons Gemini: satu percakapan bersambung + beberapa pesan lepas. Hasil ke hasil/uji_gemini.txt"""
from pipeline import proses_pesan
from konfigurasi import GEMINI_MODEL
print('Model:', GEMINI_MODEL)
print('\n===== PERCAKAPAN BERSAMBUNG =====')
riwayat = []
for t in ['aku capek banget akhir-akhir ini, skripsi gak selesai-selesai',
          'iya, dosen pembimbingku susah dihubungi terus temen-temenku udah pada sidang',
          'aku jadi ngerasa bodoh dan ketinggalan jauh',
          'terus aku harus gimana ya']:
    h = proses_pesan(t, riwayat=riwayat)
    riwayat += [{'peran': 'pengguna', 'teks': t}, {'peran': 'chatbot', 'teks': h['respons']}]
    print(f"\nPengguna: {t}\n[{h['jalur']} | {h['label']} {h['probs'][h['label']]:.0%} | {h['sumber_respons']} | {h['waktu_detik']} dtk | {len(h['respons'].split())} kata]\nChatbot : {h['respons']}")
print('\n===== PESAN LEPAS =====')
for t in ['aku kesel banget sama temen sekelompok yang gak ngerjain apa-apa',
          'alhamdulillah akhirnya aku diterima magang di perusahaan impianku',
          'aku takut banget besok sidang']:
    h = proses_pesan(t)
    print(f"\nPengguna: {t}\n[{h['label']} {h['probs'][h['label']]:.0%} | {h['sumber_respons']} | {len(h['respons'].split())} kata]\nChatbot : {h['respons']}")
