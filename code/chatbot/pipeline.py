"""Alur pemrosesan satu pesan:

  pesan -> [model deteksi krisis] --p >= ambang krisis--> pesan rujukan, selesai
              | tidak
              v
          [model emosi IndoBERTweet] -> (label, probabilitas)
              |
              v
          [modul respons Gemini] <- HANYA peta emosi (tanpa teks pengguna) + penanda hati-hati bila p >= ambang hati

Riwayat percakapan disimpan tanpa teks pengguna: giliran pengguna hanya berisi label dan probabilitas emosi,
giliran chatbot berisi balasan chatbot. Gunakan tambah_riwayat() untuk memperbarui riwayat."""
import time
import krisis
import respons as modul_respons

MAKS_GILIRAN = 20


def proses_pesan(teks, riwayat=None, pemeta=None, generator=None, penilai_krisis=None):
    """Memproses satu pesan pengguna. pemeta, generator, dan penilai_krisis bisa diganti saat pengujian."""
    t0 = time.time()
    teks = (teks or '').strip()
    from jenis_pesan import golongkan
    hasil = {'pesan': teks, 'jenis_pesan': golongkan(teks), 'jalur': None, 'status_krisis': None, 'p_krisis': None,
             'label': None, 'probs': None, 'respons': None, 'sumber_respons': None}
    if not teks:
        hasil.update(jalur='kosong', respons='Aku belum menerima pesanmu. Ceritakan saja apa yang sedang kamu rasakan.')
        return hasil

    # 1) deteksi krisis dengan model, sebelum klasifikasi emosi
    status, p = (penilai_krisis or krisis.periksa)(teks)
    hasil.update(status_krisis=status, p_krisis=round(p, 4))
    if status == 'krisis':
        hasil.update(jalur='krisis', respons=krisis.PESAN_KRISIS, sumber_respons='rujukan')
        hasil['waktu_detik'] = round(time.time() - t0, 3)
        return hasil

    # 2) klasifikasi emosi
    if pemeta is None:
        from emosi import petakan_emosi as pemeta
    label, probs = pemeta(teks)
    hasil.update(label=label, probs=probs)

    # 3) respons dari peta emosi (teks pengguna tidak dikirim ke Gemini)
    gen = generator or modul_respons.buat_respons
    teks_r, sumber = gen(teks, label, probs, hati_hati=(status == 'hati_hati'), riwayat=riwayat)
    hasil.update(jalur='respons', respons=teks_r, sumber_respons=sumber)
    hasil['waktu_detik'] = round(time.time() - t0, 3)
    return hasil


def tambah_riwayat(riwayat, hasil):
    """Menambahkan satu giliran ke riwayat TANPA menyimpan teks pengguna."""
    riwayat.append({'peran': 'pengguna', 'label': hasil.get('label'), 'probs': hasil.get('probs'),
                    'hati_hati': hasil.get('status_krisis') == 'hati_hati'})
    riwayat.append({'peran': 'chatbot', 'teks': hasil.get('respons') if hasil.get('jalur') == 'respons' else ''})
    del riwayat[:-2 * MAKS_GILIRAN]
    return riwayat
