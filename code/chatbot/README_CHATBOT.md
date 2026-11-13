# Prototipe Chatbot Konseling Virtual

Folder ini berisi backend chatbot yang mengintegrasikan model pemetaan emosi (IndoBERTweet hasil fine-tuning dua tahap) dengan jalur deteksi krisis dan modul respons generatif, sesuai Gambar 3.4 pada proposal.

## Alur satu pesan

Pesan pengguna lebih dulu diperiksa oleh deteksi krisis berbasis kata kunci (`krisis.cek_kata_kunci`). Bila cocok, sistem langsung mengirim pesan rujukan layanan darurat dan pesan tidak diteruskan ke model maupun Gemini. Bila tidak cocok, pesan diklasifikasikan oleh model emosi (`emosi.petakan_emosi`) yang menghasilkan label dan probabilitas lima kelas. Hasil itu diperiksa oleh mekanisme krisis kedua (`krisis.cek_emosi_dan_indikator`), yaitu label sadness atau fear dengan probabilitas minimal 0,70 yang disertai indikator kata krisis tidak langsung. Bila tidak ada indikasi krisis, teks pesan, label emosi, dan probabilitas dikirim ke modul respons (`respons.buat_respons`) yang memanggil Gemini API. Seluruh alur ada di `pipeline.proses_pesan`.

## Peran hasil pemetaan emosi

Hasil pemetaan emosi (label dan probabilitas lima kelas) diteruskan ke modul respons sebagai konteks, bukan sebagai perintah. Di dalam prompt, konteks ini dipakai untuk menentukan nada respons sesuai emosi, memberi tahu tingkat keyakinan model (di bawah 0,60 ditandai belum pasti sehingga respons lebih terbuka), dan menandai emosi campuran (emosi kedua dengan probabilitas minimal 0,25). Isi pesan pengguna tetap menjadi acuan utama. Penelitian ini tidak mengklaim bahwa konteks emosi meningkatkan kualitas respons generatif; pengujian pengaruh tersebut menjadi saran penelitian lanjutan.

Di luar prompt, hasil pemetaan emosi dipakai pada tiga bagian lain yang tidak bergantung pada model generatif: mekanisme kedua jalur deteksi krisis, pemilihan respons cadangan saat layanan generatif tidak tersedia, dan pencatatan emosi tiap pesan untuk pemantauan.

## Berkas

`konfigurasi.py` memuat pengaturan dan membaca file `.env`. `krisis.py` memuat daftar frasa krisis, normalisasi teks informal, dan pesan rujukan. `emosi.py` memuat model final dari `../Eksperimen/eksperimen/hasil/model_final`. `respons.py` memuat instruksi sistem untuk Gemini dan respons cadangan berbasis templat yang dipakai bila API key belum diisi atau panggilan gagal. `coba_chatbot.bat` menjalankan chatbot di terminal. `uji_fungsional.bat` menjalankan 30 pesan uji pada `uji_kasus.csv` dan menulis hasilnya ke `hasil/uji_fungsional.csv` dan `hasil/uji_fungsional.md`.

## Menyalakan Gemini

Salin `.env.contoh` menjadi `.env`, isi `GEMINI_API_KEY`, lalu pasang paket dengan `..\Eksperimen\eksperimen\.venv\Scripts\python.exe -m pip install google-genai`. Tanpa API key, chatbot tetap berjalan dengan respons cadangan.

## Catatan keselamatan

Daftar frasa krisis disusun mengacu pada kategori C-SSRS dan tanda peringatan WHO LIVE LIFE. Daftar ini adalah pengaman prototipe riset dan bukan alat skrining klinis. Pesan rujukan memakai layanan kesehatan jiwa Kementerian Kesehatan (119 ekstensi 8, healing119.id) dan nomor darurat 112.
