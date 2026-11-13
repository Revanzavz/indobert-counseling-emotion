# Kelas Neutral dari DailyDialog (Langkah 1 Perbaikan Dataset Tahap 2)

## Tujuan

Pada eksperimen awal, kelas neutral tahap kedua diambil dari tweet Riccosan, sedangkan empat kelas emosi berasal dari terjemahan NLLB atas EmpatheticDialogues. Perbedaan asal dan gaya bahasa ini memungkinkan model mengenali neutral dari gayanya saja. Folder ini berisi pengganti kelas neutral yang berasal dari percakapan berbahasa Inggris berlabel "no emotion" pada DailyDialog (Li et al., 2017), lalu diterjemahkan dengan model dan konfigurasi NLLB yang sama, sehingga kelima kelas melewati pipeline terjemahan yang identik.

## Sumber data

DailyDialog berisi 13.118 dialog percakapan sehari-hari yang dianotasi manual dengan tujuh label emosi, salah satunya "no emotion". Lisensinya CC BY-NC-SA 4.0 (nonkomersial), sejalan dengan EmpatheticDialogues dan NLLB-200. Situs unduhan resmi (yanran.li) sudah tidak aktif dan kini dialihkan ke domain parkir, sehingga data diambil dari salinan ConvLab di HuggingFace (`ConvLab/dailydialog`, file `data.zip`), yang memuat seluruh 13.118 dialog beserta label emosi per ujaran. Checksum SHA256 file sudah dicocokkan dengan yang tercatat di HuggingFace.

## Proses seleksi (`pilih_neutral.py`)

Dari 102.979 ujaran, 85.572 berlabel "no emotion". Kandidat kemudian disaring secara berurutan: panjang 5 sampai 45 kata (71.012 tersisa), tanpa tanda seru (68.051), tanpa kata bermuatan emosi dalam bahasa Inggris seperti happy, sad, worried, sorry, great, atau feel (59.542), deduplikasi (50.280), dan maksimal satu ujaran per dialog agar topiknya beragam (11.147). Dari kumpulan ini diambil 1.000 kandidat dengan distribusi panjang kalimat yang disamakan dengan ujaran pertama pembicara EmpatheticDialogues pada empat kelas emosi (median 14 kata dan persentil ke-90 sebesar 27 kata untuk keduanya), dan porsi kalimat tanya dibatasi sekitar 30%. Penyamaan panjang ini mencegah panjang teks menjadi pintasan bagi model.

## Terjemahan (`terjemahkan_nllb.py`)

Model yang dipakai adalah `facebook/nllb-200-distilled-600M` (checksum bobot model sudah dicocokkan dengan versi resmi), dari eng_Latn ke ind_Latn, beam search 4, maksimal 256 token baru, tanpa sampling sehingga hasilnya deterministik.

Pada percobaan pertama ditemukan bahwa NLLB sering hanya menerjemahkan satu kalimat dan membuang kalimat lain bila input berisi beberapa kalimat. Contohnya, "What do you mean, Mr. Smith? You said to make it brief." hanya diterjemahkan menjadi "Apa maksudmu, Mr. Smith?". Pada percobaan pertama, 122 dari 1.000 terjemahan memiliki rasio panjang yang mencurigakan akibat kalimat yang hilang. Karena itu skrip memecah teks per kalimat, menerjemahkan tiap kalimat, lalu menggabungkannya kembali; jumlah kasus mencurigakan turun menjadi 28. Pemecahan per kalimat memunculkan efek samping baru, yaitu kalimat yang sangat pendek (kurang dari tiga kata, seperti "Hello." atau "Thank you.") kadang dihalusinasi, misalnya "Hello." menjadi "Hei, apa yang kau lakukan?" atau "Thank you." menjadi "- Terima kasih. - Terima kasih.". Skrip mendeteksi kasus ini secara otomatis lalu menerjemahkan ulang kalimat pendek tersebut bersama kalimat sesudahnya (8 kasus). Sisa kasus kecil yang lolos, seperti "Well." menjadi "Ya, aku tahu.", diserahkan ke verifikasi manual. Hasil percobaan pertama tetap disimpan sebagai pembanding (`neutral_dailydialog_id_v1_tanpa_split.csv`).

Catatan penting: masalah ini kemungkinan juga terjadi pada terjemahan EmpatheticDialogues di eksperimen awal. Agar kelima kelas benar-benar setara, empat kelas emosi sebaiknya diterjemahkan ulang dengan skrip dan konfigurasi yang sama.

## Cek kualitas otomatis (`cek_terjemahan.py`)

Setiap hasil terjemahan diberi flag bila masih mengandung banyak kata bahasa Inggris, rasio panjang terjemahan terhadap teks asli di luar 0,6 sampai 1,8, terdapat pengulangan kata yang tidak wajar, memuat kata emosi bahasa Indonesia, atau duplikat. Dari 1.000 kandidat, 961 lolos tanpa flag (28 rasio panjang, 9 memuat kata emosi, 1 pengulangan, 1 duplikat). Sebanyak 800 sampel final diambil dari yang tidak ber-flag dengan distribusi panjang yang tetap dijaga; sisanya menjadi cadangan.

## File

`neutral_dailydialog_id_final800.csv` adalah 800 sampel neutral siap pakai (kolom `text` berisi teks bahasa Indonesia). `neutral_dailydialog_id_semua.csv` berisi seluruh 1.000 kandidat beserta teks asli, flag, dan penanda masuk final. `lembar_verifikasi_neutral.csv` dipakai untuk verifikasi manual: isi kolom `status_verifikasi` dengan OK, REVISI, atau BUANG, dan tulis perbaikan pada `text_id_revisi` bila perlu. Sampel yang dibuang dapat diganti dari cadangan. File `*_stats.json` mencatat angka setiap tahap untuk dikutip di laporan.
