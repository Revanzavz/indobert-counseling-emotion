# Pedoman Pengumpulan dan Pelabelan Data Uji Asli Berbahasa Indonesia

Dokumen ini dipakai untuk membuat data uji tambahan berisi kalimat curhat yang ditulis langsung dalam bahasa Indonesia, bukan hasil terjemahan. Data ini hanya dipakai untuk menguji model final. Data ini tidak dipakai untuk melatih atau memilih model.

## 1. Sasaran

Jumlah sasaran adalah 150 sampai 200 kalimat, dengan 30 sampai 40 kalimat untuk tiap kelas (anger, fear, happy, sadness, neutral). Jumlah per kelas tidak harus persis sama, tetapi tiap kelas sebaiknya tidak kurang dari 25 kalimat supaya nilai F1 per kelas cukup stabil.

## 2. Cara mengumpulkan

Kalimat dikumpulkan lewat formulir anonim (misalnya Google Form) yang diisi mahasiswa atau teman sebaya. Formulir tidak meminta nama, nomor telepon, atau email. Di bagian awal formulir cantumkan penjelasan bahwa tulisan dipakai untuk penelitian tugas akhir, bersifat anonim, dan pengisi boleh menulis pengalaman nyata atau pengalaman yang wajar dialami mahasiswa.

Contoh pertanyaan formulir yang bisa dipakai:

1. Tuliskan satu atau dua kalimat seperti saat kamu curhat lewat chat tentang hal yang pernah membuatmu marah atau kesal.
2. Tuliskan satu atau dua kalimat tentang hal yang pernah membuatmu takut, cemas, atau khawatir.
3. Tuliskan satu atau dua kalimat tentang hal yang pernah membuatmu senang atau bangga.
4. Tuliskan satu atau dua kalimat tentang hal yang pernah membuatmu sedih atau kecewa.
5. Tuliskan satu atau dua kalimat biasa tentang kegiatanmu hari ini tanpa perasaan tertentu.

Minta pengisi menulis dengan gaya chat sehari-hari, termasuk singkatan atau bahasa gaul bila itu memang cara mereka menulis. Kalimat tidak perlu dirapikan.

Hal yang tidak boleh dilakukan: mengambil isi chat pribadi orang lain tanpa izin, menyalin tulisan dari media sosial yang menyebut identitas, atau menulis sendiri semua kalimatnya. Bila peneliti ikut menulis sebagian kalimat, jumlahnya dicatat dan dilaporkan.

Bila ada isian yang memuat ungkapan ingin mengakhiri hidup atau menyakiti diri, kalimat itu tidak dimasukkan ke data uji emosi. Kalimat seperti itu ditangani oleh jalur deteksi krisis, bukan oleh model emosi.

## 3. Berkas

Semua kalimat yang terkumpul dimasukkan ke `data_asli.csv` dengan kolom `id`, `teks`, dan `kelas_pertanyaan` (kelas pertanyaan formulir tempat kalimat itu ditulis, hanya sebagai catatan). Setelah itu jalankan `buat_lembar_penilai.bat`. Skrip mengacak urutan kalimat dan membuat `lembar_penilai_A.csv` dan `lembar_penilai_B.csv` yang tidak memuat kolom `kelas_pertanyaan`, sehingga penilai tidak terpengaruh.

## 4. Pelabelan oleh dua penilai

Dua orang melabeli semua kalimat secara terpisah tanpa saling melihat. Tiap kalimat diberi tepat satu label di kolom `label`.

| Label | Diberikan bila kalimat terutama menunjukkan | Contoh |
|---|---|---|
| anger | marah, kesal, jengkel, tersinggung, iri | "kesel banget temen sekelompok gak ada yang ngerjain" |
| fear | takut, cemas, khawatir, gugup, panik | "deg-degan parah besok presentasi di depan dosen" |
| happy | senang, lega, bangga, bersyukur, bersemangat | "akhirnya proposal aku di-acc, seneng banget" |
| sadness | sedih, kecewa, kesepian, kehilangan, merasa bersalah | "kangen rumah, di kosan rasanya sepi banget" |
| neutral | tidak menunjukkan emosi tertentu, hanya informasi atau percakapan biasa | "besok ada kelas jam delapan" |

Aturan saat ragu:

1. Pilih emosi yang paling dominan menurut isi kalimat, bukan menurut topiknya. Kalimat tentang ujian tidak otomatis fear.
2. Bila dua emosi muncul, pilih yang paling ditekankan penulis. Bila sama kuat, pilih yang disebut terakhir.
3. Lelah atau capek tanpa keterangan lain diberi label sadness bila bernada putus asa atau kecewa, anger bila bernada kesal, dan neutral bila hanya menyatakan keadaan fisik.
4. Sarkasme dilabeli menurut perasaan yang sebenarnya dimaksud.
5. Bila kalimat benar-benar tidak bisa ditentukan, isi kolom `catatan` dengan "ragu" dan tetap pilih satu label.

## 5. Menyatukan label dan mengevaluasi

Setelah kedua lembar terisi, jalankan `evaluasi_data_asli.bat`. Skrip menghitung kesepakatan dua penilai (persentase sepakat dan Cohen's kappa), lalu menulis `label_gabungan.csv`. Kalimat yang dilabeli sama oleh kedua penilai langsung mendapat label akhir. Kalimat yang berbeda perlu diputuskan oleh penilai ketiga atau lewat diskusi, dengan mengisi kolom `label_akhir` di `label_gabungan.csv`. Kalimat yang kolom `label_akhir`-nya dibiarkan kosong tidak diikutkan dalam evaluasi dan jumlahnya dilaporkan.

Jalankan `evaluasi_data_asli.bat` sekali lagi untuk menguji model final pada kalimat yang sudah punya label akhir. Hasilnya tersimpan di `hasil_evaluasi_data_asli.md` dan `prediksi_data_asli.csv`.

Sebagai pegangan umum, kappa di atas 0,60 biasanya dianggap kesepakatan yang baik, dan di atas 0,80 sangat baik. Nilai kappa dilaporkan di Bab 3 atau Bab 4 bersama jumlah kalimat yang tidak disepakati.
