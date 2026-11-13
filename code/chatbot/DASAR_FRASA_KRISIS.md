# Dasar Penyusunan Daftar Frasa Deteksi Krisis

Dokumen ini menjelaskan asal tiap kelompok frasa pada `krisis.py`, cara daftar itu diuji, dan batasannya. Dokumen ini juga menjadi pengantar lembar tinjauan untuk ahli (`lembar_tinjauan_frasa_krisis.csv`).

## Kedudukan jalur krisis dalam sistem

Jalur krisis adalah lapisan pengaman pada prototipe riset. Tujuannya mengalihkan pesan yang berindikasi krisis ke pesan rujukan layanan profesional, sehingga pesan itu tidak dijawab oleh model generatif. Jalur ini bukan alat skrining klinis dan tidak dimaksudkan untuk menilai tingkat risiko seseorang.

## Kelompok frasa dan acuannya

Pengelompokan mengikuti susunan Columbia Suicide Severity Rating Scale (C-SSRS), yang membedakan keinginan untuk mati, pikiran bunuh diri, pikiran yang disertai cara atau rencana, serta perilaku menyakiti diri. Frasa bahasa Indonesia di dalam tiap kelompok disusun oleh peneliti, termasuk bentuk tidak baku dan singkatan yang lazim di percakapan daring. C-SSRS dipakai sebagai kerangka pengelompokan, bukan sebagai sumber frasa.

| Kelompok | Padanan kategori C-SSRS | Contoh frasa | Mekanisme |
|---|---|---|---|
| Keinginan untuk mati | Wish to be dead | ingin mati, lebih baik aku mati, tidak ingin hidup lagi | 1 (kata kunci) |
| Keinginan pasif | Wish to be dead (bentuk pasif) | capek hidup, tidur selamanya, lebih baik tidak pernah lahir, untuk apa aku hidup | 1 (kata kunci) |
| Pikiran bunuh diri | Non-specific active suicidal thoughts | bunuh diri, mengakhiri hidup, bundir | 1 (kata kunci) |
| Cara atau rencana | Active ideation with method or plan | gantung diri, minum racun, loncat dari gedung, cara mati | 1 (kata kunci) |
| Menyakiti diri | Non-suicidal self-injurious behavior | menyakiti diri, self harm, menyayat tangan | 1 (kata kunci) |
| Indikator tidak langsung | Tidak ada padanan langsung. Disusun dari tanda peringatan yang umum disebut dalam literatur pencegahan bunuh diri: putus asa, merasa menjadi beban, ingin menghilang, menyiapkan sarana | putus asa, cuma jadi beban, tidak kuat lagi, kalau aku sudah tidak ada | 2 (label emosi sadness atau fear dengan probabilitas minimal 0,70) |

Catatan untuk penulisan skripsi: rujukan untuk baris "indikator tidak langsung" perlu dipastikan sendiri dari sumber yang dikutip (misalnya panduan WHO LIVE LIFE yang disebut di proposal), karena pengelompokan ini disusun peneliti dan belum ditinjau ahli.

## Cara kerja dua mekanisme

Mekanisme pertama memeriksa frasa eksplisit sebelum model klasifikasi dijalankan. Bila cocok, pesan langsung dialihkan. Mekanisme kedua dijalankan setelah model klasifikasi: pesan dialihkan bila memuat indikator tidak langsung dan model memberi label sadness atau fear dengan probabilitas minimal 0,70. Syarat emosi pada mekanisme kedua dimaksudkan untuk mengurangi salah tangkap pada frasa yang juga muncul di percakapan biasa, misalnya "tidak kuat lagi".

Sebelum dicocokkan, pesan dinormalisasi: huruf kecil, huruf berulang dirapatkan ("matiii" menjadi "mati"), dan bentuk tidak baku diganti ("gak" menjadi "tidak", "pengen" menjadi "ingin", "bundir" menjadi "bunuh diri").

## Pengujian

Daftar diuji dengan 50 pesan pada `uji_krisis.csv`, terdiri atas 30 pesan berindikasi krisis dengan variasi ungkapan dan 20 pesan yang memuat kata serupa tetapi bukan krisis (misalnya "mati lampu", "bunuh waktu"). Pesan uji disusun peneliti.

Pada pengujian pertama dengan mekanisme kata kunci saja, 21 dari 30 pesan krisis terdeteksi (recall 70%) dengan satu salah tangkap. Pesan yang lolos sebagian besar berupa variasi ejaan ("cepet", "lompat") dan ungkapan tersirat. Setelah daftar dan normalisasi diperbaiki, mekanisme kata kunci mendeteksi 27 dari 30 (recall 90%) tanpa salah tangkap. Hasil sebelum perbaikan disimpan di `hasil/uji_krisis_sebelum_perbaikan.md`. Hasil dengan alur lengkap (kata kunci dan model emosi) ada di `hasil/uji_krisis.md` setelah `uji_krisis.bat` dijalankan.

Angka setelah perbaikan tidak boleh dibaca sebagai performa di lapangan, karena daftar diperbaiki dengan melihat pesan uji yang sama. Angka itu menunjukkan bahwa daftar menangani variasi yang sudah diketahui, bukan variasi yang belum terpikirkan.

## Keterbatasan

1. Pendekatan kata kunci tidak menangkap ungkapan tersirat yang tidak memuat frasa dalam daftar. Tiga pesan uji tetap lolos dari mekanisme kata kunci, misalnya "aku udah nyiapin obat tidur banyak banget".
2. Mekanisme kedua bergantung pada keyakinan model emosi. Pada pesan pendek, keyakinan model sering di bawah 0,70 sehingga mekanisme kedua tidak aktif.
3. Daftar tidak membedakan pesan tentang diri sendiri dan tentang orang lain. Pesan seperti "temanku ingin bunuh diri" juga dialihkan ke rujukan. Ini dipilih dengan sengaja karena salah tangkap dinilai lebih ringan akibatnya daripada pesan krisis yang lolos.
4. Daftar belum ditinjau oleh psikolog atau konselor, dan belum diuji pada pesan pengguna nyata.
5. Layanan rujukan yang dicantumkan (119 ekstensi 8) tidak beroperasi 24 jam.

Untuk pesan yang memuat indikator tidak langsung tetapi tidak memenuhi syarat mekanisme kedua, modul respons tetap diberi catatan agar menanggapi dengan hati-hati dan menyebut layanan profesional.
