# Kerangka Eksperimen IndoBERT: Pemetaan Emosi Lima Kelas

Folder ini berisi seluruh kode untuk mencari konfigurasi IndoBERT terbaik secara otomatis, dari pemilihan backbone sampai strategi fine-tuning dua tahap. Semua keputusan metode sudah dikunci di dalam kode, sehingga hasil dapat direplikasi dan langsung dipindahkan ke Bab 3 dan Bab 4.

## Cara menjalankan di laptop

Siapkan lingkungan Python (disarankan conda) dengan PyTorch versi CUDA terlebih dahulu, misalnya `pip install torch --index-url https://download.pytorch.org/whl/cu121`, lalu pasang paket lain dengan `pip install -r requirements.txt`. Laptop perlu terhubung ke internet saat pertama kali menjalankan, karena bobot setiap backbone diunduh otomatis dari HuggingFace lalu disimpan di cache.

Setelah itu cukup jalankan `jalankan.bat` (atau `python jalankan_eksperimen.py`) dari folder ini. Runner menjalankan enam fase secara berurutan. Setiap run yang sudah selesai dilewati, jadi proses aman dihentikan kapan saja (laptop dimatikan, sleep, atau Ctrl+C) dan dilanjutkan dengan perintah yang sama. Run yang gagal akan dicoba lagi pada pemanggilan berikutnya. Untuk melihat daftar run tanpa melatih apa pun, gunakan `python jalankan_eksperimen.py --rencana`. Setelah selesai, jalankan `python rekap_hasil.py` untuk menghasilkan tabel dan grafik di folder `hasil/`.

## Data

Semua file data berada di `../data_final/` dan dibangun oleh `../pipeline/02_bangun_dataset.py`. Data tahap 1 adalah Riccosan yang sudah dideduplikasi dan dibersihkan dari teks berlabel ganda, dengan kelas love digabung ke happy. Data tahap 2 (domain konseling) berisi 800 sampel per kelas: empat kelas emosi dari ujaran pertama pembicara EmpatheticDialogues dan kelas neutral dari DailyDialog, semuanya diterjemahkan dengan NLLB-200 memakai konfigurasi yang identik. Kedua dataset dibagi 80/10/10 secara stratified dengan seed 42. File `rehearsal_tahap1.csv` berisi 10% data latih tahap 1 untuk varian rehearsal, `satu_tahap_*.csv` adalah gabungan kedua dataset untuk baseline satu tahap, dan `tahap2alt_*.csv` adalah versi tahap 2 dengan neutral dari tweet Riccosan untuk ablasi.

## Ruang pencarian

Sepuluh backbone dari keluarga IndoBERT diuji: IndoBERT base, large, lite-base, dan lite-large (masing-masing fase p1 dan p2) dari IndoNLU, serta IndoBERT base dan IndoBERTweet dari IndoLEM.

Lima varian classification head dibandingkan. Head `cls` memakai representasi [CLS] layer terakhir, sesuai rancangan proposal. Head `mean` memakai rata-rata semua token, `cls_mean` menggabungkan keduanya, `bilstm` menambahkan BiLSTM dengan pooling rata-rata dan maksimum, dan `bilstm_attn` menambahkan BiLSTM dengan attention pooling. Dua varian BiLSTM mengikuti pendekatan hybrid Ahmadian et al. [1].

Empat strategi fine-tuning dibandingkan. Strategi `full` melatih semua parameter dengan satu learning rate. Strategi `llrd` (layer-wise learning rate decay) memberi learning rate yang mengecil 0,9 kali per layer ke arah bawah. Strategi `gradual_unfreeze` mengikuti ULMFiT [11]: epoch pertama hanya melatih head, lalu tiap epoch membuka satu blok layer dari atas. Strategi `freeze_half` membekukan embedding dan setengah layer bawah. Pada IndoBERT-lite (ALBERT), seluruh layer berbagi satu blok parameter, sehingga pembagian layer hanya terdiri dari embedding dan encoder bersama.

Empat fungsi loss dibandingkan: cross-entropy biasa, cross-entropy berbobot kelas, focal loss (gamma 2), dan label smoothing 0,1. Grid hyperparameter mengikuti Tabel 3.3 proposal, yaitu learning rate 1e-5, 2e-5, 3e-5, dan 5e-5, batch size 16 dan 32, serta dropout 0,1, 0,2, dan 0,3, dengan AdamW, weight decay 0,01, warmup linear 10%, dan maksimal 5 epoch dengan early stopping pada validation loss (patience 2). Epoch tidak dimasukkan ke grid karena early stopping sudah memilih epoch terbaik secara otomatis; epoch terbaik setiap run dicatat.

## Enam fase turnamen

Mencoba seluruh kombinasi secara penuh akan menghasilkan ribuan run, sehingga pencarian dilakukan bertahap. Fase 1 melatih kesepuluh backbone dengan tiga learning rate (30 run) memakai head [CLS] dan full fine-tuning. Fase 2 mengambil tiga backbone terbaik dan mencoba seluruh kombinasi lima head dan empat strategi fine-tuning (60 run); opsi `--semua-backbone` menjalankannya untuk kesepuluh backbone (200 run). Fase 3 menguji empat fungsi loss pada tiga konfigurasi terbaik. Fase 4 menjalankan grid hyperparameter proposal (24 run) pada konfigurasi terbaik. Keempat fase ini dilatih pada data tahap 1 dan dipilih berdasarkan weighted F1 validasi, sesuai proposal.

Fase 5 menjawab pertanyaan penelitian pertama. Dengan konfigurasi terbaik dan tiga seed (42, 43, 44), dibandingkan model tahap 1 saja yang langsung diuji di data konseling, baseline satu tahap (dilatih pada gabungan data), model tahap 2 saja (IndoBERT langsung ke data konseling tanpa tahap 1), dan fine-tuning dua tahap dalam delapan varian. Kedelapan varian itu adalah kombinasi learning rate tahap 2 (sama atau setengah), strategi tahap 2 (full atau freeze_half), dan ada atau tidaknya rehearsal. Pemilihan varian terbaik memakai rata-rata macro F1 validasi domain konseling.

Fase 6 berisi ablasi pada pipeline terbaik: pengaruh normalisasi teks informal (kamus Colloquial Indonesian Lexicon), pengaruh sumber kelas neutral (DailyDialog versus tweet Riccosan seperti rancangan awal), dan pembanding IndoBERT-base-p1 yang disebut di proposal bila backbone terbaik ternyata berbeda.

Data uji tidak pernah dipakai untuk memilih konfigurasi. Setiap run mencatat accuracy, precision, recall, dan F1 berbobot, macro F1, macro F1 empat kelas emosi, F1 per kelas, confusion matrix, serta Expected Calibration Error sebelum dan sesudah temperature scaling. Nilai temperatur dihitung pada data validasi dan dipakai untuk probabilitas yang dikirim ke modul chatbot dan jalur deteksi krisis.

## Menyesuaikan dengan GPU laptop

Runner membaca kapasitas VRAM secara otomatis. Batch efektif selalu mengikuti grid, tetapi pada VRAM kecil batch dipecah menjadi batch mikro dengan akumulasi gradien. Model large (sekitar 335 juta parameter) dijalankan dengan batch mikro 4, gradient checkpointing, dan optimizer AdamW 8-bit (paket `bitsandbytes`). Bila tetap kehabisan memori, run ditandai OOM di `runs/<nama_run>/status.json` dan turnamen berlanjut tanpa backbone tersebut. Fakta ini dilaporkan sebagai keterbatasan perangkat keras, atau run tersebut dapat dijalankan di Google Colab dengan menyalin folder ini.

## Struktur keluaran

Setiap run disimpan di `runs/<nama_run>/` berisi `config.json`, `log.txt`, `metrics.json`, dan folder `prediksi/` yang memuat prediksi serta probabilitas per sampel uji untuk analisis kesalahan. Model hanya disimpan (`model.pt`) untuk run tahap 1 yang menjadi titik awal tahap 2 dan untuk model final. Rekap semua run ada di `hasil/semua_run.csv`, dan `rekap_hasil.py` menghasilkan tabel per fase (CSV, Excel, dan Markdown siap salin) beserta grafik perbandingan backbone, heatmap head x strategi, dan confusion matrix model final.

## Referensi tambahan

Kamus normalisasi: N. A. Salsabila, Y. A. Winatmoko, A. A. Septiandri, and A. Jamal, "Colloquial Indonesian Lexicon," 2018 International Conference on Asian Language Processing (IALP), 2018, pp. 226–229. IndoBERT dan IndoBERTweet dari IndoLEM: F. Koto, A. Rahimi, J. H. Lau, and T. Baldwin, "IndoLEM and IndoBERT: A Benchmark Dataset and Pre-trained Language Model for Indonesian NLP," COLING 2020; F. Koto, J. H. Lau, and T. Baldwin, "IndoBERTweet: A Pretrained Language Model for Indonesian Twitter with Effective Domain-Specific Vocabulary Initialization," EMNLP 2021.
