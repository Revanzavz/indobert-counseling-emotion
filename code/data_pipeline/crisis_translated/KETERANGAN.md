# Data dan model deteksi krisis (dataset Kaggle)

Sumber: "Sentiment Analysis for Mental Health" (Kaggle, suchintikasarkar), file `Combined Data.csv`.
Kelas positif (krisis) diambil dari status **Suicidal**. Kelas negatif diambil dari Normal, Depression, Anxiety,
Stress, dan Bipolar, ditambah kalimat emosi biasa dari data latih domain konseling.

Cara menjalankan: taruh `Combined Data.csv` di folder ini, lalu klik dua kali `jalankan_krisis.bat`.

1. `siapkan_kaggle.py` memilih 2.500 kalimat Suicidal dan 2.500 kalimat bukan krisis (5 sampai 70 kata) -> `kandidat_en.csv`
2. `terjemahkan_nllb.py` (konfigurasi NLLB-200 yang sama dengan data emosi) -> `kandidat_id.csv`
3. `susun_data_krisis.py` membuang terjemahan bermasalah, menambah 750 kalimat emosi biasa sebagai negatif, membagi 80/10/10
4. `Eksperimen/eksperimen/latih_krisis.py` melatih IndoBERTweet biner, mengkalibrasi, memilih ambang, dan mengevaluasi
   pada data uji Kaggle serta 50 pesan uji berbahasa Indonesia asli (`Chatbot/uji_krisis.csv`).

Model tersimpan di `Eksperimen/eksperimen/hasil/model_krisis/` dan dipakai oleh `Chatbot/krisis.py`.
