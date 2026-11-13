# EmoT sebagai data uji tambahan

Sumber: EmoT (Saputri et al., 2018), bagian dari IndoNLU, diunduh dari github.com/IndoNLP/indonlu.
Isi: tweet berbahasa Indonesia asli berlabel anger, fear, happy, love, sadness.

- `train_preprocess.csv`, `valid_preprocess.csv`: berkas asli berlabel.
- `test_preprocess_masked_label.csv`: berkas test resmi, labelnya disamarkan sehingga TIDAK dipakai.
- `siapkan_emot.py`: gabung train+valid, love -> happy, buang duplikat dan teks berlabel ganda,
  buang tweet yang sama dengan data Riccosan. Hasil: `emot_uji.csv`, statistik di `emot_stats.json`.
- `evaluasi_emot.bat`: menguji model final, hasil di `hasil_evaluasi_emot.md` dan `prediksi_emot.csv`.

Catatan: EmoT tidak pernah dipakai untuk melatih atau memilih model. EmoT tidak punya kelas neutral,
sehingga hasil dilaporkan untuk empat kelas emosi.
