# Evaluasi model deteksi krisis (dengan penyamaran kata pemicu, peluang 0.3)

Ambang krisis 0.23, ambang hati-hati 0.1, temperatur 1.323, epoch terbaik 2.

## Data uji Kaggle terjemahan (568 kalimat)

| Ambang | Accuracy | Precision krisis | Recall krisis | F1 krisis | CM [[TN,FP],[FN,TP]] |
|---|---|---|---|---|---|
| 0.23 | 0,847 | 0,781 | 0,898 | 0,836 | [[260, 62], [25, 221]] |
| 0.1 | 0,799 | 0,696 | 0,951 | 0,804 | [[220, 102], [12, 234]] |
| 0.5 | 0,856 | 0,863 | 0,793 | 0,826 | [[291, 31], [51, 195]] |

AUC 0,934

## 50 pesan uji berbahasa Indonesia asli

| Ambang | Accuracy | Precision krisis | Recall krisis | F1 krisis | CM [[TN,FP],[FN,TP]] |
|---|---|---|---|---|---|
| 0.23 | 0,800 | 0,833 | 0,833 | 0,833 | [[15, 5], [5, 25]] |
| 0.1 | 0,700 | 0,703 | 0,867 | 0,776 | [[9, 11], [4, 26]] |
| 0.5 | 0,720 | 0,864 | 0,633 | 0,731 | [[17, 3], [11, 19]] |

AUC 0,818

## Data uji per status (ambang krisis)

| Status | Jumlah | Terdeteksi krisis |
|---|---|---|
| Anxiety | 30 | 3 |
| Bipolar | 25 | 6 |
| Depression | 89 | 35 |
| Normal | 79 | 8 |
| Stress | 24 | 9 |
| Suicidal | 246 | 221 |
| emosi_anger | 15 | 0 |
| emosi_fear | 15 | 1 |
| emosi_happy | 15 | 0 |
| emosi_neutral | 15 | 0 |
| emosi_sadness | 15 | 0 |

## Perbandingan dengan model tanpa penyamaran (ambang krisis masing-masing)

| Data | Model | Precision | Recall | F1 | Bukan krisis yang salah dianggap krisis |
|---|---|---|---|---|---|
| Uji Kaggle | tanpa penyamaran | 0,769 | 0,907 | 0,832 | 67 dari 322 |
| Uji Kaggle | dengan penyamaran | 0,781 | 0,898 | 0,836 | 62 dari 322 |
| 50 pesan Indonesia asli | tanpa penyamaran | 0,828 | 0,800 | 0,814 | 5 dari 20 |
| 50 pesan Indonesia asli | dengan penyamaran | 0,833 | 0,833 | 0,833 | 5 dari 20 |

## 50 pesan Indonesia asli: prediksi model dengan penyamaran

| Pesan | Harapan | Deteksi | p krisis |
|---|---|---|---|
| aku pengen bunuh diri | krisis | krisis | 0.9844 |
| kadang kepikiran buat bundir aja | krisis | hati_hati | 0.1974 |
| aku mau mengakhiri hidupku malam ini | krisis | krisis | 0.9597 |
| rasanya pengen ngakhirin hidup aja | krisis | krisis | 0.7964 |
| aku pengen mati aja | krisis | krisis | 0.9503 |
| mending gue mati daripada gini terus | krisis | krisis | 0.902 |
| aku udah gak mau hidup lagi | krisis | krisis | 0.9169 |
| semoga aku cepet mati | krisis | krisis | 0.871 |
| aku capek hidup | krisis | krisis | 0.9523 |
| pengen tidur selamanya dan gak bangun lagi | krisis | krisis | 0.416 |
| aku pengen ngilang aja selamanya | krisis | krisis | 0.8709 |
| mending aku gak pernah lahir | krisis | krisis | 0.5573 |
| aku udah nyari cara biar cepet mati | krisis | krisis | 0.7468 |
| pengen loncat dari gedung kampus | krisis | bukan | 0.0787 |
| aku mau minum racun aja | krisis | krisis | 0.3839 |
| kepikiran gantung diri di kamar | krisis | krisis | 0.9283 |
| aku suka nyakitin diri sendiri kalau lagi kacau | krisis | krisis | 0.2483 |
| semalem aku nyayat tangan lagi | krisis | bukan | 0.076 |
| aku self harm lagi | krisis | krisis | 0.6793 |
| aku pengen melukai diriku sendiri | krisis | krisis | 0.834 |
| aku pengen matiii ajaaa | krisis | krisis | 0.9101 |
| gw pgn bundir | krisis | krisis | 0.2393 |
| ak ga mau idup lg | krisis | krisis | 0.6896 |
| hidupku gak ada gunanya mending aku mati | krisis | krisis | 0.9004 |
| aku pengen nyusul almarhum ibuku | krisis | bukan | 0.0764 |
| aku pengen udahan aja sama semuanya | krisis | krisis | 0.7545 |
| buat apa aku hidup | krisis | krisis | 0.8585 |
| aku pengen lompat ke rel kereta | krisis | krisis | 0.4012 |
| aku udah nyiapin obat tidur banyak banget | krisis | bukan | 0.0352 |
| kayaknya semua bakal lebih tenang kalau aku udah gak ada | krisis | krisis | 0.2305 |
| tadi malam mati lampu tiga jam | bukan | bukan | 0.0216 |
| aku mati gaya pas ditanya dosen | bukan | bukan | 0.0927 |
| hp aku mati total | bukan | krisis | 0.2828 |
| main game buat bunuh waktu | bukan | krisis | 0.8947 |
| kakiku mati rasa kelamaan duduk | bukan | bukan | 0.0388 |
| aku mati kutu di depan gebetan | bukan | krisis | 0.3891 |
| tugasnya banyak banget setengah mati ngerjainnya | bukan | hati_hati | 0.1493 |
| filmnya lucu banget aku ngakak sampai mau nangis | bukan | hati_hati | 0.1036 |
| aku capek banget hari ini kuliah sampai sore | bukan | bukan | 0.0431 |
| aku pengen tidur seharian besok | bukan | bukan | 0.0945 |
| aku pengen menghilang dari grup wa kelas | bukan | hati_hati | 0.1204 |
| aku gak mau hidup di kota ini selamanya pengen pindah | bukan | krisis | 0.8678 |
| tanaman di kosan pada mati semua | bukan | bukan | 0.0634 |
| ikan cupangku mati kemarin | bukan | bukan | 0.0299 |
| baterai laptopku cepet banget matinya | bukan | bukan | 0.0816 |
| aku gak kuat lagi makan pedes segini | bukan | bukan | 0.0603 |
| aku nyerah sama soal kalkulus ini | bukan | hati_hati | 0.1689 |
| aku lagi nonton film tentang perang banyak yang mati | bukan | hati_hati | 0.1411 |
| deadline-nya bikin mau gila | bukan | hati_hati | 0.1111 |
| aku pengen udahan sama pacarku | bukan | krisis | 0.6161 |
