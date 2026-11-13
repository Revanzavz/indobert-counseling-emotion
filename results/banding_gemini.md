## Data uji konseling (400 kalimat)

| Metrik | gemini-2.5-flash (zero-shot) | IndoBERTweet dua tahap |
|---|---|---|
| Accuracy | 0.7575 | 0.7525 |
| Macro F1 | 0.7575 | 0.7523 |
| Weighted F1 | 0.7575 | 0.7523 |
| F1 anger | 0.7218 | 0.7186 |
| F1 fear | 0.8194 | 0.7200 |
| F1 happy | 0.8049 | 0.7515 |
| F1 sadness | 0.6707 | 0.6962 |
| F1 neutral | 0.7708 | 0.8750 |

Uji McNemar: benar hanya Gemini 37, benar hanya IndoBERTweet 35, p = 0.9063. Kesepakatan prediksi kedua model 74.5%. Waktu Gemini 0.29 detik per kalimat (dikirim per 25 sampai 50 kalimat). Kalimat yang diblokir filter keamanan Gemini (dihitung salah): 0.

