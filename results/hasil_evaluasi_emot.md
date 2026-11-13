# Evaluasi model final pada EmoT (3939 tweet)

Prediksi neutral: 809 tweet (20.5%).

## (a) Lima kelas, prediksi neutral dihitung salah

| Metrik | Nilai |
|---|---|
| Accuracy | 0.5829 |
| Macro F1 (4 emosi) | 0.6390 |
| Weighted F1 (4 emosi) | 0.6419 |
| F1 anger (n=989) | 0.6045 |
| F1 fear (n=576) | 0.7807 |
| F1 happy (n=1479) | 0.7416 |
| F1 sadness (n=895) | 0.4292 |

Confusion matrix (baris aktual, kolom prediksi)

| | anger | fear | happy | sadness | neutral |
|---|---|---|---|---|---|
| anger | 645 | 20 | 18 | 10 | 296 |
| fear | 48 | 438 | 13 | 8 | 69 |
| happy | 149 | 28 | 937 | 97 | 268 |
| sadness | 303 | 60 | 80 | 276 | 176 |

## (b) Empat kelas emosi

| Metrik | Nilai |
|---|---|
| Accuracy | 0.7035 |
| Macro F1 (4 emosi) | 0.6802 |
| Weighted F1 (4 emosi) | 0.6940 |
| F1 anger (n=989) | 0.6857 |
| F1 fear (n=576) | 0.7520 |
| F1 happy (n=1479) | 0.8115 |
| F1 sadness (n=895) | 0.4717 |

Confusion matrix (baris aktual, kolom prediksi)

| | anger | fear | happy | sadness |
|---|---|---|---|---|
| anger | 875 | 70 | 28 | 16 |
| fear | 82 | 464 | 19 | 11 |
| happy | 212 | 47 | 1115 | 105 |
| sadness | 394 | 77 | 107 | 317 |
