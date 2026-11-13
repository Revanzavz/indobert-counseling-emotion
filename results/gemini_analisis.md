# Analisis Gemini vs IndoBERTweet (400 kalimat uji konseling)

## Metrik per kelas

| Kelas | P Gemini | R Gemini | F1 Gemini | P IndoBERTweet | R IndoBERTweet | F1 IndoBERTweet | Lebih baik |
|---|---|---|---|---|---|---|---|
| anger | 0,906 | 0,600 | 0,722 | 0,690 | 0,750 | 0,719 | setara |
| fear | 0,922 | 0,738 | 0,819 | 0,771 | 0,675 | 0,720 | Gemini |
| happy | 0,786 | 0,825 | 0,805 | 0,729 | 0,775 | 0,752 | Gemini |
| sadness | 0,644 | 0,700 | 0,671 | 0,705 | 0,688 | 0,696 | IndoBERTweet |
| neutral | 0,661 | 0,925 | 0,771 | 0,875 | 0,875 | 0,875 | IndoBERTweet |

## Paired t-test macro F1 (10 lipatan stratified)

Gemini 0,751 ± 0,066, IndoBERTweet 0,751 ± 0,077, t(9) = -0.000, p = 0.9997. Uji normalitas selisih (Shapiro-Wilk) p = 0.242; Wilcoxon p = 0.6250.

## Paired t-test F1 per kelas (10 lipatan)

| Kelas | F1 Gemini | F1 IndoBERTweet | t | p |
|---|---|---|---|---|
| anger | 0,692 | 0,717 | -0.456 | 0.6594 |
| fear | 0,816 | 0,716 | 4.210 | 0.0023 |
| happy | 0,804 | 0,752 | 1.245 | 0.2445 |
| sadness | 0,671 | 0,695 | -0.614 | 0.5542 |
| neutral | 0,770 | 0,873 | -4.006 | 0.0031 |

## Confusion matrix Gemini (baris aktual, kolom prediksi)

| | anger | fear | happy | sadness | neutral |
|---|---|---|---|---|---|
| anger | 48 | 3 | 2 | 19 | 8 |
| fear | 0 | 59 | 6 | 7 | 8 |
| happy | 1 | 1 | 66 | 3 | 9 |
| sadness | 3 | 1 | 7 | 56 | 13 |
| neutral | 1 | 0 | 3 | 2 | 74 |

## Confusion matrix IndoBERTweet (baris aktual, kolom prediksi)

| | anger | fear | happy | sadness | neutral |
|---|---|---|---|---|---|
| anger | 60 | 4 | 6 | 9 | 1 |
| fear | 6 | 54 | 9 | 9 | 2 |
| happy | 8 | 3 | 62 | 3 | 4 |
| sadness | 12 | 4 | 6 | 55 | 3 |
| neutral | 1 | 5 | 2 | 2 | 70 |
