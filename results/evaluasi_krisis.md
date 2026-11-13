# Evaluasi model deteksi krisis

Ambang krisis 0.21, ambang hati-hati 0.08, temperatur 1.314, epoch terbaik 2.

## Data uji Kaggle terjemahan (568 kalimat)

| Ambang | Accuracy | Precision krisis | Recall krisis | F1 krisis | CM [[TN,FP],[FN,TP]] |
|---|---|---|---|---|---|
| 0.21 | 0,842 | 0,769 | 0,907 | 0,832 | [[255, 67], [23, 223]] |
| 0.08 | 0,799 | 0,692 | 0,967 | 0,807 | [[216, 106], [8, 238]] |
| 0.5 | 0,857 | 0,867 | 0,793 | 0,828 | [[292, 30], [51, 195]] |

AUC 0,933

## 50 pesan uji berbahasa Indonesia asli

| Ambang | Accuracy | Precision krisis | Recall krisis | F1 krisis | CM [[TN,FP],[FN,TP]] |
|---|---|---|---|---|---|
| 0.21 | 0,780 | 0,828 | 0,800 | 0,814 | [[15, 5], [6, 24]] |
| 0.08 | 0,700 | 0,683 | 0,933 | 0,789 | [[7, 13], [2, 28]] |
| 0.5 | 0,700 | 0,857 | 0,600 | 0,706 | [[17, 3], [12, 18]] |

AUC 0,793

## Data uji per status (ambang krisis)

| Status | Jumlah | Terdeteksi krisis |
|---|---|---|
| Anxiety | 30 | 3 |
| Bipolar | 25 | 6 |
| Depression | 89 | 39 |
| Normal | 79 | 10 |
| Stress | 24 | 8 |
| Suicidal | 246 | 223 |
| emosi_anger | 15 | 0 |
| emosi_fear | 15 | 1 |
| emosi_happy | 15 | 0 |
| emosi_neutral | 15 | 0 |
| emosi_sadness | 15 | 0 |
