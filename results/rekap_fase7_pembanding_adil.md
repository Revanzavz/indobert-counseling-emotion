## Perbandingan skenario dengan ruang pencarian yang setara

| Skenario | Varian terpilih | Seed | Macro F1 val konseling | Macro F1 uji konseling | Accuracy uji konseling | Weighted F1 uji tweet |
|---|---|---|---|---|---|---|
| Dua tahap (terpilih dari 8 varian) | duatahap_lr2setengah_full_noreh | 3 | 0.7552 ± 0.0160 | 0.7455 ± 0.0078 | 0.7467 ± 0.0063 | 0.7344 ± 0.0071 |
| Tahap 2 saja (terpilih dari 4 varian) | tahap2saja_lrsetengah_full | 3 | 0.7617 ± 0.0160 | 0.7366 ± 0.0065 | 0.7383 ± 0.0038 | 0.5770 ± 0.0099 |
| Satu tahap gabungan (terpilih dari 4 varian) | satutahap_lrpenuh_full | 3 | 0.7423 ± 0.0074 | 0.7394 ± 0.0104 | 0.7392 ± 0.0088 | 0.8064 ± 0.0152 |
| Dua tahap, head [CLS] + full (arsitektur proposal) | duatahap_cls | 3 | 0.7631 ± 0.0092 | 0.7437 ± 0.0039 | 0.7442 ± 0.0029 | 0.7376 ± 0.0145 |
| Tahap 2 saja (konfigurasi tunggal, fase 5) | tahap2_saja | 3 | 0.7553 ± 0.0151 | 0.7411 ± 0.0127 | 0.7417 ± 0.0138 | 0.5613 ± 0.0063 |
| Satu tahap gabungan (konfigurasi tunggal, fase 5) | satu_tahap | 3 | - | 0.7131 ± 0.0050 | 0.7117 ± 0.0058 | 0.8088 ± 0.0048 |

## Uji signifikansi terhadap dua tahap terpilih

| Pembanding | Selisih macro F1 (dua tahap - pembanding) | CI 95% | McNemar (benar hanya dua tahap / hanya pembanding) | p |
|---|---|---|---|---|
| Tahap 2 saja | +0.0089 | [-0.0131; +0.0307] | 64 / 54 | 0.4075 |
| Satu tahap gabungan | +0.0061 | [-0.0140; +0.0257] | 71 / 62 | 0.4880 |
| Dua tahap head [CLS] | +0.0019 | [-0.0173; +0.0203] | 56 / 53 | 0.8482 |

## Semua varian tahap2saja (rata-rata antar seed)

| tag                               |   macro F1 val |   macro F1 uji konseling |   weighted F1 uji tweet |
|:----------------------------------|---------------:|-------------------------:|------------------------:|
| tahap2saja_lrpenuh_freeze_half    |         0.7454 |                   0.7299 |                  0.5473 |
| tahap2saja_lrpenuh_full           |         0.7553 |                   0.7411 |                  0.5613 |
| tahap2saja_lrsetengah_freeze_half |         0.7468 |                   0.7181 |                  0.5703 |
| tahap2saja_lrsetengah_full        |         0.7617 |                   0.7366 |                  0.577  |

## Semua varian satutahap (rata-rata antar seed)

| tag                              |   macro F1 val |   macro F1 uji konseling |   weighted F1 uji tweet |
|:---------------------------------|---------------:|-------------------------:|------------------------:|
| satutahap_lrpenuh_freeze_half    |         0.7167 |                   0.7131 |                  0.8088 |
| satutahap_lrpenuh_full           |         0.7423 |                   0.7394 |                  0.8064 |
| satutahap_lrsetengah_freeze_half |         0.7371 |                   0.7165 |                  0.8091 |
| satutahap_lrsetengah_full        |         0.7392 |                   0.7295 |                  0.8032 |
