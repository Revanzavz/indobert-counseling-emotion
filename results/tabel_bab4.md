## fase1_backbone

| backbone               |    lr |   val_f1_weighted |   tahap1_test_f1_weighted |   tahap1_test_f1_macro |   tahap2_test_f1_weighted |   durasi_menit |   vram_gb |
|:-----------------------|------:|------------------:|--------------------------:|-----------------------:|--------------------------:|---------------:|----------:|
| indobertweet-base      | 3e-05 |            0.8085 |                    0.8144 |                 0.8172 |                    0.4187 |           11.1 |      2.27 |
| indobert-large-p1      | 3e-05 |            0.7893 |                    0.7848 |                 0.7928 |                    0.4004 |           70.1 |      4.04 |
| indobert-base-p2       | 3e-05 |            0.7798 |                    0.7787 |                 0.7851 |                    0.3633 |            9.3 |      2.55 |
| indobert-large-p2      | 5e-05 |            0.7795 |                    0.7902 |                 0.7906 |                    0.3631 |           68.6 |      4.04 |
| indolem-indobert-base  | 3e-05 |            0.7771 |                    0.7943 |                 0.8041 |                    0.4215 |           15.3 |      2.35 |
| indobert-base-p1       | 2e-05 |            0.7738 |                    0.7903 |                 0.795  |                    0.3547 |           11.3 |      2.55 |
| indobert-lite-base-p1  | 2e-05 |            0.7724 |                    0.7759 |                 0.7817 |                    0.3154 |            4.5 |      0.49 |
| indobert-lite-large-p1 | 3e-05 |            0.7592 |                    0.7436 |                 0.7554 |                    0.4299 |           15.7 |      1.05 |
| indobert-lite-large-p2 | 2e-05 |            0.7539 |                    0.7599 |                 0.7713 |                    0.387  |           15.2 |      1.05 |
| indobert-lite-base-p2  | 2e-05 |            0.7539 |                    0.7787 |                 0.7849 |                    0.3367 |            5.5 |      0.49 |

## fase2_arsitektur

| backbone          | head        | strategy         |    lr |   val_f1_weighted |   tahap1_test_f1_weighted |   tahap1_test_f1_macro |   durasi_menit |
|:------------------|:------------|:-----------------|------:|------------------:|--------------------------:|-----------------------:|---------------:|
| indobertweet-base | bilstm_attn | freeze_half      | 3e-05 |            0.8154 |                    0.8202 |                 0.8208 |            6   |
| indobertweet-base | cls_mean    | freeze_half      | 3e-05 |            0.8119 |                    0.8192 |                 0.8279 |            8.2 |
| indobertweet-base | cls_mean    | full             | 3e-05 |            0.8103 |                    0.8216 |                 0.827  |           12   |
| indobertweet-base | cls         | freeze_half      | 3e-05 |            0.8091 |                    0.8227 |                 0.8284 |            8.2 |
| indobertweet-base | mean        | llrd             | 3e-05 |            0.8088 |                    0.8145 |                 0.8221 |           12.4 |
| indobertweet-base | cls_mean    | llrd             | 3e-05 |            0.8075 |                    0.8287 |                 0.8385 |           12.1 |
| indobertweet-base | bilstm_attn | llrd             | 3e-05 |            0.8072 |                    0.8307 |                 0.8326 |            9.6 |
| indobertweet-base | mean        | freeze_half      | 3e-05 |            0.8065 |                    0.8172 |                 0.8232 |            7.8 |
| indobertweet-base | cls         | llrd             | 3e-05 |            0.8057 |                    0.8326 |                 0.8391 |           12.3 |
| indobertweet-base | bilstm      | llrd             | 3e-05 |            0.8046 |                    0.8212 |                 0.8255 |           13.2 |
| indobertweet-base | cls         | full             | 3e-05 |            0.8043 |                    0.8158 |                 0.819  |           11.1 |
| indobertweet-base | bilstm      | freeze_half      | 3e-05 |            0.8009 |                    0.8233 |                 0.8258 |           12.3 |
| indobertweet-base | bilstm      | gradual_unfreeze | 3e-05 |            0.798  |                    0.7933 |                 0.7987 |           14.7 |
| indobertweet-base | bilstm_attn | full             | 3e-05 |            0.7978 |                    0.8304 |                 0.8339 |            8.1 |
| indobertweet-base | mean        | gradual_unfreeze | 3e-05 |            0.7975 |                    0.7934 |                 0.7991 |           12.2 |
| indobertweet-base | mean        | full             | 3e-05 |            0.7971 |                    0.8141 |                 0.82   |           12.2 |
| indobertweet-base | cls_mean    | gradual_unfreeze | 3e-05 |            0.7968 |                    0.7995 |                 0.8059 |           12.4 |
| indobert-large-p1 | cls_mean    | llrd             | 3e-05 |            0.7932 |                    0.7851 |                 0.7949 |           56.7 |
| indobert-large-p1 | cls         | llrd             | 3e-05 |            0.7929 |                    0.8025 |                 0.8106 |           60.1 |
| indobert-large-p1 | mean        | llrd             | 3e-05 |            0.7908 |                    0.7929 |                 0.8006 |           55.8 |
| indobert-large-p1 | mean        | freeze_half      | 3e-05 |            0.7898 |                    0.7904 |                 0.7996 |           18.7 |
| indobert-large-p1 | cls         | full             | 3e-05 |            0.7893 |                    0.7848 |                 0.7928 |           61   |
| indobert-large-p1 | cls         | freeze_half      | 3e-05 |            0.7887 |                    0.7925 |                 0.8017 |           25.9 |
| indobertweet-base | bilstm_attn | gradual_unfreeze | 3e-05 |            0.7875 |                    0.7995 |                 0.8071 |            9.7 |
| indobert-large-p1 | cls_mean    | freeze_half      | 3e-05 |            0.7861 |                    0.7849 |                 0.7943 |           17.7 |
| indobert-large-p1 | mean        | full             | 3e-05 |            0.7855 |                    0.7878 |                 0.7956 |           55.9 |
| indobert-base-p2  | cls_mean    | freeze_half      | 3e-05 |            0.7853 |                    0.7724 |                 0.7799 |            5.6 |
| indobert-large-p1 | bilstm      | llrd             | 3e-05 |            0.783  |                    0.7905 |                 0.7995 |           54.7 |
| indobertweet-base | cls         | gradual_unfreeze | 3e-05 |            0.783  |                    0.82   |                 0.8263 |           12.5 |
| indobert-base-p2  | cls         | freeze_half      | 3e-05 |            0.7822 |                    0.7626 |                 0.7702 |            5.5 |
| indobert-base-p2  | mean        | freeze_half      | 3e-05 |            0.7818 |                    0.7717 |                 0.7788 |            5.2 |
| indobert-large-p1 | bilstm_attn | full             | 3e-05 |            0.7809 |                    0.7872 |                 0.7928 |           47.8 |
| indobertweet-base | bilstm      | full             | 3e-05 |            0.7807 |                    0.8177 |                 0.8208 |            9.8 |
| indobert-base-p2  | cls         | llrd             | 3e-05 |            0.7806 |                    0.7737 |                 0.7813 |            9.2 |
| indobert-large-p1 | cls_mean    | full             | 3e-05 |            0.7804 |                    0.7825 |                 0.7894 |           54.4 |
| indobert-base-p2  | mean        | llrd             | 3e-05 |            0.78   |                    0.7869 |                 0.7941 |            6.7 |
| indobert-base-p2  | cls         | full             | 3e-05 |            0.7798 |                    0.7787 |                 0.7851 |            7   |
| indobert-base-p2  | bilstm      | llrd             | 3e-05 |            0.7798 |                    0.7848 |                 0.7871 |            7.2 |
| indobert-base-p2  | cls_mean    | llrd             | 3e-05 |            0.7798 |                    0.7705 |                 0.7786 |            9.4 |
| indobert-large-p1 | mean        | gradual_unfreeze | 3e-05 |            0.7786 |                    0.7761 |                 0.7828 |           55   |
| indobert-large-p1 | bilstm      | full             | 3e-05 |            0.7781 |                    0.7815 |                 0.7903 |           55.1 |
| indobert-large-p1 | bilstm_attn | freeze_half      | 3e-05 |            0.7772 |                    0.7784 |                 0.7875 |           17.9 |
| indobert-base-p2  | mean        | gradual_unfreeze | 3e-05 |            0.7748 |                    0.7842 |                 0.7945 |            8.9 |
| indobert-base-p2  | bilstm_attn | llrd             | 3e-05 |            0.7741 |                    0.7993 |                 0.803  |            7.2 |
| indobert-large-p1 | bilstm_attn | llrd             | 3e-05 |            0.7734 |                    0.7831 |                 0.7939 |           48   |
| indobert-large-p1 | cls         | gradual_unfreeze | 3e-05 |            0.7709 |                    0.7925 |                 0.7992 |           67   |
| indobert-base-p2  | cls_mean    | full             | 3e-05 |            0.7702 |                    0.7712 |                 0.777  |            9.1 |
| indobert-base-p2  | bilstm      | freeze_half      | 3e-05 |            0.7688 |                    0.7754 |                 0.7805 |            4.4 |
| indobert-base-p2  | mean        | full             | 3e-05 |            0.7675 |                    0.7916 |                 0.7989 |            9.1 |
| indobert-base-p2  | bilstm_attn | freeze_half      | 3e-05 |            0.7659 |                    0.7616 |                 0.7677 |            5.7 |
| indobert-base-p2  | cls_mean    | gradual_unfreeze | 3e-05 |            0.7658 |                    0.7643 |                 0.7675 |            9.1 |
| indobert-large-p1 | cls_mean    | gradual_unfreeze | 3e-05 |            0.7656 |                    0.7746 |                 0.7799 |           55.9 |
| indobert-large-p1 | bilstm      | freeze_half      | 3e-05 |            0.7654 |                    0.784  |                 0.7926 |           18.7 |
| indobert-base-p2  | bilstm      | full             | 3e-05 |            0.7648 |                    0.7697 |                 0.7748 |            7.2 |
| indobert-large-p1 | bilstm      | gradual_unfreeze | 3e-05 |            0.7645 |                    0.7829 |                 0.7916 |           51.1 |
| indobert-base-p2  | bilstm      | gradual_unfreeze | 3e-05 |            0.754  |                    0.7438 |                 0.7497 |            9.5 |
| indobert-base-p2  | bilstm_attn | full             | 3e-05 |            0.7532 |                    0.7839 |                 0.7928 |            9.6 |
| indobert-base-p2  | cls         | gradual_unfreeze | 3e-05 |            0.7531 |                    0.7608 |                 0.767  |            8.9 |
| indobert-base-p2  | bilstm_attn | gradual_unfreeze | 3e-05 |            0.7512 |                    0.7697 |                 0.7742 |            9.7 |
| indobert-large-p1 | bilstm_attn | gradual_unfreeze | 3e-05 |            0.75   |                    0.7697 |                 0.7753 |           65.3 |

## fase3_loss

| backbone          | head        | strategy    | loss            |   val_f1_weighted |   tahap1_test_f1_weighted |   tahap1_test_f1_macro |   best_epoch |
|:------------------|:------------|:------------|:----------------|------------------:|--------------------------:|-----------------------:|-------------:|
| indobertweet-base | bilstm_attn | freeze_half | weighted_ce     |            0.8154 |                    0.8202 |                 0.8208 |            2 |
| indobertweet-base | cls_mean    | freeze_half | weighted_ce     |            0.8119 |                    0.8192 |                 0.8279 |            2 |
| indobertweet-base | cls_mean    | full        | weighted_ce     |            0.8116 |                    0.8216 |                 0.827  |            2 |
| indobertweet-base | bilstm_attn | freeze_half | label_smoothing |            0.8115 |                    0.8203 |                 0.8238 |            2 |
| indobertweet-base | cls_mean    | full        | ce              |            0.8102 |                    0.8225 |                 0.8289 |            2 |
| indobertweet-base | cls_mean    | freeze_half | ce              |            0.8093 |                    0.8239 |                 0.8296 |            2 |
| indobertweet-base | bilstm_attn | freeze_half | ce              |            0.8091 |                    0.8127 |                 0.8155 |            2 |
| indobertweet-base | cls_mean    | freeze_half | focal           |            0.8073 |                    0.8126 |                 0.8202 |            2 |
| indobertweet-base | cls_mean    | full        | label_smoothing |            0.8057 |                    0.8364 |                 0.8401 |            2 |
| indobertweet-base | cls_mean    | freeze_half | label_smoothing |            0.8024 |                    0.8253 |                 0.8326 |            2 |
| indobertweet-base | cls_mean    | full        | focal           |            0.7934 |                    0.8145 |                 0.8236 |            2 |
| indobertweet-base | bilstm_attn | freeze_half | focal           |            0.7932 |                    0.8257 |                 0.8296 |            3 |

## fase4_hyperparameter

|    lr |   batch_efektif |   dropout |   val_f1_weighted |   tahap1_test_f1_weighted |   tahap1_test_f1_macro |   best_epoch |
|------:|----------------:|----------:|------------------:|--------------------------:|-----------------------:|-------------:|
| 3e-05 |              16 |       0.3 |            0.8179 |                    0.8079 |                 0.8085 |            2 |
| 3e-05 |              16 |       0.1 |            0.8154 |                    0.8202 |                 0.8208 |            2 |
| 1e-05 |              16 |       0.3 |            0.8135 |                    0.8147 |                 0.8209 |            2 |
| 2e-05 |              16 |       0.1 |            0.8128 |                    0.8161 |                 0.8195 |            2 |
| 1e-05 |              16 |       0.1 |            0.812  |                    0.8161 |                 0.8211 |            2 |
| 2e-05 |              16 |       0.3 |            0.8114 |                    0.8145 |                 0.8179 |            2 |
| 3e-05 |              16 |       0.2 |            0.8109 |                    0.824  |                 0.8259 |            2 |
| 2e-05 |              16 |       0.2 |            0.8108 |                    0.8212 |                 0.8244 |            2 |
| 3e-05 |              32 |       0.1 |            0.8085 |                    0.8192 |                 0.8217 |            2 |
| 2e-05 |              32 |       0.2 |            0.8068 |                    0.8168 |                 0.8213 |            2 |
| 5e-05 |              16 |       0.2 |            0.8056 |                    0.8109 |                 0.8113 |            2 |
| 2e-05 |              32 |       0.3 |            0.8056 |                    0.8168 |                 0.8235 |            2 |
| 5e-05 |              16 |       0.3 |            0.8053 |                    0.8069 |                 0.81   |            2 |
| 2e-05 |              32 |       0.1 |            0.8048 |                    0.8136 |                 0.8187 |            2 |
| 5e-05 |              32 |       0.3 |            0.8047 |                    0.8138 |                 0.8183 |            2 |
| 5e-05 |              32 |       0.2 |            0.8032 |                    0.8172 |                 0.8187 |            2 |
| 1e-05 |              16 |       0.2 |            0.8027 |                    0.8153 |                 0.8216 |            2 |
| 3e-05 |              32 |       0.2 |            0.8023 |                    0.8166 |                 0.823  |            2 |
| 5e-05 |              16 |       0.1 |            0.8015 |                    0.8196 |                 0.8232 |            2 |
| 3e-05 |              32 |       0.3 |            0.8004 |                    0.8129 |                 0.8171 |            2 |
| 1e-05 |              32 |       0.1 |            0.7906 |                    0.8125 |                 0.815  |            3 |
| 1e-05 |              32 |       0.3 |            0.7901 |                    0.8152 |                 0.8192 |            3 |
| 1e-05 |              32 |       0.2 |            0.7853 |                    0.808  |                 0.8116 |            3 |
| 5e-05 |              32 |       0.1 |            0.783  |                    0.8015 |                 0.8073 |            1 |

## fase5_tahap2

| skenario                               | tahap2_test_accuracy   | tahap2_test_f1_weighted   | tahap2_test_f1_macro   | tahap2_test_f1_macro_emosi   | tahap2_test_f1_neutral   | tahap1_test_f1_weighted   | tahap2_test_ece   | tahap2_test_ece_kalibrasi   |   n_seed |
|:---------------------------------------|:-----------------------|:--------------------------|:-----------------------|:-----------------------------|:-------------------------|:--------------------------|:------------------|:----------------------------|---------:|
| duatahap_lr2penuh_full_noreh           | 0.7583 � 0.0080        | 0.7582 � 0.0085           | 0.7582 � 0.0085        | 0.7283 � 0.0083              | 0.8777 � 0.0133          | 0.7106 � 0.0051           | 0.0674 � 0.0082   | 0.0556 � 0.0057             |        3 |
| duatahap_lr2setengah_full_noreh        | 0.7467 � 0.0063        | 0.7455 � 0.0078           | 0.7455 � 0.0078        | 0.7129 � 0.0095              | 0.8761 � 0.0025          | 0.7344 � 0.0071           | 0.0593 � 0.0064   | 0.0478 � 0.0061             |        3 |
| duatahap_lr2penuh_freeze_half_noreh    | 0.7442 � 0.0029        | 0.7439 � 0.0023           | 0.7439 � 0.0023        | 0.7099 � 0.0022              | 0.8803 � 0.0089          | 0.7261 � 0.0053           | 0.0784 � 0.0061   | 0.0448 � 0.0070             |        3 |
| tahap2_saja                            | 0.7417 � 0.0138        | 0.7411 � 0.0127           | 0.7411 � 0.0127        | 0.7044 � 0.0154              | 0.8880 � 0.0021          | 0.5613 � 0.0063           | 0.0660 � 0.0063   | 0.0456 � 0.0054             |        3 |
| duatahap_lr2penuh_full_reh             | 0.7333 � 0.0118        | 0.7309 � 0.0104           | 0.7309 � 0.0104        | 0.6917 � 0.0166              | 0.8877 � 0.0164          | 0.7793 � 0.0182           | 0.0776 � 0.0277   | 0.0427 � 0.0172             |        3 |
| duatahap_lr2setengah_freeze_half_noreh | 0.7308 � 0.0052        | 0.7300 � 0.0053           | 0.7300 � 0.0053        | 0.6902 � 0.0037              | 0.8894 � 0.0119          | 0.7410 � 0.0097           | 0.0529 � 0.0181   | 0.0470 � 0.0061             |        3 |
| duatahap_lr2penuh_freeze_half_reh      | 0.7283 � 0.0123        | 0.7271 � 0.0122           | 0.7271 � 0.0122        | 0.6892 � 0.0143              | 0.8788 � 0.0235          | 0.7791 � 0.0176           | 0.0633 � 0.0247   | 0.0441 � 0.0196             |        3 |
| duatahap_lr2setengah_freeze_half_reh   | 0.7258 � 0.0038        | 0.7249 � 0.0048           | 0.7249 � 0.0048        | 0.6876 � 0.0062              | 0.8742 � 0.0035          | 0.7882 � 0.0090           | 0.0824 � 0.0213   | 0.0500 � 0.0099             |        3 |
| duatahap_lr2setengah_full_reh          | 0.7233 � 0.0275        | 0.7226 � 0.0264           | 0.7226 � 0.0264        | 0.6870 � 0.0343              | 0.8652 � 0.0137          | 0.7875 � 0.0087           | 0.0674 � 0.0070   | 0.0406 � 0.0125             |        3 |
| satu_tahap                             | 0.7117 � 0.0058        | 0.7131 � 0.0050           | 0.7131 � 0.0050        | 0.6788 � 0.0045              | 0.8505 � 0.0227          | 0.8088 � 0.0048           | 0.0719 � 0.0074   | 0.0542 � 0.0103             |        3 |
| tahap1                                 | 0.4258 � 0.0115        | 0.4254 � 0.0148           | 0.4254 � 0.0148        | 0.4253 � 0.0181              | 0.4257 � 0.0026          | 0.8088 � 0.0124           | 0.3904 � 0.0382   | 0.3222 � 0.0364             |        3 |

## fase6_ablasi

| skenario                  | tahap2_test_accuracy   | tahap2_test_f1_weighted   | tahap2_test_f1_macro   | tahap2_test_f1_macro_emosi   | tahap2_test_f1_neutral   | tahap1_test_f1_weighted   | tahap2_test_ece   | tahap2_test_ece_kalibrasi   |   n_seed |
|:--------------------------|:-----------------------|:--------------------------|:-----------------------|:-----------------------------|:-------------------------|:--------------------------|:------------------|:----------------------------|---------:|
| duatahap_norm             | 0.7392 � 0.0076        | 0.7386 � 0.0072           | 0.7386 � 0.0072        | 0.7006 � 0.0075              | 0.8906 � 0.0062          | 0.7304 � 0.0055           | 0.0590 � 0.0138   | 0.0527 � 0.0026             |        3 |
| duatahap_p1               | 0.7317 � 0.0095        | 0.7329 � 0.0079           | 0.7329 � 0.0079        | 0.6964 � 0.0103              | 0.8789 � 0.0146          | 0.6859 � 0.0218           | 0.0718 � 0.0168   | 0.0439 � 0.0101             |        3 |
| duatahap_neutral_riccosan | 0.6050 � 0.0100        | 0.5747 � 0.0167           | 0.5747 � 0.0167        | 0.6498 � 0.0061              | 0.2742 � 0.0781          | 0.4688 � 0.0157           | 0.1690 � 0.0460   | 0.1105 � 0.0280             |        3 |
| tahap1_norm               | 0.4442 � 0.0208        | 0.4416 � 0.0237           | 0.4416 � 0.0237        | 0.4399 � 0.0271              | 0.4486 � 0.0112          | 0.8037 � 0.0250           | 0.3589 � 0.0381   | 0.2953 � 0.0439             |        3 |
| tahap1_p1                 | 0.3850 � 0.0132        | 0.3604 � 0.0134           | 0.3604 � 0.0134        | 0.3465 � 0.0154              | 0.4160 � 0.0057          | 0.7837 � 0.0105           | 0.4886 � 0.0085   | 0.4262 � 0.0136             |        3 |

