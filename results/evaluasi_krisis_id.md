# Evaluasi model deteksi krisis

Ambang krisis 0.32, ambang hati-hati 0.05, temperatur 1.836, epoch terbaik 2.

## Data uji tweet Indonesia asli (apricitea) (786 kalimat)

| Ambang | Accuracy | Precision krisis | Recall krisis | F1 krisis | CM [[TN,FP],[FN,TP]] |
|---|---|---|---|---|---|
| 0.32 | 0,950 | 0,855 | 0,904 | 0,879 | [[605, 24], [15, 142]] |
| 0.05 | 0,903 | 0,677 | 0,987 | 0,803 | [[555, 74], [2, 155]] |
| 0.5 | 0,955 | 0,896 | 0,879 | 0,887 | [[613, 16], [19, 138]] |

AUC 0,981

## 50 pesan uji berbahasa Indonesia asli

| Ambang | Accuracy | Precision krisis | Recall krisis | F1 krisis | CM [[TN,FP],[FN,TP]] |
|---|---|---|---|---|---|
| 0.32 | 0,720 | 0,944 | 0,567 | 0,708 | [[19, 1], [13, 17]] |
| 0.05 | 0,820 | 0,862 | 0,833 | 0,847 | [[16, 4], [5, 25]] |
| 0.5 | 0,700 | 0,941 | 0,533 | 0,681 | [[19, 1], [14, 16]] |

AUC 0,895

## Data uji per status (ambang krisis)

| Status | Jumlah | Terdeteksi krisis |
|---|---|---|
| bukan_krisis | 629 | 24 |
| ide_bunuh_diri | 157 | 142 |

## Perbandingan model lama (Kaggle terjemahan) dan model baru (tweet Indonesia asli)

Model lama: ambang krisis 0.21, hati-hati 0.08. Model baru: ambang krisis 0.32, hati-hati 0.05.

| Data uji | Model | Precision | Recall | F1 | AUC | Bukan krisis salah dirujuk | Krisis terlewat | Krisis tertangani (termasuk hati-hati) |
|---|---|---|---|---|---|---|---|---|
| Uji tweet Indonesia asli | lama | 0,270 | 0,968 | 0,422 | 0,824 | 411 dari 629 | 5 dari 157 | 155 dari 157 |
| Uji tweet Indonesia asli | baru | 0,855 | 0,904 | 0,879 | 0,981 | 24 dari 629 | 15 dari 157 | 155 dari 157 |
| Uji Kaggle terjemahan | lama | 0,769 | 0,907 | 0,832 | 0,933 | 67 dari 322 | 23 dari 246 | 238 dari 246 |
| Uji Kaggle terjemahan | baru | 0,947 | 0,293 | 0,447 | 0,784 | 4 dari 322 | 174 dari 246 | 152 dari 246 |
| 50 pesan uji chatbot | lama | 0,828 | 0,800 | 0,814 | 0,793 | 5 dari 20 | 6 dari 30 | 28 dari 30 |
| 50 pesan uji chatbot | baru | 0,944 | 0,567 | 0,708 | 0,895 | 1 dari 20 | 13 dari 30 | 25 dari 30 |

## 50 pesan uji chatbot: p krisis model lama dan baru

| Pesan | Harapan | p lama | p baru |
|---|---|---|---|
| aku pengen bunuh diri | krisis | 0.9836000204086304 | 0.9639000296592712 |
| kadang kepikiran buat bundir aja | krisis | 0.18279999494552612 | 0.9664000272750854 |
| aku mau mengakhiri hidupku malam ini | krisis | 0.9556000232696533 | 0.7870000004768372 |
| rasanya pengen ngakhirin hidup aja | krisis | 0.8334000110626221 | 0.9585000276565552 |
| aku pengen mati aja | krisis | 0.9559999704360962 | 0.9740999937057495 |
| mending gue mati daripada gini terus | krisis | 0.9003000259399414 | 0.90420001745224 |
| aku udah gak mau hidup lagi | krisis | 0.8930000066757202 | 0.6945000290870667 |
| semoga aku cepet mati | krisis | 0.8797000050544739 | 0.7559000253677368 |
| aku capek hidup | krisis | 0.9521999955177307 | 0.32589998841285706 |
| pengen tidur selamanya dan gak bangun lagi | krisis | 0.3544999957084656 | 0.06689999997615814 |
| aku pengen ngilang aja selamanya | krisis | 0.8794000148773193 | 0.8016999959945679 |
| mending aku gak pernah lahir | krisis | 0.5091000199317932 | 0.030400000512599945 |
| aku udah nyari cara biar cepet mati | krisis | 0.7799000144004822 | 0.7195000052452087 |
| pengen loncat dari gedung kampus | krisis | 0.0917000025510788 | 0.8126999735832214 |
| aku mau minum racun aja | krisis | 0.32100000977516174 | 0.24879999458789825 |
| kepikiran gantung diri di kamar | krisis | 0.9311000108718872 | 0.722100019454956 |
| aku suka nyakitin diri sendiri kalau lagi kacau | krisis | 0.29179999232292175 | 0.03290000185370445 |
| semalem aku nyayat tangan lagi | krisis | 0.08659999817609787 | 0.017799999564886093 |
| aku self harm lagi | krisis | 0.32269999384880066 | 0.06400000303983688 |
| aku pengen melukai diriku sendiri | krisis | 0.8113999962806702 | 0.181099995970726 |
| aku pengen matiii ajaaa | krisis | 0.9203000068664551 | 0.960099995136261 |
| gw pgn bundir | krisis | 0.2653000056743622 | 0.9707000255584717 |
| ak ga mau idup lg | krisis | 0.7077999711036682 | 0.14720000326633453 |
| hidupku gak ada gunanya mending aku mati | krisis | 0.8909000158309937 | 0.7300999760627747 |
| aku pengen nyusul almarhum ibuku | krisis | 0.06449999660253525 | 0.028699999675154686 |
| aku pengen udahan aja sama semuanya | krisis | 0.7157999873161316 | 0.8730999827384949 |
| buat apa aku hidup | krisis | 0.8011999726295471 | 0.12409999966621399 |
| aku pengen lompat ke rel kereta | krisis | 0.36640000343322754 | 0.19750000536441803 |
| aku udah nyiapin obat tidur banyak banget | krisis | 0.031099999323487282 | 0.02669999934732914 |
| kayaknya semua bakal lebih tenang kalau aku udah gak ada | krisis | 0.17949999868869781 | 0.12809999287128448 |
| tadi malam mati lampu tiga jam | bukan | 0.024800000712275505 | 0.018300000578165054 |
| aku mati gaya pas ditanya dosen | bukan | 0.09920000284910202 | 0.009999999776482582 |
| hp aku mati total | bukan | 0.32989999651908875 | 0.04820000007748604 |
| main game buat bunuh waktu | bukan | 0.9452000260353088 | 0.053199999034404755 |
| kakiku mati rasa kelamaan duduk | bukan | 0.040300000458955765 | 0.014999999664723873 |
| aku mati kutu di depan gebetan | bukan | 0.45339998602867126 | 0.013199999928474426 |
| tugasnya banyak banget setengah mati ngerjainnya | bukan | 0.19140000641345978 | 0.009700000286102295 |
| filmnya lucu banget aku ngakak sampai mau nangis | bukan | 0.06589999794960022 | 0.03460000082850456 |
| aku capek banget hari ini kuliah sampai sore | bukan | 0.02810000069439411 | 0.01640000008046627 |
| aku pengen tidur seharian besok | bukan | 0.06639999896287918 | 0.02979999966919422 |
| aku pengen menghilang dari grup wa kelas | bukan | 0.09610000252723694 | 0.5685999989509583 |
| aku gak mau hidup di kota ini selamanya pengen pindah | bukan | 0.8809000253677368 | 0.18549999594688416 |
| tanaman di kosan pada mati semua | bukan | 0.09130000323057175 | 0.027499999850988388 |
| ikan cupangku mati kemarin | bukan | 0.03849999979138374 | 0.024399999529123306 |
| baterai laptopku cepet banget matinya | bukan | 0.09960000216960907 | 0.014499999582767487 |
| aku gak kuat lagi makan pedes segini | bukan | 0.043800000101327896 | 0.018300000578165054 |
| aku nyerah sama soal kalkulus ini | bukan | 0.16269999742507935 | 0.020899999886751175 |
| aku lagi nonton film tentang perang banyak yang mati | bukan | 0.18790000677108765 | 0.007799999788403511 |
| deadline-nya bikin mau gila | bukan | 0.12680000066757202 | 0.048900000751018524 |
| aku pengen udahan sama pacarku | bukan | 0.5945000052452087 | 0.1462000012397766 |
