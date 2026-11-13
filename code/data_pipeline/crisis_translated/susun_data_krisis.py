"""Langkah 3 data deteksi krisis: memeriksa hasil terjemahan, menambah contoh negatif berbahasa Indonesia
dari data latih domain konseling, lalu membagi data menjadi latih, validasi, dan uji (80/10/10).

Contoh negatif tambahan diambil dari tahap2_train.csv (kalimat emosi hasil terjemahan EmpatheticDialogues dan
DailyDialog), hanya dari bagian data latih supaya data uji emosi tidak ikut terpakai. Tujuannya agar model belajar
bahwa kalimat sedih, takut, atau marah biasa bukan pesan krisis. Contoh ini melewati penerjemahan NLLB yang sama
sehingga gaya bahasanya setara dengan data Kaggle.

Keluaran: krisis_train.csv, krisis_val.csv, krisis_test.csv (id, text, label, status, text_en)."""
import os, re, json
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_EMOSI = os.path.join(HERE, '..', '..', 'Eksperimen', 'data_final', 'tahap2_train.csv')
SEED = 42
N_ED_PER_KELAS = 150

d = pd.read_csv(os.path.join(HERE, 'kandidat_id.csv'))
EN = set('the a an is are was were you your i my me we they this that what with would could should will have has had '
         'do does did not of to in on for at from and or but if so just like really'.split())


def sisa_inggris(t):
    w = re.findall(r"[a-zA-Z']+", str(t).lower())
    return sum(x in EN for x in w) / max(len(w), 1)


def ulang(t):
    w = str(t).lower().split()
    return any(w[i] == w[i + 1] == w[i + 2] for i in range(len(w) - 2)) or (len(w) > 8 and len(set(w)) / len(w) < 0.35)


d['nw_id'] = d.text_id.astype(str).str.split().str.len()
d['rasio'] = d.nw_id / d.nw
d['flag'] = ''
d.loc[d.text_id.map(sisa_inggris) > 0.25, 'flag'] += 'sisa_inggris,'
d.loc[(d.rasio < 0.5) | (d.rasio > 2.0), 'flag'] += 'rasio_panjang,'
d.loc[d.text_id.map(ulang), 'flag'] += 'pengulangan,'
d.loc[d.text_id.isna() | (d.nw_id < 3), 'flag'] += 'kosong,'
flag = d.flag.str.strip(',').value_counts().to_dict()
d = d[d.flag == ''].copy()
d = d.drop_duplicates('text_id')
kg = d.rename(columns={'text_id': 'text'})[['id', 'text', 'label', 'status', 'text_en']]

ed = pd.read_csv(DATA_EMOSI)
ed = pd.concat([g.sample(min(N_ED_PER_KELAS, len(g)), random_state=SEED) for _, g in ed.groupby('label')])
ed = pd.DataFrame({'id': 'ed_' + ed.id.astype(str), 'text': ed.text, 'label': 'bukan',
                   'status': 'emosi_' + ed.label, 'text_en': ed.get('text_en', '')})

semua = pd.concat([kg, ed], ignore_index=True).dropna(subset=['text'])
strata = semua.label + '|' + semua.status
tr, sisa = train_test_split(semua, test_size=0.2, stratify=strata, random_state=SEED)
va, te = train_test_split(sisa, test_size=0.5, stratify=strata.loc[sisa.index], random_state=SEED)
for nama, x in [('train', tr), ('val', va), ('test', te)]:
    x.to_csv(os.path.join(HERE, f'krisis_{nama}.csv'), index=False)
st = {'terjemahan_ditandai_dibuang': flag, 'kaggle_lolos': len(kg), 'tambahan_negatif_emosi': len(ed),
      'total': len(semua), 'per_label': semua.label.value_counts().to_dict(),
      'per_status': semua.status.value_counts().to_dict(),
      'split': {'train': len(tr), 'val': len(va), 'test': len(te)}}
json.dump(st, open(os.path.join(HERE, 'statistik_data_krisis.json'), 'w'), indent=1)
print(json.dumps(st, indent=1))
