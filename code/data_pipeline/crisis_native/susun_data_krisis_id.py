"""Menyusun data deteksi krisis berbahasa Indonesia asli dari dataset apricitea (tweet X/Twitter, tiga anotator).

Sumber : final_dataset.xlsx, sheet "10k" (label 1 = ide bunuh diri, 0 = bukan; label akhir suara terbanyak dari
         tiga anotator ACN, BME, AIP pada sheet "current_data").
Langkah: perbaikan emoji yang salah decode, hapus URL dan mention, rapikan spasi, buang teks < 3 kata,
         buang duplikat (setelah normalisasi) dan teks berlabel ganda, buang teks yang sama dengan 50 pesan uji
         chatbot, lalu dibagi bertingkat 80/10/10 (seed 42).
Keluaran: krisis_{train,val,test}.csv (kolom id, text, label, status) dan statistik_seleksi.json"""
import os, re, json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'final_dataset.xlsx')
UJI = os.path.join(HERE, '..', '..', 'Chatbot', 'uji_krisis.csv')


def perbaiki(t):
    try:
        return t.encode('cp1252').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return t


def bersihkan(t):
    t = perbaiki(str(t))
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)
    t = re.sub(r'@\w+', ' ', t)
    t = re.sub(r'&amp;', '&', t)
    return re.sub(r'\s+', ' ', t).strip()


kunci = lambda t: re.sub(r'[^a-z0-9 ]', '', t.lower()).strip()

d = pd.read_excel(SRC, sheet_name='10k')
st = {'awal': len(d), 'awal_per_label': d.label.value_counts().to_dict()}
ann = pd.read_excel(SRC, sheet_name='current_data')
st['kappa_anotator'] = {f'{a}-{b}': round(cohen_kappa_score(*ann[[a, b]].dropna().T.values), 3)
                        for a, b in [('ACN', 'BME'), ('ACN', 'AIP'), ('BME', 'AIP')]}
d['text'] = d.tweet.map(bersihkan)
d = d[d.text.str.split().str.len() >= 3]
st['setelah_buang_pendek'] = len(d)
d['k'] = d.text.map(kunci)
ganda = d.groupby('k').label.nunique()
d = d[~d.k.isin(ganda[ganda > 1].index)]
st['buang_label_ganda_teks'] = int((ganda > 1).sum())
d = d.drop_duplicates('k')
st['setelah_buang_duplikat'] = len(d)
if os.path.exists(UJI):
    uk = set(pd.read_csv(UJI).pesan.map(kunci))
    n0 = len(d); d = d[~d.k.isin(uk)]; st['buang_sama_dengan_uji_chatbot'] = n0 - len(d)
d['label'] = d.label.map({1: 'krisis', 0: 'bukan'})
d['status'] = d.label.map({'krisis': 'ide_bunuh_diri', 'bukan': 'bukan_krisis'})
d['id'] = ['ap_%05d' % i for i in d['index']]
d = d[['id', 'text', 'label', 'status']].reset_index(drop=True)
tr, sisa = train_test_split(d, test_size=0.2, stratify=d.label, random_state=42)
va, te = train_test_split(sisa, test_size=0.5, stratify=sisa.label, random_state=42)
for n, x in [('train', tr), ('val', va), ('test', te)]:
    x.to_csv(os.path.join(HERE, f'krisis_{n}.csv'), index=False, encoding='utf-8')
    st[n] = {'n': len(x), **x.label.value_counts().to_dict()}
json.dump(st, open(os.path.join(HERE, 'statistik_seleksi.json'), 'w'), indent=1)
print(json.dumps(st, indent=1))
