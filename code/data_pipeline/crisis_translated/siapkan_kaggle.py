"""Langkah 1 data deteksi krisis: memilih kalimat dari dataset Kaggle "Sentiment Analysis for Mental Health"
(suchintikasarkar), file Combined Data.csv, kolom statement dan status.

Kelas positif (krisis) diambil dari status Suicidal. Kelas negatif (bukan krisis) diambil dari status lain,
termasuk Depression dan Anxiety, supaya model belajar membedakan ide bunuh diri dari kesedihan atau
kecemasan biasa. Kalimat yang terlalu pendek atau terlalu panjang dibuang karena pesan chatbot umumnya
pendek dan kalimat panjang mahal diterjemahkan.

Keluaran: kandidat_en.csv (id, status, label, text_en) untuk diterjemahkan dengan NLLB-200."""
import os, re, sys, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
N_KRISIS = 2500
N_BUKAN = {'Normal': 800, 'Depression': 900, 'Anxiety': 300, 'Stress': 250, 'Bipolar': 250}   # total 2500
MIN_KATA, MAKS_KATA = 5, 70

src = os.path.join(HERE, 'Combined Data.csv')
if not os.path.exists(src):
    sys.exit('File "Combined Data.csv" belum ada di folder ini. Unduh dari Kaggle lalu ekstrak ke sini.')
d = pd.read_csv(src)
kol_teks = next(c for c in d.columns if c.lower() in ('statement', 'text'))
kol_status = next(c for c in d.columns if c.lower() in ('status', 'label'))
d = d[[kol_teks, kol_status]].rename(columns={kol_teks: 'text_en', kol_status: 'status'}).dropna()
d['status'] = d.status.astype(str).str.strip()
awal = d.status.value_counts().to_dict()


def bersih(t):
    t = re.sub(r'https?://\S+|www\.\S+', ' ', str(t))
    t = re.sub(r'[@#]\w+', ' ', t)
    t = re.sub(r'(.)\1{3,}', r'\1\1', t)               # huruf berulang berlebihan
    return ' '.join(t.split())


d['text_en'] = d.text_en.map(bersih)
d['nw'] = d.text_en.str.split().str.len()
d = d[(d.nw >= MIN_KATA) & (d.nw <= MAKS_KATA)]
d['norm'] = d.text_en.str.lower().str.replace(r'[^a-z ]', '', regex=True).str.strip()
konflik = d.groupby('norm').status.nunique()
d = d[~d.norm.isin(konflik[konflik > 1].index)].drop_duplicates('norm')
d = d[d.text_en.str.contains(r'[a-zA-Z]{3}')]           # buang baris tanpa kata bahasa Inggris
setelah = d.status.value_counts().to_dict()

pos = d[d.status == 'Suicidal']
pos = pos.sample(min(N_KRISIS, len(pos)), random_state=SEED).assign(label='krisis')
# Pembersihan derau label: sebagian teks berstatus Depression atau lainnya justru memuat ide bunuh diri eksplisit
# (misalnya "I want to die"). Teks seperti itu tidak dipakai sebagai contoh negatif supaya model tidak belajar
# bahwa ide bunuh diri adalah "bukan krisis". Pola ini hanya dipakai untuk membersihkan data, bukan untuk deteksi.
IDE = (r"\b(?:kill(?:ing)? my ?self|suicid\w*|want(?:ed)? to die|wanna die|end (?:my|it all|my life)|take my (?:own )?life|"
       r"don'?t want to (?:live|be alive|exist)|better off dead|overdos\w*|hang my ?self|slit\w*)\b")
derau = (d.status != 'Suicidal') & d.text_en.str.contains(IDE, case=False, regex=True)
n_derau = d[derau].status.value_counts().to_dict()
d = d[~derau]
neg = []
for s, n in N_BUKAN.items():
    pool = d[d.status == s]
    neg.append(pool.sample(min(n, len(pool)), random_state=SEED))
neg = pd.concat(neg).assign(label='bukan')
out = pd.concat([pos, neg]).sample(frac=1, random_state=SEED).reset_index(drop=True)
out.insert(0, 'id', [f'kg_{i:05d}' for i in range(len(out))])
out[['id', 'status', 'label', 'text_en', 'nw']].to_csv(os.path.join(HERE, 'kandidat_en.csv'), index=False)
st = {'jumlah_awal_per_status': awal, 'setelah_filter': setelah, 'negatif_dibuang_karena_memuat_ide_bunuh_diri': n_derau, 'dipilih_per_status': out.status.value_counts().to_dict(),
      'dipilih_per_label': out.label.value_counts().to_dict(), 'median_kata': float(out.nw.median()),
      'filter': f'{MIN_KATA}-{MAKS_KATA} kata, tanpa URL/mention, tanpa duplikat, tanpa teks berlabel ganda'}
json.dump(st, open(os.path.join(HERE, 'statistik_seleksi.json'), 'w'), indent=1)
print(json.dumps(st, indent=1))
