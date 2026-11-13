"""Membangun dataset final untuk seluruh eksperimen.

Input:
  riccosan_dir/            6 CSV Riccosan (GitHub resmi)
  ed_kandidat_id.csv       hasil terjemahan 01 + terjemahkan_nllb.py
  neutral_semua.csv        neutral_dailydialog_id_semua.csv (1.000 kandidat + flag)
Output (folder data_final/):
  tahap1_{train,val,test}.csv          Riccosan bersih, 5 kelas
  tahap2_{train,val,test}.csv          domain konseling: 4 kelas ED + neutral DailyDialog
  rehearsal_tahap1.csv                 10% data latih tahap 1 (untuk varian rehearsal)
  satu_tahap_{train,val}.csv           gabungan train/val tahap 1 + tahap 2 (baseline satu tahap)
  verifikasi_tahap2.csv                lembar verifikasi manual (test 100%, train+val sampel 12%)
  dataset_stats.json
Semua split stratified dengan seed 42.
"""
import re, sys, json, glob, os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
import warnings; warnings.filterwarnings('ignore')
RIC_DIR, ED_FILE, NEU_FILE = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = sys.argv[4] if len(sys.argv) > 4 else 'data_final'
os.makedirs(OUT, exist_ok=True)
LABELS = ['anger', 'fear', 'happy', 'sadness', 'neutral']
st = {}

def norm(s):
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', str(s).lower())).strip()

def split3(df, strat):
    tr, tmp = train_test_split(df, test_size=0.2, stratify=df[strat], random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.5, stratify=tmp[strat], random_state=SEED)
    return tr, va, te

# ---------------- Tahap 1: Riccosan ----------------
fr = []
for f in sorted(glob.glob(os.path.join(RIC_DIR, '*.csv'))):
    d = pd.read_csv(f, sep='\t', dtype=str, keep_default_na=False, quoting=3)
    d.columns = [c.strip() for c in d.columns]
    fr.append(d)
ric = pd.concat(fr, ignore_index=True)
ric['text'] = ric.Tweet.str.strip()
ric['label'] = ric.Label.str.strip().map({'Anger': 'anger', 'Fear': 'fear', 'Joy': 'happy', 'Love': 'happy',
                                          'Sad': 'sadness', 'Neutral': 'neutral'})
ric['label_asli'] = ric.Label.str.strip()
st['riccosan_mentah'] = len(ric)
ric['norm'] = ric.text.map(norm)
ric = ric[ric.norm != '']
conflict = ric.groupby('norm').label.nunique()
conflict = set(conflict[conflict > 1].index)
st['riccosan_teks_label_konflik_dibuang'] = int(ric.norm.isin(conflict).sum())
ric = ric[~ric.norm.isin(conflict)]
ric = ric.drop_duplicates('norm')
st['riccosan_bersih'] = len(ric)
st['riccosan_bersih_per_kelas'] = ric.label.value_counts().to_dict()
ric['source'] = 'riccosan'
ric['id'] = [f'ric_{i:05d}' for i in range(len(ric))]
t1 = split3(ric, 'label')
for name, d in zip(['train', 'val', 'test'], t1):
    d[['id', 'text', 'label', 'label_asli', 'source']].to_csv(f'{OUT}/tahap1_{name}.csv', index=False)
st['tahap1_split'] = {n: {'total': len(d), **d.label.value_counts().to_dict()} for n, d in zip(['train', 'val', 'test'], t1)}

# ---------------- Tahap 2: ED (4 kelas) + DailyDialog neutral ----------------
EN = set('the a an is are was were you your i my me we our they their this that what which with would could should '
         'will have has had do does did not no yes please of to in on for at from and or but if so'.split())
def en_ratio(t):
    w = re.findall(r"[a-zA-Z']+", str(t).lower()); return sum(x in EN for x in w) / max(len(w), 1)
def repeat(t):
    w = str(t).lower().split()
    return any(w[i] == w[i + 1] == w[i + 2] for i in range(len(w) - 2)) or (len(w) > 6 and len(set(w)) / len(w) < 0.4)

ed = pd.read_csv(ED_FILE)
ed['nw_id'] = ed.text_id.astype(str).str.split().str.len()
ed['rasio'] = ed.nw_id / ed.nw
ed['flag'] = ''
ed.loc[ed.text_id.map(en_ratio) > 0.25, 'flag'] += 'sisa_inggris,'
ed.loc[(ed.rasio < 0.6) | (ed.rasio > 1.8), 'flag'] += 'rasio_panjang,'
ed.loc[ed.text_id.map(repeat), 'flag'] += 'pengulangan,'
ed['norm'] = ed.text_id.map(norm)
ed.loc[ed.duplicated('norm'), 'flag'] += 'duplikat,'
ed['flag'] = ed.flag.str.strip(',')
st['ed_kandidat'] = len(ed)
st['ed_flag'] = {k: int(ed.flag.str.contains(k).sum()) for k in ['sisa_inggris', 'rasio_panjang', 'pengulangan', 'duplikat']}
PER = {'anger': 200, 'fear': 200, 'happy': 100, 'sadness': 100}
pick = []
for la, g in ed.groupby('label_asli'):
    k = PER[g.label.iloc[0]]
    clean = g[g.flag == '']
    if len(clean) >= k:
        pick.append(clean.sample(k, random_state=SEED))
    else:   # kekurangan: lengkapi dengan yang ber-flag paling ringan (rasio_panjang saja)
        extra = g[g.flag == 'rasio_panjang'].sample(min(k - len(clean), (g.flag == 'rasio_panjang').sum()), random_state=SEED)
        pick.append(pd.concat([clean, extra]))
ed_sel = pd.concat(pick)
ed_sel = ed_sel.assign(text=ed_sel.text_id, source='empatheticdialogues')
st['ed_terpilih_per_kelas'] = ed_sel.label.value_counts().to_dict()

neu = pd.read_csv(NEU_FILE)
neu = neu[neu.masuk_final800].assign(text=lambda d: d.text_id, label_asli='no emotion', source='dailydialog')
neu = neu.rename(columns={'dialogue_id': 'conv_id'})
st['neutral_terpilih'] = len(neu)

cols = ['id', 'text', 'label', 'label_asli', 'source', 'text_en', 'conv_id']
t2all = pd.concat([ed_sel[cols], neu[cols]], ignore_index=True)
# cegah teks yang sama muncul di tahap 1 dan tahap 2
t2all = t2all[~t2all.text.map(norm).isin(set(ric.norm))]
vc = t2all.label_asli.value_counts()
t2all['strat'] = t2all.label_asli if vc.min() >= 10 else t2all.label
t2 = split3(t2all, 'strat')
for name, d in zip(['train', 'val', 'test'], t2):
    d[cols].to_csv(f'{OUT}/tahap2_{name}.csv', index=False)
st['tahap2_split'] = {n: {'total': len(d), **d.label.value_counts().to_dict()} for n, d in zip(['train', 'val', 'test'], t2)}

# ---------------- Varian ablasi: neutral tahap 2 dari tweet Riccosan (rancangan awal) ----------------
# 4 kelas emosi identik dengan tahap2_*; neutral diganti tweet neutral Riccosan dengan porsi yang
# sejajar: latih dari tahap1_train, validasi dari tahap1_val, uji dari tahap1_test. Dengan begitu
# tweet neutral di data validasi/uji ablasi tidak pernah dilihat model saat pelatihan tahap 1.
a_, b_ = int(0.8 * len(neu)), int(0.9 * len(neu))
nparts = [t1[k][t1[k].label == 'neutral'].sample(n, random_state=SEED).assign(text_en='', conv_id='')
          for k, n in zip(range(3), [a_, b_ - a_, len(neu) - b_])]
for name, d, rn in zip(['train', 'val', 'test'], t2, nparts):
    alt = pd.concat([d[d.label != 'neutral'][cols], rn[cols]])
    alt.to_csv(f'{OUT}/tahap2alt_{name}.csv', index=False)
st['tahap2alt'] = 'sama dengan tahap2, kelas neutral diganti tweet neutral Riccosan (latih/val/uji dari porsi tahap 1 yang sejajar)'

# ---------------- Rehearsal & baseline satu tahap ----------------
def strat_sample(d, col, frac):
    return pd.concat([g.sample(frac=frac, random_state=SEED) for _, g in d.groupby(col)])

reh = strat_sample(t1[0], 'label', 0.10)
reh[['id', 'text', 'label', 'label_asli', 'source']].to_csv(f'{OUT}/rehearsal_tahap1.csv', index=False)
st['rehearsal'] = len(reh)
c = ['id', 'text', 'label', 'label_asli', 'source']
pd.concat([t1[0][c], t2[0][c]]).to_csv(f'{OUT}/satu_tahap_train.csv', index=False)
pd.concat([t1[1][c], t2[1][c]]).to_csv(f'{OUT}/satu_tahap_val.csv', index=False)

# ---------------- Lembar verifikasi manual ----------------
te = t2[2].assign(bagian='test')
trv = strat_sample(pd.concat([t2[0], t2[1]]), 'label_asli', 0.12).assign(bagian='train/val (sampel 12%)')
ver = pd.concat([te, trv])[['id', 'bagian', 'label', 'label_asli', 'text_en', 'text']]
ver['status_verifikasi'] = ''; ver['text_revisi'] = ''; ver['label_revisi'] = ''; ver['catatan'] = ''
ver.to_csv(f'{OUT}/verifikasi_tahap2.csv', index=False)
st['verifikasi'] = {'test': len(te), 'sampel_train_val': len(trv)}

json.dump(st, open(f'{OUT}/dataset_stats.json', 'w'), indent=1, default=str)
print(json.dumps(st, indent=1, default=str))
