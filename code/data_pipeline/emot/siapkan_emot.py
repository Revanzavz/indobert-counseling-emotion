"""Menyiapkan EmoT (Saputri et al., 2018) sebagai data uji tambahan berbahasa Indonesia asli.
Hanya train+valid yang dipakai karena label data test resmi disamarkan.
love digabung ke happy. Tweet yang juga ada di data Riccosan (tahap 1) dibuang."""
import os, re, json, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(H, '..', '..', 'Eksperimen', 'data_final')
e = pd.concat([pd.read_csv(os.path.join(H, f)) for f in ['train_preprocess.csv', 'valid_preprocess.csv']])
e['label'] = e.label.str.strip().str.lower().replace({'love': 'happy', 'sad': 'sadness'})
e = e.rename(columns={'tweet': 'text'})
norm = lambda s: re.sub(r'\[username\]|\[url\]|[^a-z0-9]', '', str(s).lower())
e['n'] = e.text.map(norm)
awal = len(e)
e = e[~e.n.duplicated(keep=False) | ~e.duplicated(['n', 'label'])]
konflik = e.groupby('n').label.nunique(); e = e[~e.n.isin(konflik[konflik > 1].index)].drop_duplicates('n')
ric = pd.concat([pd.read_csv(os.path.join(D, f'tahap1_{s}.csv')) for s in ['train', 'val', 'test']])
kol = 'text' if 'text' in ric.columns else ric.columns[0]
rn = set(ric[kol].map(norm))
overlap = e.n.isin(rn)
e2 = e[~overlap & (e.n.str.len() > 0)]
e2[['label', 'text']].to_csv(os.path.join(H, 'emot_uji.csv'), index=False, encoding='utf-8-sig')
st = {'baris_awal': awal, 'setelah_dedup': len(e), 'tumpang_tindih_riccosan': int(overlap.sum()),
      'final': len(e2), 'per_kelas': e2.label.value_counts().to_dict()}
json.dump(st, open(os.path.join(H, 'emot_stats.json'), 'w'), indent=1); print(json.dumps(st, indent=1))
