"""Cek kualitas otomatis hasil terjemahan neutral dan pilih 800 sampel final.

Flag otomatis (untuk memprioritaskan verifikasi manual):
  sisa_inggris   : masih banyak kata bahasa Inggris
  rasio_panjang  : panjang ID / EN < 0.6 atau > 1.8 (kemungkinan terpotong atau halusinasi)
  pengulangan    : ada kata yang berulang >= 4 kali berturut/berlebihan
  kata_emosi_id  : terjemahan memuat kata emosi bahasa Indonesia (bisa bukan netral lagi)
Sampel tanpa flag diprioritaskan masuk 800 final; sisanya jadi cadangan.
Output:
  neutral_dailydialog_id_semua.csv   (1000 baris + flag)
  neutral_dailydialog_id_final800.csv
  lembar_verifikasi_neutral.csv      (untuk dicek manual: status OK/REVISI/BUANG)
"""
import re, json, pandas as pd, numpy as np
SEED = 42; N_FINAL = 800
df = pd.read_csv('neutral_dailydialog_id.csv')
EN = set('the a an is are was were you your i my me we our they their this that what which with would could should will have has had do does did not no yes please of to in on for at from and or but if so'.split())
EMO_ID = re.compile(r'\b(senang|bahagia|gembira|sedih|kecewa|marah|kesal|jengkel|benci|takut|cemas|khawatir|gugup|panik|'
                    r'bangga|syukur|bersyukur|cinta|sayang|rindu|kangen|kesepian|malu|bersalah|menyesal|terkejut|'
                    r'kaget|jijik|stres|frustrasi|hebat|luar biasa|mengerikan|buruk sekali|sayangnya|untungnya)\b', re.I)

def en_ratio(t):
    w = re.findall(r"[a-zA-Z']+", str(t).lower())
    return sum(x in EN for x in w) / max(len(w), 1)

def repeat(t):
    w = str(t).lower().split()
    return any(w[i] == w[i + 1] == w[i + 2] for i in range(len(w) - 2)) or (len(w) > 6 and len(set(w)) / len(w) < 0.4)

df['nw_id'] = df.text_id.astype(str).str.split().str.len()
df['rasio_panjang'] = (df.nw_id / df.nw).round(2)
df['f_sisa_inggris'] = df.text_id.map(en_ratio) > 0.25
df['f_rasio_panjang'] = (df.rasio_panjang < 0.6) | (df.rasio_panjang > 1.8)
df['f_pengulangan'] = df.text_id.map(repeat)
df['f_kata_emosi_id'] = df.text_id.str.contains(EMO_ID)
fl = ['f_sisa_inggris', 'f_rasio_panjang', 'f_pengulangan', 'f_kata_emosi_id']
df['flag'] = df[fl].apply(lambda r: ','.join(c[2:] for c in fl if r[c]), axis=1)
df['bersih'] = df.flag == ''
df['norm_id'] = df.text_id.str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True).str.strip()
df['dup_id'] = df.duplicated('norm_id')
df.loc[df.dup_id, 'flag'] = (df.loc[df.dup_id, 'flag'] + ',duplikat').str.strip(',')
df.loc[df.dup_id, 'bersih'] = False

# 800 final: ambil yang bersih, jaga distribusi panjang (per bin) tetap seperti kandidat
bins = [5, 8, 11, 14, 17, 20, 25, 30, 46]
df['bin'] = pd.cut(df.nw, bins=bins, right=False, labels=False)
prop = df.bin.value_counts(normalize=True).sort_index()
parts = []
for b, p in prop.items():
    k = int(round(p * N_FINAL)); pool = df[(df.bin == b) & df.bersih]
    parts.append(pool.sample(min(k, len(pool)), random_state=SEED))
final = pd.concat(parts)
if len(final) < N_FINAL:
    rest = df[df.bersih & ~df.id.isin(final.id)]
    final = pd.concat([final, rest.sample(N_FINAL - len(final), random_state=SEED)])
final = final.head(N_FINAL)
df['masuk_final800'] = df.id.isin(final.id)

cols = ['id', 'label', 'text_en', 'text_id', 'nw', 'nw_id', 'rasio_panjang', 'flag', 'masuk_final800',
        'is_q', 'dialogue_id', 'data_split', 'topic', 'nllb_config']
df[cols].to_csv('neutral_dailydialog_id_semua.csv', index=False)
final[['id', 'label', 'text_id', 'text_en', 'dialogue_id', 'data_split', 'topic']].rename(
    columns={'text_id': 'text'}).sort_values('id').to_csv('neutral_dailydialog_id_final800.csv', index=False)
v = df[cols[:8] + ['masuk_final800']].copy()
v['status_verifikasi'] = ''; v['text_id_revisi'] = ''; v['catatan'] = ''
v.sort_values(['masuk_final800', 'id'], ascending=[False, True]).to_csv('lembar_verifikasi_neutral.csv', index=False)

stat = {'total': len(df), 'bersih': int(df.bersih.sum()), 'final': len(final),
        'flag_counts': {c[2:]: int(df[c].sum()) for c in fl}, 'duplikat_id': int(df.dup_id.sum()),
        'median_kata_en_final': float(final.nw.median()), 'median_kata_id_final': float(final.nw_id.median()),
        'pertanyaan_pct_final': round(100 * final.is_q.mean(), 1)}
json.dump(stat, open('neutral_terjemahan_stats.json', 'w'), indent=1)
print(json.dumps(stat, indent=1))
print(df[~df.bersih][['text_en', 'text_id', 'flag']].head(12).to_string())
print(final.sample(12, random_state=3)[['text_en', 'text_id']].to_string())
