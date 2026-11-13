"""Ekstraksi kandidat 4 kelas emosi dari EmpatheticDialogues untuk dataset tahap 2.

Aturan (dikunci):
- Hanya ujaran pertama pembicara (utterance_idx = 1), karena label emosi ED berlaku pada
  pengalaman pembicara, bukan pada balasan pendengar.
- Pemetaan 24 label asli ke 4 kelas mengikuti Tabel 3.1 proposal; 8 label lain dibuang.
- Filter panjang 5-45 kata (sama dengan kelas neutral DailyDialog), deduplikasi.
- Sampling proporsional per label asli agar tidak didominasi satu label:
    anger & fear  : 4 label x 250 = 1.000 kandidat per kelas
    happy & sadness: 8 label x 125 = 1.000 kandidat per kelas
  (800 per kelas dipakai, sisanya cadangan untuk yang gugur di cek kualitas/verifikasi)
Output: ed_kandidat_en.csv
"""
import sys, json, pandas as pd

ED_DIR = sys.argv[1] if len(sys.argv) > 1 else '../EmpatheticDialogues/empatheticdialogues'
SEED = 42
EMAP = {'anger': ['angry', 'furious', 'jealous', 'annoyed'],
        'fear': ['afraid', 'anxious', 'terrified', 'apprehensive'],
        'happy': ['joyful', 'excited', 'proud', 'grateful', 'content', 'hopeful', 'confident', 'caring'],
        'sadness': ['sad', 'devastated', 'lonely', 'disappointed', 'ashamed', 'guilty', 'nostalgic', 'sentimental']}
PER_LABEL = {'anger': 250, 'fear': 250, 'happy': 125, 'sadness': 125}
INV = {e: k for k, v in EMAP.items() for e in v}

rows = []
for sp in ['train', 'valid', 'test']:
    with open(f'{ED_DIR}/{sp}.csv', encoding='utf-8') as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip('\n').split(',')
            if len(p) < 6 or p[1] != '1' or p[2] not in INV:
                continue
            rows.append(dict(conv_id=p[0], ed_split=sp, label_asli=p[2], label=INV[p[2]],
                             prompt=' '.join(p[3].replace('_comma_', ',').split()),
                             text_en=' '.join(p[5].replace('_comma_', ',').split())))
df = pd.DataFrame(rows)
st = {'ujaran_pertama_4kelas': len(df)}
df['nw'] = df.text_en.str.split().str.len()
df = df[(df.nw >= 5) & (df.nw <= 45)]
st['setelah_filter_panjang'] = len(df)
df['norm'] = df.text_en.str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True).str.strip()
df = df.drop_duplicates('norm')
st['setelah_dedup'] = len(df)

parts = []
for lab, labels in EMAP.items():
    for la in labels:
        pool = df[df.label_asli == la]
        parts.append(pool.sample(min(PER_LABEL[lab], len(pool)), random_state=SEED))
out = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
out.insert(0, 'id', [f'ed_{i:05d}' for i in range(len(out))])
out = out[['id', 'label', 'label_asli', 'text_en', 'nw', 'conv_id', 'ed_split', 'prompt']]
out.to_csv('ed_kandidat_en.csv', index=False)
st['kandidat_per_kelas'] = out.label.value_counts().to_dict()
st['median_kata'] = float(out.nw.median())
json.dump(st, open('ed_seleksi_stats.json', 'w'), indent=1)
print(json.dumps(st, indent=1))
