"""Rekap fase 7 (pembanding adil untuk pertanyaan penelitian 1).

Tiap keluarga skenario dipilih variannya dengan cara yang sama: rata-rata macro F1 validasi domain
konseling tertinggi. Data uji tidak dipakai untuk memilih. Setelah itu skenario terpilih dibandingkan
dengan dua tahap terpilih memakai bootstrap (selang kepercayaan 95% selisih macro F1) dan uji McNemar.
Keluaran: hasil/rekap_fase7_pembanding_adil.md dan .json
"""
import glob, json, os, re
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS, H = os.path.join(HERE, 'runs'), os.path.join(HERE, 'hasil')


def muat(pola):
    rows = []
    for p in glob.glob(os.path.join(RUNS, pola, 'metrics.json')):
        m = json.load(open(p)); d = os.path.dirname(p); rid = os.path.basename(d)
        rows.append({'dir': d, 'seed': m['config']['seed'], 'tag': re.sub(r'.*_s4[0-9]_', '', rid),
                     'val': m['val']['f1_macro'], 't2': m['test']['tahap2_test']['f1_macro'],
                     't2_acc': m['test']['tahap2_test']['accuracy'], 't1': m['test']['tahap1_test']['f1_weighted']})
    return pd.DataFrame(rows)


def terbaik(df, awalan):
    d = df[df.tag.str.startswith(awalan)]
    g = d.groupby('tag').val.agg(['mean', 'count'])
    g = g[g['count'] == g['count'].max()]
    tag = g['mean'].idxmax()
    return tag, d[d.tag == tag].sort_values('seed'), d.groupby('tag')[['val', 't2', 't1']].mean().round(4)


def pred(dirs):
    return [pd.read_csv(os.path.join(d, 'prediksi', 'tahap2_test.csv')) for d in dirs]


def banding(a, b, n_boot=2000):
    """a, b: daftar DataFrame prediksi per seed (urutan seed sama)."""
    y = a[0].label.values; rng = np.random.default_rng(0); idx = rng.integers(0, len(y), (n_boot, len(y)))
    A = [x.pred.values for x in a]; B = [x.pred.values for x in b]; k = min(len(A), len(B))
    bs = np.array([np.mean([f1_score(y[r], A[i][r], average='macro') - f1_score(y[r], B[i][r], average='macro')
                            for i in range(k)]) for r in idx])
    h_a = sum(int(((A[i] == y) & (B[i] != y)).sum()) for i in range(k))
    h_b = sum(int(((A[i] != y) & (B[i] == y)).sum()) for i in range(k))
    return {'selisih': float(np.mean([f1_score(y, A[i], average='macro') - f1_score(y, B[i], average='macro') for i in range(k)])),
            'ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            'mcnemar': {'benar_hanya_a': h_a, 'benar_hanya_b': h_b, 'p': float(binomtest(h_a, h_a + h_b).pvalue) if h_a + h_b else 1.0}}


f5, f7 = muat('F5_*'), muat('F7_*')
dua_tag = json.load(open(os.path.join(H, 'model_final.json')))['skenario']
dua = f5[f5.tag == dua_tag].sort_values('seed')
out = {'dua_tahap': {'tag': dua_tag, 'n_seed': len(dua)}}
baris = [('Dua tahap (terpilih dari 8 varian)', dua_tag, dua)]
tabel_varian = {}
for nama, awalan in [('Tahap 2 saja (terpilih dari 4 varian)', 'tahap2saja_'), ('Satu tahap gabungan (terpilih dari 4 varian)', 'satutahap_')]:
    if not f7.tag.str.startswith(awalan).any():
        continue
    tag, d, semua = terbaik(f7, awalan)
    baris.append((nama, tag, d)); tabel_varian[awalan] = semua
    out[awalan.rstrip('_')] = {'tag': tag, 'n_seed': len(d), 'vs_dua_tahap': banding(pred(dua.dir), pred(d.dir))}
cls = f7[f7.tag == 'duatahap_cls'].sort_values('seed')
if len(cls):
    baris.append(('Dua tahap, head [CLS] + full (arsitektur proposal)', 'duatahap_cls', cls))
    out['duatahap_cls'] = {'n_seed': len(cls), 'vs_dua_tahap': banding(pred(dua.dir), pred(cls.dir))}
lama = {'Tahap 2 saja (konfigurasi tunggal, fase 5)': 'tahap2_saja', 'Satu tahap gabungan (konfigurasi tunggal, fase 5)': 'satu_tahap'}
ms = lambda s: f'{s.mean():.4f} ± {s.std(ddof=1):.4f}' if len(s) > 1 else f'{s.mean():.4f}'
with open(os.path.join(H, 'rekap_fase7_pembanding_adil.md'), 'w', encoding='utf-8') as f:
    f.write('## Perbandingan skenario dengan ruang pencarian yang setara\n\n')
    f.write('| Skenario | Varian terpilih | Seed | Macro F1 val konseling | Macro F1 uji konseling | Accuracy uji konseling | Weighted F1 uji tweet |\n|---|---|---|---|---|---|---|\n')
    for nama, tag, d in baris:
        f.write(f'| {nama} | {tag} | {len(d)} | {ms(d.val)} | {ms(d.t2)} | {ms(d.t2_acc)} | {ms(d.t1)} |\n')
    for nama, tag in lama.items():
        d = f5[f5.tag == tag]
        if len(d):
            f.write(f'| {nama} | {tag} | {len(d)} | {"-" if tag == "satu_tahap" else ms(d.val)} | {ms(d.t2)} | {ms(d.t2_acc)} | {ms(d.t1)} |\n')
    f.write('\n## Uji signifikansi terhadap dua tahap terpilih\n\n| Pembanding | Selisih macro F1 (dua tahap - pembanding) | CI 95% | McNemar (benar hanya dua tahap / hanya pembanding) | p |\n|---|---|---|---|---|\n')
    for k, nama in [('tahap2saja', 'Tahap 2 saja'), ('satutahap', 'Satu tahap gabungan'), ('duatahap_cls', 'Dua tahap head [CLS]')]:
        if k in out:
            v = out[k]['vs_dua_tahap']
            f.write(f"| {nama} | {v['selisih']:+.4f} | [{v['ci95'][0]:+.4f}; {v['ci95'][1]:+.4f}] | {v['mcnemar']['benar_hanya_a']} / {v['mcnemar']['benar_hanya_b']} | {v['mcnemar']['p']:.4f} |\n")
    for awalan, t in tabel_varian.items():
        f.write(f'\n## Semua varian {awalan.rstrip("_")} (rata-rata antar seed)\n\n{t.rename(columns={"val": "macro F1 val", "t2": "macro F1 uji konseling", "t1": "weighted F1 uji tweet"}).to_markdown()}\n')
json.dump(out, open(os.path.join(H, 'rekap_fase7_pembanding_adil.json'), 'w'), indent=1)
print(open(os.path.join(H, 'rekap_fase7_pembanding_adil.md'), encoding='utf-8').read())
