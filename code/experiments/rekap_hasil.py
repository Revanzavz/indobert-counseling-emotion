"""Rekap hasil eksperimen menjadi tabel (CSV/Excel/Markdown) dan grafik untuk Bab 4.

  python rekap_hasil.py
Output di folder hasil/:
  rekap_fase1_backbone.csv       performa tiap backbone (lr terbaik)
  rekap_fase2_arsitektur.csv     head x strategi fine-tuning
  rekap_fase3_loss.csv, rekap_fase4_hyperparameter.csv
  rekap_fase5_tahap2.csv         rata-rata +- std 3 seed per skenario (menjawab RQ1)
  rekap_fase6_ablasi.csv
  rekap_semua.xlsx               semua tabel dalam satu file
  tabel_bab4.md                  tabel siap salin
  grafik_*.png                   grafik perbandingan dan confusion matrix model final
"""
import os, json, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
H = os.environ.get('TA_HASIL_DIR', os.path.join(HERE, 'hasil')); os.makedirs(H, exist_ok=True)
RUNS = os.environ.get('TA_RUNS_DIR', os.path.join(HERE, 'runs'))
df = pd.read_csv(os.path.join(H, 'semua_run.csv'))
df['lr'] = df['lr'].map(lambda v: f'{v:g}' if pd.notna(v) else v)
LABELS = ['anger', 'fear', 'happy', 'sadness', 'neutral']
BLUE, TXT, SEC, BG = '#2a78d6', '#0b0b0b', '#52514e', '#fcfcfb'
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': BG,
                     'axes.facecolor': BG, 'axes.edgecolor': '#c9c8c2', 'xtick.color': SEC, 'ytick.color': SEC})
tables = {}
r4 = lambda d: d.round(4)

# ---- fase 1
f1 = df[df.fase == 'F1']
if len(f1):
    t = f1.sort_values('val_f1_weighted', ascending=False).groupby('backbone').head(1)
    t = t[['backbone', 'lr', 'val_f1_weighted', 'tahap1_test_f1_weighted', 'tahap1_test_f1_macro',
           'tahap2_test_f1_weighted', 'durasi_menit', 'vram_gb']].sort_values('val_f1_weighted', ascending=False)
    tables['fase1_backbone'] = r4(t)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(t.backbone[::-1], t.val_f1_weighted[::-1], color=BLUE, height=0.6)
    for y, v in enumerate(t.val_f1_weighted[::-1]):
        ax.text(v, y, f' {v:.3f}', va='center', color=TXT, fontsize=9)
    ax.set_xlabel('Weighted F1 validasi tahap 1'); ax.set_title('Perbandingan backbone IndoBERT', loc='left', color=TXT)
    ax.set_xlim(max(0, t.val_f1_weighted.min() - 0.1), min(1, t.val_f1_weighted.max() + 0.05))
    fig.tight_layout(); fig.savefig(os.path.join(H, 'grafik_fase1_backbone.png'), dpi=150); plt.close(fig)

# ---- fase 2
f2 = df[df.fase == 'F2']
if len(f2):
    tables['fase2_arsitektur'] = r4(f2[['backbone', 'head', 'strategy', 'lr', 'val_f1_weighted', 'tahap1_test_f1_weighted',
                                        'tahap1_test_f1_macro', 'durasi_menit']].sort_values('val_f1_weighted', ascending=False))
    for b, g in f2.groupby('backbone'):
        pv = g.pivot_table(index='head', columns='strategy', values='val_f1_weighted')
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        im = ax.imshow(pv.values, cmap='Blues', aspect='auto')
        ax.set_xticks(range(pv.shape[1]), pv.columns, rotation=20); ax.set_yticks(range(pv.shape[0]), pv.index)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = pv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=8,
                            color='white' if v > np.nanmean(pv.values) else TXT)
        ax.set_title(f'{b}: weighted F1 validasi (head x strategi)', loc='left', color=TXT, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout()
        fig.savefig(os.path.join(H, f'grafik_fase2_{b}.png'), dpi=150); plt.close(fig)

for f, name, cols in [('F3', 'fase3_loss', ['backbone', 'head', 'strategy', 'loss']),
                      ('F4', 'fase4_hyperparameter', ['lr', 'batch_efektif', 'dropout'])]:
    d = df[df.fase == f]
    if len(d):
        tables[name] = r4(d[cols + ['val_f1_weighted', 'tahap1_test_f1_weighted', 'tahap1_test_f1_macro', 'best_epoch']]
                          .sort_values('val_f1_weighted', ascending=False))

# ---- fase 5 & 6: rata-rata +- std antar seed
def agg(d):
    d = d.copy()
    d['skenario'] = d.run_id.str.replace(r'.*_s4[0-9]_', '', regex=True)
    mets = [c for c in ['tahap2_test_accuracy', 'tahap2_test_f1_weighted', 'tahap2_test_f1_macro', 'tahap2_test_f1_macro_emosi',
                        'tahap2_test_f1_neutral', 'tahap1_test_f1_weighted', 'tahap2_test_ece', 'tahap2_test_ece_kalibrasi']
            if c in d.columns]
    g = d.groupby('skenario')[mets].agg(['mean', 'std', 'count'])
    out = pd.DataFrame(index=g.index)
    for m in mets:
        out[m] = g[(m, 'mean')].map('{:.4f}'.format) + ' ± ' + g[(m, 'std')].fillna(0).map('{:.4f}'.format)
    out['n_seed'] = g[(mets[0], 'count')]
    out['_sort'] = g[('tahap2_test_f1_macro', 'mean')] if 'tahap2_test_f1_macro' in mets else 0
    return out.sort_values('_sort', ascending=False).drop(columns='_sort').reset_index()

for f, name in [('F5', 'fase5_tahap2'), ('F6', 'fase6_ablasi')]:
    d = df[df.fase == f]
    if len(d):
        tables[name] = agg(d)

# ---- confusion matrix model final (dua tahap terbaik, seed 42)
f5 = df[(df.fase == 'F5') & df.run_id.str.contains('duatahap')]
if len(f5):
    best_tag = f5.assign(t=f5.run_id.str.replace(r'.*_s4[0-9]_', '', regex=True)).groupby('t').val_f1_macro.mean().idxmax()
    rid = f5[f5.run_id.str.endswith(best_tag)].sort_values('seed').run_id.iloc[0]
    m = json.load(open(os.path.join(RUNS, rid, 'metrics.json')))
    cm = np.array(m['test']['tahap2_test']['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(5), LABELS, rotation=30); ax.set_yticks(range(5), LABELS)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else TXT)
    ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
    ax.set_title('Confusion matrix model dua tahap (test konseling)', loc='left', color=TXT, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(H, 'grafik_confusion_matrix_final.png'), dpi=150); plt.close(fig)
    json.dump({'run_final': rid, 'skenario': best_tag}, open(os.path.join(H, 'model_final.json'), 'w'), indent=1)

with pd.ExcelWriter(os.path.join(H, 'rekap_semua.xlsx')) as xw:
    for n, t in tables.items():
        t.to_csv(os.path.join(H, f'rekap_{n}.csv'), index=False)
        t.to_excel(xw, sheet_name=n[:31], index=False)
with open(os.path.join(H, 'tabel_bab4.md'), 'w') as fh:
    for n, t in tables.items():
        fh.write(f'## {n}\n\n{t.to_markdown(index=False)}\n\n')
print('Rekap tersimpan di', H, '|', ', '.join(tables))
