"""Menyatukan label dua penilai, menghitung kesepakatan, lalu menguji model final pada data uji asli."""
import os, sys, json
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, confusion_matrix

LABELS = ['anger', 'fear', 'happy', 'sadness', 'neutral']
HERE = os.path.dirname(os.path.abspath(__file__))
EKS = os.path.join(HERE, '..', '..', 'Eksperimen', 'eksperimen')
CHAT = os.path.join(HERE, '..', '..', 'Chatbot')


def baca(nama):
    d = pd.read_csv(os.path.join(HERE, f'lembar_penilai_{nama}.csv'))
    d['label'] = d.label.astype(str).str.strip().str.lower()
    salah = d[~d.label.isin(LABELS)]
    if len(salah):
        sys.exit(f'Lembar {nama}: {len(salah)} baris belum diisi atau labelnya tidak dikenal, contoh id {salah.id.head(5).tolist()}')
    return d[['id', 'teks', 'label']]


a, b = baca('A'), baca('B')
g = a.merge(b[['id', 'label']], on='id', suffixes=('_A', '_B'))
kappa = cohen_kappa_score(g.label_A, g.label_B, labels=LABELS)
sepakat = (g.label_A == g.label_B).mean()
p = os.path.join(HERE, 'label_gabungan.csv')
lama = pd.read_csv(p)[['id', 'label_akhir']] if os.path.exists(p) else pd.DataFrame(columns=['id', 'label_akhir'])
g = g.merge(lama, on='id', how='left')
g['label_akhir'] = g.label_akhir.astype(str).str.strip().str.lower().where(g.label_akhir.notna(), '')
g.loc[g.label_A == g.label_B, 'label_akhir'] = g.label_A
g.loc[~g.label_akhir.isin(LABELS), 'label_akhir'] = ''
g.to_csv(p, index=False, encoding='utf-8-sig')
siap = g[g.label_akhir != '']
tulis = [f'## Kesepakatan dua penilai\n\nJumlah kalimat {len(g)}. Sepakat {sepakat:.1%} ({int((g.label_A == g.label_B).sum())} kalimat). '
         f"Cohen's kappa {kappa:.3f}. Belum punya label akhir {len(g) - len(siap)} kalimat.\n"]
print(tulis[0])
if len(g) - len(siap):
    print('Isi kolom label_akhir di label_gabungan.csv untuk kalimat yang belum disepakati, lalu jalankan lagi.')

sys.path.insert(0, os.path.abspath(EKS)); sys.path.insert(0, os.path.abspath(CHAT))
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
from prediksi import petakan_emosi
import krisis
rows = []
for r in siap.itertuples():
    lab, pr = petakan_emosi(str(r.teks))
    rows.append({'id': r.id, 'teks': r.teks, 'label': r.label_akhir, 'pred': lab, **{f'p_{k}': v for k, v in pr.items()},
                 'kata_kunci_krisis': krisis.cek_kata_kunci(str(r.teks)) or ''})
d = pd.DataFrame(rows); d.to_csv(os.path.join(HERE, 'prediksi_data_asli.csv'), index=False, encoding='utf-8-sig')
f1k = f1_score(d.label, d.pred, average=None, labels=LABELS, zero_division=0)
cm = confusion_matrix(d.label, d.pred, labels=LABELS)
tulis.append(f'## Performa model final pada data uji asli ({len(d)} kalimat)\n\n| Metrik | Nilai |\n|---|---|\n'
             f'| Accuracy | {accuracy_score(d.label, d.pred):.4f} |\n| Macro F1 | {f1_score(d.label, d.pred, average="macro", labels=LABELS, zero_division=0):.4f} |\n'
             f'| Weighted F1 | {f1_score(d.label, d.pred, average="weighted", labels=LABELS, zero_division=0):.4f} |\n'
             + ''.join(f'| F1 {l} (n={int((d.label == l).sum())}) | {v:.4f} |\n' for l, v in zip(LABELS, f1k)))
tulis.append('## Confusion matrix (baris aktual, kolom prediksi)\n\n| | ' + ' | '.join(LABELS) + ' |\n|---|' + '---|' * 5 + '\n'
             + ''.join(f'| {l} | ' + ' | '.join(str(x) for x in cm[i]) + ' |\n' for i, l in enumerate(LABELS)))
tulis.append(f"Kalimat yang memicu kata kunci krisis: {int((d.kata_kunci_krisis != '').sum())}.\n")
open(os.path.join(HERE, 'hasil_evaluasi_data_asli.md'), 'w', encoding='utf-8').write('\n'.join(tulis))
print('\n'.join(tulis[1:]))
