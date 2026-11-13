"""Menguji model final pada EmoT (tweet emosi berbahasa Indonesia asli, Saputri et al. 2018).
EmoT tidak punya kelas neutral, jadi dilaporkan dua cara:
  (a) apa adanya: prediksi neutral dihitung salah,
  (b) empat kelas: prediksi diambil dari emosi berprobabilitas tertinggi di antara anger/fear/happy/sadness."""
import os, sys, time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

L4 = ['anger', 'fear', 'happy', 'sadness']; L5 = L4 + ['neutral']
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(H, '..', '..', 'Eksperimen', 'eksperimen')))
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
from prediksi import petakan_emosi

d = pd.read_csv(os.path.join(H, 'emot_uji.csv'))
rows, t0 = [], time.time()
for i, r in enumerate(d.itertuples(), 1):
    lab, pr = petakan_emosi(str(r.text))
    rows.append({'label': r.label, 'text': r.text, 'pred': lab, 'pred4': max(L4, key=lambda k: pr[k]),
                 **{f'p_{k}': v for k, v in pr.items()}})
    if i % 250 == 0: print(f'{i}/{len(d)} ({time.time() - t0:.0f} detik)')
p = pd.DataFrame(rows); p.to_csv(os.path.join(H, 'prediksi_emot.csv'), index=False, encoding='utf-8-sig')

def blok(judul, pred, labs):
    f1 = f1_score(p.label, pred, average=None, labels=L4, zero_division=0)
    cm = confusion_matrix(p.label, pred, labels=labs)
    s = (f'## {judul}\n\n| Metrik | Nilai |\n|---|---|\n| Accuracy | {accuracy_score(p.label, pred):.4f} |\n'
         f'| Macro F1 (4 emosi) | {f1_score(p.label, pred, average="macro", labels=L4, zero_division=0):.4f} |\n'
         f'| Weighted F1 (4 emosi) | {f1_score(p.label, pred, average="weighted", labels=L4, zero_division=0):.4f} |\n'
         + ''.join(f'| F1 {l} (n={int((p.label == l).sum())}) | {v:.4f} |\n' for l, v in zip(L4, f1)))
    s += ('\nConfusion matrix (baris aktual, kolom prediksi)\n\n| | ' + ' | '.join(labs) + ' |\n|---|' + '---|' * len(labs) + '\n'
          + ''.join(f'| {l} | ' + ' | '.join(str(x) for x in cm[i]) + ' |\n' for i, l in enumerate(labs) if l in L4))
    return s

out = [f'# Evaluasi model final pada EmoT ({len(p)} tweet)\n',
       f'Prediksi neutral: {int((p.pred == "neutral").sum())} tweet ({100 * (p.pred == "neutral").mean():.1f}%).\n',
       blok('(a) Lima kelas, prediksi neutral dihitung salah', p.pred, L5),
       blok('(b) Empat kelas emosi', p.pred4, L4)]
open(os.path.join(H, 'hasil_evaluasi_emot.md'), 'w', encoding='utf-8').write('\n'.join(out))
print('\n'.join(out))
