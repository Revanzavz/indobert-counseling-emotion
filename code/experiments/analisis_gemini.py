"""Analisis lanjutan perbandingan Gemini vs IndoBERTweet pada data uji konseling:
metrik per kelas, confusion matrix, dan paired t-test pada 10 lipatan stratified.
Pemakaian: python analisis_gemini.py <banding_gemini_konseling_lengkap.csv> <folder_keluaran>"""
import sys, os, json
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats

L = ['anger', 'fear', 'happy', 'sadness', 'neutral']
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
d = pd.read_csv(src)
y, g, b = d.label.values, d.pred_gemini.values, d.pred_indobert.values
res = {'n': len(d)}
rows = []
for nama, p in [('Gemini', g), ('IndoBERTweet', b)]:
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=L, zero_division=0)
    for i, l in enumerate(L):
        rows.append({'model': nama, 'kelas': l, 'precision': pr[i], 'recall': rc[i], 'f1': f1[i], 'support': int(sup[i])})
    res[nama] = {'accuracy': accuracy_score(y, p), 'macro_f1': f1_score(y, p, labels=L, average='macro', zero_division=0),
                 'cm': confusion_matrix(y, p, labels=L).tolist()}
per = pd.DataFrame(rows); per.to_csv(os.path.join(out, 'gemini_per_kelas.csv'), index=False)

# 10 lipatan stratified, macro F1 per lipatan
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fg, fb = [], []
fk = {l: ([], []) for l in L}
for _, idx in skf.split(d, y):
    fg.append(f1_score(y[idx], g[idx], labels=L, average='macro', zero_division=0))
    fb.append(f1_score(y[idx], b[idx], labels=L, average='macro', zero_division=0))
    a1 = f1_score(y[idx], g[idx], labels=L, average=None, zero_division=0); a2 = f1_score(y[idx], b[idx], labels=L, average=None, zero_division=0)
    for i, l in enumerate(L): fk[l][0].append(a1[i]); fk[l][1].append(a2[i])
fg, fb = np.array(fg), np.array(fb)
t, p = stats.ttest_rel(fg, fb)
sh = stats.shapiro(fg - fb)
res['ttest_macro'] = {'gemini_mean': fg.mean(), 'gemini_sd': fg.std(ddof=1), 'indobert_mean': fb.mean(), 'indobert_sd': fb.std(ddof=1),
                      'selisih_mean': (fg - fb).mean(), 't': t, 'df': len(fg) - 1, 'p': p, 'shapiro_p': sh.pvalue,
                      'wilcoxon_p': stats.wilcoxon(fg, fb).pvalue, 'lipatan_gemini': fg.tolist(), 'lipatan_indobert': fb.tolist()}
res['ttest_kelas'] = {}
for l in L:
    a, c = np.array(fk[l][0]), np.array(fk[l][1]); tt = stats.ttest_rel(a, c)
    res['ttest_kelas'][l] = {'gemini_mean': a.mean(), 'indobert_mean': c.mean(), 't': tt.statistic, 'p': tt.pvalue}
json.dump(res, open(os.path.join(out, 'gemini_ttest.json'), 'w'), indent=1, default=float)

f = lambda x: f'{x:.3f}'.replace('.', ',')
md = [f"# Analisis Gemini vs IndoBERTweet ({len(d)} kalimat uji konseling)\n", '## Metrik per kelas\n',
      '| Kelas | P Gemini | R Gemini | F1 Gemini | P IndoBERTweet | R IndoBERTweet | F1 IndoBERTweet | Lebih baik |', '|---|---|---|---|---|---|---|---|']
for l in L:
    a = per[(per.model == 'Gemini') & (per.kelas == l)].iloc[0]; c = per[(per.model == 'IndoBERTweet') & (per.kelas == l)].iloc[0]
    lb = 'Gemini' if a.f1 > c.f1 + 0.005 else ('IndoBERTweet' if c.f1 > a.f1 + 0.005 else 'setara')
    md.append(f'| {l} | {f(a.precision)} | {f(a.recall)} | {f(a.f1)} | {f(c.precision)} | {f(c.recall)} | {f(c.f1)} | {lb} |')
r = res['ttest_macro']
md += ['', '## Paired t-test macro F1 (10 lipatan stratified)\n',
       f"Gemini {f(r['gemini_mean'])} ± {f(r['gemini_sd'])}, IndoBERTweet {f(r['indobert_mean'])} ± {f(r['indobert_sd'])}, "
       f"t({r['df']}) = {r['t']:.3f}, p = {r['p']:.4f}. Uji normalitas selisih (Shapiro-Wilk) p = {r['shapiro_p']:.3f}; Wilcoxon p = {r['wilcoxon_p']:.4f}.\n",
       '## Paired t-test F1 per kelas (10 lipatan)\n', '| Kelas | F1 Gemini | F1 IndoBERTweet | t | p |', '|---|---|---|---|---|']
for l in L:
    k = res['ttest_kelas'][l]; md.append(f"| {l} | {f(k['gemini_mean'])} | {f(k['indobert_mean'])} | {k['t']:.3f} | {k['p']:.4f} |")
for nama in ['Gemini', 'IndoBERTweet']:
    md += ['', f'## Confusion matrix {nama} (baris aktual, kolom prediksi)\n', '| | ' + ' | '.join(L) + ' |', '|---|' + '---|' * 5]
    md += [f'| {l} | ' + ' | '.join(str(x) for x in res[nama]['cm'][i]) + ' |' for i, l in enumerate(L)]
open(os.path.join(out, 'gemini_analisis.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
print('\n'.join(md))
