"""Ensemble antar-seed untuk skenario dua tahap terbaik (dipilih dari macro F1 validasi).
Probabilitas (setelah kalibrasi temperatur) dari 3 seed dirata-rata, lalu dievaluasi di data uji.
Tidak ada pemilihan yang memakai data uji. Output: hasil/ensemble.json dan hasil/ensemble_prediksi_*.csv
"""
import os, json, glob
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get('TA_RUNS_DIR', os.path.join(HERE, 'runs'))
H = os.environ.get('TA_HASIL_DIR', os.path.join(HERE, 'hasil'))
LABELS = ['anger', 'fear', 'happy', 'sadness', 'neutral']
rows = []
for p in glob.glob(os.path.join(RUNS, 'F5_*duatahap*', 'metrics.json')):
    m = json.load(open(p)); rid = m['config']['run_id']
    rows.append({'rid': rid, 'dir': os.path.dirname(p), 'skenario': rid.split('_s4')[-1][2:], 'val': m['val']['f1_macro']})
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit('Belum ada run dua tahap (fase 5).')
best = df.groupby('skenario').val.mean().idxmax()
sel = df[df.skenario == best]
out = {'skenario': best, 'run': sel.rid.tolist()}
for test in ['tahap2_test', 'tahap1_test']:
    preds = [pd.read_csv(os.path.join(d, 'prediksi', f'{test}.csv')) for d in sel.dir]
    P = np.mean([p[[f'p_{l}' for l in LABELS]].values for p in preds], axis=0)
    y = preds[0].label.values; yp = np.array(LABELS)[P.argmax(1)]
    tunggal = [f1_score(p.label, p.pred, average='macro') for p in preds]
    out[test] = {'accuracy': accuracy_score(y, yp), 'f1_weighted': f1_score(y, yp, average='weighted'),
                 'f1_macro': f1_score(y, yp, average='macro'),
                 'f1_macro_model_tunggal_rata2': float(np.mean(tunggal)),
                 'per_kelas': classification_report(y, yp, labels=LABELS, output_dict=True, zero_division=0),
                 'confusion_matrix': confusion_matrix(y, yp, labels=LABELS).tolist()}
    pdf = preds[0][['id', 'text', 'label']].copy(); pdf['pred'] = yp
    for i, l in enumerate(LABELS):
        pdf[f'p_{l}'] = P[:, i].round(4)
    pdf.to_csv(os.path.join(H, f'ensemble_prediksi_{test}.csv'), index=False)
json.dump(out, open(os.path.join(H, 'ensemble.json'), 'w'), indent=1, default=float)
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)}) for k, v in out.items()}, indent=1))

# lanjut: simpan model final ke hasil/model_final dan bersihkan checkpoint yang tidak terpilih
import subprocess, sys
subprocess.run([sys.executable, os.path.join(HERE, 'finalisasi_model.py')], cwd=HERE)
