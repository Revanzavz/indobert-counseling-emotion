"""Menyimpan model final untuk chatbot dan membersihkan checkpoint yang tidak terpilih.

Model final = skenario dua tahap dengan rata-rata macro F1 validasi tertinggi (fase 5), seed dengan
macro F1 validasi tertinggi di skenario itu. Hasilnya disalin ke hasil/model_final/ berisi model.pt,
tokenizer/, config.json, dan info.json (label, temperatur kalibrasi, metrik). Checkpoint dua tahap
lain dihapus untuk menghemat disk (metrics.json dan prediksinya tetap disimpan).
"""
import os, json, glob, shutil
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs'); H = os.path.join(HERE, 'hasil')
rows = []
for p in glob.glob(os.path.join(RUNS, 'F5_*duatahap*', 'metrics.json')):
    m = json.load(open(p)); rid = m['config']['run_id']
    rows.append({'dir': os.path.dirname(p), 'skenario': rid.split('_s4')[-1][2:], 'val': m['val']['f1_macro'], 'm': m})
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit('Belum ada run dua tahap.')
best_s = df.groupby('skenario').val.mean().idxmax()
best = df[df.skenario == best_s].sort_values('val', ascending=False).iloc[0]
out = os.path.join(H, 'model_final'); os.makedirs(out, exist_ok=True)
shutil.copy2(os.path.join(best.dir, 'model.pt'), os.path.join(out, 'model.pt'))
if os.path.exists(os.path.join(out, 'tokenizer')):
    shutil.rmtree(os.path.join(out, 'tokenizer'))
shutil.copytree(os.path.join(best.dir, 'tokenizer'), os.path.join(out, 'tokenizer'))
m = best.m
json.dump(m['config'], open(os.path.join(out, 'config.json'), 'w'), indent=1)
json.dump({'run': os.path.basename(best.dir), 'skenario': best_s, 'labels': ['anger', 'fear', 'happy', 'sadness', 'neutral'],
           'temperature': m['temperature'], 'backbone': m['config']['backbone'], 'head': m['config']['head'],
           'max_len': m['config']['max_len'], 'dropout': m['config']['dropout'],
           'val_f1_macro': m['val']['f1_macro'],
           'test': {k: {'f1_weighted': v['f1_weighted'], 'f1_macro': v['f1_macro']} for k, v in m['test'].items()}},
          open(os.path.join(out, 'info.json'), 'w'), indent=1)
# hapus checkpoint dua tahap yang tidak terpilih (kecuali 3 seed skenario terbaik, untuk ensemble)
keep = set(df[df.skenario == best_s].dir)
freed = 0
for d in df.dir:
    f = os.path.join(d, 'model.pt')
    if d not in keep and os.path.exists(f):
        freed += os.path.getsize(f); os.remove(f)
print(f'Model final: {os.path.basename(best.dir)} -> {out}  | disk dibebaskan {freed/1e9:.1f} GB')
