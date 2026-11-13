"""Melatih model deteksi krisis (biner: krisis / bukan) dengan IndoBERTweet.

Data: Dataset/Krisis_Kaggle/krisis_{train,val,test}.csv, hasil terjemahan kelas Suicidal dan kelas lain dari
dataset Kaggle "Sentiment Analysis for Mental Health", ditambah kalimat emosi biasa sebagai contoh negatif.

Langkah: fine-tuning penuh dengan lapisan [CLS], early stopping berdasarkan loss validasi, kalibrasi temperatur,
lalu pemilihan dua ambang pada data validasi:
  AMBANG_KRISIS : F1 tertinggi dengan syarat recall krisis minimal 0,90 -> pesan dialihkan ke rujukan
  AMBANG_HATI   : ambang lebih rendah dengan recall minimal 0,97 -> Gemini diberi penanda kehati-hatian
Evaluasi: data uji Kaggle (terjemahan) dan 50 pesan uji krisis berbahasa Indonesia asli (Chatbot/uji_krisis.csv).
Keluaran: hasil/model_krisis/ (model.pt, info.json) dan hasil/evaluasi_krisis.md

Varian MASKING=1 (penyamaran kata pemicu): selama pelatihan, pada kalimat yang memuat kata kematian atau kehidupan
(mati, bunuh, hidup, nyawa, dan bentuk turunannya) seluruh kata itu diganti token [MASK] dengan peluang P_MASK per
kalimat, diacak ulang setiap epoch, di kedua kelas. Tujuannya mengurangi shortcut learning, yaitu model menganggap
kata "mati" atau "bunuh" pasti berarti krisis tanpa melihat konteks. Data validasi dan uji tidak disamarkan.
Keluaran varian ini disimpan terpisah: hasil/model_krisis_masking/ dan hasil/evaluasi_krisis_masking.md"""
import os, json, random, re, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, confusion_matrix
from transformers import get_linear_schedule_with_warmup
from modeling import EmotionClassifier, load_tokenizer, BACKBONES

HERE = os.path.dirname(os.path.abspath(__file__))
NAMA_DATA = os.environ.get('DATA_KRISIS', 'Krisis_Kaggle')
DATA = os.path.join(HERE, '..', '..', 'Dataset', NAMA_DATA)
DATA_ID = NAMA_DATA != 'Krisis_Kaggle'
UJI_ASLI = os.path.join(HERE, '..', '..', 'Chatbot', 'uji_krisis.csv')
MASKING = os.environ.get('MASKING', '0') == '1'
P_MASK = float(os.environ.get('P_MASK', '0.3'))
SUF = os.environ.get('SUFIKS') or ('_masking' if MASKING else '')
PEMICU = re.compile(r'\b(?:ke|me|ter|di|ber)?(?:mati|matikan|kematian|meninggal|bunuh|membunuh|hidup|kehidupan|nyawa)(?:ku|mu|nya|lah|kan)?\b', re.I)
OUT = os.path.join(HERE, 'hasil', 'model_krisis' + SUF); os.makedirs(OUT, exist_ok=True)
BACKBONE, HEAD, LR, BS, EPOCH, MAXLEN, DROPOUT, SEED = 'indobertweet-base', 'cls', 2e-5, 16, 4, 128, 0.1, 42
LAB = ['bukan', 'krisis']

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); amp = dev.type == 'cuda'
tok = load_tokenizer(BACKBONES[BACKBONE])
rd = lambda n: pd.read_csv(os.path.join(DATA, f'krisis_{n}.csv')).dropna(subset=['text']).reset_index(drop=True)
tr, va, te = rd('train'), rd('val'), rd('test')
print(f'latih {len(tr)}, validasi {len(va)}, uji {len(te)} | {dev}', flush=True)
if MASKING:
    ada = tr.text.astype(str).str.contains(PEMICU)
    print(f'penyamaran kata pemicu aktif (peluang {P_MASK}); kalimat latih yang memuat kata pemicu: '
          f'krisis {int((ada & (tr.label == "krisis")).sum())}, bukan {int((ada & (tr.label == "bukan")).sum())}', flush=True)


def samarkan(teks):
    """Mengganti semua kata pemicu dalam satu kalimat dengan [MASK], dengan peluang P_MASK per kalimat."""
    if PEMICU.search(teks) and random.random() < P_MASK:
        return PEMICU.sub(tok.mask_token, teks)
    return teks


def batches(df, shuffle, samar=False):
    idx = np.random.permutation(len(df)) if shuffle else np.arange(len(df))
    for k in range(0, len(idx), BS):
        b = df.iloc[idx[k:k + BS]]
        teks = [samarkan(t) for t in b.text.astype(str)] if samar else b.text.astype(str).tolist()
        enc = tok(teks, truncation=True, max_length=MAXLEN, padding=True, return_tensors='pt')
        enc = {x: v.to(dev) for x, v in enc.items() if x in ('input_ids', 'attention_mask', 'token_type_ids')}
        y = torch.tensor([LAB.index(l) for l in b.label]) if 'label' in b else None
        yield enc, y


@torch.no_grad()
def logits_of(model, df):
    model.eval(); out = []
    for enc, _ in batches(df, False):
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            out.append(model(**enc).float().cpu())
    return torch.cat(out)


model = EmotionClassifier(BACKBONES[BACKBONE], HEAD, DROPOUT, n_labels=2).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
langkah = EPOCH * ((len(tr) + BS - 1) // BS)
sch = get_linear_schedule_with_warmup(opt, int(0.1 * langkah), langkah)
scaler = torch.cuda.amp.GradScaler(enabled=amp); ce = nn.CrossEntropyLoss()
y_va = torch.tensor([LAB.index(l) for l in va.label])
terbaik, sabar, t0 = 1e9, 0, time.time()
for ep in range(1, EPOCH + 1):
    model.train()
    for i, (enc, y) in enumerate(batches(tr, True, samar=MASKING)):
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            loss = ce(model(**enc).float(), y.to(dev))
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
        if i % 100 == 0:
            print(f'epoch {ep} batch {i} loss {loss.item():.4f} ({time.time() - t0:.0f} dtk)', flush=True)
    vl = ce(logits_of(model, va), y_va).item()
    print(f'epoch {ep}: loss validasi {vl:.4f}', flush=True)
    if vl < terbaik:
        terbaik, sabar = vl, 0; torch.save(model.state_dict(), os.path.join(OUT, 'model.pt')); ep_terbaik = ep
    else:
        sabar += 1
        if sabar >= 1:
            break
model.load_state_dict(torch.load(os.path.join(OUT, 'model.pt'), map_location=dev, weights_only=True))

# kalibrasi temperatur
lv = logits_of(model, va)
T = nn.Parameter(torch.ones(1)); o = torch.optim.LBFGS([T], lr=0.05, max_iter=200)
def cl():
    o.zero_grad(); l = ce(lv / T.clamp(min=0.05), y_va); l.backward(); return l
o.step(cl); T = float(T.clamp(min=0.05).item())
pv = torch.softmax(lv / T, -1)[:, 1].numpy(); yv = y_va.numpy()


def pada(p, y, t):
    pr = (p >= t).astype(int)
    P, R, F, _ = precision_recall_fscore_support(y, pr, labels=[1], zero_division=0)
    return P[0], R[0], F[0]


kandidat = np.round(np.arange(0.05, 0.96, 0.01), 2)
ok = [(pada(pv, yv, t)[2], t) for t in kandidat if pada(pv, yv, t)[1] >= 0.90]
AMBANG_KRISIS = float(max(ok)[1] if ok else max(kandidat, key=lambda t: pada(pv, yv, t)[1]))
hati = [t for t in kandidat if t <= AMBANG_KRISIS and pada(pv, yv, t)[1] >= 0.97]
AMBANG_HATI = float(max(hati) if hati else min(kandidat))


def laporan(df, nama, p):
    y = np.array([LAB.index(l) for l in df.label])
    hasil = {}
    for t, n in [(AMBANG_KRISIS, 'ambang_krisis'), (AMBANG_HATI, 'ambang_hati'), (0.5, 'ambang_0.5')]:
        pr = (p >= t).astype(int); P, R, F, _ = precision_recall_fscore_support(y, pr, labels=[1], zero_division=0)
        hasil[n] = {'ambang': t, 'accuracy': accuracy_score(y, pr), 'precision': P[0], 'recall': R[0], 'f1': F[0],
                    'cm': confusion_matrix(y, pr, labels=[0, 1]).tolist()}
    if len(set(y)) > 1:
        hasil['auc'] = roc_auc_score(y, p)
    return hasil


pt = torch.softmax(logits_of(model, te) / T, -1)[:, 1].numpy()
ev = {'test_kaggle': laporan(te, 'uji', pt)}
te.assign(p_krisis=pt).to_csv(os.path.join(HERE, 'hasil', 'prediksi_krisis_test' + SUF + '.csv'), index=False)
# per status pada data uji
ev['test_per_status'] = {s: {'n': int((te.status == s).sum()), 'terdeteksi_krisis': int((pt[te.status.values == s] >= AMBANG_KRISIS).sum())}
                         for s in sorted(te.status.unique())}
if os.path.exists(UJI_ASLI):
    ua = pd.read_csv(UJI_ASLI).rename(columns={'pesan': 'text'})
    ua['label'] = ua.harapan.map(lambda h: 'krisis' if h == 'krisis' else 'bukan')
    pu = torch.softmax(logits_of(model, ua) / T, -1)[:, 1].numpy()
    ev['uji_indonesia_asli'] = laporan(ua, 'asli', pu)
    ua.assign(p_krisis=pu.round(4), deteksi=np.where(pu >= AMBANG_KRISIS, 'krisis', np.where(pu >= AMBANG_HATI, 'hati_hati', 'bukan'))
              ).to_csv(os.path.join(HERE, 'hasil', 'prediksi_krisis_uji_asli' + SUF + '.csv'), index=False)

info = {'data': NAMA_DATA, 'masking': MASKING, 'p_mask': P_MASK if MASKING else None, 'backbone': BACKBONE, 'head': HEAD, 'max_len': MAXLEN, 'dropout': DROPOUT, 'labels': LAB, 'temperature': T,
        'ambang_krisis': AMBANG_KRISIS, 'ambang_hati': AMBANG_HATI, 'epoch_terbaik': ep_terbaik, 'lr': LR, 'batch': BS,
        'n_data': {'train': len(tr), 'val': len(va), 'test': len(te)}, 'evaluasi': ev,
        'durasi_menit': round((time.time() - t0) / 60, 1)}
json.dump(info, open(os.path.join(OUT, 'info.json'), 'w'), indent=1, default=float)

f = lambda x: f'{x:.3f}'.replace('.', ',')
md = [f'# Evaluasi model deteksi krisis{" (dengan penyamaran kata pemicu, peluang " + str(P_MASK) + ")" if MASKING else ""}\n', f'Ambang krisis {AMBANG_KRISIS}, ambang hati-hati {AMBANG_HATI}, temperatur {T:.3f}, epoch terbaik {ep_terbaik}.\n']
for k, judul in [('test_kaggle', f'Data uji {"tweet Indonesia asli (apricitea)" if DATA_ID else "Kaggle terjemahan"} ({len(te)} kalimat)'), ('uji_indonesia_asli', '50 pesan uji berbahasa Indonesia asli')]:
    if k in ev:
        md += [f'## {judul}\n', '| Ambang | Accuracy | Precision krisis | Recall krisis | F1 krisis | CM [[TN,FP],[FN,TP]] |', '|---|---|---|---|---|---|']
        for n in ['ambang_krisis', 'ambang_hati', 'ambang_0.5']:
            h = ev[k][n]; md.append(f"| {h['ambang']} | {f(h['accuracy'])} | {f(h['precision'])} | {f(h['recall'])} | {f(h['f1'])} | {h['cm']} |")
        if 'auc' in ev[k]:
            md.append(f"\nAUC {f(ev[k]['auc'])}\n")
md += ['## Data uji per status (ambang krisis)\n', '| Status | Jumlah | Terdeteksi krisis |', '|---|---|---|']
md += [f"| {s} | {v['n']} | {v['terdeteksi_krisis']} |" for s, v in ev['test_per_status'].items()]
# perbandingan dengan model tanpa penyamaran
dasar_p = os.path.join(HERE, 'hasil', 'model_krisis', 'info.json')
if MASKING and os.path.exists(dasar_p):
    dasar = json.load(open(dasar_p))['evaluasi']
    md += ['\n## Perbandingan dengan model tanpa penyamaran (ambang krisis masing-masing)\n',
           '| Data | Model | Precision | Recall | F1 | Bukan krisis yang salah dianggap krisis |', '|---|---|---|---|---|---|']
    for k, judul in [('test_kaggle', 'Uji Kaggle'), ('uji_indonesia_asli', '50 pesan Indonesia asli')]:
        for nama, e in [('tanpa penyamaran', dasar.get(k)), ('dengan penyamaran', ev.get(k))]:
            if e:
                h = e['ambang_krisis']
                md.append(f"| {judul} | {nama} | {f(h['precision'])} | {f(h['recall'])} | {f(h['f1'])} | {h['cm'][0][1]} dari {sum(h['cm'][0])} |")
    asli_p = os.path.join(HERE, 'hasil', 'prediksi_krisis_uji_asli' + SUF + '.csv')
    if os.path.exists(asli_p):
        ua2 = pd.read_csv(asli_p)
        md += ['\n## 50 pesan Indonesia asli: prediksi model dengan penyamaran\n', '| Pesan | Harapan | Deteksi | p krisis |', '|---|---|---|---|']
        md += [f"| {x.text} | {x.label} | {x.deteksi} | {x.p_krisis} |" for x in ua2.itertuples()]
# perbandingan silang dengan model lama (data Kaggle terjemahan), masing-masing dengan temperatur dan ambangnya sendiri
if DATA_ID and os.path.exists(dasar_p) and os.path.exists(os.path.join(HERE, 'hasil', 'model_krisis', 'model.pt')):
    il = json.load(open(dasar_p))
    lama = EmotionClassifier(BACKBONES[il['backbone']], il['head'], il['dropout'], n_labels=2).to(dev)
    lama.load_state_dict(torch.load(os.path.join(HERE, 'hasil', 'model_krisis', 'model.pt'), map_location=dev, weights_only=True))
    kg = pd.read_csv(os.path.join(HERE, '..', '..', 'Dataset', 'Krisis_Kaggle', 'krisis_test.csv')).dropna(subset=['text']).reset_index(drop=True)
    p_lama = lambda df: torch.softmax(logits_of(lama, df) / il['temperature'], -1)[:, 1].numpy()
    p_baru = lambda df: torch.softmax(logits_of(model, df) / T, -1)[:, 1].numpy()
    set_uji = [('Uji tweet Indonesia asli', te), ('Uji Kaggle terjemahan', kg)]
    if os.path.exists(UJI_ASLI):
        set_uji.append(('50 pesan uji chatbot', ua))
    def ringkas(df, p, ak, ah):
        y = np.array([LAB.index(l) for l in df.label]); pr = (p >= ak).astype(int)
        P, R, F, _ = precision_recall_fscore_support(y, pr, labels=[1], zero_division=0); cm = confusion_matrix(y, pr, labels=[0, 1])
        return {'precision': P[0], 'recall': R[0], 'f1': F[0], 'auc': roc_auc_score(y, p), 'fp': int(cm[0][1]), 'n_bukan': int(cm[0].sum()),
                'fn': int(cm[1][0]), 'n_krisis': int(cm[1].sum()), 'tertangani': int(((p >= ah) & (y == 1)).sum())}
    silang = {}
    md += ['\n## Perbandingan model lama (Kaggle terjemahan) dan model baru (tweet Indonesia asli)\n',
           f"Model lama: ambang krisis {il['ambang_krisis']}, hati-hati {il['ambang_hati']}. Model baru: ambang krisis {AMBANG_KRISIS}, hati-hati {AMBANG_HATI}.\n",
           '| Data uji | Model | Precision | Recall | F1 | AUC | Bukan krisis salah dirujuk | Krisis terlewat | Krisis tertangani (termasuk hati-hati) |',
           '|---|---|---|---|---|---|---|---|---|']
    for judul, df in set_uji:
        for nama, p, ak, ah in [('lama', p_lama(df), il['ambang_krisis'], il['ambang_hati']), ('baru', p_baru(df), AMBANG_KRISIS, AMBANG_HATI)]:
            h = ringkas(df, p, ak, ah); silang[f'{judul} | {nama}'] = h
            md.append(f"| {judul} | {nama} | {f(h['precision'])} | {f(h['recall'])} | {f(h['f1'])} | {f(h['auc'])} | {h['fp']} dari {h['n_bukan']} | {h['fn']} dari {h['n_krisis']} | {h['tertangani']} dari {h['n_krisis']} |")
            if judul == '50 pesan uji chatbot':
                ua['p_' + nama] = p.round(4)
    if 'p_lama' in ua:
        md += ['\n## 50 pesan uji chatbot: p krisis model lama dan baru\n', '| Pesan | Harapan | p lama | p baru |', '|---|---|---|---|']
        md += [f"| {x.text} | {x.label} | {x.p_lama} | {x.p_baru} |" for x in ua.itertuples()]
    json.dump(silang, open(os.path.join(OUT, 'perbandingan_silang.json'), 'w'), indent=1, default=float)
open(os.path.join(HERE, 'hasil', 'evaluasi_krisis' + SUF + '.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
print('\n'.join(md))
