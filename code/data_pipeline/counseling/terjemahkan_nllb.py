"""Terjemahan EN -> ID dengan NLLB-200 distilled-600M.

Dipakai untuk kelas neutral (DailyDialog) dan dapat dipakai ulang untuk EmpatheticDialogues
agar kelima kelas melewati pipeline terjemahan yang identik.

Pemakaian:
  python terjemahkan_nllb.py input.csv output.csv --model <folder_atau_hf_id> [--col text_en]

Konfigurasi dikunci (harus sama untuk semua kelas):
  model facebook/nllb-200-distilled-600M, src eng_Latn, tgt ind_Latn,
  num_beams=4, max_new_tokens=256, greedy=False, tanpa sampling (deterministik).
"""
import argparse, re, time, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ap = argparse.ArgumentParser()
ap.add_argument('inp'); ap.add_argument('out')
ap.add_argument('--model', default='facebook/nllb-200-distilled-600M')
ap.add_argument('--col', default='text_en')
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--beams', type=int, default=4)
ap.add_argument('--only-ids', default=None, help='file teks berisi id yang ingin diterjemahkan ulang')
ap.add_argument('--no-split', action='store_true', help='jangan pecah per kalimat (tidak disarankan)')
a = ap.parse_args()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(a.model, src_lang='eng_Latn')
model = AutoModelForSeq2SeqLM.from_pretrained(a.model).to(dev).eval()
if dev == 'cuda':
    model = model.half()

# ID token bahasa dihitung dari daftar resmi NLLB supaya tidak bergantung versi tokenizer.
try:
    from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES as L
    SRC, TGT = 256001 + L.index('eng_Latn'), 256001 + L.index('ind_Latn')
except Exception:
    SRC, TGT = tok.convert_tokens_to_ids('eng_Latn'), tok.convert_tokens_to_ids('ind_Latn')
assert (SRC, TGT) == (256047, 256075), (SRC, TGT)

# NLLB cenderung hanya menerjemahkan kalimat pertama bila input berisi beberapa kalimat.
# Karena itu teks dipecah per kalimat, diterjemahkan satu per satu, lalu digabung kembali.
ABBR = ''.join(r'(?<!\b' + x + r')' for x in [r'Mr\.', r'Mrs\.', r'Ms\.', r'Dr\.', r'St\.', r'Jr\.', r'Sr\.', r'vs\.', r'e\.g\.', r'i\.e\.', r'a\.m\.', r'p\.m\.', r'No\.'])
SPLIT = re.compile(ABBR + r'(?<=[.?!])\s+(?=[A-Z"\'(])')
def split_sent(t):
    t = ' '.join(str(t).split())
    t = re.sub(r'(\d)\. (\d)', r'\1.\2', t)          # "$1. 50" -> "$1.50"
    if a.no_split:
        return [t]
    return [s for s in SPLIT.split(t) if s.strip()]

df = pd.read_csv(a.inp)
segs = [split_sent(t) for t in df[a.col].astype(str)]
owner = [i for i, ss in enumerate(segs) for _ in ss]
texts = [s for ss in segs for s in ss]
print(f'{len(df)} baris -> {len(texts)} kalimat', flush=True)
def translate(texts, label=''):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    res = [None] * len(texts)
    t0 = time.time()
    for k in range(0, len(order), a.batch):
        idx = order[k:k + a.batch]
        x = tok([texts[i] for i in idx], return_tensors='pt', padding=True, truncation=True, max_length=256)
        x['input_ids'][:, 0] = SRC
        x = {kk: v.to(dev) for kk, v in x.items()}
        with torch.inference_mode():
            y = model.generate(**x, forced_bos_token_id=TGT, num_beams=a.beams, max_new_tokens=256, do_sample=False)
        for i, t in zip(idx, tok.batch_decode(y, skip_special_tokens=True)):
            res[i] = t.strip()
        print(f'{label}{min(k + a.batch, len(order))}/{len(order)}  {time.time() - t0:.0f}s', flush=True)
    return res

res = translate(texts)

# Pengaman halusinasi: kalimat sangat pendek (< 3 kata, mis. "Hello.", "Thank you.") kadang
# diterjemahkan melebar ("Hei, apa yang kau lakukan?") atau diulang ("- Terima kasih. - Terima kasih.").
# Kalimat seperti itu diterjemahkan ulang bersama kalimat sesudahnya dalam satu input.
def suspicious(src, out):
    ns, no = len(src.split()), len(out.split())
    return ns < 3 and (no > 2 * ns + 2 or out.count('- ') >= 2 or re.search(r'(\b\w+\b.{0,20})\1', out.lower()) is not None)

fix_rows = sorted({owner[j] for j in range(len(texts)) if suspicious(texts[j], res[j])})
n_fixed = 0
if fix_rows:
    pairs = []   # (row, seg_start) -> gabungan seg_start + seg_start+1
    for r in fix_rows:
        js = [j for j in range(len(texts)) if owner[j] == r]
        for pos, j in enumerate(js):
            if suspicious(texts[j], res[j]):
                nb = js[pos + 1] if pos + 1 < len(js) else (js[pos - 1] if pos > 0 else None)
                if nb is not None:
                    pairs.append((j, nb))
    merged = [texts[min(j, nb)] + ' ' + texts[max(j, nb)] for j, nb in pairs]
    out2 = translate(merged, 'pengaman ')
    for (j, nb), t in zip(pairs, out2):
        res[min(j, nb)] = t; res[max(j, nb)] = ''
        n_fixed += 1
print('kalimat pendek yang diterjemahkan ulang (digabung):', n_fixed)

joined = [[] for _ in range(len(df))]
for i, t in zip(owner, res):
    if t:
        joined[i].append(t)
df['n_kalimat'] = [len(s) for s in segs]
df['text_id'] = [' '.join(j) for j in joined]
df['nllb_config'] = f'nllb-200-distilled-600M|eng_Latn->ind_Latn|beams={a.beams}|max_new_tokens=256|split_kalimat={not a.no_split}'
df.to_csv(a.out, index=False)
print('selesai', a.out)
