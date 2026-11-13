"""Seleksi kandidat kelas neutral dari DailyDialog (label "no emotion").

Langkah:
1. Ambil semua ujaran berlabel "no emotion".
2. Filter: 5-45 kata, tanpa tanda seru, tanpa kata emosi eksplisit (bahasa Inggris),
   bukan basa-basi sangat pendek, deduplikasi.
3. Maksimal satu ujaran per dialog agar topik beragam.
4. Batasi porsi kalimat tanya maksimal 30%.
5. Sampling dengan distribusi panjang (jumlah kata) yang disamakan dengan ujaran pertama
   pembicara EmpatheticDialogues pada 4 kelas emosi target, agar panjang teks tidak menjadi
   pintasan bagi model.
Output: neutral_dailydialog_kandidat_en.csv (N_TARGET baris, belum diterjemahkan).
"""
import json, re, random, sys
import pandas as pd, numpy as np

SEED = 42
N_TARGET = 1000          # 800 dipakai + cadangan untuk yang gugur saat verifikasi manual
DD_JSON = sys.argv[1] if len(sys.argv) > 1 else 'data/dialogues.json'
ED_DIR = sys.argv[2] if len(sys.argv) > 2 else '../EmpatheticDialogues/empatheticdialogues'

EMO_WORDS = r"""\b(happy|happily|happiness|glad|joy|joyful|excited|exciting|thrilled|delighted|love|loved|lovely|
adore|proud|grateful|thankful|hope|hopeful|wonderful|great|awesome|amazing|fantastic|terrific|fun|
sad|sadly|sadness|unhappy|upset|depressed|lonely|miss|missed|cry|cried|crying|tears|sorry|regret|
disappointed|disappointing|heartbroken|devastated|ashamed|guilty|embarrassed|
angry|anger|mad|furious|annoyed|annoying|irritated|hate|hated|jealous|frustrated|frustrating|
afraid|scared|scary|fear|feared|terrified|terrible|horrible|awful|worried|worry|worrying|anxious|
nervous|panic|stress|stressed|stressful|shocked|surprised|surprise|disgusted|disgusting|
feel|feeling|felt|emotional|nice|beautiful|pleased|enjoy|enjoyed|poor|unfortunately|fortunately|
congratulations|wow|oh|ah|hmm|haha|oops|damn|god|gosh)\b"""
EMO_RE = re.compile(EMO_WORDS.replace('\n', ''), re.I)

def nwords(s):
    return len(s.split())

# ---------- 1. DailyDialog ----------
dd = json.load(open(DD_JSON))
rows = []
for dlg in dd:
    for t in dlg['turns']:
        rows.append(dict(dialogue_id=dlg['dialogue_id'], data_split=dlg['data_split'],
                         topic='|'.join(dlg.get('domains', [])), utt_idx=t['utt_idx'],
                         text_en=' '.join(t['utterance'].split()), emotion=t['emotion']))
df = pd.DataFrame(rows)
stat = {'dd_dialog': len(dd), 'dd_utt': len(df),
        'dd_no_emotion': int((df.emotion == 'no emotion').sum())}
ne = df[df.emotion == 'no emotion'].copy()
ne['nw'] = ne.text_en.map(nwords)
ne = ne[(ne.nw >= 5) & (ne.nw <= 45)]
stat['after_len'] = len(ne)
ne = ne[~ne.text_en.str.contains('!', regex=False)]
stat['after_exclaim'] = len(ne)
ne = ne[~ne.text_en.str.contains(EMO_RE)]
stat['after_emoword'] = len(ne)
ne['norm'] = ne.text_en.str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True).str.strip()
ne = ne.drop_duplicates('norm')
stat['after_dedup'] = len(ne)
ne['is_q'] = ne.text_en.str.strip().str.endswith('?')
rng = np.random.default_rng(SEED)
ne = ne.sample(frac=1, random_state=SEED).drop_duplicates('dialogue_id')   # satu per dialog
stat['after_one_per_dialog'] = len(ne)

# ---------- 2. target distribusi panjang dari ED (ujaran pertama pembicara, 4 kelas) ----------
emap = {'anger': ['angry', 'furious', 'jealous', 'annoyed'],
        'fear': ['afraid', 'anxious', 'terrified', 'apprehensive'],
        'happy': ['joyful', 'excited', 'proud', 'grateful', 'content', 'hopeful', 'confident', 'caring'],
        'sadness': ['sad', 'devastated', 'lonely', 'disappointed', 'ashamed', 'guilty', 'nostalgic', 'sentimental']}
keep = {e for v in emap.values() for e in v}
edw = []
for sp in ['train', 'valid', 'test']:
    with open(f'{ED_DIR}/{sp}.csv', encoding='utf-8') as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip('\n').split(',')
            if len(p) >= 6 and p[1] == '1' and p[2] in keep:
                edw.append(nwords(p[5].replace('_comma_', ',')))
edw = np.array(edw)
bins = [5, 8, 11, 14, 17, 20, 25, 30, 46]
ed_hist = np.histogram(np.clip(edw, 5, 45), bins=bins)[0]
ed_prop = ed_hist / ed_hist.sum()
ne['bin'] = pd.cut(ne.nw, bins=bins, right=False, labels=False)

# ---------- 3. sampling per bin, pertanyaan maks 30% ----------
picked = []
for b, p in enumerate(ed_prop):
    k = int(round(p * N_TARGET))
    pool = ne[ne.bin == b]
    q = pool[pool.is_q]; s = pool[~pool.is_q]
    kq = min(len(q), int(k * 0.30)); ks = min(len(s), k - kq)
    if ks < k - kq:   # kekurangan pernyataan -> isi dengan pertanyaan
        kq = min(len(q), k - ks)
    picked.append(s.sample(ks, random_state=SEED)); picked.append(q.sample(kq, random_state=SEED))
out = pd.concat(picked).sample(frac=1, random_state=SEED).reset_index(drop=True)
out.insert(0, 'id', [f'dd_neu_{i:04d}' for i in range(len(out))])
out['label'] = 'neutral'
out = out[['id', 'label', 'text_en', 'nw', 'is_q', 'dialogue_id', 'data_split', 'topic', 'utt_idx']]
out.to_csv('neutral_dailydialog_kandidat_en.csv', index=False)

stat['final'] = len(out)
stat['final_question_pct'] = round(100 * out.is_q.mean(), 1)
stat['len_median_final'] = float(out.nw.median()); stat['len_median_ed'] = float(np.median(edw))
stat['len_p90_final'] = float(out.nw.quantile(.9)); stat['len_p90_ed'] = float(np.quantile(edw, .9))
stat['topics'] = out.topic.value_counts().to_dict()
json.dump(stat, open('neutral_seleksi_stats.json', 'w'), indent=1)
print(json.dumps(stat, indent=1))
print(out.sample(15, random_state=1)[['text_en']].to_string())
