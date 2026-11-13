import pandas as pd, numpy as np, re, json, glob, os, collections
OUT = '.'; os.makedirs(OUT, exist_ok=True)
R = {}

# ---------------- Riccosan ----------------
rdir = '../Riccosan/Emotion Dataset from Indonesian Public Opinion'
frames = []
for f in sorted(glob.glob(rdir + '/*.csv')):
    d = pd.read_csv(f, sep='\t', dtype=str, keep_default_na=False, quoting=3)
    d['file'] = os.path.basename(f)
    frames.append(d)
ric = pd.concat(frames, ignore_index=True)
ric.columns = [c.strip() for c in ric.columns]
ric['Tweet'] = ric['Tweet'].str.strip(); ric['Label'] = ric['Label'].str.strip()
R['ric_total'] = len(ric)
R['ric_label_counts6'] = ric['Label'].value_counts().to_dict()
R['ric_file_label_mismatch'] = int((ric['file'].str.replace('Data.csv', '') != ric['Label']).sum())
map5 = {'Anger': 'anger', 'Fear': 'fear', 'Joy': 'happy', 'Love': 'happy', 'Sad': 'sadness', 'Neutral': 'neutral'}
ric['label5'] = ric['Label'].map(map5)
c5 = ric['label5'].value_counts()
R['ric_label_counts5'] = c5.to_dict()
R['ric_pct5'] = (c5 / c5.sum() * 100).round(2).to_dict()
R['ric_imbalance_ratio5'] = round(c5.max() / c5.min(), 2)
c6 = ric['Label'].value_counts()
R['ric_imbalance_ratio6'] = round(c6.max() / c6.min(), 2)
def norm_entropy(c):
    p = c / c.sum(); return round(float(-(p * np.log(p)).sum() / np.log(len(c))), 4)
R['ric_balance_entropy5'] = norm_entropy(c5)
R['ric_balance_entropy6'] = norm_entropy(c6)

def normtxt(s):
    s = s.lower(); s = re.sub(r'\s+', ' ', s); s = re.sub(r'[^\w\s]', '', s); return s.strip()
ric['norm'] = ric['Tweet'].map(normtxt)
R['ric_empty'] = int((ric['norm'] == '').sum())
R['ric_exact_dup_rows'] = int(ric.duplicated(['Tweet']).sum())
R['ric_norm_dup_rows'] = int(ric.duplicated(['norm']).sum())
g = ric.groupby('norm')['Label'].nunique()
conflict = g[g > 1].index
R['ric_conflicting_label_texts'] = int(len(conflict))
R['ric_conflict_examples'] = ric[ric['norm'].isin(conflict)].sort_values('norm')[['Tweet', 'Label']].head(12).values.tolist()
# conflicts that survive the love->happy merge
g5 = ric.groupby('norm')['label5'].nunique()
R['ric_conflicting_after_merge'] = int((g5 > 1).sum())
dups_within = ric[ric.duplicated('norm', keep=False)]
R['ric_dup_by_label'] = dups_within.groupby('Label').size().to_dict()

ric['nwords'] = ric['Tweet'].str.split().str.len()
R['ric_len_words'] = ric['nwords'].describe(percentiles=[.5, .9, .95, .99]).round(1).to_dict()
R['ric_len_by_label'] = ric.groupby('label5')['nwords'].median().to_dict()
R['ric_short_le2'] = int((ric['nwords'] <= 2).sum())
R['ric_short_examples'] = ric[ric['nwords'] <= 2][['Tweet', 'Label']].sample(12, random_state=1).values.tolist()
R['ric_url'] = int(ric['Tweet'].str.contains(r'http|www\.', regex=True).sum())
R['ric_mention'] = int(ric['Tweet'].str.contains(r'@\w', regex=True).sum())
R['ric_hashtag'] = int(ric['Tweet'].str.contains(r'#\w', regex=True).sum())
R['ric_nonascii'] = int(ric['Tweet'].str.contains(r'[^\x00-\x7F]', regex=True).sum())
R['ric_uppercase'] = int(ric['Tweet'].str.contains(r'[A-Z]', regex=True).sum())

# keyword shortcut: fraction of class containing its "trigger" word
trig = {'anger': r'\b(marah|kesel|kesal|emosi|benci|anjir|anjing|bangsat|goblok|bodoh)\b',
        'fear': r'\b(takut|cemas|khawatir|was-was|ngeri|panik|gelisah)\b',
        'happy': r'\b(senang|seneng|bahagia|suka|sayang|cinta|gembira|syukur|bangga)\b',
        'sadness': r'\b(sedih|kecewa|nangis|galau|hancur|sakit hati|kangen|rindu)\b',
        'neutral': r'$^'}
kw = {}
for lab, pat in trig.items():
    if lab == 'neutral': continue
    m = ric['Tweet'].str.lower().str.contains(pat, regex=True)
    kw[lab] = {'in_class_pct': round(100 * m[ric['label5'] == lab].mean(), 1),
               'in_other_pct': round(100 * m[ric['label5'] != lab].mean(), 1)}
R['ric_keyword_shortcut'] = kw

# 80/10/10 stratified expected test support
from sklearn.model_selection import train_test_split
tr, tmp = train_test_split(ric, test_size=0.2, stratify=ric['label5'], random_state=42)
va, te = train_test_split(tmp, test_size=0.5, stratify=tmp['label5'], random_state=42)
R['ric_split_sizes'] = [len(tr), len(va), len(te)]
R['ric_test_support'] = te['label5'].value_counts().to_dict()
# leakage: normalized text in both train and test
R['ric_train_test_overlap'] = int(te['norm'].isin(set(tr['norm'])).sum())
R['ric_train_test_overlap_after_dedup'] = 0

# ---------------- EmpatheticDialogues ----------------
def load_ed(path):
    rows = []
    with open(path, encoding='utf-8') as fh:
        header = fh.readline()
        for line in fh:
            p = line.rstrip('\n').split(',')
            if len(p) < 6: continue
            rows.append(dict(conv_id=p[0], utt_idx=int(p[1]), context=p[2], prompt=p[3].replace('_comma_', ','),
                             speaker=p[4], utterance=p[5].replace('_comma_', ','), nfields=len(p)))
    return pd.DataFrame(rows)
eds = {s: load_ed(f'../EmpatheticDialogues/empatheticdialogues/{s}.csv') for s in ['train', 'valid', 'test']}
ed = pd.concat([d.assign(split=s) for s, d in eds.items()], ignore_index=True)
R['ed_rows'] = {s: len(d) for s, d in eds.items()}
R['ed_convs'] = {s: d['conv_id'].nunique() for s, d in eds.items()}
R['ed_bad_field_rows'] = int((ed['nfields'] != 8).sum())
conv = ed.groupby(['split', 'conv_id']).first().reset_index()
R['ed_n_emotions'] = conv['context'].nunique()
R['ed_conv_per_emotion'] = conv['context'].value_counts().to_dict()
R['ed_utts_per_conv'] = ed.groupby('conv_id').size().describe().round(2).to_dict()

emap = {'anger': ['angry', 'furious', 'jealous', 'annoyed'],
        'fear': ['afraid', 'anxious', 'terrified', 'apprehensive'],
        'happy': ['joyful', 'excited', 'proud', 'grateful', 'content', 'hopeful', 'confident', 'caring'],
        'sadness': ['sad', 'devastated', 'lonely', 'disappointed', 'ashamed', 'guilty', 'nostalgic', 'sentimental']}
inv = {e: k for k, v in emap.items() for e in v}
conv['target'] = conv['context'].map(inv)
ed['target'] = ed['context'].map(inv)
used = set(inv)
R['ed_unmapped_labels'] = sorted(set(conv['context']) - used)
R['ed_unmapped_conv_count'] = int(conv['target'].isna().sum())
R['ed_conv_per_target'] = conv['target'].value_counts().to_dict()
R['ed_conv_per_target_by_source'] = {k: {e: int((conv['context'] == e).sum()) for e in v} for k, v in emap.items()}
# utterance-level
R['ed_utt_per_target'] = ed['target'].value_counts().to_dict()
first = ed[ed['utt_idx'] == 1]
R['ed_first_utt_per_target'] = first['target'].value_counts().to_dict()
# speaker vs listener: listener turns (even utt_idx) do not express the labelled emotion
R['ed_listener_share_pct'] = round(100 * (ed['utt_idx'] % 2 == 0).mean(), 1)
ed['nwords'] = ed['utterance'].str.split().str.len()
conv['pwords'] = conv['prompt'].str.split().str.len()
R['ed_utt_len_words'] = ed['nwords'].describe(percentiles=[.5, .9, .95]).round(1).to_dict()
R['ed_first_len_words'] = ed[ed['utt_idx'] == 1]['nwords'].describe(percentiles=[.5, .9, .95]).round(1).to_dict()
R['ed_prompt_len_words'] = conv['pwords'].describe(percentiles=[.5, .9, .95]).round(1).to_dict()
# duplicate prompts / utterances
R['ed_dup_first_utt'] = int(first.duplicated('utterance').sum())
# Listener examples to show label noise
lis = ed[(ed['utt_idx'] == 2) & ed['target'].notna()].sample(6, random_state=3)
R['ed_listener_examples'] = lis[['context', 'utterance']].values.tolist()
spk = ed[(ed['utt_idx'] == 1) & ed['target'].notna()].sample(6, random_state=3)
R['ed_speaker_examples'] = spk[['context', 'utterance']].values.tolist()
# ED emotion ambiguity reported by the dataset itself (Rashkin: ~ 'content' vs 'joyful' etc) not computed.

# neutral in 2nd stage: from Riccosan train portion
R['neutral_train_available'] = int((tr['label5'] == 'neutral').sum())

json.dump(R, open(f'{OUT}/stats.json', 'w'), indent=1, ensure_ascii=False, default=str)
ric.to_pickle(f'{OUT}/ric.pkl'); ed.to_pickle(f'{OUT}/ed.pkl'); conv.to_pickle(f'{OUT}/conv.pkl')
print(json.dumps(R, indent=1, ensure_ascii=False, default=str)[:12000])
import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, json
S=json.load(open('./stats.json')); ric=pd.read_pickle('./ric.pkl'); ed=pd.read_pickle('./ed.pkl')
BLUE='#2a78d6'; ORANGE='#eb6834'; TXT='#0b0b0b'; SEC='#52514e'; BG='#fcfcfb'
plt.rcParams.update({'font.size':10,'axes.edgecolor':'#c9c8c2','axes.labelcolor':SEC,'xtick.color':SEC,'ytick.color':SEC,'figure.facecolor':BG,'axes.facecolor':BG,'axes.spines.top':False,'axes.spines.right':False})
def bars(ax,d,title,color=BLUE):
    k=list(d.keys()); v=list(d.values())
    b=ax.bar(k,v,color=color,width=0.6)
    for r,x in zip(b,v): ax.text(r.get_x()+r.get_width()/2,x,f'{x:,}'.replace(',','.'),ha='center',va='bottom',color=TXT,fontsize=9)
    ax.set_title(title,loc='left',color=TXT,fontsize=11); ax.grid(axis='y',color='#ecebe6'); ax.set_axisbelow(True)
fig,ax=plt.subplots(1,2,figsize=(11,3.8))
bars(ax[0],S['ric_label_counts6'],'Riccosan: 6 label asli')
o=['happy','neutral','anger','sadness','fear']; bars(ax[1],{k:S['ric_label_counts5'][k] for k in o},'Riccosan: setelah love digabung ke happy (5 kelas)')
fig.tight_layout(); fig.savefig('./01_distribusi_riccosan.png',dpi=150)
fig,ax=plt.subplots(1,2,figsize=(11,3.8))
o=['happy','sadness','anger','fear']; bars(ax[0],{k:S['ed_conv_per_target'][k] for k in o},'EmpatheticDialogues: percakapan per kelas target')
bars(ax[1],{'dipakai (4 kelas)':sum(S['ed_conv_per_target'].values()),'dibuang (8 label)':S['ed_unmapped_conv_count']},'EmpatheticDialogues: cakupan pemetaan label',color=BLUE)
fig.tight_layout(); fig.savefig('./02_distribusi_ed.png',dpi=150)
fig,ax=plt.subplots(1,2,figsize=(11,3.8))
for i,(lab) in enumerate(['anger','fear','happy','sadness','neutral']): pass
ax[0].hist(ric['nwords'],bins=30,color=BLUE); ax[0].set_title('Riccosan: panjang tweet (kata)',loc='left',color=TXT); ax[0].axvline(ric['nwords'].quantile(.95),color=SEC,ls='--',lw=1); ax[0].text(ric['nwords'].quantile(.95)+1,ax[0].get_ylim()[1]*.85,'P95',color=SEC)
f=ed[ed.utt_idx==1]; ax[1].hist(f['nwords'].clip(upper=80),bins=40,color=BLUE); ax[1].set_title('ED: panjang ujaran pertama pembicara (kata, EN)',loc='left',color=TXT); ax[1].axvline(f['nwords'].quantile(.95),color=SEC,ls='--',lw=1); ax[1].text(f['nwords'].quantile(.95)+1,ax[1].get_ylim()[1]*.85,'P95',color=SEC)
for a in ax: a.grid(axis='y',color='#ecebe6'); a.set_axisbelow(True)
fig.tight_layout(); fig.savefig('./03_panjang_teks.png',dpi=150)
