import pandas as pd,numpy as np,re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
def norm(s): return re.sub(r'\s+',' ',re.sub(r'[^\w\s]','',str(s).lower())).strip()
emo=pd.read_csv('Dataset/EmoT/emot_uji.csv'); print(emo.columns.tolist(), len(emo))
D='Eksperimen/data_final/'
tr=pd.concat([pd.read_csv(D+f) for f in ['tahap1_train.csv','tahap1_val.csv','tahap1_test.csv','tahap2_train.csv','tahap2_val.csv','tahap2_test.csv']])
tcol='text'
E_=emo[tcol].map(norm).values; T=tr.text.map(norm).values
print('exact overlap', len(set(E_)&set(T)))
v=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=1).fit(np.concatenate([E_,T]))
XE=v.transform(E_); XT=v.transform(T)
nn=NearestNeighbors(n_neighbors=1,metric='cosine').fit(XT); dist,ind=nn.kneighbors(XE)
sim=1-dist[:,0]
for th in [0.95,0.9,0.8]: print('near>=',th,(sim>=th).sum())
o=np.argsort(-sim)[:5]
for i in o: print(round(sim[i],3),'|',E_[i][:90],'||',T[ind[i,0]][:90])
# crisis: 750 in which split
for sp in ['train','val','test']:
    k=pd.read_csv(f'Dataset/Krisis_Kaggle/krisis_{sp}.csv'); print(sp, k.columns.tolist()[:6], k.status.str.startswith('emosi').sum() if 'status' in k else '')
# overlap 750 with tahap2 test/val
k=pd.concat([pd.read_csv(f'Dataset/Krisis_Kaggle/krisis_{sp}.csv') for sp in ['train','val','test']])
em=k[k.status.str.startswith('emosi')].text.map(norm)
t2te=set(pd.concat([pd.read_csv(D+'tahap2_test.csv'),pd.read_csv(D+'tahap2_val.csv')]).text.map(norm))
print('750 overlap tahap2 val/test', em.isin(t2te).sum())
# chatbot 50 msgs overlap with any crisis training
ua=pd.read_csv('Eksperimen/eksperimen/hasil/prediksi_krisis_uji_asli_id.csv').text.map(norm)
ki=pd.concat([pd.read_csv(f'Dataset/Krisis_ID/krisis_{sp}.csv') for sp in ['train','val','test']])
print('50 msgs exact in krisis_id', ua.isin(set(ki.text.map(norm))).sum(), 'in kaggle', ua.isin(set(k.text.map(norm))).sum())
v2=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5)).fit(pd.concat([ua,ki.text.map(norm)]))
nn2=NearestNeighbors(n_neighbors=1,metric='cosine').fit(v2.transform(ki.text.map(norm))); d2,i2=nn2.kneighbors(v2.transform(ua))
s2=1-d2[:,0]; print('50 msgs max sim to native crisis data', round(s2.max(),3), (s2>=0.8).sum())
kil=ki.text.map(norm).values
for i in np.argsort(-s2)[:4]: print(round(s2[i],3),ua.iloc[i],'||',kil[i2[i,0]][:80])
