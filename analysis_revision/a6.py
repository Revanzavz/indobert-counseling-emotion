import pandas as pd,numpy as np,json
from tokenizers import Tokenizer
tok=Tokenizer.from_file('Eksperimen/eksperimen/hasil/model_final/tokenizer/tokenizer.json'); tok.no_padding(); tok.no_truncation()
D='Eksperimen/data_final/'
def lens(t): return np.array([len(e.ids) for e in tok.encode_batch([str(x) for x in t])])
out={}
for nm,f in [('stage1',['tahap1_train.csv','tahap1_val.csv','tahap1_test.csv']),('stage2',['tahap2_train.csv','tahap2_val.csv','tahap2_test.csv'])]:
    d=pd.concat([pd.read_csv(D+x) for x in f]); l=lens(d.text); d['l']=l
    out[nm]={'p95':int(np.percentile(l,95)),'max':int(l.max()),'trunc':{k:round(float((g.l>128).mean()*100),2) for k,g in d.groupby('label')},'trunc_all':int((l>128).sum())}
for nm,pth in [('crisis_native','Dataset/Krisis_ID/krisis_{}.csv'),('crisis_tr','Dataset/Krisis_Kaggle/krisis_{}.csv')]:
    d=pd.concat([pd.read_csv(pth.format(s)) for s in ['train','val','test']]); l=lens(d.text); d['l']=l
    out[nm]={'p95':int(np.percentile(l,95)),'max':int(l.max()),'trunc':{k:round(float((g.l>128).mean()*100),2) for k,g in d.groupby('label')},'trunc_all':int((l>128).sum()),'n':len(d)}
e=pd.read_csv('Dataset/EmoT/emot_uji.csv'); l=lens(e.text); out['emot']={'p95':int(np.percentile(l,95)),'trunc_all':int((l>128).sum())}
print(json.dumps(out,indent=1))
