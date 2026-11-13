import glob,json,os,re,numpy as np,pandas as pd
from sklearn.metrics import f1_score
from scipy.stats import binomtest
R='Eksperimen/eksperimen/runs'
def muat(p):
    rows=[]
    for f in glob.glob(f'{R}/{p}/metrics.json'):
        m=json.load(open(f)); d=os.path.dirname(f); rid=os.path.basename(d)
        rows.append(dict(dir=d,seed=m['config']['seed'],tag=re.sub(r'.*_s4[0-9]_','',rid),val=m['val']['f1_macro'],t2=m['test']['tahap2_test']['f1_macro'],t1=m['test']['tahap1_test']['f1_weighted']))
    return pd.DataFrame(rows)
f5,f7=muat('F5_*'),muat('F7_*')
print(f5.groupby('tag')[['val','t2','t1']].mean().round(4).sort_values('val'))
dua=f5[f5.tag=='duatahap_lr2setengah_full_noreh'].sort_values('seed')
def pred(ds): return [pd.read_csv(os.path.join(d,'prediksi','tahap2_test.csv')) for d in ds]
A=pred(dua.dir)
comps={'satutahap':'satutahap_lrpenuh_full','tahap2saja':'tahap2saja_lrsetengah_full','cls':'duatahap_cls'}
y=A[0].label.values
res={}
for k,tag in comps.items():
    d=f7[f7.tag==tag].sort_values('seed'); Bp=pred(d.dir)
    assert all((a.id.values==b.id.values).all() for a,b in zip(A,Bp))
    per=[]
    for i in range(3):
        a,b=A[i].pred.values,Bp[i].pred.values
        diff=f1_score(y,a,average='macro')-f1_score(y,b,average='macro')
        ha=int(((a==y)&(b!=y)).sum()); hb=int(((a!=y)&(b==y)).sum())
        per.append((round(diff,4),ha,hb,round(binomtest(ha,ha+hb).pvalue,3)))
    rng=np.random.default_rng(0); idx=rng.integers(0,len(y),(5000,len(y)))
    bs=np.array([np.mean([f1_score(y[r],A[i].pred.values[r],average='macro')-f1_score(y[r],Bp[i].pred.values[r],average='macro') for i in range(3)]) for r in idx])
    # item-level majority-correct McNemar: per item, correct count out of 3 -> paired sign test on mean correctness
    ca=np.mean([A[i].pred.values==y for i in range(3)],0); cb=np.mean([Bp[i].pred.values==y for i in range(3)],0)
    w=int((ca>cb).sum()); l=int((ca<cb).sum())
    res[k]=dict(per_seed=per,mean=round(np.mean([p[0] for p in per]),4),ci=[round(np.percentile(bs,2.5),4),round(np.percentile(bs,97.5),4)],sign_items=(w,l,round(binomtest(w,w+l).pvalue,3)))
    print(k,res[k])
json.dump(res,open('../ana/fase7.json','w'),indent=1)
