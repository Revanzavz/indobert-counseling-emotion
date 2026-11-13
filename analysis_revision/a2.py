import json,numpy as np,pandas as pd
from sklearn.metrics import f1_score
L=['anger','fear','happy','sadness','neutral']
d=pd.read_csv('Eksperimen/eksperimen/hasil/banding_gemini_konseling_lengkap.csv')
y,g,b=d.label.values,d.pred_gemini.values,d.pred_indobert.values
full=f1_score(y,g,labels=L,average='macro')-f1_score(y,b,labels=L,average='macro')
pc=f1_score(y,g,labels=L,average=None)-f1_score(y,b,labels=L,average=None)
rng=np.random.default_rng(0); n=len(y); B=10000
idx=rng.integers(0,n,(B,n))
bm=np.empty(B); bc=np.empty((B,5))
for k,r in enumerate(idx):
    fg=f1_score(y[r],g[r],labels=L,average=None,zero_division=0); fb=f1_score(y[r],b[r],labels=L,average=None,zero_division=0)
    bc[k]=fg-fb; bm[k]=fg.mean()-fb.mean()
ci=lambda a:(round(np.percentile(a,2.5),4),round(np.percentile(a,97.5),4))
pval=lambda a:min(1,2*min((a<=0).mean(),(a>=0).mean()))
out={'macro_diff':round(full,4),'macro_ci':ci(bm),'macro_p':pval(bm)}
ps=[pval(bc[:,i]) for i in range(5)]
# Holm
order=np.argsort(ps); adj=[None]*5; m=5; run=0
for rank,i in enumerate(order):
    v=min(1,(m-rank)*ps[i]); run=max(run,v); adj[i]=run
out['class']={L[i]:dict(diff=round(pc[i],4),ci=ci(bc[:,i]),p=round(ps[i],4),p_holm=round(adj[i],4)) for i in range(5)}
t=json.load(open('Eksperimen/eksperimen/hasil/gemini_ttest.json'))
dd=np.array(t['ttest_macro']['lipatan_gemini'])-np.array(t['ttest_macro']['lipatan_indobert'])
from scipy import stats
se=dd.std(ddof=1)/np.sqrt(10); tc=stats.t.ppf(.975,9)
out['fold']=dict(g_mean=round(np.mean(t['ttest_macro']['lipatan_gemini']),4),b_mean=round(np.mean(t['ttest_macro']['lipatan_indobert']),4),diff=round(dd.mean(),5),ci=(round(dd.mean()-tc*se,4),round(dd.mean()+tc*se,4)),t=t['ttest_macro']['t'],p=t['ttest_macro']['p'])
out['fold_class']={k:dict(t=round(v['t'],3),p=round(v['p'],4)) for k,v in t['ttest_kelas'].items()}
# Holm on fold class tests
fp=[t['ttest_kelas'][l]['p'] for l in L]; order=np.argsort(fp); run=0; adj2={}
for rank,i in enumerate(order):
    v=min(1,(5-rank)*fp[i]); run=max(run,v); adj2[L[i]]=round(run,4)
out['fold_class_holm']=adj2
print(json.dumps(out,indent=1,default=float)); json.dump(out,open('../ana/gemini.json','w'),indent=1,default=float)
