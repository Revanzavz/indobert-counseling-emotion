import pandas as pd,numpy as np,json
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import binomtest
a=pd.read_csv('Eksperimen/eksperimen/hasil/prediksi_krisis_uji_asli.csv'); b=pd.read_csv('Eksperimen/eksperimen/hasil/prediksi_krisis_uji_asli_id.csv')
assert (a.text==b.text).all()
y=(a.label=='krisis').values
def cat(p,tk,th): return np.where(p>=tk,'ref',np.where(p>=th,'cau','none'))
ca=cat(a.p_krisis.values,0.21,0.08); cb=cat(b.p_krisis.values,0.32,0.05)
w=lambda k,n:tuple(round(x,2) for x in proportion_confint(k,n,method='wilson'))
out={}
for nm,c in [('tr',ca),('na',cb)]:
    o={}
    for grp,m in [('crisis',y),('non',~y)]:
        n=m.sum(); o[grp]={k:(int((c[m]==k).sum()),w(int((c[m]==k).sum()),n)) for k in ['ref','cau','none']}
    out[nm]=o
print(json.dumps(out,indent=1))
# paired direct referral on crisis msgs
ra=(ca=='ref')[y]; rb=(cb=='ref')[y]
print('ref only tr',int((ra&~rb).sum()),'only na',int((~ra&rb).sum()), binomtest(int((ra&~rb).sum()),int((ra&~rb).sum()+(~ra&rb).sum())).pvalue)
# common false-referral criterion: at most 1 false referral among 20 non-crisis -> threshold per model
for nm,p in [('tr',a.p_krisis.values),('na',b.p_krisis.values)]:
    pn=np.sort(p[~y])[::-1]; thr=pn[1]+1e-9 if len(pn)>1 else 0  # allow 1 FP: threshold just above 2nd highest
    # threshold such that FP<=1: > second highest non-crisis
    thr=np.nextafter(pn[1],1)
    rec=(p[y]>=thr).sum(); fp=(p[~y]>=thr).sum()
    print(nm,'thr',round(thr,3),'refs',rec,'/30 fp',fp)
    # common recall 0.8 (24/30): threshold = 24th highest crisis
    pc=np.sort(p[y])[::-1]; t2=pc[23]; print(nm,'recall24 thr',round(t2,3),'fp',(p[~y]>=t2).sum())
# paired at FP<=1
def refs(p,thr): return p>=thr
pa=a.p_krisis.values; pb=b.p_krisis.values
ta=np.nextafter(np.sort(pa[~y])[::-1][1],1); tb=np.nextafter(np.sort(pb[~y])[::-1][1],1)
ra2=refs(pa,ta)[y]; rb2=refs(pb,tb)[y]
x1=int((ra2&~rb2).sum()); x2=int((~ra2&rb2).sum()); print('paired fp<=1: only tr',x1,'only na',x2,binomtest(x1,x1+x2).pvalue if x1+x2 else 1)
# categories of crisis messages: check columns
print(a.columns.tolist()); 
