import pandas as pd,numpy as np,re
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.sparse import hstack,csr_matrix
D='Eksperimen/data_final/'
SW=set('''yang di dan itu ini dengan untuk tidak ada ke dari dalam akan pada juga saya aku kamu kau anda dia ia kita kami mereka sudah telah bisa harus karena jadi apa siapa kapan mengapa kenapa bagaimana gimana tapi tetapi namun atau jika kalau maka saat ketika sangat sekali lebih paling hanya saja masih belum lagi pun sih kok dong ya nya mu ku adalah ialah oleh bagi seperti agar supaya sedang lalu kemudian setelah sebelum sejak sampai hingga tentang antara para banyak semua setiap beberapa sesuatu tak bukan ga gak nggak enggak kan deh lah tuh nih mau ingin punya buat bikin'''.split())
def style_feats(texts):
    f=[]
    for t in texts:
        t=str(t); w=t.split()
        f.append([len(w),t.count('?'),t.count(','),t.count('.'),t.count('!'),sum(c.isupper() for c in t)/max(len(t),1),len(re.findall(r'[.!?]',t)),np.mean([len(x) for x in w]) if w else 0])
    return np.array(f)
def run(train,test,name):
    tr=pd.read_csv(D+train); te=pd.read_csv(D+test)
    cv=TfidfVectorizer(vocabulary=sorted(SW),token_pattern=r'(?u)\b\w+\b',lowercase=True)
    Xtr=hstack([cv.fit_transform(tr.text),csr_matrix(style_feats(tr.text))]).tocsr(); Xte=hstack([cv.transform(te.text),csr_matrix(style_feats(te.text))]).tocsr()
    from sklearn.preprocessing import MaxAbsScaler
    sc=MaxAbsScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xte=sc.transform(Xte)
    m=LogisticRegression(max_iter=3000,C=1.0,class_weight='balanced').fit(Xtr,tr.label); p=m.predict(Xte)
    L=['anger','fear','happy','sadness','neutral']
    pc=f1_score(te.label,p,labels=L,average=None)
    # binary neutral vs rest
    yb=(te.label=='neutral'); mb=LogisticRegression(max_iter=3000,class_weight='balanced').fit(Xtr,tr.label=='neutral'); pb=mb.predict(Xte)
    print(name,'macro',round(f1_score(te.label,p,average='macro'),3),'neutral(5cls)',round(pc[4],3),'neutral-vs-rest F1',round(f1_score(yb,pb),3), 'emo classes',dict(zip(L[:4],pc[:4].round(3))))
run('tahap2_train.csv','tahap2_test.csv','DailyDialog neutral')
run('tahap2alt_train.csv','tahap2alt_test.csv','Tweet neutral')
