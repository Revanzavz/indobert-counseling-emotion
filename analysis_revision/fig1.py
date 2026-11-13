import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
plt.rcParams.update({'font.family':['Times New Roman','Liberation Serif'],'font.size':8})
W=3.45
fig,ax=plt.subplots(figsize=(W,5.3)); ax.set_xlim(0,10.35); ax.set_ylim(1.3,17.0); ax.axis('off')
FS=7.0
def box(x,y,w,h,t,fc='white',lw=0.8,ls='-'):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0,rounding_size=0.15',fc=fc,ec='black',lw=lw,ls=ls))
    ax.text(x,y,t,ha='center',va='center',fontsize=FS,linespacing=1.12)
def dia(x,y,w,h,t):
    ax.add_patch(Polygon([[x,y+h/2],[x+w/2,y],[x,y-h/2],[x-w/2,y]],fc='white',ec='black',lw=0.8))
    ax.text(x,y,t,ha='center',va='center',fontsize=FS,linespacing=1.05)
def arr(x1,y1,x2,y2,ls='-'):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>,head_length=0.3,head_width=0.15',lw=0.8,ls=ls,color='black',shrinkA=0,shrinkB=0))
def line(xs,ys,ls='-'): ax.plot(xs,ys,color='black',lw=0.8,ls=ls)
def lab(x,y,t,ha='left'): ax.text(x,y,t,fontsize=7,style='italic',ha=ha,va='center')
xc=3.3; bw=6.4; rx=8.45; rw=2.9
box(xc,16.4,bw,0.8,'User message',fc='#e6e6e6')
arr(xc,16.0,xc,15.5)
box(xc,14.95,bw,1.1,'Crisis detector (IndoBERTweet)\ncalibrated crisis probability p')
arr(xc,14.4,xc,13.95)
dia(xc,13.25,4.3,1.4,'p ≥ 0.32?')
box(rx,13.25,rw,1.4,'Direct referral\n(119 ext. 8),\nno Gemini call',lw=1.2)
arr(xc+2.15,13.25,rx-rw/2,13.25); lab(xc+2.25,13.55,'yes')
arr(xc,12.55,xc,12.05); lab(xc+0.15,12.3,'no')
dia(xc,11.35,4.3,1.4,'p ≥ 0.05?')
box(rx,11.35,rw,1.6,'Caution flag;\nrepeated flag if\nthe last turn was\nalso flagged',ls='--')
arr(xc+2.15,11.35,rx-rw/2,11.35); lab(xc+2.25,11.65,'yes')
arr(xc,10.65,xc,10.15); lab(xc+0.15,10.4,'no')
box(xc,9.6,bw,1.1,'Emotion model (IndoBERTweet with\nBiLSTM head), calibrated probabilities')
arr(xc,9.05,xc,8.6)
box(xc,7.95,bw,1.3,'Conversation map (fixed schema): turn,\nstage, dialogue act, emotions, flag')
line([rx,rx],[11.35-0.8,7.95],ls='--'); arr(rx,7.95,xc+bw/2,7.95,ls='--')
arr(xc,7.3,xc,6.85)
dia(xc,6.15,4.3,1.4,'Privacy\ncheck passed?')
arr(xc,5.45,xc,4.95); lab(xc+0.15,5.2,'yes')
box(xc,4.4,bw,1.1,'Response module (Gemini API) receives\nthe map only; on API error, template')
box(rx,4.4,rw,1.3,'Template reply\nby emotion and\ndialogue act')
line([xc+2.15,rx],[6.15,6.15]); arr(rx,6.15,rx,5.05); lab(xc+2.25,6.45,'no')
arr(xc+bw/2,4.4,rx-rw/2,4.4,ls=':'); 
arr(xc,3.85,xc,3.2)
box(xc,2.8,bw,0.8,'Reply to user',fc='#e6e6e6')
line([rx,rx],[3.75,2.8]); arr(rx,2.8,xc+bw/2,2.8)
line([rx+rw/2,10.2],[13.25,13.25]); line([10.2,10.2],[13.25,1.6]); line([xc,10.2],[1.6,1.6]); arr(xc,1.6,xc,2.4)
plt.savefig('fig1_alur.png',dpi=300,bbox_inches='tight',pad_inches=0.02)
