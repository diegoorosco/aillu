import cv2, json, numpy as np, torch, sys, collections
sys.path.insert(0,'/home/claude/hack/logic')
from common import *; from train import Net, to_t, render, te
from grade import evaluate as grade_eval
import random
net=Net(); net.load_state_dict(torch.load('net.pt')); net.eval()
TH=json.load(open('/home/claude/hack/logic/config.json'))['umbral_prob_grano']
# 1) per-bean test
r=random.Random(5)
X=to_t([render(m,r,False) for m in te]); y=np.array([CLASSES.index(m['cls']) for m in te])
with torch.no_grad(): P=torch.softmax(net(X),1).numpy()
pred=P.argmax(1); print('acc grano (test, 11 clases):',(pred==y).mean().round(3))
sd=(y==0); print('sano vs defecto: acc',((pred==0)==sd).mean().round(3),' recall defecto',((pred!=0)&~sd).sum()/(~sd).sum())
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y,pred,labels=range(11),target_names=CLASSES,digits=2,zero_division=0))
# 2) end-to-end on synthetic samples
gt=json.load(open('muestras/gt.json')); rows=[]
for s in gt:
    img=cv2.imread('muestras/'+s['file']); th,cnts=segment(img)
    img=white_balance(img,th==0); crops=[crop_norm(img,c) for c in cnts]
    with torch.no_grad(): p=torch.softmax(net(to_t(crops)),1).numpy()
    conf=p.max(1)>=TH; am=p.argmax(1); soft={c:float(((am==i)&conf).sum()+p[~conf,i].sum()) for i,c in enumerate(CLASSES)}; dud=int((~conf).sum())
    g_true=collections.Counter(b['cls'] for b in s['beans'])
    rp=grade_eval(soft,dud); rt=grade_eval({c:g_true.get(c,0) for c in CLASSES},0)
    rows.append(dict(n_gt=len(s['beans']),n_seg=len(cnts),def_gt=len(s['beans'])-g_true['sano'],def_pred=round(sum(soft.values())-soft['sano'],1),dud=dud,
        g_true=rt['grado'],g_pred=rp['grado'],seguro=rp['seguro'],motivo=rp['motivo']))
import pandas as pd
df=pd.DataFrame(rows); print(df.head(8).to_string())
print('segmentación: granos detectados / reales =',round(df.n_seg.sum()/df.n_gt.sum(),3))
print('error medio en nº de defectos por muestra:',(df.def_pred-df.def_gt).abs().mean().round(2),' (sesgo',(df.def_pred-df.def_gt).mean().round(2),')')
print('grado correcto:',(df.g_true==df.g_pred).mean().round(3))
seg=df[df.seguro]; print('respuestas "seguras":',len(seg),'/',len(df),' de ellas correctas:',(seg.g_true==seg.g_pred).mean().round(3))
print('derivadas a humano:',(~df.seguro).sum(), df[~df.seguro].motivo.value_counts().to_dict())
df.to_csv('eval_muestras.csv',index=False)
