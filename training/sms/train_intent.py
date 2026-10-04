import json, numpy as np, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from parser import parse, norm
tr=json.load(open('sms_train.json')); te=json.load(open('sms_test.json'))
vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4),max_features=4000,sublinear_tf=True,preprocessor=norm)
X=vec.fit_transform([x['texto'] for x in tr]); y=[x['intencion'] for x in tr]
lr=LogisticRegression(C=4,max_iter=3000,class_weight='balanced').fit(X,y)
def clf(t):
    p=lr.predict_proba(vec.transform([t]))[0]; k=p.argmax(); return lr.classes_[k],float(p[k])
pred=[parse(x['texto'],clf) for x in te]
print(classification_report([x['intencion'] for x in te],[p['intencion'] for p in pred],digits=3))
of=[(x,p) for x,p in zip(te,pred) if x['intencion']=='oferta']
def eq(a,b): return (a is None and b is None) or (a is not None and b is not None and abs(float(a)-float(b))<1e-6)
print('ofertas test',len(of))
print('cantidad exacta',np.mean([eq(x['cantidad'],p['cantidad']) for x,p in of]).round(3))
print('unidad exacta',np.mean([x['unidad']==p['unidad'] for x,p in of]).round(3))
print('mes exacto (cuando hay mes)',np.mean([x['mes']==p['mes'] for x,p in of if x['mes']]).round(3))
print('derivado a humano',np.mean([p['requiere_humano'] for x,p in of]).round(3))
# errores silenciosos: cantidad/unidad incorrecta y NO derivado a humano
sil=[(x,p) for x,p in of if not p['requiere_humano'] and (not eq(x['cantidad'],p['cantidad']) or x['unidad']!=p['unidad'])]
print('errores silenciosos',len(sil),'/',len(of))
for x,p in sil[:12]: print('  ',x['texto'],'| gt',x['cantidad'],x['unidad'],'| pred',p['cantidad'],p['unidad'])
# export for JS
voc=vec.vocabulary_; inv=sorted(voc,key=voc.get)
json.dump({'vocab':inv,'idf':vec.idf_.round(4).tolist(),'classes':lr.classes_.tolist(),'coef':lr.coef_.round(4).tolist(),'intercept':lr.intercept_.round(4).tolist(),'ngram':[2,4]},open('intent_model.json','w'))
