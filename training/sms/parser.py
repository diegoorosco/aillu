import re, json, unicodedata
NUM_ES={'medio':0.5,'un':1,'uno':1,'una':1,'dos':2,'tres':3,'cuatro':4,'cinco':5,'seis':6,'siete':7,'ocho':8,'nueve':9,'diez':10,'once':11,'doce':12,'quince':15,'veinte':20,'treinta':30,'cuarenta':40,'cincuenta':50,'cien':100,'ciento':100}
NUM_QU={'huk':1,'hoq':1,'iskay':2,'ishkay':2,'kimsa':3,'kinsa':3,'tawa':4,'pichqa':5,'pisqa':5,'pichqa':5,'suqta':6,'soqta':6,'qanchis':7,'qanchik':7,'pusaq':8,'pusac':8,'isqun':9,'isqon':9,'chunka':10,'chunca':10,'pachak':100,'pachaq':100}
UNITS=[('quintal',r'\b(q\.?q\.?|qq|quint\w*|kintal\w*)\b'),('arroba',r'\b(arroba\w*|@)'),('kg',r'\b(kg|kgs|kilo\w*|quilo\w*)\b'),('saco',r'\b(saco\w*|costal\w*)\b'),('lata',r'\b(lata\w*)\b')]
KG={'quintal':46,'arroba':11.5,'kg':1,'lata':2.5,'saco':None}  # lata: ~12.5 kg cereza ≈ 2.5 kg pergamino (*supuesto); saco: ambiguo
MESES=['enero','febrero','marzo','abril','mayo','junio','julio','agosto','setiembre','octubre','noviembre','diciembre']
def norm(t): return ''.join(c for c in unicodedata.normalize('NFD',t.lower()) if unicodedata.category(c)!='Mn')
def words_to_num(tok):
    t=tok.rstrip('n')  # pichqan -> pichqa
    return NUM_ES.get(tok, NUM_QU.get(tok, NUM_QU.get(t)))
def numbers(t):
    out=[]
    toks=re.findall(r'\d+(?:[.,]\d+)?|[a-zñ]+',t)
    i=0
    while i<len(toks):
        tk=toks[i]
        if re.match(r'\d',tk): v=float(tk.replace(',','.'))
        else: v=words_to_num(tk)
        if v is not None:
            # quechua compounds: iskay chunka (20), chunka pichqa(yuq) (15)
            j=i
            if j+1<len(toks) and toks[j+1] in ('chunka','chunca') and v<10 and not re.match(r'\d',tk): v*=10; j+=1
            if toks[j] in ('chunka','chunca') and j+1<len(toks) and (words_to_num(toks[j+1]) or 99)<10: v+=words_to_num(toks[j+1]); j+=1
            if j+2<len(toks) and toks[j+1]=='y' and toks[j+2] in ('medio','media'): v+=0.5; j+=2
            elif j+3<len(toks) and toks[j+2]=='y' and toks[j+3] in ('medio','media'): v+=0.5
            out.append((i,v,tk)); i=j
        i+=1
    return toks,out
def mes(t, hoy=10):
    for k,m in enumerate(MESES):
        if re.search(r'\b'+m[:4],t) or (m=='setiembre' and 'septiem' in t): return k+1
    if re.search(r'todos ?santos|noviem',t): return 11
    if re.search(r'fin de mes|esta semana|la otra semana|proxima semana|semana que viene|ya\b|ahora|listo|hoy|manana|kunan|paqarin',t): return hoy
    return None
def parse(text, clf=None, hoy=10):
    t=norm(text); toks,nums=numbers(t)
    unidad=None
    for u,rx in UNITS:
        if re.search(rx,t): unidad=u; break
    # filter numbers that look like ids/dates (socio 214, 15 de julio)
    cand=[]
    for i,v,tk in nums:
        prev=toks[i-1] if i>0 else ''; nxt=toks[i+1] if i+1<len(toks) else ''
        if prev in ('socio','soc','codigo','cod','n','nro') or nxt=='de' and any(nxt2.startswith(m[:4]) for m in MESES for nxt2 in toks[i+2:i+3]): continue
        if v>=1000: continue
        cand.append((i,v))
    # codigos de socio: numero en los 2 primeros tokens (o pegado a un nombre) seguido de otro numero
    while len(cand)>1 and cand[0][0]<=2 and re.match(r'\d',toks[cand[0][0]]) and not re.search(r'\b(a|y|o|entre)\b',' '.join(toks[cand[0][0]+1:cand[1][0]])):
        cand=cand[1:]
    if len(cand)>1 and re.match(r'\d',toks[cand[0][0]]) and cand[0][0]<=2 and toks[cand[0][0]+1:cand[0][0]+2]==['a']: cand=cand[1:]
    cantidad=None; rango=None
    if cand:
        m=re.search(r'\b(mejor|no son|corrijo|digo)\b',t)
        if m:
            after=[v for i,v in cand if i>=len(re.findall(r'\d+(?:[.,]\d+)?|[a-zñ]+',t[:m.start()]))]
            cantidad=after[0] if after else cand[-1][1]
        else:
            cantidad=cand[0][1]
            if len(cand)>1 and re.search(r'\b(a|y|o|entre)\b',t) and cand[1][0]-cand[0][0]<=2: rango=(cand[0][1],cand[1][1]); cantidad=(cand[0][1]+cand[1][1])/2
    producto='cereza' if re.search(r'cerez',t) else ('pergamino' if re.search(r'perga',t) else None)
    intent,p=('oferta',1.0)
    if clf: intent,p=clf(text)
    kg=None
    if cantidad is not None and unidad and KG.get(unidad): kg=cantidad*KG[unidad]*(0.2 if producto=='cereza' and unidad!='lata' else 1)
    falta=[]
    if intent=='oferta':
        if cantidad is None: falta.append('cantidad')
        if unidad is None: falta.append('unidad')
        elif unidad=='saco': falta.append('unidad_ambigua')
    return {'intencion':intent,'confianza':round(p,3),'cantidad':cantidad,'rango':rango,'unidad':unidad,'producto':producto or ('pergamino' if intent=='oferta' else None),'mes':mes(t,hoy),'kg_pergamino':kg,'falta':falta,
            'requiere_humano': p<0.6 or bool(falta)}
