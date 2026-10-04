import json, random, sys
sys.path.insert(0,'/home/claude/hack/sms'); sys.path.insert(0,'/home/claude/hack/logic')
from parser import parse
r=random.Random(7)
NOM=['Noor','Rosa','Juana','Maria','Lucia','Elena','Teofila','Justina','Felicitas','Marcelina','Julio','Pedro','Santiago','Hilario','Mario','Wilber','Rene','Edgar','Fortunato','Valentin','Celestino','Ruben','Alejandro','Victor','Nilda']
APE=['Quispe','Mamani','Huaman','Condori','Ccahuana','Choque','Puma','Callañaupa','Yupanqui','Huallpa','Ttito','Sullca','Apaza','Cusi','Champi','Ccama','Rojas','Flores','Gutierrez','Vargas']
SECT=['Huayopata','Santa Teresa','Maranura','Echarate','Quellouno','Ocobamba']
G=['especial','premium','estandar','bajo','fuera']
socios=[]
for i in range(50):
    nom='Noor' if i==0 else r.choice(NOM[1:]); ape=r.choice(APE)+' '+r.choice(APE)
    ha=round(r.uniform(0.8,4.5),1)
    hist=[r.choices(G,[3,15,45,27,10])[0] for _ in range(3)]
    cumplimiento=round(min(1.15,max(0.55,r.gauss(0.88,0.12))),2)  # entregado / reportado, historico (sintetico)
    socios.append({'id':f'SOC-{101+i}','nombre':f'{nom} {ape}','sector':r.choice(SECT),'hectareas':ha,'idioma':r.choices(['qu','es'],[6,4])[0],
                   'telefono':f'+51 9{r.randint(10,99)} *** {r.randint(100,999)}','consentimiento_datos':r.random()>0.06,
                   'historial_grados':hist,'cumplimiento_hist':cumplimiento,'sintetico':True})
socios[0].update({'nombre':'Noor Quispe Huaman','sector':'Huayopata','hectareas':2.0,'idioma':'qu','consentimiento_datos':True,'historial_grados':['estandar','estandar','bajo']})
sms=[x for x in json.load(open('/home/claude/hack/sms/sms_train.json')) if x['intencion']=='oferta' and x['unidad'] in ('quintal','arroba','kg') and x['cantidad']]
reportes=[]; tot=0
for s in socios[1:]:
    if r.random()<0.62:
        m=r.choice(sms); p=parse(m['texto'])
        if p['kg_pergamino'] and 20<=p['kg_pergamino']<=1500 and tot<12600:
            g=r.choices(['premium','estandar','bajo'],[2,6,2])[0]
            reportes.append({'socio':s['id'],'sms':m['texto'],'kg':p['kg_pergamino'],'mes':p['mes'] or 11,'grado_ultima_muestra':g}); tot+=p['kg_pergamino']
json.dump({'socios':socios,'reportes':reportes,'comprador':{'nombre':'Comprador aliado (ejemplo)','idioma':'en','lote_min_qq':300,'grado_min':'estandar','entrega':'noviembre 2026','_supuesto':True}},open('coop.json','w'),ensure_ascii=False,indent=0)
print(len(reportes),'reportes',round(tot/46),'qq')
