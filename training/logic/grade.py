import json, numpy as np
C=json.load(open('/home/claude/hack/logic/config.json'))
CLASSES=list(C['clases'].keys())
def grade_from(full, cat1):
    for g in C['grados']:
        if full<=g['max_defectos'] and (g['cat1_permitidos'] or cat1<0.5): return g['id']
def evaluate(soft_counts, n_dudosos=0, draws=4000, seed=0):
    """soft_counts: dict clase->conteo esperado (suma de probabilidades). Devuelve distribución de grados."""
    rng=np.random.default_rng(seed); a=np.array([soft_counts.get(c,0)+(1.0 if c=='sano' else 0.02) for c in CLASSES])
    P=rng.dirichlet(a,draws); N=C['granos_en_350g']
    full=np.zeros(draws); cat1=np.zeros(draws)
    for i,c in enumerate(CLASSES):
        info=C['clases'][c]
        if info['cat']==0: continue
        full+=P[:,i]*N/info['granos_por_defecto']
        if info['cat']==1: cat1+=P[:,i]*N
    gr=[grade_from(f,k) for f,k in zip(full,cat1)]
    dist={g['id']:gr.count(g['id'])/draws for g in C['grados']}
    best=max(dist,key=dist.get); n=sum(soft_counts.values())
    seguro= dist[best]>=C['umbral_confianza_grado'] and n_dudosos/max(n,1)<=C['umbral_dudosos']
    motivo=None
    if n_dudosos/max(n,1)>C['umbral_dudosos']: motivo='granos_dudosos'
    elif dist[best]<C['umbral_confianza_grado']: motivo='grado_incierto'
    if best=='especial': seguro=False; motivo='especial_requiere_catador'
    return {'motivo':motivo,'grado':best,'prob':round(dist[best],3),'dist':{k:round(v,3) for k,v in dist.items()},'defectos_350g_mediana':float(np.median(full)),
            'defectos_350g_p10_p90':[float(np.percentile(full,10)),float(np.percentile(full,90))],'n_granos':round(n,1),'dudosos':n_dudosos,'seguro':bool(seguro)}
def precio(grado_ids):
    p=C['precio']; base=p['arabica_usd_kg']
    out=[]
    for g in grado_ids:
        usd_oro=base+p['diferencial_usd_kg_oro'][g]
        soles_kg_perg=usd_oro*p['rendimiento_oro_pergamino']*p['tipo_cambio']*p['participacion_productor']
        out.append(soles_kg_perg*46)
    return round(min(out)/10)*10, round(max(out)/10)*10
def precio_rango(res):
    gs=[g for g,v in res['dist'].items() if v>=0.15] or [res['grado']]
    return precio(gs)
if __name__=='__main__':
    for sc in [{'sano':97,'broca_leve':2,'inmaduro':1},{'sano':90,'negro':3,'vinagre':2,'broca_severa':5},{'sano':100}]:
        r=evaluate(sc); print(sc,'->',r['grado'],r['prob'],r['dist'],round(r['defectos_350g_mediana']),precio_rango(r),'seguro' if r['seguro'] else 'NO SEGURO')
