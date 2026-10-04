import sys, json, random, concurrent.futures as cf
sys.path.insert(0,'/home/claude/hack'); from llm import ask, extract_json, USAGE
SYS="""Generas datos SINTÉTICOS para entrenar un lector de SMS de una cooperativa cafetalera del Cusco (Perú), La Convención. Los socios son pequeños productores; escriben SMS cortos desde teléfonos básicos, con faltas de ortografía, abreviaturas, sin tildes, mezclando español y quechua cusqueño. Devuelve SOLO un arreglo JSON."""
TPL="""Genera {n} SMS distintos. Estilo de este lote: {style}. Idioma: {lang}.
Mezcla de intenciones: ~65% "oferta" (avisa cuánto café tendrá y cuándo), ~15% "consulta_precio", ~10% "ayuda" (pide visita técnica o reporta problema: broca, roya, secado), ~10% "otro" (saludos, cosas no relacionadas).
Unidades posibles: quintal/qq (46 kg), kg/kilos, arroba (11.5 kg), saco (ambiguo), lata (de cereza). Producto: pergamino (por defecto) o cereza. Incluye a veces números en palabras (español o quechua: huk, iskay, kimsa, tawa, pichqa, suqta, qanchis, pusaq, isqun, chunka), fechas como "fin de mes", "pa junio", "la otra semana", "15 de julio", "despues de la fiesta".
Cada elemento: {{"texto": str, "idioma": "es"|"qu"|"mix", "intencion": "oferta"|"consulta_precio"|"ayuda"|"otro", "cantidad": number|null, "unidad": "quintal"|"kg"|"arroba"|"saco"|"lata"|null, "producto": "pergamino"|"cereza"|null, "cuando_texto": str|null, "mes": 1-12|null}}
Reglas: "cantidad" es el número tal como lo dice el socio (en la unidad que usa). "mes" solo si se puede deducir; hoy es octubre de 2026 (la cosecha principal es abril-septiembre, hay rezagos hasta noviembre). No repitas plantillas; varía longitud (3 a 25 palabras)."""
STYLES=["como lo dictaria una persona mayor a su nieto que escribe por ella","con jerga local de la selva de Cusco","muy corto tipo telegrama","con faltas de ortografía y sin espacios a veces","cortés y largo, con saludo","con números en palabras","con códigos o nombre del socio al inicio","con dudas o rangos (entre 5 y 6 qq)","con cambios de unidad o correcciones (no, mejor 8)","mezclando quechua y castellano a mitad de frase"]
LANGS=["español rural","quechua cusqueño (Collao/Qosqo) con préstamos del español","mezcla quechua-español"]
jobs=[(STYLES[i%len(STYLES)],LANGS[i%len(LANGS)]) for i in range(int(sys.argv[2]))]
model=sys.argv[3] if len(sys.argv)>3 else 'claude-haiku-4-5-20251001'
def run(j):
    try: return extract_json(ask(TPL.format(n=50,style=j[0],lang=j[1]),model=model,system=SYS))
    except Exception as e: print('err',e); return []
out=[]
with cf.ThreadPoolExecutor(8) as ex:
    for res in ex.map(run,jobs): out+=res
for o in out: o['sintetico']=True; o['generador']=model
json.dump(out,open(sys.argv[1],'w'),ensure_ascii=False,indent=0)
print(len(out),USAGE)
