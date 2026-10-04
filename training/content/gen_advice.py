import sys, json; sys.path.insert(0,'/home/claude/hack'); from llm import ask, extract_json, USAGE
BASE="""Eres agrónomo especialista en café arábica de La Convención (Cusco, Perú) y redactas mensajes para pequeños productores que reciben SMS en teléfonos básicos y audios. Lenguaje simple, respetuoso, concreto, sin tecnicismos. NO inventes cifras que no estén en las pautas. Si recomiendas algo que requiere insumos o químicos, sugiere consultar al técnico de la cooperativa.
Pautas técnicas validadas (usa SOLO esto):
- negro: viene de cerezas sobremaduras, caídas al suelo o estrés por sequía. Cosechar solo maduro, no mezclar café del suelo, separar los flotes en agua.
- vinagre: fermentación excesiva o demora entre cosecha y despulpado. Despulpar el mismo día, controlar el tiempo de fermentación (revisar el mucílago hasta que se desprenda), lavar bien, tanques y despulpadora limpios.
- cereza_seca: cerezas secas en la planta o que no pasaron por la despulpadora. Separar por flotación antes de despulpar, revisar la despulpadora.
- hongo: secado lento o almacenamiento húmedo. Secar en capa delgada, remover varias veces al día, guardar seco en sacos limpios sobre tarimas, lejos de paredes y del piso.
- broca_severa y broca_leve: daño de la broca del café (insecto). Hacer la cosecha sanitaria (repase), recoger frutos del suelo y de la planta después de la cosecha, usar trampas con alcohol, pedir al técnico control biológico (hongo Beauveria). La broca también baja el rendimiento de la cosecha.
- concha: grano malformado; se separa con flotación y zarandas. Poco control en campo.
- inmaduro: cosecha de frutos verdes. Cosechar solo frutos maduros (rojos), hacer varias pasadas.
- pergamino: grano mal trillado (problema de la trilladora, no del productor). Avisar a la cooperativa.
- partido_cortado: despulpadora mal calibrada o golpes; también sobresecado. Calibrar la despulpadora, no secar de más.
"""
P1=BASE+"""
Para cada clase [negro, vinagre, cereza_seca, hongo, broca_severa, broca_leve, concha, inmaduro, pergamino, partido_cortado] devuelve un objeto JSON con:
"sms_es": consejo en español, máximo 150 caracteres, sin tildes ni ñ (compatibilidad GSM), empieza diciendo qué se encontró.
"audio_es": guion de audio en español, 35-50 palabras, cálido, dice la causa probable y 2 acciones concretas.
"causa_es": causa en 1 frase corta.
"accion_campo": true si el productor puede actuar en su chacra, false si no.
"buyer_en": descripción técnica en inglés para el comprador, máx 15 palabras.
Devuelve {"clase": {...}, ...}."""
P2="""Traduce al quechua cusqueño (Qusqu-Qullaw, alfabeto oficial de 3 vocales) los siguientes textos para productores cafetaleros. Usa préstamos del español cuando el término técnico no exista (broca, cooperativa, pergamino, despulpadora). Frases cortas y naturales, como se habla en el campo. Devuelve el MISMO JSON con las claves "sms_qu" y "audio_qu" añadidas a cada clase y marca "revision_hablante": true. Solo JSON.
"""
adv=extract_json(ask(P1,model='claude-sonnet-4-6',max_tokens=6000,temperature=0.3))
sub={k:{'sms_es':v['sms_es'],'audio_es':v['audio_es']} for k,v in adv.items()}
qu=extract_json(ask(P2+json.dumps(sub,ensure_ascii=False),model='claude-sonnet-4-6',max_tokens=8000,temperature=0.2))
for k in adv: adv[k].update({kk:qu.get(k,{}).get(kk) for kk in ('sms_qu','audio_qu')}); adv[k]['revision_hablante']=True; adv[k]['borrador_validar_agronomo']=True
json.dump(adv,open('consejos.json','w'),ensure_ascii=False,indent=1); print(USAGE)
