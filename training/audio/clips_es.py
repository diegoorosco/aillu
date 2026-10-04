import json, os, urllib.request, concurrent.futures as cf
for l in open('/home/claude/hack/.env'):
    if '=' in l: k,v=l.strip().split('=',1); os.environ[k]=v
VOICE='FGY2WhTYpPnrIDTdsKH5'
T={
 'saludo':'Hola, le habla su cooperativa de café. Este es el resultado de su muestra.',
 'grado_especial':'Su café salió categoría Especial.','grado_premium':'Su café salió categoría Premium.',
 'grado_estandar':'Su café salió categoría Estándar.','grado_bajo':'Su café salió categoría Bajo estándar.','grado_fuera':'Su café salió fuera de grado.',
 'precio_intro':'El precio de referencia está entre','y':'y','soles_qq':'soles por quintal de pergamino.',
 'no_oferta':'Este precio es solo una referencia, no es una oferta. Usted decide a quién vender.',
 'duda':'No estamos seguros del resultado de su muestra. Por favor, lleve su café al catador de la cooperativa para que lo revise.',
 'consejo_intro':'Un consejo para mejorar su precio en la próxima cosecha.',
 'sms_ok':'Gracias, registramos su aviso de cosecha.',
 'sms_aclarar':'No entendimos bien su mensaje. ¿Cuántos quintales tendrá, y para qué mes?',
 'sms_saco':'¿De cuántos kilos es su saco?',
 'lote':'Buena noticia: la cooperativa ya juntó el lote para el comprador. Pronto le avisamos la fecha de entrega.',
}
NUM={'n_mil':'mil','n_dosmil':'dos mil','n_100':'cien','n_ciento':'ciento','n_200':'doscientos','n_300':'trescientos','n_400':'cuatrocientos','n_500':'quinientos','n_600':'seiscientos','n_700':'setecientos','n_800':'ochocientos','n_900':'novecientos',
     'n_10':'diez','n_20':'veinte','n_30':'treinta','n_40':'cuarenta','n_50':'cincuenta','n_60':'sesenta','n_70':'setenta','n_80':'ochenta','n_90':'noventa'}
adv=json.load(open('/home/claude/hack/content/consejos.json'))
for k,v in adv.items(): T['consejo_'+k]=v['audio_es']
T.update(NUM)
def tts(item):
    k,txt=item; fn=f'es/{k}.mp3'
    if os.path.exists(fn): return k,'skip'
    body=json.dumps({'text':txt,'model_id':'eleven_multilingual_v2','language_code':'es','voice_settings':{'stability':0.6,'similarity_boost':0.75}}).encode()
    req=urllib.request.Request(f'https://api.elevenlabs.io/v1/text-to-speech/{VOICE}?output_format=mp3_22050_32',data=body,headers={'xi-api-key':os.environ['ELEVENLABS_API_KEY'],'content-type':'application/json'})
    open(fn,'wb').write(urllib.request.urlopen(req,timeout=120).read()); return k,'ok'
os.makedirs('es',exist_ok=True)
with cf.ThreadPoolExecutor(3) as ex:
    for r in ex.map(tts,T.items()): pass
json.dump(T,open('es/textos.json','w'),ensure_ascii=False,indent=0)
print(len(T),'clips', sum(len(v) for v in T.values()),'caracteres')
