import json, os, urllib.request, concurrent.futures as cf
for l in open('/home/claude/hack/.env'):
    if '=' in l: k,v=l.strip().split('=',1); os.environ[k]=v
M=json.load(open('/home/claude/hack/content/mensajes.json'))
VOICE='FGY2WhTYpPnrIDTdsKH5'; OUT='/home/claude/hack/app/audio2'
def tts(job):
    lang,k,txt=job; fn=f'{OUT}/{lang}/{k}.mp3'
    if os.path.exists(fn) and os.path.getsize(fn)>1000: return 'skip'
    body=json.dumps({'text':txt,'model_id':'eleven_v3','language_code':lang,'voice_settings':{'stability':0.5,'similarity_boost':0.8}}).encode()
    req=urllib.request.Request(f'https://api.elevenlabs.io/v1/text-to-speech/{VOICE}?output_format=mp3_22050_32',data=body,headers={'xi-api-key':os.environ['ELEVENLABS_API_KEY'],'content-type':'application/json'})
    for a in range(3):
        try: open(fn,'wb').write(urllib.request.urlopen(req,timeout=180).read()); return 'ok'
        except Exception as e: err=e
    return f'ERR {k} {err}'
jobs=[(l,k,v) for l in ('es','en') for k,v in M[l].items()]
with cf.ThreadPoolExecutor(3) as ex: res=list(ex.map(tts,jobs))
print(len(jobs),[r for r in res if r.startswith('ERR')], sum(len(j[2]) for j in jobs),'chars')
