import json, sys, os, torch, numpy as np, soundfile as sf
sys.path.insert(0,'/home/claude/hack'); from llm import ask, extract_json, USAGE
T=json.load(open('es/textos.json'))
tpl={k:v for k,v in T.items() if not k.startswith('n_') and not k.startswith('consejo_') or k=='consejo_intro'}
if not os.path.exists('qu_textos.json'):
    qu=extract_json(ask("Traduce al quechua cusqueño (Qusqu-Qullaw, alfabeto oficial) estos mensajes de voz de una cooperativa cafetalera a sus socios. Frases cortas y naturales del campo; usa préstamos del español para términos técnicos (cooperativa, categoría, Premium, Especial, quintal, pergamino, catador, lote). Devuelve SOLO un JSON con las mismas claves.\n"+json.dumps(tpl,ensure_ascii=False),model='claude-sonnet-4-6',max_tokens=3000,temperature=0.2))
    adv=json.load(open('/home/claude/hack/content/consejos.json'))
    for k,v in adv.items(): qu['consejo_'+k]=v['audio_qu']
    NUMQ={'q_1':'huk','q_2':'iskay','q_3':'kimsa','q_4':'tawa','q_5':'pichqa','q_6':'suqta','q_7':'qanchis','q_8':'pusaq','q_9':'isqun','q_chunka':'chunka','q_pachak':'pachak','q_waranqa':'waranqa','q_sol':'sol'}
    qu.update(NUMQ); json.dump(qu,open('qu_textos.json','w'),ensure_ascii=False,indent=0)
qu=json.load(open('qu_textos.json'))
from transformers import VitsModel, AutoTokenizer
tok=AutoTokenizer.from_pretrained('facebook/mms-tts-quz'); m=VitsModel.from_pretrained('facebook/mms-tts-quz').eval()
os.makedirs('qu',exist_ok=True); torch.manual_seed(0)
for k,txt in qu.items():
    fn=f'qu/{k}.wav'
    if os.path.exists(fn): continue
    with torch.no_grad(): w=m(**tok(txt.lower(),return_tensors='pt')).waveform[0].numpy()
    sf.write(fn,w,m.config.sampling_rate)
print(len(qu),'clips qu',USAGE)
