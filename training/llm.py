import os, json, urllib.request, time
def load_env():
    for l in open('/home/claude/hack/.env'):
        if '=' in l: k,v=l.strip().split('=',1); os.environ[k]=v
load_env()
USAGE={'in':0,'out':0}
def ask(prompt, model='claude-haiku-4-5-20251001', max_tokens=8000, system=None, temperature=1.0):
    body={'model':model,'max_tokens':max_tokens,'temperature':temperature,'messages':[{'role':'user','content':prompt}]}
    if system: body['system']=system
    req=urllib.request.Request('https://api.anthropic.com/v1/messages',data=json.dumps(body).encode(),headers={
        'x-api-key':os.environ['ANTHROPIC_API_KEY'],'anthropic-version':'2023-06-01',
        'anthropic-workspace-id':os.environ['ANTHROPIC_WORKSPACE_ID'],'content-type':'application/json'})
    for a in range(4):
        try:
            d=json.load(urllib.request.urlopen(req,timeout=300)); break
        except Exception as e:
            if a==3: raise
            time.sleep(5*(a+1))
    USAGE['in']+=d['usage']['input_tokens']; USAGE['out']+=d['usage']['output_tokens']
    return ''.join(b.get('text','') for b in d['content'])
def extract_json(t):
    s=min([i for i in (t.find('['),t.find('{')) if i>=0]); e=max(t.rfind(']'),t.rfind('}'))
    return json.loads(t[s:e+1])
