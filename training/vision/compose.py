import cv2, numpy as np, json, os, random
B='/home/claude/hack/vision'
meta=json.load(open(f'{B}/beans.json'))
rng=random.Random(42)
# stratified split
by={}
for m in meta: by.setdefault(m['cls'],[]).append(m)
split={}
for c,L in by.items():
    L=sorted(L,key=lambda m:m['file']); rng.shuffle(L); n=max(3,round(0.3*len(L)))
    for i,m in enumerate(L): split[m['file']]='test' if i<n else 'train'
json.dump(split,open(f'{B}/split.json','w'))

def background(W,H,r):
    base=r.uniform(205,245); tint=np.array([r.uniform(-8,8),r.uniform(-5,5),r.uniform(-10,4)])
    img=np.ones((H,W,3),np.float32)*base+tint
    # lighting gradient
    gx=np.linspace(0,1,W)[None,:,None]; gy=np.linspace(0,1,H)[:,None,None]
    a,b=r.uniform(-30,30),r.uniform(-30,30); img+=a*gx+b*gy
    # paper texture noise
    n=np.random.default_rng(r.randint(0,10**9)).normal(0,1,(H//8,W//8)).astype(np.float32)
    n=cv2.resize(n,(W,H),interpolation=cv2.INTER_CUBIC)*r.uniform(1,4); img+=n[...,None]
    return img

def compose(n_beans, defect_rate, pool, r, W=1600, H=1200, scale_len=None):
    img=background(W,H,r); gt=[]
    cols=int(np.ceil(np.sqrt(n_beans*W/H))); rows=int(np.ceil(n_beans/cols))
    cw,ch=W/cols,H/rows; L=scale_len or 0.62*min(cw,ch)
    sanos=[m for m in pool if m['cls']=='sano']; defs=[m for m in pool if m['cls']!='sano']
    cells=[(i,j) for i in range(rows) for j in range(cols)]; r.shuffle(cells)
    for k in range(n_beans):
        m=r.choice(defs) if r.random()<defect_rate else r.choice(sanos)
        rgba=cv2.imread(f'{B}/beans/{m["file"]}',cv2.IMREAD_UNCHANGED)
        s=L*r.uniform(0.85,1.1)/m['px_len']; rgba=cv2.resize(rgba,None,fx=s,fy=s,interpolation=cv2.INTER_AREA)
        ang=r.uniform(0,360); h,w=rgba.shape[:2]; M=cv2.getRotationMatrix2D((w/2,h/2),ang,1)
        c,sn=abs(M[0,0]),abs(M[0,1]); nw,nh=int(h*sn+w*c),int(h*c+w*sn); M[0,2]+=nw/2-w/2; M[1,2]+=nh/2-h/2
        rgba=cv2.warpAffine(rgba,M,(nw,nh),borderValue=(0,0,0,0))
        i,j=cells[k]; cx=int((j+0.5)*cw+r.uniform(-0.15,0.15)*cw); cy=int((i+0.5)*ch+r.uniform(-0.15,0.15)*ch)
        x0,y0=cx-nw//2,cy-nh//2
        if x0<0 or y0<0 or x0+nw>W or y0+nh>H: continue
        a=(rgba[...,3:4].astype(np.float32)/255)
        # soft shadow
        sh=cv2.GaussianBlur(a[...,0],(0,0),4)[...,None]*0.25; dx,dy=4,5
        ys,xs=slice(y0+dy,min(H,y0+dy+nh)),slice(x0+dx,min(W,x0+dx+nw))
        img[ys,xs]*= (1-sh[:ys.stop-ys.start,:xs.stop-xs.start])
        img[y0:y0+nh,x0:x0+nw]=img[y0:y0+nh,x0:x0+nw]*(1-a)+rgba[...,:3].astype(np.float32)*a
        gt.append({'cls':m['cls'],'cx':cx,'cy':cy})
    # camera effects: exposure, color temp, blur, noise, jpeg
    img=img*r.uniform(0.85,1.1)+np.array([r.uniform(-10,10),0,r.uniform(-10,10)])
    img=np.clip(img,0,255).astype(np.uint8)
    if r.random()<0.6: img=cv2.GaussianBlur(img,(0,0),r.uniform(0.3,1.5))
    img=np.clip(img+np.random.default_rng(r.randint(0,10**9)).normal(0,r.uniform(1,5),img.shape),0,255).astype(np.uint8)
    ok,enc=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,r.randint(55,90)]); img=cv2.imdecode(enc,1)
    return img,gt

if __name__=='__main__':
    os.makedirs(f'{B}/muestras',exist_ok=True)
    test=[m for m in meta if split[m['file']]=='test']
    out=[]
    for k in range(60):
        r=random.Random(1000+k); rate=r.choice([0.02,0.05,0.08,0.12,0.2,0.3])
        img,gt=compose(100,rate,test,r)
        name=f'muestra_{k:02d}.jpg'; cv2.imwrite(f'{B}/muestras/{name}',img); out.append({'file':name,'defect_rate':rate,'beans':gt})
    json.dump(out,open(f'{B}/muestras/gt.json','w')); print('ok',len(out))
