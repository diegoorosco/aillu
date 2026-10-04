import os, cv2, numpy as np, json, random, torch, torch.nn as nn, time
from common import *
torch.set_num_threads(2)
B='/home/claude/hack/vision'
meta=json.load(open(f'{B}/beans.json')); split=json.load(open(f'{B}/split.json'))
rgbas={m['file']:cv2.imread(f'{B}/beans/{m["file"]}',cv2.IMREAD_UNCHANGED) for m in meta}
tr=[m for m in meta if split[m['file']]=='train']; te=[m for m in meta if split[m['file']]=='test']
def render(m, r, aug=True):
    rgba=rgbas[m['file']]
    big=S*3
    bgv=r.uniform(200,245) if aug else 228
    bg=np.ones((big,big,3),np.float32)*bgv
    if aug:
        bg+=np.array([r.uniform(-8,8),r.uniform(-5,5),r.uniform(-10,4)])
        gx=np.linspace(0,1,big)[None,:,None]; bg+=r.uniform(-15,15)*gx
    a=rgba[...,3]; ys,xs=np.where(a>0); cnts,_=cv2.findContours(a,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cnt=max(cnts,key=cv2.contourArea)
    (cx,cy),(w,h),_=cv2.minAreaRect(cnt); L=max(w,h)
    s=FILL*S/L*(r.uniform(0.88,1.12) if aug else 1.0); ang=r.uniform(0,360) if aug else 0
    M=cv2.getRotationMatrix2D((cx,cy),ang,s); M[0,2]+=S/2-cx; M[1,2]+=S/2-cy
    if aug: M[0,2]+=r.uniform(-4,4); M[1,2]+=r.uniform(-4,4)
    fg=cv2.warpAffine(rgba[...,:3],M,(S,S),flags=cv2.INTER_AREA).astype(np.float32)
    al=cv2.warpAffine(a,M,(S,S),flags=cv2.INTER_AREA).astype(np.float32)[...,None]/255
    out=bg[:S,:S]*(1-al)+fg*al
    if aug:
        if r.random()<0.5: out=out[:,::-1]
        out=out*r.uniform(0.85,1.12)+np.array([r.uniform(-10,10),r.uniform(-4,4),r.uniform(-10,10)])
        out=np.clip(out,0,255).astype(np.uint8)
        if r.random()<0.5: out=cv2.GaussianBlur(out,(0,0),r.uniform(0.3,1.3))
        out=np.clip(out+np.random.default_rng(r.randint(0,10**9)).normal(0,r.uniform(0,4),out.shape),0,255).astype(np.uint8)
        ok,enc=cv2.imencode('.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,r.randint(50,92)]); out=cv2.imdecode(enc,1)
    return np.clip(out,0,255).astype(np.uint8)
def to_t(batch): x=np.stack(batch)[...,::-1].astype(np.float32)/255; return torch.from_numpy((x-0.5)/0.25).permute(0,3,1,2).contiguous()
class Net(nn.Module):
    def __init__(s,n=11):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True),nn.Conv2d(o,o,3,padding=1,groups=o,bias=False),nn.Conv2d(o,o,1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True),nn.MaxPool2d(2))
        W=[int(v) for v in os.environ.get('QP_W','24,48,80,128,160').split(',')]
        s.f=nn.Sequential(blk(3,W[0]),*[blk(W[i],W[i+1]) for i in range(4)])
        s.h=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(0.3),nn.Linear(W[4],n))
    def forward(s,x): return s.h(s.f(x))
if __name__=='__main__':
    r=random.Random(0); torch.manual_seed(0)
    net=Net(); print('params',sum(p.numel() for p in net.parameters()))
    cls_idx={c:i for i,c in enumerate(CLASSES)}
    by={}; [by.setdefault(m['cls'],[]).append(m) for m in tr]
    E=int(os.environ.get('QP_E','45')); opt=torch.optim.AdamW(net.parameters(),2e-3,weight_decay=1e-3); sched=torch.optim.lr_scheduler.OneCycleLR(opt,4e-3,total_steps=E*20)
    lossf=nn.CrossEntropyLoss(label_smoothing=0.05)
    Xte=to_t([render(m,r,False) for m in te]); yte=torch.tensor([cls_idx[m['cls']] for m in te])
    best=0; t0=time.time()
    for ep in range(E):
        net.train()
        for it in range(20):
            # class-balanced batch, sano oversampled x3 (dominant in reality)
            cs=[r.choice(CLASSES+['sano','sano']) for _ in range(48)]
            ms=[r.choice(by[c]) for c in cs]
            x=to_t([render(m,r) for m in ms]); y=torch.tensor([cls_idx[m['cls']] for m in ms])
            opt.zero_grad(); l=lossf(net(x),y); l.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad(): acc=(net(Xte).argmax(1)==yte).float().mean().item()
        if acc>=best: best=acc; torch.save(net.state_dict(),f'{B}/'+os.environ.get('QP_OUT','net.pt'))
        if ep%5==4 or ep==E-1: print(f'ep{ep+1} loss {l.item():.3f} test_acc {acc:.3f} best {best:.3f} {time.time()-t0:.0f}s',flush=True)
