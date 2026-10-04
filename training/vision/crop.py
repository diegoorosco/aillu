import cv2, numpy as np, os, json
SRC='/home/claude/hack/data/GreenCoffeeBeansDataset/ImageDataset'
OUT='/home/claude/hack/vision/beans'; os.makedirs(OUT,exist_ok=True)
CLS={'Normales':'sano','Negros':'negro','MarronAVinagre':'vinagre','CerezaSeca':'cereza_seca','DXHongo':'hongo',
     'BrocadoSevero':'broca_severa','BrocadoLeve':'broca_leve','Concha':'concha','Inmaduro':'inmaduro',
     'Pergamino':'pergamino','PMordidoCortado':'partido_cortado'}
meta=[]
for fn,c in CLS.items():
    im=cv2.imread(f'{SRC}/{fn}.jpg')
    g=cv2.GaussianBlur(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(5,5),0)
    _,th=cv2.threshold(g,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    th=cv2.morphologyEx(th,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    th=cv2.morphologyEx(th,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    # white balance from background
    bg=im[th==0].reshape(-1,3).astype(np.float32); gain=np.median(bg,0); imwb=np.clip(im.astype(np.float32)*(235.0/gain),0,255).astype(np.uint8)
    cnts,_=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    areas=[cv2.contourArea(x) for x in cnts]; med=np.median(sorted(areas,reverse=True)[:20])
    k=0
    for cn,a in zip(cnts,areas):
        if a<0.25*med: continue
        x,y,w,h=cv2.boundingRect(cn); p=int(0.15*max(w,h))
        x0,y0=max(0,x-p),max(0,y-p); x1,y1=min(im.shape[1],x+w+p),min(im.shape[0],y+h+p)
        mask=np.zeros(th.shape,np.uint8); cv2.drawContours(mask,[cn],-1,255,-1)
        crop=imwb[y0:y1,x0:x1]; m=mask[y0:y1,x0:x1]
        rgba=np.dstack([crop,m]); name=f'{c}_{k:03d}.png'
        cv2.imwrite(f'{OUT}/{name}',rgba); meta.append({'file':name,'cls':c,'src':fn,'px_len':max(w,h)}); k+=1
    print(c,k)
json.dump(meta,open('/home/claude/hack/vision/beans.json','w'))
