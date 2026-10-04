import cv2, numpy as np
import os
S=int(os.environ.get("QP_S","96")); FILL=0.62
CLASSES=['sano','negro','vinagre','cereza_seca','hongo','broca_severa','broca_leve','concha','inmaduro','pergamino','partido_cortado']
def white_balance(img, mask_bg):
    bg=img[mask_bg].reshape(-1,3).astype(np.float32); g=np.median(bg,0)+1e-3
    return np.clip(img.astype(np.float32)*(235.0/g),0,255).astype(np.uint8)
def segment(img):
    g=cv2.GaussianBlur(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),(5,5),0)
    _,th=cv2.threshold(g,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    th=cv2.morphologyEx(th,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    th=cv2.morphologyEx(th,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    cnts,_=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    areas=np.array([cv2.contourArea(c) for c in cnts]) if cnts else np.array([])
    if len(areas)==0: return th,[]
    med=np.median(areas[areas>np.percentile(areas,50)]) if len(areas)>4 else np.median(areas)
    keep=[c for c,a in zip(cnts,areas) if 0.2*med<a<3.0*med]
    return th,keep
def crop_norm(img, cnt):
    """crop centred on contour, scale so long side = FILL*S, output SxS BGR"""
    (cx,cy),(w,h),_=cv2.minAreaRect(cnt); L=max(w,h,1)
    s=FILL*S/L; M=np.float32([[s,0,S/2-s*cx],[0,s,S/2-s*cy]])
    return cv2.warpAffine(img,M,(S,S),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_REPLICATE)
