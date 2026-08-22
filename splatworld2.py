#!/usr/bin/env python3
"""SplatWorld2 — no-retraining local transport + identity-lock explorer.

Uses the original SplatWorld `splat_decoder.onnx`.  The model is not changed.
At startup we probe local latent directions and keep the ones whose +/- image
change is best explained by optical flow (transport) instead of repainting.
During live motion, detail is always warped from ONE immutable anchor frame;
the previous displayed frame is never reused, so recursive blur cannot build.
"""
from __future__ import annotations
import argparse, math, os, time
from pathlib import Path
import numpy as np

LATENT = 128


def cv2():
    import cv2 as cv
    return cv


def rgb(x):
    return np.clip(np.transpose(x, (1, 2, 0)).astype(np.float32), 0, 1)


def gray(x):
    return cv2().cvtColor(x.astype(np.float32), cv2().COLOR_RGB2GRAY)


def find_model(explicit=None):
    here = Path(__file__).resolve().parent
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates += [here/'splat_decoder.onnx', here.parent/'SplatWorld'/'splat_decoder.onnx',
                   Path.cwd()/'splat_decoder.onnx', Path.cwd().parent/'SplatWorld'/'splat_decoder.onnx']
    for p in candidates:
        if p.is_file(): return p
    raise FileNotFoundError('splat_decoder.onnx not found; pass --model PATH')


class OnnxDecoder:
    def __init__(self, path, cpu=False):
        import onnxruntime as ort
        providers = ['CPUExecutionProvider']
        if not cpu and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        self.s = ort.InferenceSession(str(path), providers=providers)
        self.i = self.s.get_inputs()[0].name
        self.o = self.s.get_outputs()[0].name
        self.backend = self.s.get_providers()[0]
    def __call__(self, z):
        z = np.ascontiguousarray(z, dtype=np.float32)
        return self.s.run([self.o], {self.i: z})[0]


class MockDecoder:
    """Synthetic two-transport-axis decoder for --selftest."""
    def __init__(self, h=64):
        self.h=h
        y,x=np.mgrid[0:h,0:h].astype(np.float32)
        r=np.sqrt((x-.46*h)**2+(y-.52*h)**2)
        self.base=(.25+.30*np.exp(-(r/(.30*h))**2)+.12*np.sin(.55*x)*np.exp(-((y-.60*h)/(.18*h))**2)).astype(np.float32)
        rng=np.random.default_rng(3)
        self.noise=rng.standard_normal((6,h,h)).astype(np.float32)
        cv=cv2()
        for i in range(6): self.noise[i]=cv.GaussianBlur(self.noise[i],(0,0),2)
        self.noise/=np.std(self.noise,axis=(1,2),keepdims=True)+1e-6
    def __call__(self,zs):
        cv=cv2(); out=[]
        for z in np.asarray(zs,np.float32):
            M=np.array([[1,0,2.7*z[0]],[0,1,2.4*z[1]]],np.float32)
            g=cv.warpAffine(self.base,M,(self.h,self.h),borderMode=cv.BORDER_REFLECT)
            if np.any(z[2:8]): g += .012*np.tensordot(z[2:8],self.noise,axes=(0,0))
            g=np.clip(g,0,1); out.append(np.stack([g,g*.95+.02,g*.90+.04],axis=0))
        return np.asarray(out,np.float32)


def orthodirs(seed, n):
    rng=np.random.default_rng(seed); n=int(np.clip(n,2,LATENT))
    q,_=np.linalg.qr(rng.standard_normal((LATENT,n)))
    return q[:,:n].T.astype(np.float32)


def warp(src, backward_flow, out_size=None):
    cv=cv2(); h0,w0=backward_flow.shape[:2]
    h,w=src.shape[:2] if out_size is None else (out_size,out_size)
    f=backward_flow.astype(np.float32)
    if (h,w)!=(h0,w0):
        f=cv.resize(f,(w,h)); f[...,0]*=w/w0; f[...,1]*=h/h0
    y,x=np.mgrid[0:h,0:w].astype(np.float32)
    return cv.remap(src,x+f[...,0],y+f[...,1],cv.INTER_LINEAR,borderMode=cv.BORDER_REFLECT)


def transport_metric(a,b):
    """Return score,motion,explained,raw,residual for RGB images a->b."""
    cv=cv2(); ga=gray(a)*255.; gb=gray(b)*255.
    fwd=cv.calcOpticalFlowFarneback(ga,gb,None,.5,3,15,3,5,1.1,0)
    back=cv.calcOpticalFlowFarneback(gb,ga,None,.5,3,15,3,5,1.1,0)
    wa=warp(a,back)
    raw=float(np.mean(np.abs(a-b))); residual=float(np.mean(np.abs(wa-b)))
    motion=float(np.mean(np.linalg.norm(fwd,axis=2)))
    explained=float(np.clip(1-residual/(raw+1e-6),0,1))
    score=motion*explained/(residual+.02)
    return score,motion,explained,raw,residual


def probe(dec,z0,n=32,k=2,eps=.35,seed=7,directions=None,verbose=True):
    D=orthodirs(seed,n) if directions is None else np.asarray(directions,np.float32); n=len(D)
    outs=dec(np.concatenate([z0[None]-eps*D,z0[None]+eps*D],axis=0))
    metrics=[transport_metric(rgb(outs[i]),rgb(outs[n+i])) for i in range(n)]
    order=sorted(range(n),key=lambda i:metrics[i][0],reverse=True)
    chosen=order[:max(1,min(k,n))]
    B=D[chosen]; q,_=np.linalg.qr(B.T); B=q[:,:len(chosen)].T.astype(np.float32)
    if verbose:
        print('idx   score   motion  explained  raw->flow_residual')
        for i in order[:min(10,n)]:
            s,m,e,r,rr=metrics[i]
            print(f'{i:3d} {s:7.3f} {m:7.3f} {100*e:8.1f}%  {r:.4f}->{rr:.4f}')
        print('selected transport axes:',chosen)
    return B,chosen,metrics


class IdentityLock:
    """Current guide + high-frequency detail warped from immutable anchor."""
    def __init__(self,anchor,size=640,anchor_image=None,sigma=1.2,gain=1.0,conf_sigma=.10):
        cv=cv2(); self.anchor=anchor.copy(); self.size=size; self.gain=gain; self.conf_sigma=conf_sigma
        if anchor_image is None:
            tex=cv.resize(anchor,(size,size),interpolation=cv.INTER_LANCZOS4)
        else:
            im=cv.imread(anchor_image)
            if im is None: raise FileNotFoundError(anchor_image)
            im=cv.cvtColor(im,cv.COLOR_BGR2RGB).astype(np.float32)/255.
            h,w=im.shape[:2]; s=min(h,w); im=im[(h-s)//2:(h+s)//2,(w-s)//2:(w+s)//2]
            tex=cv.resize(im,(size,size),interpolation=cv.INTER_LANCZOS4)
        self.texture=tex.copy()
        self.detail=tex-cv.GaussianBlur(tex,(0,0),sigma)
        self.conf=1.0
    def __call__(self,current):
        cv=cv2(); gc=gray(current)*255.; ga=gray(self.anchor)*255.
        back=cv.calcOpticalFlowFarneback(gc,ga,None,.5,3,17,3,5,1.1,0)
        d=warp(self.detail,back,self.size)
        warped_anchor=warp(self.anchor,back)
        err=np.mean(np.abs(warped_anchor-current),axis=2)
        c=np.exp(-err/max(self.conf_sigma,1e-4)).astype(np.float32)
        c=cv.GaussianBlur(c,(0,0),1); self.conf=float(c.mean())
        c=cv.resize(c,(self.size,self.size))[...,None]
        base=cv.resize(current,(self.size,self.size),interpolation=cv.INTER_CUBIC)
        return np.clip(base+self.gain*c*d,0,1)


class Controller:
    def __init__(self,z0,B,span=3):
        self.z0=z0.copy(); self.B=B.copy(); self.a=np.zeros(len(B),np.float32); self.ta=self.a.copy(); self.span=span
    def drag(self,dx,dy,fine=False):
        g=.012*(.2 if fine else 1.)
        if len(self.ta)>0:self.ta[0]+=g*dx
        if len(self.ta)>1:self.ta[1]-=g*dy
        self.ta[:]=np.clip(self.ta,-self.span,self.span)
    def tick(self): self.a += .28*(self.ta-self.a)
    def reset(self): self.ta[:]=0
    def z(self): return (self.z0+self.a@self.B).astype(np.float32)


def selftest():
    ok=True
    def check(name,cond,note=''):
        nonlocal ok; ok &= bool(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name} {note}")
    dec=MockDecoder(); z0=np.zeros(LATENT,np.float32)
    D=np.zeros((8,LATENT),np.float32); D[0,0]=1;D[1,1]=1;D[2:,2:8]=np.eye(6,dtype=np.float32)
    B,sel,_=probe(dec,z0,n=8,k=2,eps=.5,directions=D,verbose=False)
    check('transport axes selected',set(sel)=={0,1},str(sel))
    check('basis orthonormal',np.allclose(B@B.T,np.eye(2),atol=1e-5))
    anchor=rgb(dec(z0[None])[0]); lock=IdentityLock(anchor,size=256)
    before=lock.texture.copy(); moved=rgb(dec((z0+B[0])[None])[0])
    for _ in range(10): out=lock(moved)
    check('render finite',np.isfinite(out).all()); check('immutable anchor',np.array_equal(before,lock.texture))
    c=Controller(z0,B); c.drag(90,-40); [c.tick() for _ in range(20)]
    dz=c.z()-z0; outside=dz-B.T@(B@dz)
    check('control stays in measured subspace',np.linalg.norm(outside)<1e-5,f'{np.linalg.norm(outside):.2e}')
    print('selftest:','ALL PASS' if ok else 'FAILURES ABOVE'); return 0 if ok else 1


def live(dec,args):
    cv=cv2(); rng=np.random.default_rng(args.seed)
    z0=(rng.standard_normal(LATENT)*args.anchor_std).astype(np.float32)
    anchor=rgb(dec(z0[None])[0]); print(f'anchor |z|={np.linalg.norm(z0):.3f}')
    B,_,_=probe(dec,z0,args.probe_dirs,args.k,args.probe_eps,args.seed)
    ctl=Controller(z0,B,args.span)
    lock=IdentityLock(anchor,args.size,args.anchor_image,args.detail_sigma,args.detail_gain,args.confidence_sigma)
    win='SplatWorld2  drag=transport  L=lock/raw  A=auto  R=reset  S=save  Q=quit'
    cv.namedWindow(win,cv.WINDOW_NORMAL); mouse={'b':0,'x':0,'y':0}
    def cb(ev,x,y,flags,_):
        if ev in (cv.EVENT_LBUTTONDOWN,cv.EVENT_RBUTTONDOWN): mouse.update(b=1 if ev==cv.EVENT_LBUTTONDOWN else 2,x=x,y=y)
        elif ev in (cv.EVENT_LBUTTONUP,cv.EVENT_RBUTTONUP): mouse['b']=0
        elif ev==cv.EVENT_MOUSEMOVE and mouse['b']:
            ctl.drag(x-mouse['x'],y-mouse['y'],mouse['b']==2); mouse.update(x=x,y=y)
    cv.setMouseCallback(win,cb)
    locked=True; auto=args.auto; t0=time.time(); frames=0; last=time.time(); fps=0.
    while True:
        if auto and len(ctl.ta)>=2:
            t=time.time()-t0; ctl.ta[0]=args.span*.72*math.sin(.55*t); ctl.ta[1]=args.span*.58*math.sin(.41*t+.8)
        ctl.tick(); guide=rgb(dec(ctl.z()[None])[0])
        shown=lock(guide) if locked else cv.resize(guide,(args.size,args.size),interpolation=cv.INTER_CUBIC)
        im=cv.cvtColor((shown*255).astype(np.uint8),cv.COLOR_RGB2BGR)
        frames+=1; now=time.time()
        if now-last>.5: fps=frames/(now-last);frames=0;last=now
        hud=f"{'LOCK' if locked else 'RAW '} fps {fps:4.1f} a={np.array2string(ctl.a[:2],precision=2)} conf={lock.conf:.2f}"
        cv.putText(im,hud,(12,args.size-15),cv.FONT_HERSHEY_PLAIN,1.2,(0,255,0),1,cv.LINE_AA); cv.imshow(win,im)
        key=cv.waitKeyEx(1)
        if key<0: continue
        k=key&0xff
        if k==ord('q'):break
        if k==ord('l'):locked=not locked
        elif k==ord('a'):auto=not auto
        elif k==ord('r'):ctl.reset()
        elif k==ord('s'):
            fn=f'splatworld2_{int(time.time())}.png';cv.imwrite(fn,im);print('saved',fn)
    cv.destroyAllWindows(); return 0


def main():
    p=argparse.ArgumentParser(description='SplatWorld2 local transport / identity lock')
    p.add_argument('--model');p.add_argument('--cpu',action='store_true');p.add_argument('--selftest',action='store_true');p.add_argument('--probe',action='store_true')
    p.add_argument('--seed',type=int,default=7);p.add_argument('--anchor_std',type=float,default=.60)
    p.add_argument('--probe_dirs',type=int,default=32);p.add_argument('--probe_eps',type=float,default=.35);p.add_argument('--k',type=int,default=2)
    p.add_argument('--span',type=float,default=3.0);p.add_argument('--size',type=int,default=640);p.add_argument('--anchor_image')
    p.add_argument('--detail_sigma',type=float,default=1.2);p.add_argument('--detail_gain',type=float,default=1.0);p.add_argument('--confidence_sigma',type=float,default=.10)
    p.add_argument('--auto',action='store_true');a=p.parse_args()
    if a.selftest:return selftest()
    m=find_model(a.model);print('model:',m);dec=OnnxDecoder(m,a.cpu);print('backend:',dec.backend)
    if a.probe:
        z0=(np.random.default_rng(a.seed).standard_normal(LATENT)*a.anchor_std).astype(np.float32);probe(dec,z0,a.probe_dirs,a.k,a.probe_eps,a.seed);return 0
    return live(dec,a)

if __name__=='__main__': raise SystemExit(main())
