#!/usr/bin/env python3
"""webcam_drive.py — drive SplatWorld2's measured transport plane from a webcam.

No encoder is added. The existing decoder is first probed exactly as in
splatworld2.py. The selected latent transport axes are then measured in image
space: for each axis we estimate the similarity motion (x, y, log-scale, roll)
caused by a small +/- latent step.

A webcam tracker estimates the same four quantities from the user's face.
A regularized least-squares map converts that physical pose delta into
coefficients on the measured latent transport axes.

Rendering stays SplatWorld2's:
    z0 + transport_basis @ coefficients -> ONNX guide
    immutable anchor detail --flow--> current guide
The previous displayed frame is never fed back, so this driver does not
re-introduce recursive blur.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

import splatworld2 as SW

POSE_NAMES = ("x", "y", "logS", "roll")


def affine_pose(M, width, height):
    """2x3 partial-affine -> [tx/W, ty/H, log(scale), rotation_rad]."""
    if M is None:
        return np.zeros(4, np.float32)
    a, b, tx = [float(v) for v in M[0]]
    c, d, ty = [float(v) for v in M[1]]
    scale = math.sqrt(max(1e-12, 0.5 * (a*a + b*b + c*c + d*d)))
    rot = math.atan2(c - b, a + d)
    return np.array([tx / max(width, 1), ty / max(height, 1),
                     math.log(max(scale, 1e-8)), rot], np.float32)


def dense_similarity(src, dst):
    """Estimate global similarity motion inside dense flow src->dst."""
    cv = SW.cv2()
    ga = SW.gray(src) * 255.0
    gb = SW.gray(dst) * 255.0
    flow = cv.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 17, 3, 5, 1.1, 0)
    h, w = ga.shape
    step = max(3, min(h, w) // 14)
    yy, xx = np.mgrid[step//2:h:step, step//2:w:step].astype(np.float32)
    p = np.stack([xx.ravel(), yy.ravel()], axis=1)
    f = flow[yy.astype(np.int32), xx.astype(np.int32)].reshape(-1, 2)
    q = p + f
    mag = np.linalg.norm(f, axis=1)
    if len(mag) >= 12:
        cap = np.quantile(mag, 0.90) + 1e-6
        keep = np.isfinite(mag) & (mag <= cap)
        p, q = p[keep], q[keep]
    if len(p) < 6:
        return np.zeros(4, np.float32), 0.0
    M, inliers = cv.estimateAffinePartial2D(
        p, q, method=cv.RANSAC, ransacReprojThreshold=1.5,
        maxIters=500, confidence=0.99, refineIters=10
    )
    pose = affine_pose(M, w, h)
    quality = float(inliers.mean()) if inliers is not None and len(inliers) else 0.0
    return pose, quality


def measure_transport_signatures(dec, z0, basis, eps=0.55, verbose=True):
    """Return S(4,k): image-pose change per latent coefficient."""
    B = np.asarray(basis, np.float32)
    k = len(B)
    zs = np.concatenate([z0[None] - eps * B, z0[None] + eps * B], axis=0)
    outs = dec(zs)
    S = np.zeros((4, k), np.float32)
    qualities = np.zeros(k, np.float32)
    if verbose:
        print("\nmeasured transport-axis signatures")
        print("axis      dx        dy      logS      roll    inliers")
    for j in range(k):
        minus = SW.rgb(outs[j])
        plus = SW.rgb(outs[k + j])
        pose, q = dense_similarity(minus, plus)
        S[:, j] = pose / (2.0 * eps)
        qualities[j] = q
        if verbose:
            print(f"{j:4d}  {S[0,j]:+8.4f} {S[1,j]:+8.4f} "
                  f"{S[2,j]:+8.4f} {S[3,j]:+8.4f}   {q:5.2f}")
    sv = np.linalg.svd(S, compute_uv=False) if S.size else np.zeros(0)
    if verbose:
        print("signature singular values:", " ".join(f"{float(x):.5f}" for x in sv))
        if len(sv) and sv[-1] < 0.08 * max(sv[0], 1e-9):
            print("NOTE: pose signatures are close to rank deficient; "
                  "M toggles a direct XY fallback in the live window.")
    return S, qualities


def solve_coefficients(signatures, pose, weights, ridge=0.02):
    """Weighted ridge solve S a ~= pose."""
    S = np.asarray(signatures, np.float64)
    p = np.asarray(pose, np.float64)
    w = np.asarray(weights, np.float64)
    A = w[:, None] * S
    b = w * p
    H = A.T @ A + float(ridge) * np.eye(S.shape[1])
    rhs = A.T @ b
    try:
        a = np.linalg.solve(H, rhs)
    except np.linalg.LinAlgError:
        a = np.linalg.pinv(H) @ rhs
    return a.astype(np.float32)


class FaceFlowTracker:
    """OpenCV-only incremental face pose tracker.

    Haar chooses the face at calibration. Shi-Tomasi points inside it are
    tracked with pyramidal Lucas-Kanade. A partial affine increment is fitted
    each frame and composed from the neutral/calibration frame.
    """
    def __init__(self, mirror=True, max_points=140):
        cv = SW.cv2()
        self.cv = cv
        self.mirror = bool(mirror)
        self.max_points = int(max_points)
        cascade = Path(cv.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.face_cascade = cv.CascadeClassifier(str(cascade))
        if self.face_cascade.empty():
            raise RuntimeError(f"could not load Haar cascade: {cascade}")
        self.ready = False
        self.prev_gray = None
        self.prev_pts = None
        self.T = np.eye(3, dtype=np.float64)
        self.face_box = None
        self.n_good = 0
        self.last_quality = 0.0
        self.frames = 0

    def prepare(self, frame):
        return self.cv.flip(frame, 1) if self.mirror else frame

    def _largest_face(self, gray):
        eq = self.cv.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(
            eq, scaleFactor=1.12, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            return None
        return max(faces, key=lambda r: int(r[2]) * int(r[3]))

    def _feature_mask(self, gray, box):
        x, y, w, h = [int(v) for v in box]
        mask = np.zeros_like(gray)
        px, py = int(0.08*w), int(0.08*h)
        x0, y0 = max(0, x+px), max(0, y+py)
        x1, y1 = min(gray.shape[1], x+w-px), min(gray.shape[0], y+h-py)
        mask[y0:y1, x0:x1] = 255
        return mask

    def _seed_points(self, gray, box):
        return self.cv.goodFeaturesToTrack(
            gray, mask=self._feature_mask(gray, box), maxCorners=self.max_points,
            qualityLevel=0.012, minDistance=6, blockSize=7
        )

    def calibrate(self, prepared_frame):
        gray = self.cv.cvtColor(prepared_frame, self.cv.COLOR_BGR2GRAY)
        box = self._largest_face(gray)
        if box is None:
            self.ready = False
            return False
        pts = self._seed_points(gray, box)
        if pts is None or len(pts) < 12:
            self.ready = False
            return False
        self.prev_gray = gray
        self.prev_pts = pts.astype(np.float32)
        self.T = np.eye(3, dtype=np.float64)
        self.face_box = tuple(int(v) for v in box)
        self.n_good = len(pts)
        self.last_quality = 1.0
        self.frames = 0
        self.ready = True
        return True

    def _current_face_box(self):
        if self.face_box is None:
            return None
        x, y, w, h = [float(v) for v in self.face_box]
        corners = np.array([[x,y,1],[x+w,y,1],[x+w,y+h,1],[x,y+h,1]], np.float64).T
        q = (self.T @ corners).T[:, :2]
        x0, y0 = q.min(axis=0)
        x1, y1 = q.max(axis=0)
        return int(x0), int(y0), max(1, int(x1-x0)), max(1, int(y1-y0))

    def update(self, prepared_frame):
        gray = self.cv.cvtColor(prepared_frame, self.cv.COLOR_BGR2GRAY)
        if not self.ready:
            if not self.calibrate(prepared_frame):
                return np.zeros(4, np.float32), False
            return np.zeros(4, np.float32), True
        nxt, st, err = self.cv.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None,
            winSize=(25,25), maxLevel=3,
            criteria=(self.cv.TERM_CRITERIA_EPS | self.cv.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        if nxt is None or st is None:
            self.ready = False
            return np.zeros(4, np.float32), False
        good = st.reshape(-1).astype(bool)
        if err is not None:
            good &= np.isfinite(err.reshape(-1))
            good &= err.reshape(-1) < 35.0
        p = self.prev_pts.reshape(-1,2)[good]
        q = nxt.reshape(-1,2)[good]
        self.n_good = len(p)
        if len(p) < 10:
            self.ready = False
            return np.zeros(4, np.float32), False
        A, inliers = self.cv.estimateAffinePartial2D(
            p, q, method=self.cv.RANSAC, ransacReprojThreshold=3.0,
            maxIters=1000, confidence=0.995, refineIters=10
        )
        if A is None:
            self.ready = False
            return np.zeros(4, np.float32), False
        self.last_quality = float(inliers.mean()) if inliers is not None else 0.0
        A3 = np.eye(3, dtype=np.float64)
        A3[:2] = A
        self.T = A3 @ self.T
        self.prev_gray = gray
        self.prev_pts = q.reshape(-1,1,2).astype(np.float32)
        self.frames += 1
        if self.frames % 24 == 0 or len(self.prev_pts) < 45:
            box = self._current_face_box()
            if box is not None:
                fresh = self._seed_points(gray, box)
                if fresh is not None and len(fresh) >= 12:
                    self.prev_pts = fresh.astype(np.float32)
                    self.n_good = len(fresh)
        h, w = gray.shape
        return affine_pose(self.T[:2], w, h), True

    def draw_preview(self, prepared_frame, width=190):
        cv = self.cv
        f = prepared_frame.copy()
        box = self._current_face_box()
        if box is not None:
            x, y, w, h = box
            cv.rectangle(f, (x,y), (x+w,y+h), (0,255,0), 1)
        if self.prev_pts is not None:
            skip = max(1, len(self.prev_pts)//35)
            for pt in self.prev_pts.reshape(-1,2)[::skip]:
                cv.circle(f, tuple(np.round(pt).astype(int)), 2, (0,220,255), -1)
        hh, ww = f.shape[:2]
        out_h = max(1, int(width * hh / max(ww,1)))
        return cv.resize(f, (width,out_h), interpolation=cv.INTER_AREA)


class WebcamController:
    def __init__(self, signatures, span=3.0, gain=1.0, smooth=0.24,
                 ridge=0.02, weights=(1.0,1.0,0.75,0.45)):
        self.S = np.asarray(signatures, np.float32)
        self.k = self.S.shape[1]
        self.span = float(span)
        self.gain = float(gain)
        self.smooth = float(smooth)
        self.ridge = float(ridge)
        self.weights = np.asarray(weights, np.float32)
        self.a = np.zeros(self.k, np.float32)
        self.target = np.zeros(self.k, np.float32)
        self.mode = "signature"

    def set_pose(self, pose):
        p = np.asarray(pose, np.float32) * self.gain
        if self.mode == "signature":
            target = solve_coefficients(self.S, p, self.weights, self.ridge)
        else:
            target = np.zeros(self.k, np.float32)
            if self.k > 0:
                target[0] = p[0] * self.span / 0.12
            if self.k > 1:
                target[1] = p[1] * self.span / 0.12
        self.target[:] = np.clip(target, -self.span, self.span)

    def tick(self):
        self.a += self.smooth * (self.target - self.a)
        return self.a

    def reset(self):
        self.target[:] = 0
        self.a[:] = 0

    def toggle_mode(self):
        self.mode = "xy" if self.mode == "signature" else "signature"


def parse_weights(s):
    vals = [float(x.strip()) for x in str(s).split(",")]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("need x,y,scale,roll weights")
    return tuple(vals)


def selftest():
    ok = True
    def check(name, cond, note=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {note}")
    S = np.array([[.10,.01],[.00,.09],[.015,-.004],[.008,.020]], np.float32)
    truth = np.array([1.1,-0.7], np.float32)
    est = solve_coefficients(S, S @ truth, (1,1,1,1), ridge=1e-8)
    check("pose->latent inverse", np.allclose(est, truth, atol=1e-3), f"{est} vs {truth}")
    dec = SW.MockDecoder()
    z0 = np.zeros(SW.LATENT, np.float32)
    D = np.zeros((8,SW.LATENT), np.float32)
    D[0,0]=1; D[1,1]=1; D[2:,2:8]=np.eye(6,dtype=np.float32)
    B, sel, _ = SW.probe(dec,z0,n=8,k=2,eps=.5,directions=D,verbose=False)
    Sig, quality = measure_transport_signatures(dec,z0,B,eps=.5,verbose=False)
    check("transport basis retained", set(sel)=={0,1}, str(sel))
    check("signature finite", np.isfinite(Sig).all())
    check("signature has motion", float(np.linalg.norm(Sig[:2])) > 0.01,
          f"norm={np.linalg.norm(Sig[:2]):.4f}")
    check("flow fit quality", float(np.mean(quality)) > 0.3,
          f"mean={np.mean(quality):.2f}")
    print("selftest:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


def live(args):
    cv = SW.cv2()
    model = SW.find_model(args.model)
    print("model:", model)
    dec = SW.OnnxDecoder(model, args.cpu)
    print("backend:", dec.backend)
    rng = np.random.default_rng(args.seed)
    z0 = (rng.standard_normal(SW.LATENT) * args.anchor_std).astype(np.float32)
    anchor = SW.rgb(dec(z0[None])[0])
    print(f"anchor |z|={np.linalg.norm(z0):.3f}")
    B, selected, _ = SW.probe(dec,z0,args.probe_dirs,args.k,args.probe_eps,args.seed)
    signatures, qualities = measure_transport_signatures(dec,z0,B,args.signature_eps)
    wc = WebcamController(signatures,args.span,args.gain,args.smooth,
                          args.ridge,args.pose_weights)
    lock = SW.IdentityLock(anchor,args.size,args.anchor_image,args.detail_sigma,
                           args.detail_gain,args.confidence_sigma)

    backend = cv.CAP_DSHOW if hasattr(cv,"CAP_DSHOW") else 0
    cap = cv.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        cap = cv.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {args.camera}")
    cap.set(cv.CAP_PROP_FRAME_WIDTH,args.cam_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT,args.cam_height)
    cap.set(cv.CAP_PROP_BUFFERSIZE,1)

    tracker = FaceFlowTracker(mirror=not args.no_mirror)
    print("\nLook at the camera in a neutral pose. Calibration happens automatically.")
    print("C recalibrate | L lock/raw | M signature/XY | [ ] gain | R anchor | S save | Q quit")
    win = "SplatWorld2 WEBCAM  C=calibrate L=lock M=map [ ]=gain Q=quit"
    cv.namedWindow(win,cv.WINDOW_NORMAL)
    locked=True; paused=False; pose_sm=np.zeros(4,np.float32)
    last=time.time(); frames=0; fps=0.0
    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                print("camera read failed"); break
            cam = tracker.prepare(raw)
            pose, tracked = tracker.update(cam)
            if tracked and not paused:
                pose_sm += args.pose_smooth * (pose - pose_sm)
                wc.set_pose(pose_sm)
            elif not tracked:
                wc.set_pose(np.zeros(4,np.float32))
            a = wc.tick()
            z = (z0 + a @ B).astype(np.float32)
            guide = SW.rgb(dec(z[None])[0])
            shown = lock(guide) if locked else cv.resize(
                guide,(args.size,args.size),interpolation=cv.INTER_CUBIC)
            im = cv.cvtColor((shown*255).astype(np.uint8),cv.COLOR_RGB2BGR)

            prev = tracker.draw_preview(cam,width=max(150,args.size//4))
            ph,pw = prev.shape[:2]
            if ph < args.size and pw < args.size:
                im[8:8+ph,8:8+pw] = prev
                cv.rectangle(im,(7,7),(9+pw,9+ph),(255,255,255),1)

            frames += 1; now=time.time()
            if now-last >= .5:
                fps=frames/(now-last); frames=0; last=now
            hud1=(f"{'LOCK' if locked else 'RAW '} {'TRACK' if tracked else 'FIND'} "
                  f"fps {fps:4.1f} map={wc.mode} gain={wc.gain:.2f} conf={lock.conf:.2f}")
            hud2=(f"pose x/y/s/r {pose_sm[0]:+.3f} {pose_sm[1]:+.3f} "
                  f"{pose_sm[2]:+.3f} {pose_sm[3]:+.3f}  "
                  f"a={np.array2string(a,precision=2)}")
            cv.putText(im,hud1,(12,args.size-34),cv.FONT_HERSHEY_PLAIN,1.05,(0,255,0),1,cv.LINE_AA)
            cv.putText(im,hud2,(12,args.size-14),cv.FONT_HERSHEY_PLAIN,.92,(0,255,0),1,cv.LINE_AA)
            cv.imshow(win,im)

            key=cv.waitKeyEx(1)
            if key < 0: continue
            k=key & 0xff
            if k==ord('q'): break
            elif k==ord('c'):
                if tracker.calibrate(cam):
                    pose_sm[:]=0; wc.reset(); print("camera neutral recalibrated")
                else: print("no face/features found for calibration")
            elif k==ord('l'): locked=not locked
            elif k==ord('m'):
                wc.toggle_mode(); wc.reset(); pose_sm[:]=0
                if tracker.calibrate(cam): print("mapping:",wc.mode,"(neutral recalibrated)")
                else: print("mapping:",wc.mode)
            elif k==ord('r'): wc.reset(); pose_sm[:]=0
            elif k==ord('p'): paused=not paused; print("tracking","paused" if paused else "live")
            elif k==ord('['): wc.gain=max(.10,wc.gain/1.15); print("gain",wc.gain)
            elif k==ord(']'): wc.gain=min(10.,wc.gain*1.15); print("gain",wc.gain)
            elif k==ord('s'):
                fn=f"splatworld2_webcam_{int(time.time())}.png"; cv.imwrite(fn,im); print("saved",fn)
    finally:
        cap.release(); cv.destroyAllWindows()
    return 0


def main():
    p=argparse.ArgumentParser(description="Drive SplatWorld2 transport plane from webcam")
    p.add_argument("--model"); p.add_argument("--cpu",action="store_true")
    p.add_argument("--selftest",action="store_true")
    p.add_argument("--camera",type=int,default=0); p.add_argument("--cam_width",type=int,default=640)
    p.add_argument("--cam_height",type=int,default=480); p.add_argument("--no_mirror",action="store_true")
    p.add_argument("--seed",type=int,default=7); p.add_argument("--anchor_std",type=float,default=.60)
    p.add_argument("--probe_dirs",type=int,default=32); p.add_argument("--probe_eps",type=float,default=.35)
    p.add_argument("--k",type=int,default=2); p.add_argument("--span",type=float,default=3.0)
    p.add_argument("--size",type=int,default=640); p.add_argument("--anchor_image")
    p.add_argument("--detail_sigma",type=float,default=1.2); p.add_argument("--detail_gain",type=float,default=1.0)
    p.add_argument("--confidence_sigma",type=float,default=.10)
    p.add_argument("--signature_eps",type=float,default=.55); p.add_argument("--gain",type=float,default=1.0)
    p.add_argument("--smooth",type=float,default=.28,help="latent coefficient smoothing")
    p.add_argument("--pose_smooth",type=float,default=.22,help="webcam pose smoothing")
    p.add_argument("--ridge",type=float,default=.002)
    p.add_argument("--pose_weights",type=parse_weights,default=(1.0,1.0,.70,.45),
                   help="x,y,scale,roll weights e.g. 1,1,.7,.45")
    a=p.parse_args()
    if a.selftest: return selftest()
    return live(a)


if __name__ == "__main__":
    raise SystemExit(main())
