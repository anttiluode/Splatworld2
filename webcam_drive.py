#!/usr/bin/env python3
"""webcam_drive.py — drive SplatWorld2 from a webcam, using TinyAvatar-style
face framing but keeping the no-recursive-blur SplatWorld2 compositor.

What is copied forward from TinyAvatar:
  * FaceFramer: Haar-based square face crop with margin and EMA smoothing.
  * normalize_crop: make the live crop occupy a stable photometric regime.

What is kept from SplatWorld2:
  * the existing ONNX decoder is probed for a small transport basis;
  * those transport axes still generate the low-frequency face guide;
  * every displayed frame is rebuilt from an immutable anchor, never from the
    previous display, so recursive blur cannot accumulate.

New webcam guidance path:
  * calibrate a reference webcam face crop (C);
  * each frame, estimate face motion relative to that reference crop;
  * map that motion into the measured latent transport plane;
  * simultaneously use the webcam flow itself to warp anchor detail.

So the guide comes from the measured latent basis, but the crisp motion detail
comes from the actual webcam deformation field.
"""
from __future__ import annotations
import argparse, math, os, time
import numpy as np
import splatworld2 as SW

POSE_NAMES = ("x", "y", "logS", "roll")


class FaceFramer:
    """Haar face crop with TinyAvatar's framing logic and EMA smoothing."""
    def __init__(self, margin=0.35, ema=0.30, every=2):
        cv = SW.cv2()
        cpath = os.path.join(cv.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.det = cv.CascadeClassifier(cpath)
        if self.det.empty():
            self.det = None
        self.margin, self.ema, self.every = float(margin), float(ema), int(every)
        self.box = None
        self.f = 0
        self.last_rect = None

    def crop(self, fr):
        cv = SW.cv2(); H, W = fr.shape[:2]
        if self.det is not None and self.f % self.every == 0:
            g = cv.cvtColor(fr, cv.COLOR_BGR2GRAY)
            det = self.det.detectMultiScale(g, 1.15, 5, minSize=(80, 80))
            if len(det):
                x, y, w, h = max(det, key=lambda b: b[2] * b[3])
                m = self.margin * max(w, h)
                cx, cy = x + w / 2.0, y + h / 2.0
                half = max(w, h) / 2.0 + m
                if self.box is None:
                    self.box = (cx, cy, half)
                else:
                    a = self.ema
                    self.box = (a * cx + (1 - a) * self.box[0],
                                a * cy + (1 - a) * self.box[1],
                                a * half + (1 - a) * self.box[2])
                self.last_rect = (int(x), int(y), int(w), int(h))
        self.f += 1
        if self.box is None:
            s = min(H, W)
            self.last_rect = None
            return fr[(H - s)//2:(H + s)//2, (W - s)//2:(W + s)//2]
        cx, cy, half = self.box
        s = int(half)
        x0, x1 = int(max(cx - s, 0)), int(min(cx + s, W))
        y0, y1 = int(max(cy - s, 0)), int(min(cy + s, H))
        c = fr[y0:y1, x0:x1]
        if c.size:
            return c
        s = min(H, W); self.last_rect = None
        return fr[(H - s)//2:(H + s)//2, (W - s)//2:(W + s)//2]


def normalize_crop(x, tgt_mean=0.52, tgt_std=0.26):
    m, s = x.mean(), x.std() + 1e-6
    return np.clip((x - m) / s * tgt_std + tgt_mean, 0, 1)


def affine_pose(M, width, height):
    if M is None:
        return np.zeros(4, np.float32)
    a, b, tx = [float(v) for v in M[0]]
    c, d, ty = [float(v) for v in M[1]]
    scale = math.sqrt(max(1e-12, 0.5 * (a*a + b*b + c*c + d*d)))
    rot = math.atan2(c - b, a + d)
    return np.array([tx / max(width, 1), ty / max(height, 1),
                     math.log(max(scale, 1e-8)), rot], np.float32)


def estimate_motion(ref_rgb, cur_rgb):
    """Pose delta + backward flow current->reference using the cropped face."""
    cv = SW.cv2()
    g0 = (SW.gray(ref_rgb) * 255.0).astype(np.uint8)
    g1 = (SW.gray(cur_rgb) * 255.0).astype(np.uint8)
    h, w = g0.shape

    p0 = cv.goodFeaturesToTrack(g0, maxCorners=120, qualityLevel=0.01,
                                minDistance=5, blockSize=5)
    pose = np.zeros(4, np.float32)
    quality = 0.0
    if p0 is not None and len(p0) >= 8:
        p1, st, _ = cv.calcOpticalFlowPyrLK(
            g0, g1, p0, None, winSize=(17, 17), maxLevel=3,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        if p1 is not None:
            st = st.reshape(-1) > 0
            P = p0.reshape(-1, 2)[st]
            Q = p1.reshape(-1, 2)[st]
            if len(P) >= 6:
                M, inl = cv.estimateAffinePartial2D(
                    P, Q, method=cv.RANSAC, ransacReprojThreshold=2.0,
                    maxIters=600, confidence=0.99, refineIters=10
                )
                pose = affine_pose(M, w, h)
                quality = float(inl.mean()) if inl is not None and len(inl) else 0.0

    back = cv.calcOpticalFlowFarneback(
        (SW.gray(cur_rgb) * 255.0).astype(np.float32),
        (SW.gray(ref_rgb) * 255.0).astype(np.float32),
        None, 0.5, 3, 17, 3, 5, 1.1, 0
    )
    pred = SW.warp(ref_rgb, back)
    photo_err = float(np.mean(np.abs(pred - cur_rgb)))
    return pose, quality, back, photo_err


def measure_transport_signatures(dec, z0, basis, eps=0.55, verbose=True):
    """How each selected latent axis moves the generated face in image space."""
    B = np.asarray(basis, np.float32); k = len(B)
    zs = np.concatenate([z0[None] - eps * B, z0[None] + eps * B], axis=0)
    outs = dec(zs)
    S = np.zeros((4, k), np.float32)
    qualities = np.zeros(k, np.float32)
    for i in range(k):
        a = SW.rgb(outs[i]); b = SW.rgb(outs[k + i])
        pose, q, _, _ = estimate_motion(a, b)
        S[:, i] = pose / max(2 * eps, 1e-6)
        qualities[i] = q
    if verbose:
        print("measured transport-axis signatures")
        print("axis      dx        dy      logS      roll    fitQ")
        for i in range(k):
            print(f" {i:2d}  {S[0,i]:+8.4f} {S[1,i]:+8.4f} {S[2,i]:+8.4f} {S[3,i]:+8.4f}   {qualities[i]:.2f}")
        sv = np.linalg.svd(S, compute_uv=False)
        print("signature singular values:", " ".join(f"{v:.4f}" for v in sv))
    return S, qualities


def pose_to_coeffs(S, pose, ridge=0.15):
    S = np.asarray(S, np.float32)
    k = S.shape[1]
    A = S.T @ S + float(ridge) * np.eye(k, dtype=np.float32)
    b = S.T @ np.asarray(pose, np.float32)
    return np.linalg.solve(A, b).astype(np.float32)


class WebcamMotionLock:
    """Warp immutable avatar detail with the actual webcam deformation field."""
    def __init__(self, anchor, size=768, anchor_image=None,
                 detail_sigma=1.2, detail_gain=1.0, confidence_sigma=0.10,
                 low_sigma=1.8, low_mix=0.28):
        cv = SW.cv2()
        self.size = int(size)
        self.gain = float(detail_gain)
        self.conf_sigma = float(confidence_sigma)
        self.low_sigma = float(low_sigma)
        self.low_mix = float(low_mix)
        if anchor_image is None:
            tex = cv.resize(anchor, (self.size, self.size), interpolation=cv.INTER_LANCZOS4)
        else:
            im = cv.imread(anchor_image)
            if im is None:
                raise FileNotFoundError(anchor_image)
            im = cv.cvtColor(im, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
            h, w = im.shape[:2]; s = min(h, w)
            im = im[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
            tex = cv.resize(im, (self.size, self.size), interpolation=cv.INTER_LANCZOS4)
        self.anchor_tex = tex.copy()
        self.anchor_detail = tex - cv.GaussianBlur(tex, (0, 0), detail_sigma)
        self.ref_crop = None
        self.conf = 1.0

    def set_reference(self, ref_crop_rgb):
        self.ref_crop = ref_crop_rgb.copy()

    def __call__(self, guide_rgb, webcam_backflow, current_crop_rgb):
        cv = SW.cv2()
        base = cv.resize(guide_rgb, (self.size, self.size), interpolation=cv.INTER_CUBIC)
        if webcam_backflow is None or self.ref_crop is None or current_crop_rgb is None:
            return np.clip(base, 0, 1)
        flow_big = cv.resize(webcam_backflow, (self.size, self.size), interpolation=cv.INTER_LINEAR)
        h0, w0 = webcam_backflow.shape[:2]
        flow_big[..., 0] *= self.size / max(w0, 1)
        flow_big[..., 1] *= self.size / max(h0, 1)
        warped_anchor = SW.warp(self.anchor_tex, flow_big, out_size=self.size)
        warped_low = cv.GaussianBlur(warped_anchor, (0, 0), self.low_sigma)
        warped_detail = SW.warp(self.anchor_detail, flow_big, out_size=self.size)

        pred_crop = SW.warp(self.ref_crop, webcam_backflow)
        err = np.mean(np.abs(pred_crop - current_crop_rgb), axis=2)
        c = np.exp(-err / max(self.conf_sigma, 1e-4)).astype(np.float32)
        c = cv.GaussianBlur(c, (0, 0), 1.0)
        self.conf = float(c.mean())
        c_big = cv.resize(c, (self.size, self.size), interpolation=cv.INTER_LINEAR)[..., None]

        mixed = (1.0 - self.low_mix) * base + self.low_mix * warped_low
        out = mixed + self.gain * c_big * warped_detail
        return np.clip(out, 0, 1)


class LatentFollower:
    def __init__(self, z0, basis, span=3.0, smooth=0.28):
        self.z0 = z0.copy()
        self.B = np.asarray(basis, np.float32)
        self.a = np.zeros(len(self.B), np.float32)
        self.ta = self.a.copy()
        self.span = float(span)
        self.smooth = float(smooth)

    def set_target(self, a):
        a = np.asarray(a, np.float32)
        self.ta[:len(a)] = np.clip(a, -self.span, self.span)

    def reset(self):
        self.a[:] = 0; self.ta[:] = 0

    def tick(self):
        self.a += self.smooth * (self.ta - self.a)

    def z(self):
        return (self.z0 + self.a @ self.B).astype(np.float32)


def crop_for_model(frame_bgr, framer, H, W, norm=True):
    cv = SW.cv2()
    crop = framer.crop(frame_bgr)
    rgb = cv.resize(crop, (W, H), interpolation=cv.INTER_CUBIC)[:, :, ::-1].astype(np.float32) / 255.0
    if norm:
        rgb = normalize_crop(rgb)
    return rgb, crop


def selftest():
    ok = True
    def check(name, cond, note=''):
        nonlocal ok; ok &= bool(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name} {note}")
    cv = SW.cv2()
    h = 96
    y, x = np.mgrid[0:h, 0:h].astype(np.float32)
    ref = np.zeros((h, h, 3), np.float32)
    ref[..., 0] = np.exp(-(((x-38)/16)**2 + ((y-50)/18)**2))
    ref[..., 1] = np.exp(-(((x-58)/12)**2 + ((y-45)/14)**2))
    ref[..., 2] = np.exp(-(((x-48)/10)**2 + ((y-68)/8)**2))
    M = cv.getRotationMatrix2D((h/2, h/2), 8.0, 1.07); M[:,2] += [6.0, -4.0]
    cur = cv.warpAffine(ref, M, (h, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)
    pose, q, back, _ = estimate_motion(ref, cur)
    check('motion fit quality', q > 0.35, f'{q:.2f}')
    check('translation x sign', pose[0] > 0, f'{pose[0]:+.3f}')
    check('translation y sign', pose[1] < 0, f'{pose[1]:+.3f}')
    check('scale positive', pose[2] > 0, f'{pose[2]:+.3f}')

    S = np.array([[1.0, 0.0], [0.0, 2.0], [0.2, 0.1], [0.0, 0.5]], np.float32)
    a0 = np.array([0.35, -0.40], np.float32)
    a1 = pose_to_coeffs(S, S @ a0, ridge=1e-6)
    check('pose inverse', np.allclose(a0, a1, atol=1e-4), f'{a0} vs {a1}')

    dec = SW.MockDecoder(); z0 = np.zeros(SW.LATENT, np.float32)
    anchor = SW.rgb(dec(z0[None])[0])
    lock = WebcamMotionLock(anchor, size=256)
    lock.set_reference(ref)
    before = lock.anchor_tex.copy()
    guide = SW.rgb(dec((z0 + 0.5*np.eye(SW.LATENT, dtype=np.float32)[0])[None])[0])
    out = lock(guide, back, cur)
    check('compositor finite', np.isfinite(out).all() and out.shape == (256,256,3))
    check('immutable anchor', np.array_equal(before, lock.anchor_tex))
    print('selftest:', 'ALL PASS' if ok else 'FAILURES ABOVE')
    return 0 if ok else 1


def live(args):
    cv = SW.cv2()
    model = SW.find_model(args.model); print('model:', model)
    dec = SW.OnnxDecoder(model, args.cpu); print('backend:', dec.backend)
    rng = np.random.default_rng(args.seed)
    z0 = (rng.standard_normal(SW.LATENT) * args.anchor_std).astype(np.float32)
    anchor = SW.rgb(dec(z0[None])[0])
    print(f'anchor |z|={np.linalg.norm(z0):.3f}')

    basis, _, _ = SW.probe(dec, z0, args.probe_dirs, args.k, args.probe_eps, args.seed)
    S, _ = measure_transport_signatures(dec, z0, basis, args.signature_eps, verbose=True)
    follower = LatentFollower(z0, basis, args.span, smooth=args.smooth)
    lock = WebcamMotionLock(anchor, args.size, args.anchor_image, args.detail_sigma,
                            args.detail_gain, args.confidence_sigma,
                            low_sigma=args.low_sigma, low_mix=args.low_mix)

    cap = cv.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError('webcam failed to open')
    framer = FaceFramer(margin=args.margin, ema=args.ema, every=args.detect_every)
    ref_crop = None
    map_mode = 'signature'
    locked = True
    gain = 1.0
    fps = 0.0; t_last = time.time(); frames = 0
    avatar_size = args.size
    inset_w = max(220, avatar_size // 4); inset_h = inset_w
    pad = 12
    print('keys: C=calibrate  L=lock/raw  M=map  [ ]=gain  Q=quit')
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop_small, crop_show = crop_for_model(frame, framer, anchor.shape[0], anchor.shape[1], norm=True)
        if ref_crop is None:
            ref_crop = crop_small.copy(); lock.set_reference(ref_crop)
        pose, track_q, webcam_back, _ = estimate_motion(ref_crop, crop_small)
        if map_mode == 'signature':
            target = gain * pose_to_coeffs(S, pose, ridge=args.ridge)
        else:
            target = np.zeros(len(basis), np.float32)
            if len(target) > 0: target[0] = gain * args.direct_scale * pose[0]
            if len(target) > 1: target[1] = -gain * args.direct_scale * pose[1]
        conf_gate = float(np.clip(0.15 + 1.2 * track_q, 0.0, 1.0))
        follower.set_target(conf_gate * target)
        follower.tick()
        guide = SW.rgb(dec(follower.z()[None])[0])
        shown = lock(guide, webcam_back, crop_small) if locked else cv.resize(guide, (avatar_size, avatar_size), interpolation=cv.INTER_CUBIC)

        canvas = cv.cvtColor((shown * 255).astype(np.uint8), cv.COLOR_RGB2BGR)
        cam_bgr = cv.resize(crop_show, (inset_w, inset_h), interpolation=cv.INTER_AREA)
        canvas[pad:pad+inset_h, pad:pad+inset_w] = cam_bgr
        frames += 1; now = time.time()
        if now - t_last > 0.5:
            fps = frames / max(now - t_last, 1e-6); frames = 0; t_last = now
        hud1 = f"{'LOCK TRACK' if locked else 'RAW GUIDE '} fps {fps:4.1f} map={map_mode} gain={gain:.2f} conf={lock.conf:.2f}"
        hud2 = (f"pose x/y/s/r {pose[0]:+0.3f} {pose[1]:+0.3f} {pose[2]:+0.3f} {pose[3]:+0.3f}  "
                f"a={np.array2string(follower.a[:min(4,len(follower.a))], precision=2)}")
        cv.putText(canvas, hud1, (16, avatar_size - 42), cv.FONT_HERSHEY_PLAIN, 1.35, (0,255,0), 2, cv.LINE_AA)
        cv.putText(canvas, hud2, (16, avatar_size - 16), cv.FONT_HERSHEY_PLAIN, 1.15, (0,255,0), 1, cv.LINE_AA)
        cv.imshow('SplatWorld2 WEBCAM  C=calibrate L=lock M=map [ ]=gain Q=quit', canvas)
        key = cv.waitKeyEx(1)
        if key < 0: continue
        k = key & 0xFF
        if k == ord('q'): break
        elif k == ord('c'):
            ref_crop = crop_small.copy(); lock.set_reference(ref_crop); follower.reset(); print('calibrated')
        elif k == ord('l'): locked = not locked
        elif k == ord('m'):
            map_mode = 'direct' if map_mode == 'signature' else 'signature'; print('map mode ->', map_mode)
        elif k == ord(']'): gain *= 1.10
        elif k == ord('['): gain /= 1.10
    cap.release(); cv.destroyAllWindows(); return 0


def main():
    p = argparse.ArgumentParser(description='SplatWorld2 webcam driver')
    p.add_argument('--model'); p.add_argument('--cpu', action='store_true'); p.add_argument('--selftest', action='store_true')
    p.add_argument('--camera', type=int, default=0); p.add_argument('--seed', type=int, default=7); p.add_argument('--anchor_std', type=float, default=0.60)
    p.add_argument('--probe_dirs', type=int, default=32); p.add_argument('--probe_eps', type=float, default=0.35); p.add_argument('--signature_eps', type=float, default=0.55); p.add_argument('--k', type=int, default=2)
    p.add_argument('--span', type=float, default=3.0); p.add_argument('--smooth', type=float, default=0.28); p.add_argument('--ridge', type=float, default=0.15); p.add_argument('--direct_scale', type=float, default=8.0)
    p.add_argument('--size', type=int, default=900); p.add_argument('--anchor_image'); p.add_argument('--detail_sigma', type=float, default=1.2); p.add_argument('--detail_gain', type=float, default=1.0); p.add_argument('--confidence_sigma', type=float, default=0.10); p.add_argument('--low_sigma', type=float, default=1.8); p.add_argument('--low_mix', type=float, default=0.28)
    p.add_argument('--margin', type=float, default=0.35); p.add_argument('--ema', type=float, default=0.30); p.add_argument('--detect_every', type=int, default=2)
    a = p.parse_args()
    if a.selftest: raise SystemExit(selftest())
    raise SystemExit(live(a))


if __name__ == '__main__':
    main()
