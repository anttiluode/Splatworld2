# SplatWorld2

A no-retraining experiment built from the original **SplatWorld** decoder.

The old model is kept exactly as it is: `splat_decoder.onnx` still maps a 128-D latent point to a 96x96 Gabor-rendered face. SplatWorld2 changes the *way we move through and display* that model.

The question is narrow:

> Can we move a face locally without letting identity/detail dissolve just because the decoder's easiest directions mix motion with appearance?

## What changed

The original explorer lets you surf arbitrary tangent directions. SplatWorld2 first **measures** the local decoder around one frozen face.

For many orthonormal latent directions `d` it renders:

```text
z0 - eps*d    and    z0 + eps*d
```

Then it asks how much of that change can be explained by **optical transport** rather than repainting the face. Directions are ranked by a transport score:

```text
latent direction
      |
      v
render minus / plus
      |
      v
dense optical flow
      |
      +--> how much raw image error disappears after warping?
```

The best local directions become the live control plane.

The second change is the important anti-blur rule:

> **Every displayed frame is reconstructed from the same immutable anchor. Never from the previous displayed frame.**

The ONNX render supplies current low-frequency geometry. High-frequency detail is extracted once from the anchor and warped into the current guide with backward optical flow. A photometric confidence mask suppresses detail where the flow is unreliable.

So there is no recurrence of the form

```text
frame N -> lossy representation -> frame N+1 -> lossy representation -> ...
```

and therefore no mechanism for recursive resampling blur to compound frame after frame.

This does **not** turn a 96px decoder into a high-resolution identity model. It only tries to preserve detail already present in the anchor while making local changes.

## Use the existing model

You do **not** need to train anything. `splat_decoder.onnx` can sit directly in this repo beside the Python files.

Install:

```bash
pip install -r requirements.txt
```

Run the synthetic smoke test first:

```bash
python splatworld2.py --selftest
```

Then inspect what the real ONNX considers its most transport-like local axes:

```bash
python splatworld2.py --probe
```

And run the mouse-driven explorer:

```bash
python splatworld2.py
```

Controls:

- **left drag** — move in the two measured transport directions
- **right drag** — fine movement
- **L** — A/B raw ONNX render vs identity-lock compositor
- **A** — automatic two-axis motion loop
- **R** — return to the immutable anchor
- **S** — save frame
- **Q** — quit

Try hands-free immediately:

```bash
python splatworld2.py --auto
```

## Webcam driver — webcam pose -> measured transport plane

`webcam_drive.py` uses the same decoder probe and the same immutable-anchor compositor. It does **not** add an image encoder and it does not send webcam pixels through the face model.

At startup it first finds SplatWorld2's best local transport axes. It then measures what each selected axis actually does in image space as a four-number motion signature:

```text
[ horizontal shift, vertical shift, log-scale, roll ]
```

The webcam estimates those same four quantities from your face using only OpenCV: Haar finds the face at neutral calibration, Shi-Tomasi points are seeded inside it, Lucas-Kanade tracks them, and a partial affine transform accumulates the motion. A small ridge solve then asks:

```text
which mixture of the measured latent transport axes best matches
this webcam motion?
```

That coefficient vector drives the existing ONNX guide. The displayed face is still reconstructed from the **same immutable anchor every frame**, so webcam control does not restore the old recursive-blur path.

Run the webcam selftest:

```bash
python webcam_drive.py --selftest
```

Then run camera 0:

```bash
python webcam_drive.py
```

Or choose another camera:

```bash
python webcam_drive.py --camera 1
```

Webcam controls:

- **C** — calibrate the current head pose as neutral
- **L** — raw ONNX / identity-lock A/B
- **M** — measured pose-signature mapping / direct XY fallback
- **[ / ]** — reduce / increase webcam gain
- **R** — return latent control to the anchor
- **P** — pause/resume tracking
- **S** — save frame
- **Q** — quit

The live window contains a small camera inset with the tracked points. If the tracker drifts, look forward and press **C**. Start close to neutral and use small motions; the selected decoder directions were measured locally, not across the whole latent universe.

The interesting diagnostic is printed before the window opens:

```text
measured transport-axis signatures
axis      dx        dy      logS      roll    inliers
   0      ...       ...      ...       ...      ...
   1      ...       ...      ...       ...      ...
signature singular values: ...
```

If those signatures are nearly rank deficient, the decoder's two best optical-transport directions do not span two clean pose controls. Press **M** to compare the deliberately crude direct-XY fallback. That is a measurement outcome, not an error to hide.

Useful webcam knobs:

```text
--gain 1.0              webcam-motion gain
--pose_smooth .22       smooth physical pose before inversion
--smooth .28            smooth latent coefficients
--ridge .002            regularization of pose -> latent inverse
--pose_weights 1,1,.7,.45
--signature_eps .55     +/- step used to measure each selected axis
--span 3.0              maximum coefficient on each transport axis
--no_mirror             use literal rather than mirror-view webcam motion
```

## Optional one-photo identity texture

You can give either live program a sharper face image:

```bash
python splatworld2.py --anchor_image face.jpg
python webcam_drive.py --anchor_image face.jpg
```

The current version center-crops that photo and uses it only as an immutable texture reservoir. This is **not** LivePortrait and is not claimed to be a general reenactment system. It tests a simpler proposition: *can a learned low-dimensional motion field move a fixed identity texture without repeatedly destroying it?*

## Knobs worth touching

```text
--probe_dirs 32        orthonormal latent directions measured at startup
--probe_eps 0.35       +/- local step used for transport measurement
--span 3.0             max live coefficient on each selected direction
--detail_sigma 1.2     which anchor frequencies count as retained detail
--detail_gain 1.0      amount of warped detail added back
--confidence_sigma .10 how quickly bad-flow regions lose anchor detail
--size 640             display/compositor resolution
```

The ONNX has a dynamic batch axis, so startup probes are batched. `onnxruntime` is required; CUDA is used automatically if the installed runtime exposes `CUDAExecutionProvider`.

## The experiment / gate

The first useful run is an A/B, not a beauty contest.

1. Pick one anchor.
2. Move it with mouse, auto mode, or webcam.
3. Toggle **L** every few seconds.
4. Watch eyes, brows, mouth edges, hairline and texture rather than overall smoothness.
5. Reduce `--span` or webcam gain if flow confidence collapses.

A win would be: comparable motion, visibly better retention of anchor detail, no progressive softening simply because the loop ran longer, and exact return to the same anchor.

This version is killed if the ONNX has no useful transport-dominant directions, optical flow between nearby decoder renders is wrong, the detail looks pasted-on while geometry changes, or the 96px guide is simply too poor to supply a useful deformation field.

Those are acceptable outcomes. The point is to **measure the decoder as an operator** instead of assuming every latent direction is equally suitable for motion.

## Why this exists

The original SplatWorld README already identified the relevant failure mode: additive representations often move features by crossfading them, while phase/transport can move structure without destroying it. SplatWorld2 takes that inference-time consequence literally:

```text
model  = motion / geometry guide
anchor = identity / detail reservoir
frame  = guide + transported anchor detail
```

not

```text
frame N+1 = another lossy transformation of frame N
```

No new physics. No claim that optical flow is a new AI architecture. Just one falsifiable attempt to stop a specific contraction mechanism without retraining the decoder.

## Provenance

Descendant of `anttiluode/SplatWorld`; reuses its existing `splat_decoder.onnx`. The original model was trained on CelebA, so check CelebA's own terms before commercial use of model outputs.
