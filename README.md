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

You do **not** need to train anything.

Copy `splat_decoder.onnx` from the original SplatWorld repo next to `splatworld2.py`, or point to it directly:

```bash
python splatworld2.py --model ..\SplatWorld\splat_decoder.onnx
```

The program also automatically checks `../SplatWorld/splat_decoder.onnx`.

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
python splatworld2.py --model ..\SplatWorld\splat_decoder.onnx --probe
```

And run live:

```bash
python splatworld2.py --model ..\SplatWorld\splat_decoder.onnx
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
python splatworld2.py --model ..\SplatWorld\splat_decoder.onnx --auto
```

## Optional one-photo identity texture

You can give the lock compositor a sharper face image:

```bash
python splatworld2.py --model ..\SplatWorld\splat_decoder.onnx --anchor_image face.jpg
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
2. Turn **A** on.
3. Toggle **L** every few seconds.
4. Watch eyes, brows, mouth edges, hairline and texture rather than overall smoothness.
5. Reduce `--span` if flow confidence collapses.

A win would be: comparable motion, visibly better retention of anchor detail, no progressive softening simply because the loop ran longer, and exact return to the same anchor with **R**.

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

Descendant of `anttiluode/SplatWorld`; intended to reuse its existing `splat_decoder.onnx`. The original model was trained on CelebA, so check CelebA's own terms before commercial use of model outputs.
