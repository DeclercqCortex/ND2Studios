# Node-backend reference — algorithm packages for the nodegraph v2 catalog

Distilled from a package survey supplied 2026-07-20 (feature enhancement + 3D voxel
reconstruction). This is the **backend menu for the Phase-2 node port** (V2.00 §14.3):
each package is a candidate compute backend behind a nodegraph node. It does **not**
change the data model — nodes wrap these; the Dataset/domain/transfer core is agnostic.

> **Caveat.** Existence / algorithm / dimensionality claims are primary-sourced and
> high-confidence. Version + maintenance details are as-of **2026-07-20** and the survey's
> adversarial re-verification did not complete — **re-check before adding any dependency.**

## The five operations → node families

| Node family | What the node does | True-3D backends (voxel) | Stack-of-2D / 2D backends | GPU |
|---|---|---|---|---|
| **Deconvolution** | invert the PSF blur (RL, RL-TV; blind) | RedLionfish (3D-only, OpenCL), cudaDecon (CUDA), Flowdec (3D, TF — *unmaintained*), ITK (n-D, incl. **blind**), DeconvolutionLab2 (Java, many solvers) | scikit-image `richardson_lucy`/`wiener` (n-D) | RedLionfish, cudaDecon, Flowdec |
| **Regularized denoise (TV)** | edge-preserving denoise (ROF/TGV) | `skimage.denoise_tv_chambolle` (**true n-D**), proxTV (n-D), PyProximal `TV` | `denoise_tv_bregman` | — (CPU) |
| **Multiscale enhancement** | separate structure/background across scales; boost faint features | PyWavelets DWT (n-D), PySAP/Sparse2D **starlet à-trous** (2D+3D, shift-invariant — best for background flatten), curvelops (2D+3D, hard install) | `skimage.denoise_wavelet` (**stack-of-2D only** — the classic 3D trap) | — |
| **Registration / stacking / fusion** | sub-pixel align, stack (SNR ↑√N), stitch tiles, fuse views | ANTsPy (SyN deformable), elastix (B-spline), multiview-stitcher, BigStitcher, `skimage phase_cross_correlation` (n-D) | pyStackReg (**stack-of-2D**, drift correction) | ANTs/elastix partial |
| **Sparse reconstruction** | recover volume from incomplete/noisy data (compressed sensing) | BART (3D, CUDA — now on Codeberg), ODL (tomographic 3D), SPORCO (dictionary/CSC), PySAP | SMILI (2D-oriented) | BART, SPORCO (`sporco-cuda`) |

## Deep-learning restoration
CSBDeep / CARE (content-aware CNN, 2D+3D, needs paired training data) — for the hardest low-signal cases; least plug-and-play.

## Two facts that shaped the plan (see V2.02 §7b)

1. **True-3D voxel vs stack-of-2D is a first-class node attribute.** Many backends are genuinely
   volumetric (RedLionfish, cudaDecon, 3D TV/starlet, ANTs/elastix, BART/ODL); others loop 2D over Z
   (`denoise_wavelet`, pyStackReg). A node must **declare** `WHOLE_VOLUME` vs `WHOLE_PLANE`-looped —
   the granularity taxonomy in V2.02 §7b encodes exactly this.
2. **Registration/stacking/fusion change `AxisSizes`** (stack T→1; stitch M→1 + grow Y,X) and produce
   **Frame-domain transform attributes** — handled by axis-changing nodes (V2.02 §7b), not a model change.

## Metadata-intelligent PSF (flagship for the V1.91 unit/derive framework)
A deconvolution node's **PSF can be synthesized from optics metadata** — NA, emission wavelength,
`pixel_size_um`, `z_step_um` — via the Gibson–Lanni model (Flowdec ships this). This is the ideal
showcase for unit/derive metadata-adaptivity: the PSF and its sampling **derive** from the loaded
image's calibration, no manual entry. Add a `deconvolve` node whose PSF params carry `unit`/`derive`.

## Representative end-to-end 3D chain (all from the menu above)
register+stack (pyStackReg / phase_cross_correlation → multiview-stitcher) → background/feature
enhance (PySAP starlet) → deconvolve (RedLionfish GPU) → TV denoise (`denoise_tv_chambolle` 3D) →
(if undersampled) reconstruct (BART/ODL). Python-native + GPU-capable end to end.

## Sources
Primary repos fetched 2026-07-20 (see the original survey for the full list): RedLionfish,
DeconvolutionLab2, Flowdec, cudaDecon, ITK, scikit-image, CSBDeep, proxTV, PyProximal, ProxImaL,
ODL, PyWavelets, curvelops, PySAP/Sparse2D, pyStackReg, multiview-stitcher, BigStitcher, ANTsPy,
elastix, SMILI, BART (codeberg.org/mrirecon/bart), SPORCO.
