# Temporal stability research for DLSS5FSR

Research date: 2026-10-22. Local source baseline: `05f86cc`.
Scope: remaining Cyberpunk trails, generated-detail flicker and two-pass instability.
This is a research assessment and experiment specification, not a demonstrated quality fix.

Implementation follow-up: the two default-off candidates and measured tradeoffs
are now described in [RESIDENT_C512.md](../RESIDENT_C512.md). The original
research assessment below is preserved to distinguish hypotheses from results.

## Findings that change the next experiment

The strongest new mechanism to investigate is **motion-correlated Gaussian noise that preserves spatial independence**. Our fixed-seed study reduced still-surface edit variation but did not materially improve moving surfaces. Moving the noise with those surfaces is a relevant next hypothesis. It must preserve the noise distribution expected by the existing network.

A second, narrower candidate is **appearance-aware confidence in our optional output stabilizer**. A current peer uses input-appearance rejection and a neighborhood edit clamp. Our stabilizer currently has depth checks and a fixed edit-change clamp. This matters when stabilization is enabled; it cannot explain the current stabilization-off result.

New peer anti-ghosting work on **cached edits** is useful engineering evidence, but its published improvement is against an older cache that skips model evaluations. Our current history-zero, every-frame path has a different cause. Pre-upscale placement is a separate integration experiment for the trails already present in the game image.

## 1. What our measurements already establish

These are local observations, independent of the online sources:

- S349: some foot trails exist in the game image before our network.
- S361h: the second pass increases edit strength and often increases trails. A peer detail-only cascade and disabling later-pass history were tried and rejected.
- S362–S364: using first-pass output on newly uncovered pixels helps some horizons, but larger fills and independent pass fills can soften or contaminate surfaces. The residual-only repair still loses on the car clip.
- S365: dormant pass history and a first-pass noise-counter reset on extra-pass allocation were reproduced and fixed. They do not explain continuous two-pass results after the fix.
- S366: history-zero two-pass edits are 1.48–1.57 times stronger than one pass. Strength-matched orthogonal transient change remains 1.08–1.31 times greater. This diagnostic includes legitimate changing edits and is not ground-truth artifact measurement.
- S367: fixed seed reduced still-surface edit RMS by 22–28% across eight clips, while primary moving-surface changes ranged from -1.85% to +0.88%. Edit strength retained 99.78–100.03%. Samples require quiet input, valid motion and depth agreement; fast pans, changing lighting and many edges are excluded.

Current live configuration uses the small model, history0, first-pass uncover fallback and same-frame present processing. Optional stabilization is off. Fixed seed is implemented and validated as an opt-in candidate; its live A/B presets have not been applied.

## 2. Noise that follows the scene: strongest research lead

[How I Warped Your Noise](https://warpyournoise.github.io/) (ICLR 2024) treats a noise pixel as an integral over a white-noise field, transporting the field while addressing the distribution changes caused by ordinary interpolation.

[Go-with-the-Flow](https://arxiv.org/html/2501.08331v2) (CVPR 2025) develops an efficient transport algorithm using flow, density, expansion/contraction and conditional noise. It separately demonstrates training-free applications to image diffusion models; its motion-controllable video models are fine-tuned. The published algorithm requires both forward and backward flow. The later paper also challenges the earlier method's Gaussianity under some deformations, so the earlier guarantee should not be assumed universally.

**Fit to our implementation, inferred from source:** `front_resident.h::k_front_noise` generates three Gaussian channels using PCG and Box–Muller. We control these channels without changing model weights. Our captured motion is current-to-previous; forward flow, collision handling and newly uncovered regions require an explicit design. Negating backward motion at the same coordinate does not produce a general forward flow.

Do not substitute a persistent bilinear-warped texture and call it distribution preserving. A zero-mean Gaussian marginal at each pixel is insufficient: neighboring samples can become correlated.

### Independent CPU feasibility check

A local probe used 512×1024×3 independent samples, seed368, periodic boundaries, and half-pixel translation in both axes. It executes no neural model.

| Transport | Variance | Horizontal correlation |
|---|---:|---:|
| Independent starting field | 1.0019 | 0.0004 |
| Integer translation | 1.0019 | 0.0004 |
| One bilinear half-pixel step | 0.2505 | 0.5001 |
| Same step, variance multiplied back to its initial scale | 1.0018 | 0.5001 |
| Four steps, measured variance normalized each time | 1.0000 | 0.8001 |

Analytically, averaging four independent unit-variance samples with equal weights gives variance1/4; rescaling restores variance but leaves shared-tap covariance. Applying the first-step scale repeatedly assumes independence that is already lost: the probe's fourth-step variance reaches19.18.

Private reproducible evidence: `build/s368/noise_interpolation_probe.py` and its JSON. This checks the interpolation objection, not the paper algorithm or DLSS output quality.

**Next test:** implement and validate transport independently of inference first. Cover identity, integer/subpixel translation, zoom expansion/contraction, collisions, out-of-bounds, depth rejection and reset. Measure variance, spatial/channel correlations, temporal correspondence and long-run drift. Only then compare it with dynamic and fixed seed on the same eight clips, identical controls/history/repair. Decide per-pass ownership explicitly. No performance or quality benefit has yet been measured on our network.

Primary implementation inspected read-only:
[Eyeline-Labs project](https://github.com/Eyeline-Labs/Go-with-the-Flow/tree/018f35148b3afae9bf57aa915d79c4e2d055552c),
which delegates noise transport to
[CommonSource/noise_warp.py](https://github.com/RyannDaGreat/CommonSource/blob/005a67a3efdfcb30491515955101d0a8cfdd62d6/noise_warp.py).
The practical helper and the mathematical pseudocode are not interchangeable specifications. No dependency installer was run.

## 3. A specific confidence gap in our optional stabilizer

The fresh DLSSNR-AMD
[RenoDX composition shader](https://github.com/mochizuki0323/DLSSNR-AMD/blob/c71b0f422239dacf7c3036bc6ee2fc2d0b6ed8a9/linux/shaders/renodx/look_compose.cs_5_1.hlsl)
checks whether the carried input appearance remains within the current 3×3 input range, then clips carried edits to current neighborhood values. Its comments explain why a pointwise appearance comparison rejects high-contrast edges during subpixel motion.

Our `k_stab_apply` uses motion/depth rejection and clamps history around the current low-band edit with a fixed tolerance. It does not retain an input-appearance guide for equivalent confidence testing. Its optional full-band still-pixel history also deserves testing on animated textures and reflections whose geometry remains still.

**Proposed adaptation:** retain a same-frame input guide, reject appearance changes before blending, and compare a current-edit neighborhood clamp against the existing fixed clamp. Preserve a byte-identical off path. Test real lighting changes, equal-depth moving shadows, animated surfaces, disocclusions, subpixel edges and camera cuts. This is a hypothesis about better stabilization, not a cure for stabilization-off ghosting. A wide neighborhood can admit the wrong surface; depth and tap validity still matter.

The peer's
[detail-only cascade](https://github.com/mochizuki0323/DLSSNR-AMD/blob/c71b0f422239dacf7c3036bc6ee2fc2d0b6ed8a9/linux/shaders/passes/cascade_feed.comp)
still removes the low-frequency log-ratio before feeding the next pass. This is the mechanism already tested in S361h; fresh availability does not justify repeating it unchanged.

## 4. New edit-cache work: useful, different problem

Skynizz's
[edit-cache design](https://github.com/Skynizz/optiscaler-dlss5/blob/fc702f8ca22f97b01bd6a02f3386bffb3484fc35/OptiScaler/dlssnr/design/edit-cache.md)
carries a keyframe's luminance/chroma fingerprint and surrounding appearance with its edit. It can invalidate changes that accumulate too slowly for adjacent-frame checks. It also restricts filling to similar surfaces and shortens refresh blending. The accompanying
[host code](https://github.com/Skynizz/optiscaler-dlss5/blob/fc702f8ca22f97b01bd6a02f3386bffb3484fc35/OptiScaler/shaders/dlssnr/DlssNr_EditCache_Dx12.cpp)
contains the fingerprint/meta/context resources and switches.

The reported largest trail reduction compares its new defaults with its older eight-frame cache, using every-frame model output as reference. It is not a measurement of our backend or of error already present in the model's fresh answer.

**Use if we revisit skipped inference or delayed-map reuse.** Carry provenance from the evaluation that produced an edit; repeated adjacent-frame agreement can otherwise retain a stale edit. Do not add a long cache to our current every-frame path merely to adopt its repairs. The peer also reports cases where stronger smoothing/filtering loses correct detail.

## 5. Pipeline placement and missing guides

Fresh [lmxxf documentation](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/c24e3a97cdf596d0c997cee7deda91c981d520ce/README.md)
places NR before the host upscaler. The
[wilsjo2 host](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/blob/1bd39091337cc07ba961c8e59ded57e21dc95b18/README.md)
offers before/after placement and describes pre-Ray-Reconstruction use as experimental.

Our current present path sees the finished game picture. **Inference:** a matched before/after-upscaler comparison could distinguish editing an existing reconstruction trail from producing an additional trail. Pre-upscale placement can also let the upscaler accumulate generated variation, so the outcome cannot be inferred from placement alone. It requires a different hook, correct exposure/encoding, jitter and motion conventions, and render/display extent handling. Start with one pass and frame-matched captures.

AMD's [FSR2 integration documentation](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/super-resolution-temporal/#reactive-mask)
recommends explicit reactive masks for alpha-blended content and composition masks for effects such as reflections and animated textures. RGB, depth and motion alone do not reliably identify transparency. Its automatic helper requires opaque-only and fully composited color.

**Implication:** first inventory which guides the host actually supplies. Our ordinary saved captures do not provide an opaque-only image or a genuine per-material reactive mask. A heuristic from the available image may help, but must not be represented as equivalent engine data. Missing guides limit what a generic present-stage filter can reliably reject.

## 6. Other research worth understanding, lower immediate priority

- [NVIDIA's DLSS5 research overview](https://research.nvidia.com/labs/adlr/DLSS5/) describes a one-step pixel-space diffusion model conditioned on motion, temporal state and artistic controls, trained for temporal stability. Turning history off removes part of that inference setup. This supports auditing state validity/conditioning, not assuming history0 is a complete stability solution. The linked report PDF returned403; no conclusions here depend on its unread contents.
- AMD's [temporally stable generative illumination](https://gpuopen.com/learn/temporally-stable-generative-illumination/) uses a trained temporal decoder with reprojected prior output and geometry/material conditions. It is a relevant architectural example, but requires training and inputs our current model path does not offer. It is not a replacement decoder we can enable in the existing weights.
- [Spatiotemporal blue-noise masks](https://research.nvidia.com/publication/2022-07_spatiotemporal-blue-noise-masks) improve sampling error and temporal convergence in stochastic rendering. Spatial blue-noise correlation changes the distribution presented to our pretrained network. This does not establish that replacing its Gaussian white-noise channels is beneficial; defer behind distribution-preserving transport.
- [NRD history-confidence guidance](https://github.com/NVIDIA-RTX/NRD#history-confidence) distinguishes true lighting change from sampling variation by retracing with an earlier RNG seed, with geometry/normal-aware processing. We cannot reproduce that radiance test from a final color image. It reinforces the need for causal confidence rather than additional blind accumulation.
- [Intel CGVQM](https://www.intel.com/content/www/us/en/developer/articles/technical/cgvqm-d-computer-graphics-video-quality.html) is a full-reference perceptual video metric with error maps. It may complement our diagnostics where a valid reference exists. Neither NR-off video nor a different NR style is ground truth for the intended edited picture; the metric cannot supply that missing reference.

## 7. Ranked experiments and decision gates

| Priority | Experiment | Resolves | Required evidence before retention |
|---|---|---|---|
| Immediate live check | Existing S367 dynamic/fixed-seed presets | Whether measured still stability is perceptually useful | Normal-speed A/B, moving textured surfaces, reflections, preserved edit strength |
| First new mechanism | Distribution-preserving motion-correlated noise | Screen-fixed noise versus scene-following noise | Noise-statistics contracts first; then native eight-clip comparison and live motion QA |
| Second, conditional | Input-appearance confidence and neighborhood clamp for stabilization | Wrong temporal reuse with stable geometry | Identical off path; compare stabilization at equal strength; no delayed real lighting or disocclusion trails |
| Third, integration | Matched pre-/post-upscale one-pass capture | Existing upstream trails versus NR placement | Same scene, guide/exposure audit, fresh input and edited output from both stages |
| Later performance branch | Fingerprint-validated edit cache | Stale corrections when inference is skipped | Compare against fresh model output, report retained effect and motion failures separately |

For any model-output experiment, keep one/two-pass controls, report raw and edit-strength-matched stability, preserve fast-motion sample counts, and inspect full-speed clips. Quiet-input RMS alone does not establish ghost reduction. New shadow, reflection and transparency clips are needed because S367 deliberately filters many changing-input pixels out.

## 8. Preservation and reproducibility

Research used primary author papers/pages and pinned peer source. No peer binaries, weights or installers were downloaded or executed. No GPU inference was needed for this research pass.

Verified unchanged during this pass:

- Deployed engine SHA256: `294821062CEC5E1F98810A4B08D9BD253C6388E521594CD01636A546148DE198`.
- Cyberpunk shim INI SHA256: `A592672A291BBF5E884E55B78BFA86292680C634EDE6C1366A666AEE0FF94AAB`.

S367 live presets remain unapplied. This report adds no production option or quality claim. The CPU probe measures interpolation only; previous engine regressions remain the applicable evidence for the unchanged implementation.

