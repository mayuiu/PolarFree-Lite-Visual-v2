# AGENTS.md — PolarFree-Lite-Visual v2

## 1. Project identity

This repository is a lightweight polarization-image reflection removal project.

The project goal is **PolarFree-Lite-Visual v2**:

- Use 4-angle polarization images as input.
- Build physical priors from polarization cues.
- Use a lightweight deep model to predict clean image, reflection residual, confidence masks, and visual fusion outputs.
- Improve strong reflection, strong glare, low-frequency glass haze, line-band reflection, building/window reflection, and colored reflection removal.
- Prioritize **visual quality for undergraduate thesis defense demonstration**, while keeping the method explainable and runnable on limited hardware.

This project is not trying to reproduce the full PolarFree diffusion model. It is a low-compute, interpretable, polarization-prior-guided reflection removal system.

---

## 2. Main objective for current development

The current problem is:

- Strong glare regions are not removed aggressively enough.
- Horizontal light bands remain visible.
- Large low-frequency glass haze is under-suppressed.
- Window/building reflections remain obvious.
- `visual_plus` can be too similar to `raw` or `prior`.
- `DirectClean` may improve the image, but later fusion may pull the result back toward `raw`, `reference`, or `prior`.

The global principle for all modifications is:

> Strong reflection / strong glare regions should be handled aggressively by `DirectClean` and residual subtraction.  
> Non-reflection background regions should preserve color, texture, and structure.  
> Final `visual_plus` / `visual_extreme` / `showcase` outputs should have strong visual impact.

Do not optimize only for PSNR/SSIM. For showcase mode, visual suppression of reflection is more important.

---

## 3. Important files

Prioritize these files:

- `main.py`
- `polarfree_lite_glass_v2.py`
- `pfl_config.py`
- `pfl_data.py`
- `pfl_model.py`
- `pfl_losses.py`
- `pfl_eval.py`

If the active main file is named differently, such as `main(5).py`, inspect imports carefully and avoid breaking module names.

Do not create a new project from scratch. Modify the existing codebase.

---

## 4. Safety rules

Never do the following unless the user explicitly asks:

- Do not run training.
- Do not run inference.
- Do not run long experiments.
- Do not delete datasets.
- Do not delete `D:\jibi\data`.
- Do not delete `D:\jibi\train`.
- Do not clear all caches automatically.
- Do not install packages.
- Do not use internet/network access.
- Do not change unrelated files.
- Do not perform broad refactors unrelated to the task.

Allowed lightweight syntax check:

```powershell
E:\ana\envs\pytorch\python.exe -m py_compile .\main.py .\polarfree_lite_glass_v2.py .\pfl_config.py .\pfl_data.py .\pfl_model.py .\pfl_losses.py .\pfl_eval.py
```

Only run this syntax check if the environment and file names are valid.

If command execution requires approval, ask first.

---

## 5. Compatibility rules

Before changing channels, heads, or cache keys, inspect the current config.

Pay special attention to:

- `MODEL_INPUT_CHANNELS`
- `MODEL_OUTPUT_CHANNELS`
- `PRIOR_CACHE_VERSION`
- `PACKAGE_CACHE_VERSION`
- `PACKAGE_REQUIRED_KEYS`
- `DEBUG_PANEL_EXTRA_FIELDS`
- `METRIC_FIELDS`

If changing input channels, output channels, package keys, or cached tensor content, update cache version strings.

If not changing those, do not unnecessarily change cache versions.

Preserve old CLI arguments as much as possible.

Do not remove existing functions unless they are definitely unused and removal is safe.

---

## 6. Core algorithm direction

The desired algorithm structure is:

```text
4-angle polarization images
        ↓
polarization feature extraction
        ↓
safe_prior / visual_prior / aggressive_prior
        ↓
multi-source reflection detection
        ↓
Reflection-aware UNet
        ↓
DirectClean + predicted reflection residual + confidence masks
        ↓
strong-reflection takeover fusion
        ↓
visual_plus / visual_extreme / showcase output
```

The most important principle:

```text
If a region is high-confidence reflection or strong glare:
    final output should strongly favor DirectClean / residual clean.
Else:
    final output should preserve safe background.
```

---

## 7. Strong glare and reflection mask improvement

Current glare detection may be too dependent on:

```text
bright * low_sat
```

This can miss colored glare, yellow/green glare, large glass haze, and line-band reflection.

Improve glare detection using multiple cues:

- HSV value high score.
- RGB max high score.
- Relaxed saturation score.
- Local over-exposure score.
- Low-frequency veil from `reference_rgb - prior_rgb`.
- Prior difference.
- DirectClean difference when available.
- `build_line_glare_response`.
- Reflection area mask.
- Low-frequency reflection mask.

Add or strengthen masks such as:

- `strong_glare_core`
- `highlight_core_mask`
- `glare_veil_mask`
- `takeover_mask`

The mask should be slightly blurred and/or dilated to avoid harsh boundaries.

Colored glare must not be suppressed by overly strict low-saturation filtering.

---

## 8. Loss improvement

In `pfl_losses.py`, strengthen strong glare, low-frequency reflection, and line-band supervision.

Current focus should be adjusted so glare and glare core receive higher priority.

Add or strengthen these loss terms if compatible with existing code:

- `highlight_residual_loss`
- `highlight_clean_loss`
- `highlight_lowfreq_loss`
- `highlight_chroma_loss`
- `line_band_suppression_loss`
- `glare_core_l1`
- `strong_glare_direct_clean_l1`
- `visualplus_should_differ_from_raw_in_reflection_region`
- `prior_escape_loss` in reflection regions
- `background_protection_loss` only in safe non-reflection regions

Important loss principle:

```text
Strong reflection regions:
    allow aggressive correction.
    do not over-penalize deviation from prior/reference.

Non-reflection regions:
    strongly protect structure, color, and texture.
```

All new loss terms must be included in the returned `loss_items` dictionary for logging.

Do not remove existing loss logic without a clear reason.

---

## 9. DirectClean and residual strengthening

Inspect:

- `decode_reflection_heads`
- `compose_prediction`
- any logic related to `direct_clean`
- any logic related to `positive_reflection`
- any logic related to `signed_clean_delta`

Desired behavior:

```text
direct_clean ≈ reference_rgb - positive_reflection * support * strength + signed_clean_delta
```

In strong glare / high-confidence reflection / low-frequency reflection / line-band regions:

- Increase residual subtraction strength.
- Increase `DirectClean` influence.
- Allow stronger signed correction.
- Do not average away DirectClean using prior or raw prediction.

If `abs_clean` exists, keep it as auxiliary support, but do not let it weaken the residual-clean path in strong reflection areas.

---

## 10. Fusion improvement

This is the most important part.

In `compose_prediction` and evaluation rendering functions, implement or strengthen strong-reflection takeover fusion.

Recommended logic:

```python
takeover = max(
    reflection_confidence,
    line_band_confidence,
    text_reflection_confidence,
    lowfreq_reflection_confidence,
    dark_shadow_confidence,
    reflection_area_mask,
    glare_mask,
    strong_glare_core,
)
```

Fusion principle:

```text
takeover low:
    use safe background / protected result

takeover medium:
    mix DirectClean, raw prediction, and prior

takeover high:
    strongly favor DirectClean

strong_glare_core high:
    output should be almost DirectClean
```

In strong glare core regions:

- Allow `force_direct` up to `0.98` or `0.995`.
- Reduce raw/prior contribution.
- Avoid pulling the result back to reference/prior.

For `visual_plus`, increase DirectClean contribution in strong reflection areas.

Optionally add a new render mode:

```text
visual_extreme
```

This mode may be more aggressive and intended for thesis defense showcase images.

---

## 11. Evaluation and debug outputs

In `pfl_eval.py`, add or strengthen outputs that help visually judge reflection suppression.

Recommended debug outputs:

- `strong_glare_core_gray_path`
- `takeover_mask_path`
- `visual_extreme_result_path`
- `direct_clean_path`
- `raw_result_path`
- `result_path`
- `direct_clean_minus_prior_path`
- `visual_plus_minus_prior_path`
- `blend_gate_path`
- `final_mask_path`

Recommended metrics:

- `highlight_region_l1`
- `glare_suppression_ratio`
- `line_band_suppression_ratio`
- `lowfreq_suppression_ratio`
- `foreground_damage_ratio`
- `raw_visualplus_diff_in_shadow`
- `direct_vs_prior_focus_improvement`
- `visual_vs_prior_focus_improvement`

If adding metric fields, update `pfl_config.py`.

Do not remove PSNR/SSIM/L1; keep them for thesis reporting.

---

## 12. Hard case mechanism

Preserve and strengthen hard case weighting.

Important failure types include:

- `line_band`
- `horizontal_band`
- `diagonal_band`
- `text_reflection`
- `green_tint`
- `yellow_tint`
- `scene_reflection`
- `tree_shadow`
- `building_reflection`
- `dark_reflection`
- `lowfreq_reflection`
- `shadow_veil`
- `reflection_shadow_mix`
- `under_suppressed_shadow`
- `missed_glare`
- `over_smooth`
- `prior_like_output`
- `visual_equals_raw`

If `hard_case_weight` is available, make sure it actually increases the influence of difficult examples in relevant loss terms.

It should affect, when possible:

- highlight residual loss
- highlight low-frequency loss
- line-band suppression loss
- residual clean loss
- DirectClean loss
- visual-plus-different-from-raw loss

---

## 13. Training command output requirement

After modifying code, provide recommended commands for these stages:

1. DirectClean pretraining.
2. Visual main training.
3. Strong glare / hard case fine-tuning.
4. Showcase fine-tuning.
5. Evaluation / image saving command if available.

Use the user's environment:

```text
OS: Windows 11
Conda env: pytorch
Python path: E:\ana\envs\pytorch\python.exe
Project path: C:\Users\86166\Desktop\PythonProject
Train root: D:\jibi\train
Test root: D:\jibi\data\test
Output root: D:\jibi\output
GPU: RTX 4050 Laptop GPU
```

Preferred training settings:

```text
--model quality
--img_size 384
--batch_size 1 or 2
--grad_accum_steps 4 or 2
--amp
--channels_last
--prior_profile visual
--mask_profile aggressive
--loss_profile visual
--render-mode visual_plus or visual_extreme
--package_cache
--save_eval_images
--test_limit 20
```

Do not run these commands unless the user explicitly asks.

---

## 14. Response format after code changes

After modifications, always report:

1. Files modified.
2. Summary of each file's changes.
3. Whether channels or cache keys changed.
4. Whether old checkpoints remain compatible.
5. Whether cache clearing is required.
6. New CLI arguments, if any.
7. Recommended training commands.
8. Risk points.
9. Rollback strategy.
10. Whether `py_compile` passed, if it was run.

Do not claim training success unless training was actually run.

Do not claim visual improvement unless images were actually generated and inspected.

---

## 15. Coding style

- Keep changes focused.
- Keep function signatures backward compatible when possible.
- Add comments for new algorithmic logic.
- Avoid magic constants without short comments.
- Prefer small helper functions over duplicating large blocks.
- Avoid unrelated formatting-only changes.
- Do not remove Chinese or English comments unless they are wrong.
- Preserve existing logging style.

---

## 16. Final goal

The final implementation should make the system behave like:

```text
PolarFree-Lite-Visual v2
=
multi-source strong glare detection
+ reflection residual learning
+ DirectClean takeover fusion
+ strong glare / low-frequency / line-band losses
+ hard case weighting
+ visual_plus / visual_extreme showcase rendering
+ background protection outside reflection regions
```

The most important expected result:

```text
visual_plus / visual_extreme should be visibly stronger than raw and prior in strong reflection regions,
while non-reflection background should remain natural.
```
