import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import main
from pfl_config import DEBUG_PANEL_EXTRA_FIELDS, METRIC_FIELDS
from pfl_data import PolarFreeLiteDataset, SampleRecord, build_line_glare_response
from pfl_losses import average_pool_ssim, weighted_l1
from pfl_model import compose_prediction, compute_blend_gate, decode_reflection_heads

EPS = 1e-6


def compute_metrics(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    l1 = F.l1_loss(prediction, target)
    focus_l1 = weighted_l1(prediction, target, mask)
    ssim = average_pool_ssim(prediction, target)
    mse = torch.mean((prediction - target) ** 2).detach().cpu().item()
    psnr = 99.0 if mse <= EPS else 10.0 * math.log10(1.0 / mse)
    return {
        "l1": float(l1.detach().cpu().item()),
        "focus_l1": float(focus_l1.detach().cpu().item()),
        "psnr": psnr,
        "ssim": float(ssim.detach().cpu().item()),
    }

def tensor_to_numpy_rgb(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    return np.transpose(np.clip(array, 0.0, 1.0), (1, 2, 0)).astype(np.float32)

def tensor_to_numpy_mask(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if array.ndim == 3:
        array = np.transpose(array, (1, 2, 0))
    return main.ensure_single_channel(np.clip(array, 0.0, 1.0).astype(np.float32))

def blur_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    mask_2d = main.ensure_single_channel(mask).squeeze(-1).astype(np.float32)
    blurred = cv2.GaussianBlur(mask_2d, (0, 0), sigma)
    return np.clip(blurred[..., np.newaxis], 0.0, 1.0)

def build_glare_mask(reference_rgb: np.ndarray) -> np.ndarray:
    reference_u8 = (np.clip(reference_rgb, 0.0, 1.0) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(reference_u8, cv2.COLOR_RGB2HSV)
    value = hsv[..., 2].astype(np.float32) / 255.0
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    bright = main.robust_percentile_normalize(value, 62, 99)
    rgb_max = np.max(np.clip(reference_rgb, 0.0, 1.0), axis=2).astype(np.float32)
    rgb_high = main.robust_percentile_normalize(rgb_max, 60, 99)
    relaxed_sat = np.clip((0.84 - saturation) / 0.84, 0.0, 1.0)
    local_over = main.robust_percentile_normalize(np.clip(value - cv2.GaussianBlur(value, (0, 0), 5.5), 0.0, 1.0), 58, 99)
    colored_glare = bright * np.clip((saturation - 0.08) / 0.66, 0.0, 1.0)
    glare = np.clip(
        0.44 * bright * (0.25 + 0.75 * relaxed_sat)
        + 0.22 * rgb_high
        + 0.18 * local_over
        + 0.16 * colored_glare,
        0.0,
        1.0,
    )
    return blur_mask(glare[..., np.newaxis], sigma=5.0)

def gradient_strength_np(image: np.ndarray) -> np.ndarray:
    image = np.clip(image.astype(np.float32), 0.0, 1.0)
    gray = np.mean(image, axis=2).astype(np.float32)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(dx * dx + dy * dy)[..., np.newaxis]

def compute_visual_takeover_debug_masks(
    reference_rgb: np.ndarray,
    prior_rgb: np.ndarray,
    raw_prediction_rgb: np.ndarray,
    reflection_mask: np.ndarray,
    direct_clean_rgb: Optional[np.ndarray] = None,
    reflection_confidence: Optional[np.ndarray] = None,
    line_band_confidence: Optional[np.ndarray] = None,
    text_reflection_confidence: Optional[np.ndarray] = None,
    lowfreq_reflection_confidence: Optional[np.ndarray] = None,
    dark_shadow_confidence: Optional[np.ndarray] = None,
    reflection_area_mask: Optional[np.ndarray] = None,
    lowfreq_reflection_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    direct_source = direct_clean_rgb if direct_clean_rgb is not None else raw_prediction_rgb
    mask = blur_mask(reflection_mask, sigma=5.2)
    area = blur_mask(reflection_area_mask if reflection_area_mask is not None else reflection_mask, sigma=4.0)
    glare_mask = build_glare_mask(reference_rgb)
    prior_diff = blur_mask(np.mean(np.abs(reference_rgb - prior_rgb), axis=2, keepdims=True), sigma=7.0)
    direct_diff = blur_mask(np.mean(np.abs(direct_source - prior_rgb), axis=2, keepdims=True), sigma=4.8)
    line_score = build_line_glare_response(reference_rgb)
    veil_score = blur_mask(np.mean(np.clip(reference_rgb - prior_rgb, 0.0, 1.0), axis=2, keepdims=True), sigma=16.0)

    conf = np.clip(
        0.24 * mask
        + 0.18 * area
        + 0.18 * glare_mask
        + 0.15 * line_score
        + 0.12 * veil_score
        + 0.08 * prior_diff
        + 0.05 * direct_diff,
        0.0,
        1.0,
    )
    for extra in (
        reflection_confidence,
        line_band_confidence,
        text_reflection_confidence,
        lowfreq_reflection_confidence,
        dark_shadow_confidence,
        lowfreq_reflection_mask,
    ):
        if extra is not None:
            conf = np.maximum(conf, 0.72 * blur_mask(extra, sigma=3.0))
    strong_core = blur_mask(np.clip((conf - 0.46) / 0.34, 0.0, 1.0), sigma=2.0)
    strong_core = np.maximum(
        strong_core,
        blur_mask(np.clip((np.maximum(glare_mask, line_score) - 0.44) / 0.40, 0.0, 1.0), sigma=1.6),
    )
    takeover = blur_mask(np.clip(0.54 * conf + 0.42 * strong_core + 0.22 * area + 0.18 * mask, 0.0, 1.0), sigma=3.2)
    return {
        "glare_mask": glare_mask.astype(np.float32),
        "strong_glare_core": strong_core.astype(np.float32),
        "takeover_mask": takeover.astype(np.float32),
        "line_score": line_score.astype(np.float32),
    }

def visual_strong_render_components(
    reference_rgb: np.ndarray,
    prior_rgb: np.ndarray,
    visual_prior_rgb: np.ndarray,
    raw_prediction_rgb: np.ndarray,
    reflection_mask: np.ndarray,
    direct_clean_rgb: Optional[np.ndarray] = None,
    showcase_mode: bool = False,
    showcase_hard_mode: bool = False,
    detail_strength: float = 0.10,
    smooth_strength: float = 0.14,
    tint_suppress: float = 0.12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    direct_source = direct_clean_rgb if direct_clean_rgb is not None else raw_prediction_rgb
    mask = blur_mask(reflection_mask, sigma=5.2)
    glare_mask = build_glare_mask(reference_rgb)
    prior_diff = blur_mask(np.mean(np.abs(reference_rgb - prior_rgb), axis=2, keepdims=True), sigma=7.0)
    direct_diff = blur_mask(np.mean(np.abs(direct_source - prior_rgb), axis=2, keepdims=True), sigma=4.8)
    line_score = build_line_glare_response(reference_rgb)
    veil_score = blur_mask(np.mean(np.clip(reference_rgb - prior_rgb, 0.0, 1.0), axis=2, keepdims=True), sigma=16.0)

    confidence = np.clip(
        0.34 * mask
        + 0.18 * prior_diff
        + 0.16 * glare_mask
        + 0.13 * line_score
        + 0.11 * veil_score
        + 0.08 * direct_diff,
        0.0,
        1.0,
    )
    boundary = blur_mask(np.clip((confidence - 0.18) / 0.46, 0.0, 1.0), sigma=3.8)
    core = blur_mask(np.clip((confidence - 0.38) / 0.38, 0.0, 1.0), sigma=2.4)
    strong_core = blur_mask(np.clip((confidence - 0.62) / 0.28, 0.0, 1.0), sigma=1.7)
    debug_masks = compute_visual_takeover_debug_masks(reference_rgb, prior_rgb, raw_prediction_rgb, reflection_mask, direct_clean_rgb=direct_source)
    strong_core = np.maximum(strong_core, debug_masks["strong_glare_core"])
    boundary = np.maximum(boundary, np.clip(0.55 * debug_masks["takeover_mask"] + 0.45 * strong_core, 0.0, 1.0))
    core = np.maximum(core, np.clip((debug_masks["takeover_mask"] - 0.26) / 0.50, 0.0, 1.0))

    veil = blur_mask(np.mean(np.clip(reference_rgb - prior_rgb, 0.0, 1.0), axis=2, keepdims=True), sigma=12.0)
    glare_veil = blur_mask(np.mean(np.clip(reference_rgb - 0.72 * prior_rgb, 0.0, 1.0), axis=2, keepdims=True), sigma=18.0)
    clean_mix = np.clip(0.42 * raw_prediction_rgb + 0.46 * direct_source + 0.08 * visual_prior_rgb + 0.04 * prior_rgb, 0.0, 1.0)
    suppressed = np.clip(
        clean_mix
        - (0.045 + 0.095 * core) * veil
        - (0.035 + 0.075 * glare_mask + 0.045 * line_score) * glare_veil,
        0.0,
        1.0,
    )
    candidate = np.clip(clean_mix * (1.0 - 0.42 * core) + suppressed * (0.42 * core), 0.0, 1.0)

    prior_luma = np.mean(prior_rgb, axis=2, keepdims=True)
    candidate_luma = np.mean(candidate, axis=2, keepdims=True)
    reference_luma = np.mean(reference_rgb, axis=2, keepdims=True)
    dark_penalty = np.clip((prior_luma - candidate_luma - 0.10) / 0.22, 0.0, 1.0)
    lowfreq_shift = np.abs(
        blur_mask(candidate_luma - reference_luma, sigma=15.0)
        - blur_mask(prior_luma - reference_luma, sigma=15.0)
    )
    lowfreq_penalty = np.clip((lowfreq_shift - 0.035) / 0.16, 0.0, 1.0)
    candidate_center = candidate - candidate_luma
    prior_center = prior_rgb - prior_luma
    color_penalty = np.clip((np.mean(np.abs(candidate_center - prior_center), axis=2, keepdims=True) - 0.040) / 0.16, 0.0, 1.0)
    prior_grad = blur_mask(gradient_strength_np(prior_rgb), sigma=3.0)
    candidate_grad = blur_mask(gradient_strength_np(candidate), sigma=3.0)
    blur_penalty = np.clip((prior_grad - candidate_grad - 0.018) / 0.09, 0.0, 1.0)
    quality = np.clip(
        1.0 - 0.34 * dark_penalty - 0.28 * lowfreq_penalty - 0.20 * color_penalty - 0.22 * blur_penalty,
        0.10,
        1.0,
    )
    alpha = np.clip(0.04 + 0.22 * boundary + 0.30 * core + 0.14 * strong_core, 0.0, 0.70)
    alpha = np.clip(alpha * quality, 0.0, 0.70)
    outside = np.clip(1.0 - np.maximum(boundary, mask), 0.0, 1.0)
    alpha = np.clip(alpha * (1.0 - 0.78 * outside), 0.0, 0.70)
    alpha = np.where(strong_core > 0.75, np.clip(alpha + 0.035 * strong_core, 0.0, 0.72), alpha)

    force_direct = np.clip(0.38 * mask + 0.55 * strong_core + 0.20 * boundary, 0.0, 0.92)
    alpha = np.maximum(alpha, force_direct)
    candidate = np.clip(candidate * (1.0 - force_direct) + direct_source * force_direct, 0.0, 1.0)
    visual_plus = np.clip(raw_prediction_rgb * (1.0 - alpha) + candidate * alpha, 0.0, 1.0)
    detail = prior_rgb - cv2.GaussianBlur(prior_rgb.astype(np.float32), (0, 0), 1.0)
    visual_plus = np.clip(visual_plus + 0.016 * np.clip(1.0 - alpha, 0.0, 1.0) * detail, 0.0, 1.0)
    if showcase_mode:
        support = np.clip(np.maximum(mask, np.maximum(core, glare_mask)), 0.0, 1.0)
        support_hard = np.clip((support - 0.08) / 0.92, 0.0, 1.0)
        support_hard = cv2.dilate(main.ensure_single_channel(support_hard).squeeze(-1).astype(np.float32), np.ones((5, 5), np.uint8))[..., np.newaxis]
        if showcase_hard_mode:
            support_hard = cv2.dilate(main.ensure_single_channel(support_hard).squeeze(-1).astype(np.float32), np.ones((7, 7), np.uint8))[..., np.newaxis]
        support_color = np.clip((support - 0.25) / 0.55, 0.0, 1.0)
        force_direct = np.clip(0.55 * support + 0.80 * np.maximum(strong_core, support_hard) + 0.28 * boundary + 0.18 * glare_mask + 0.18 * mask, 0.0, 0.995)
        alpha = np.maximum(alpha, force_direct)
        visual_plus = np.clip(visual_plus * (1.0 - force_direct) + direct_source * force_direct, 0.0, 1.0)
        rgb_mean = np.mean(visual_plus, axis=2, keepdims=True)
        green_bias = np.maximum(visual_plus[..., 1:2] - rgb_mean, 0.0)
        red_bias = np.maximum(visual_plus[..., 0:1] - rgb_mean, 0.0)
        visual_plus[..., 1:2] = visual_plus[..., 1:2] - float(tint_suppress) * green_bias * support_color
        visual_plus[..., 0:1] = visual_plus[..., 0:1] - 0.45 * float(tint_suppress) * red_bias * support_color
        reference_detail = reference_rgb - cv2.GaussianBlur(reference_rgb.astype(np.float32), (0, 0), 1.2)
        safe_detail_mask = np.clip(1.0 - support, 0.0, 1.0)
        visual_plus = visual_plus + float(detail_strength) * reference_detail * safe_detail_mask
        low_showcase = cv2.GaussianBlur(visual_plus.astype(np.float32), (0, 0), 1.0)
        smooth_alpha = np.clip(float(smooth_strength) * support_hard, 0.0, 0.45)
        visual_plus = visual_plus * (1.0 - smooth_alpha) + low_showcase * smooth_alpha
        visual_plus = np.clip(visual_plus, 0.0, 1.0)
    return visual_plus.astype(np.float32), candidate.astype(np.float32), alpha.astype(np.float32)

def render_prediction_output(
    reference_rgb: np.ndarray,
    prior_rgb: np.ndarray,
    visual_prior_rgb: np.ndarray,
    raw_prediction_rgb: np.ndarray,
    reflection_mask: np.ndarray,
    render_mode: str,
    direct_clean_rgb: Optional[np.ndarray] = None,
    showcase_mode: bool = False,
    showcase_hard_mode: bool = False,
    detail_strength: float = 0.10,
    smooth_strength: float = 0.14,
    tint_suppress: float = 0.12,
) -> np.ndarray:
    if render_mode == "raw":
        return raw_prediction_rgb.astype(np.float32)

    mask = blur_mask(reflection_mask, sigma=5.5)
    core = blur_mask((mask > 0.30).astype(np.float32), sigma=3.2)
    glare_mask = build_glare_mask(reference_rgb)
    veil = blur_mask(np.mean(np.clip(reference_rgb - prior_rgb, 0.0, 1.0), axis=2, keepdims=True), sigma=12.0)
    glare_veil = blur_mask(np.mean(np.clip(reference_rgb - 0.72 * prior_rgb, 0.0, 1.0), axis=2, keepdims=True), sigma=18.0)

    if render_mode == "hybrid":
        suppressed = np.clip(0.78 * prior_rgb + 0.22 * raw_prediction_rgb - (0.07 + 0.10 * core) * veil, 0.0, 1.0)
        return np.clip(raw_prediction_rgb * (1.0 - mask) + suppressed * mask, 0.0, 1.0).astype(np.float32)

    if render_mode not in {"visual", "visual_plus", "visual_extreme"}:
        raise ValueError(f"Unsupported render mode: {render_mode}")

    if render_mode in {"visual_plus", "visual_extreme"}:
        visual_plus, _candidate, _alpha = visual_strong_render_components(
            reference_rgb,
            prior_rgb,
            visual_prior_rgb,
            raw_prediction_rgb,
            reflection_mask,
            direct_clean_rgb=direct_clean_rgb,
            showcase_mode=showcase_mode or render_mode == "visual_extreme",
            showcase_hard_mode=showcase_hard_mode or render_mode == "visual_extreme",
            detail_strength=detail_strength,
            smooth_strength=smooth_strength,
            tint_suppress=tint_suppress,
        )
        if render_mode == "visual_extreme":
            masks = compute_visual_takeover_debug_masks(reference_rgb, prior_rgb, raw_prediction_rgb, reflection_mask, direct_clean_rgb=direct_clean_rgb)
            direct_source = direct_clean_rgb if direct_clean_rgb is not None else raw_prediction_rgb
            takeover = np.clip(0.72 * masks["takeover_mask"] + 0.42 * masks["strong_glare_core"], 0.0, 0.995)
            visual_plus = np.clip(visual_plus * (1.0 - takeover) + direct_source * takeover, 0.0, 1.0)
        return visual_plus

    stronger_prior = np.clip(0.70 * visual_prior_rgb + 0.25 * prior_rgb + 0.05 * raw_prediction_rgb, 0.0, 1.0)
    detail = raw_prediction_rgb - cv2.GaussianBlur(raw_prediction_rgb, (0, 0), 1.1)
    suppressed = np.clip(
        stronger_prior
        - (0.10 + 0.18 * core) * veil
        - (0.07 + 0.16 * glare_mask) * glare_veil
        + 0.035 * (1.0 - mask) * detail,
        0.0,
        1.0,
    )

    blended = np.clip(raw_prediction_rgb * (1.0 - 0.88 * mask) + suppressed * (0.88 * mask), 0.0, 1.0)
    blended_u8 = (blended * 255).astype(np.uint8)
    lab = cv2.cvtColor(blended_u8, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    clahe_rgb = cv2.cvtColor(cv2.merge([clahe.apply(l_channel), a_channel, b_channel]), cv2.COLOR_LAB2RGB)
    clahe_rgb = clahe_rgb.astype(np.float32) / 255.0
    contrast = np.clip(blended * (1.0 - 0.08 * mask) + clahe_rgb * (0.08 * mask), 0.0, 1.0)
    gamma_target = np.clip(1.01 + 0.08 * mask, 1.01, 1.09)
    gamma_adjusted = np.power(np.clip(contrast, 0.0, 1.0), gamma_target)
    return np.clip(contrast * (1.0 - 0.18 * mask) + gamma_adjusted * (0.18 * mask), 0.0, 1.0).astype(np.float32)

def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int = 4,
    pin_memory: bool = False,
) -> DataLoader:
    kwargs: Dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)

def batch_to_device(batch: Mapping[str, object], device: torch.device, channels_last: bool = False) -> Dict[str, object]:
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if channels_last and value.ndim == 4:
                moved[key] = value.to(device=device, non_blocking=True, memory_format=torch.channels_last)
            else:
                moved[key] = value.to(device=device, non_blocking=True)
        else:
            moved[key] = value
    return moved

def loader_runtime_config(num_workers: int, prefetch_factor: int, batch_size: int, pin_memory: bool) -> Dict[str, object]:
    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": prefetch_factor if num_workers > 0 else "disabled",
    }

def save_mask_gray(path: Path, mask: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_2d = np.clip(main.ensure_single_channel(mask).squeeze(-1), 0.0, 1.0)
    cv2.imwrite(str(path), (mask_2d * 255.0).astype(np.uint8))
    return str(path)

def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    mask_2d = np.clip(main.ensure_single_channel(mask).squeeze(-1), 0.0, 1.0)
    return np.repeat(mask_2d[..., np.newaxis], 3, axis=2).astype(np.float32)

def diff_heatmap(reference_rgb: np.ndarray, prior_rgb: np.ndarray) -> np.ndarray:
    diff = np.mean(np.abs(reference_rgb - prior_rgb), axis=2)
    diff = main.robust_percentile_normalize(diff, 20, 98)
    heat_bgr = cv2.applyColorMap((np.clip(diff, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

def save_debug_mask_panel(
    path: Path,
    scene: str,
    group: str,
    reference_rgb: np.ndarray,
    prior_rgb: np.ndarray,
    glare_mask: np.ndarray,
    reflection_area_mask: np.ndarray,
    final_mask: np.ndarray,
    blend_gate: Optional[np.ndarray],
    safe_gate: Optional[np.ndarray],
    direct_clean_rgb: Optional[np.ndarray],
    raw_rgb: np.ndarray,
    raw_minus_prior: Optional[np.ndarray],
    direct_clean_minus_prior: Optional[np.ndarray],
    target_rgb: Optional[np.ndarray],
    visual_plus_rgb: Optional[np.ndarray] = None,
    visual_plus_minus_prior: Optional[np.ndarray] = None,
    strong_glare_core: Optional[np.ndarray] = None,
    takeover_mask: Optional[np.ndarray] = None,
    visual_extreme_rgb: Optional[np.ndarray] = None,
    scene_reflection_mask: Optional[np.ndarray] = None,
    dark_reflection_mask: Optional[np.ndarray] = None,
    shadow_veil_mask: Optional[np.ndarray] = None,
    lowfreq_reflection_mask: Optional[np.ndarray] = None,
    foreground_structure_guard: Optional[np.ndarray] = None,
    reflection_confidence: Optional[np.ndarray] = None,
    line_band_confidence: Optional[np.ndarray] = None,
    text_reflection_confidence: Optional[np.ndarray] = None,
    tint_confidence: Optional[np.ndarray] = None,
    lowfreq_reflection_confidence: Optional[np.ndarray] = None,
    dark_shadow_confidence: Optional[np.ndarray] = None,
    showcase_mode: bool = False,
    showcase_hard_mode: bool = False,
) -> str:
    panels: List[Tuple[str, np.ndarray]] = [
        ("0deg", reference_rgb),
        ("Prior", prior_rgb),
        ("AbsDiff", diff_heatmap(reference_rgb, prior_rgb)),
        ("GlareMask", mask_to_rgb(glare_mask)),
        ("ReflectionMask", mask_to_rgb(reflection_area_mask)),
        ("FinalMask", mask_to_rgb(final_mask)),
    ]
    extra_masks = {
        "StrongGlareCore": strong_glare_core,
        "TakeoverMask": takeover_mask,
        "SceneReflectionMask": scene_reflection_mask,
        "DarkReflectionMask": dark_reflection_mask,
        "ShadowVeilMask": shadow_veil_mask,
        "LowfreqReflectionMask": lowfreq_reflection_mask,
        "ForegroundStructureGuard": foreground_structure_guard,
        "ReflectionConf": reflection_confidence,
        "LineBandConf": line_band_confidence,
        "TextReflectionConf": text_reflection_confidence,
        "TintConf": tint_confidence,
        "LowfreqReflectionConf": lowfreq_reflection_confidence,
        "DarkShadowConf": dark_shadow_confidence,
    }
    for label in DEBUG_PANEL_EXTRA_FIELDS:
        panel_mask = extra_masks.get(label)
        if panel_mask is not None:
            panels.append((label, mask_to_rgb(panel_mask)))
    if blend_gate is not None:
        panels.append(("BlendGate", mask_to_rgb(blend_gate)))
    if safe_gate is not None:
        panels.append(("SafeGate", mask_to_rgb(safe_gate)))
    if direct_clean_rgb is not None:
        panels.append(("DirectClean", direct_clean_rgb))
    panels.append(("Raw", raw_rgb))
    if visual_plus_rgb is not None:
        panels.append(("VisualPlus", visual_plus_rgb))
    if visual_extreme_rgb is not None:
        panels.append(("VisualExtreme", visual_extreme_rgb))
    if raw_minus_prior is not None:
        panels.append(("RawMinusPrior", raw_minus_prior))
    if direct_clean_minus_prior is not None:
        panels.append(("DirectMinusPrior", direct_clean_minus_prior))
    if visual_plus_minus_prior is not None:
        panels.append(("VisualPlusMinusPrior", visual_plus_minus_prior))
    if target_rgb is not None:
        panels.append(("GT", target_rgb))
    title_prefix = "showcase_hard debug" if showcase_mode and showcase_hard_mode else ("showcase debug" if showcase_mode else "debug")
    main.save_visual_comparison(path, tuple(panels), title=f"{title_prefix} scene={scene} group={group}")
    return str(path)

def save_test_sample_outputs(
    output_dir: Path,
    scene: str,
    group: str,
    reference_rgb: np.ndarray,
    prior_rgb: np.ndarray,
    raw_rgb: np.ndarray,
    visual_rgb: np.ndarray,
    target_rgb: Optional[np.ndarray],
    mask: np.ndarray,
    glare_mask: Optional[np.ndarray] = None,
    reflection_area_mask: Optional[np.ndarray] = None,
    blend_gate: Optional[np.ndarray] = None,
    safe_gate: Optional[np.ndarray] = None,
    direct_clean_rgb: Optional[np.ndarray] = None,
    raw_minus_prior: Optional[np.ndarray] = None,
    direct_clean_minus_prior: Optional[np.ndarray] = None,
    prediction_lowfreq: Optional[np.ndarray] = None,
    target_lowfreq: Optional[np.ndarray] = None,
    visual_plus_minus_prior: Optional[np.ndarray] = None,
    strong_glare_core: Optional[np.ndarray] = None,
    takeover_mask: Optional[np.ndarray] = None,
    visual_extreme_rgb: Optional[np.ndarray] = None,
    scene_reflection_mask: Optional[np.ndarray] = None,
    dark_reflection_mask: Optional[np.ndarray] = None,
    shadow_veil_mask: Optional[np.ndarray] = None,
    lowfreq_reflection_mask: Optional[np.ndarray] = None,
    foreground_structure_guard: Optional[np.ndarray] = None,
    reflection_confidence: Optional[np.ndarray] = None,
    line_band_confidence: Optional[np.ndarray] = None,
    text_reflection_confidence: Optional[np.ndarray] = None,
    tint_confidence: Optional[np.ndarray] = None,
    lowfreq_reflection_confidence: Optional[np.ndarray] = None,
    dark_shadow_confidence: Optional[np.ndarray] = None,
    showcase_mode: bool = False,
    showcase_hard_mode: bool = False,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{scene}_{group}"
    debug_prefix = f"{prefix}_showcase_hard" if showcase_mode and showcase_hard_mode else (f"{prefix}_showcase" if showcase_mode else prefix)
    result_path = output_dir / f"{prefix}_result.png"
    raw_path = output_dir / f"{prefix}_raw_result.png"
    comparison_path = output_dir / f"{prefix}_comparison.png"
    mask_path = output_dir / f"{prefix}_mask.png"
    mask_gray_path = output_dir / f"{prefix}_mask_gray.png"
    glare_mask_gray_path = output_dir / f"{prefix}_glare_mask_gray.png"
    strong_glare_core_gray_path = output_dir / f"{prefix}_strong_glare_core_gray.png"
    takeover_mask_path = output_dir / f"{prefix}_takeover_mask.png"
    reflection_mask_gray_path = output_dir / f"{prefix}_reflection_mask_gray.png"
    debug_mask_panel_path = output_dir / f"{debug_prefix}_debug_mask_panel.png"
    visual_extreme_result_path = output_dir / f"{prefix}_visual_extreme_result.png"
    direct_clean_path = output_dir / f"{prefix}_direct_clean.png"
    raw_minus_prior_path = output_dir / f"{prefix}_raw_minus_prior.png"
    direct_clean_minus_prior_path = output_dir / f"{prefix}_direct_clean_minus_prior.png"
    visual_plus_minus_prior_path = output_dir / f"{prefix}_visual_plus_minus_prior.png"
    blend_gate_path = output_dir / f"{prefix}_blend_gate.png"
    safe_gate_path = output_dir / f"{prefix}_safe_gate.png"
    final_mask_path = output_dir / f"{prefix}_final_mask.png"
    prediction_lowfreq_path = output_dir / f"{prefix}_prediction_lowfreq.png"
    target_lowfreq_path = output_dir / f"{prefix}_target_lowfreq.png"

    main.save_rgb_image(result_path, visual_rgb)
    main.save_rgb_image(raw_path, raw_rgb)
    main.save_mask_overlay(mask_path, reference_rgb, mask)
    save_mask_gray(mask_gray_path, mask)
    save_mask_gray(final_mask_path, mask)
    if glare_mask is not None:
        save_mask_gray(glare_mask_gray_path, glare_mask)
    if strong_glare_core is not None:
        save_mask_gray(strong_glare_core_gray_path, strong_glare_core)
    if takeover_mask is not None:
        save_mask_gray(takeover_mask_path, takeover_mask)
    if reflection_area_mask is not None:
        save_mask_gray(reflection_mask_gray_path, reflection_area_mask)
    if blend_gate is not None:
        save_mask_gray(blend_gate_path, blend_gate)
    if safe_gate is not None:
        save_mask_gray(safe_gate_path, safe_gate)
    if direct_clean_rgb is not None:
        main.save_rgb_image(direct_clean_path, direct_clean_rgb)
    if raw_minus_prior is not None:
        main.save_rgb_image(raw_minus_prior_path, raw_minus_prior)
    if direct_clean_minus_prior is not None:
        main.save_rgb_image(direct_clean_minus_prior_path, direct_clean_minus_prior)
    if visual_plus_minus_prior is not None:
        main.save_rgb_image(visual_plus_minus_prior_path, visual_plus_minus_prior)
    if visual_extreme_rgb is not None:
        main.save_rgb_image(visual_extreme_result_path, visual_extreme_rgb)
    if prediction_lowfreq is not None:
        main.save_rgb_image(prediction_lowfreq_path, prediction_lowfreq)
    if target_lowfreq is not None:
        main.save_rgb_image(target_lowfreq_path, target_lowfreq)
    debug_panel_output = ""
    if glare_mask is not None and reflection_area_mask is not None:
        debug_panel_output = save_debug_mask_panel(
            debug_mask_panel_path,
            scene,
            group,
            reference_rgb,
            prior_rgb,
            glare_mask,
            reflection_area_mask,
            mask,
            blend_gate,
            safe_gate,
            direct_clean_rgb,
            raw_rgb,
            raw_minus_prior,
            direct_clean_minus_prior,
            target_rgb,
            visual_rgb,
            visual_plus_minus_prior,
            strong_glare_core,
            takeover_mask,
            visual_extreme_rgb,
            scene_reflection_mask,
            dark_reflection_mask,
            shadow_veil_mask,
            lowfreq_reflection_mask,
            foreground_structure_guard,
            reflection_confidence,
            line_band_confidence,
            text_reflection_confidence,
            tint_confidence,
            lowfreq_reflection_confidence,
            dark_shadow_confidence,
            showcase_mode,
            showcase_hard_mode,
        )

    panels: List[Tuple[str, np.ndarray]] = [
        ("0deg", reference_rgb),
        ("Prior", prior_rgb),
        ("Raw", raw_rgb),
        ("Visual", visual_rgb),
    ]
    if target_rgb is not None:
        panels.append(("GT", target_rgb))
    if visual_extreme_rgb is not None:
        panels.append(("Extreme", visual_extreme_rgb))
    main.save_visual_comparison(
        comparison_path,
        tuple(panels),
        title=f"{'showcase_hard ' if showcase_mode and showcase_hard_mode else ('showcase ' if showcase_mode else '')}scene={scene} group={group}",
    )
    return {
        "result_path": str(result_path),
        "raw_result_path": str(raw_path),
        "comparison_path": str(comparison_path),
        "mask_path": str(mask_path),
        "mask_gray_path": str(mask_gray_path),
        "glare_mask_gray_path": str(glare_mask_gray_path) if glare_mask is not None else "",
        "strong_glare_core_gray_path": str(strong_glare_core_gray_path) if strong_glare_core is not None else "",
        "takeover_mask_path": str(takeover_mask_path) if takeover_mask is not None else "",
        "reflection_mask_gray_path": str(reflection_mask_gray_path) if reflection_area_mask is not None else "",
        "debug_mask_panel_path": debug_panel_output,
        "visual_extreme_result_path": str(visual_extreme_result_path) if visual_extreme_rgb is not None else "",
        "direct_clean_path": str(direct_clean_path) if direct_clean_rgb is not None else "",
        "raw_minus_prior_path": str(raw_minus_prior_path) if raw_minus_prior is not None else "",
        "direct_clean_minus_prior_path": str(direct_clean_minus_prior_path) if direct_clean_minus_prior is not None else "",
        "visual_plus_minus_prior_path": str(visual_plus_minus_prior_path) if visual_plus_minus_prior is not None else "",
        "blend_gate_path": str(blend_gate_path) if blend_gate is not None else "",
        "safe_gate_path": str(safe_gate_path) if safe_gate is not None else "",
        "final_mask_path": str(final_mask_path),
        "prediction_lowfreq_path": str(prediction_lowfreq_path) if prediction_lowfreq is not None else "",
        "target_lowfreq_path": str(target_lowfreq_path) if target_lowfreq is not None else "",
    }

def masked_l1_np(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    mask = main.ensure_single_channel(np.clip(mask.astype(np.float32), 0.0, 1.0))
    denom = float(np.sum(mask) * prediction.shape[2] + EPS)
    return float(np.sum(np.abs(prediction.astype(np.float32) - target.astype(np.float32)) * mask) / denom)

def masked_mean_np(value: np.ndarray, mask: np.ndarray) -> float:
    mask = main.ensure_single_channel(np.clip(mask.astype(np.float32), 0.0, 1.0))
    value = value.astype(np.float32)
    if value.ndim == 2:
        value = value[..., np.newaxis]
    denom = float(np.sum(mask) * value.shape[2] + EPS)
    return float(np.sum(value * mask) / denom)

def improvement_ratio_np(before: float, after: float) -> float:
    return float((before - after) / max(before, EPS))

def write_per_sample_metrics(output_root: Path, rows: Sequence[Dict[str, object]]) -> Dict[str, str]:
    if not rows:
        return {
            "per_sample_csv": "",
            "per_sample_json": "",
            "worst_cases_csv": "",
            "visual_ranked_cases_csv": "",
            "visual_regression_cases_csv": "",
            "visual_watch_cases_csv": "",
        }
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "per_sample_metrics.csv"
    json_path = output_root / "per_sample_metrics.json"
    worst_path = output_root / "worst_cases.csv"
    visual_ranked_path = output_root / "visual_ranked_cases.csv"
    visual_regression_path = output_root / "visual_regression_cases.csv"
    visual_watch_path = output_root / "visual_watch_cases.csv"
    fieldnames = METRIC_FIELDS
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(list(rows), fp, ensure_ascii=False, indent=2)
    worst_rows = sorted(rows, key=lambda row: float(row.get("raw_focus_l1", 0.0)), reverse=True)
    with worst_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(worst_rows)
    visual_ranked_rows = sorted(
        rows,
        key=lambda row: (
            float(row.get("visual_vs_prior_focus_improvement", 0.0)),
            float(row.get("raw_vs_prior_focus_improvement", 0.0)),
            float(row.get("raw_psnr", 0.0)),
            float(row.get("raw_ssim", 0.0)),
        ),
        reverse=True,
    )
    with visual_ranked_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(visual_ranked_rows)
    visual_regression_rows = sorted(
        rows,
        key=lambda row: (
            float(row.get("visual_vs_prior_focus_improvement", 0.0)),
            float(row.get("raw_vs_prior_focus_improvement", 0.0)),
            float(row.get("visual_ssim", 0.0)),
        ),
    )
    with visual_regression_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(visual_regression_rows)
    visual_watch_rows = sorted(
        rows,
        key=lambda row: (
            float(row.get("visual_vs_prior_focus_improvement", 0.0)),
            float(row.get("direct_vs_prior_focus_improvement", 0.0)),
            -abs(float(row.get("visual_ssim", 0.0)) - float(row.get("prior_ssim", 0.0))),
            float(row.get("raw_psnr", 0.0)),
        ),
        reverse=True,
    )
    with visual_watch_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(visual_watch_rows)
    return {
        "per_sample_csv": str(csv_path),
        "per_sample_json": str(json_path),
        "worst_cases_csv": str(worst_path),
        "visual_ranked_cases_csv": str(visual_ranked_path),
        "visual_regression_cases_csv": str(visual_regression_path),
        "visual_watch_cases_csv": str(visual_watch_path),
    }

def evaluate_records(
    model: nn.Module,
    records: Sequence[SampleRecord],
    args: argparse.Namespace,
    output_root: Path,
    prior_cache_dir: Path,
    package_cache_dir: Optional[Path],
    split_name: str,
    save_images: bool,
) -> Dict[str, float]:
    device = next(model.parameters()).device
    dataset = PolarFreeLiteDataset(
        records,
        img_size=args.img_size,
        prior_cache_dir=prior_cache_dir,
        package_cache_dir=package_cache_dir,
        use_package_cache=args.package_cache,
        mask_profile=args.mask_profile,
        prior_profile=args.prior_profile,
        augment=False,
    )
    pin_memory = device.type == "cuda"
    loader = make_loader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        prefetch_factor=args.prefetch_factor,
        pin_memory=pin_memory,
    )
    images_dir = output_root / "images"
    summary_path = output_root / "all_summary.json"

    totals: Dict[str, float] = {
        "reference_l1": 0.0,
        "reference_focus_l1": 0.0,
        "reference_psnr": 0.0,
        "reference_ssim": 0.0,
        "prior_l1": 0.0,
        "prior_focus_l1": 0.0,
        "prior_psnr": 0.0,
        "prior_ssim": 0.0,
        "direct_l1": 0.0,
        "direct_focus_l1": 0.0,
        "direct_psnr": 0.0,
        "direct_ssim": 0.0,
        "raw_l1": 0.0,
        "raw_focus_l1": 0.0,
        "raw_psnr": 0.0,
        "raw_ssim": 0.0,
        "visual_l1": 0.0,
        "visual_focus_l1": 0.0,
        "visual_psnr": 0.0,
        "visual_ssim": 0.0,
    }
    total_items = 0
    sample_outputs: List[Dict[str, str]] = []
    per_sample_rows: List[Dict[str, object]] = []

    model.eval()
    amp_enabled = bool(args.amp and device.type == "cuda")
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device=device, channels_last=args.channels_last)
            inputs = batch["inputs"]
            base_prior = batch["base_prior"]
            reference_tensor = batch["reference"]
            target = batch["target"]
            mask_tensor = batch["mask"]
            glare_mask_tensor = batch["glare_mask"]
            reflection_area_mask_tensor = batch["reflection_area_mask"]
            scene_reflection_mask_tensor = batch["scene_reflection_mask"]
            dark_reflection_mask_tensor = batch["dark_reflection_mask"]
            shadow_veil_mask_tensor = batch["shadow_veil_mask"]
            lowfreq_reflection_mask_tensor = batch["lowfreq_reflection_mask"]
            foreground_structure_guard_tensor = batch["foreground_structure_guard"]

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                residual = model(inputs)
                (
                    direct_clean_tensor,
                    _predicted_reflection,
                    signed_clean_delta_tensor,
                    _support_soft,
                    _support_hard,
                    _learned_gate,
                    reflection_confidence_tensor,
                    line_band_confidence_tensor,
                    text_reflection_confidence_tensor,
                    tint_confidence_tensor,
                    lowfreq_reflection_confidence_tensor,
                    dark_shadow_confidence_tensor,
                ) = decode_reflection_heads(
                    residual,
                    reference_tensor,
                    mask_tensor,
                    glare_mask=glare_mask_tensor,
                    reflection_area_mask=reflection_area_mask_tensor,
                    visual_strength=args.visual_strength,
                    showcase_reflection_boost=args.showcase_reflection_boost,
                    support_sharpen=args.support_sharpen,
                )
                raw_prediction = compose_prediction(
                    base_prior,
                    residual,
                    mask_tensor,
                    reference_rgb=reference_tensor,
                    glare_mask=glare_mask_tensor,
                    reflection_area_mask=reflection_area_mask_tensor,
                    scene_reflection_mask=scene_reflection_mask_tensor,
                    dark_reflection_mask=dark_reflection_mask_tensor,
                    shadow_veil_mask=shadow_veil_mask_tensor,
                    lowfreq_reflection_mask=lowfreq_reflection_mask_tensor,
                    foreground_structure_guard=foreground_structure_guard_tensor,
                    mask_profile=args.mask_profile,
                    loss_profile=args.loss_profile,
                    visual_strength=args.visual_strength,
                    showcase_mode=args.showcase_mode,
                    showcase_hard_mode=args.showcase_hard_mode,
                    showcase_reflection_boost=args.showcase_reflection_boost,
                    support_sharpen=args.support_sharpen,
                    showcase_detail_strength=args.showcase_detail_strength,
                    showcase_smooth_strength=args.showcase_smooth_strength,
                    showcase_tint_suppress=args.showcase_tint_suppress,
                )
                blend_gate_tensor = compute_blend_gate(
                    residual,
                    mask_tensor,
                    glare_mask=glare_mask_tensor,
                    reflection_area_mask=reflection_area_mask_tensor,
                    mask_profile=args.mask_profile,
                    loss_profile=args.loss_profile,
                    reference_rgb=reference_tensor,
                    base_prior=base_prior,
                    showcase_mode=args.showcase_mode,
                    visual_strength=args.visual_strength,
                    showcase_reflection_boost=args.showcase_reflection_boost,
                    support_sharpen=args.support_sharpen,
                )
                raw_metrics = compute_metrics(raw_prediction, target, mask_tensor)
                reference_metrics = compute_metrics(reference_tensor, target, mask_tensor)
                prior_metrics = compute_metrics(base_prior, target, mask_tensor)
                direct_metrics = compute_metrics(direct_clean_tensor, target, mask_tensor)

            scene = str(batch["scene"][0])
            group = str(batch["group"][0])
            reference_rgb = tensor_to_numpy_rgb(batch["reference"][0])
            prior_rgb = tensor_to_numpy_rgb(batch["base_prior"][0])
            visual_prior_rgb = tensor_to_numpy_rgb(batch["visual_prior"][0])
            raw_rgb = tensor_to_numpy_rgb(raw_prediction[0])
            direct_clean_rgb = tensor_to_numpy_rgb(direct_clean_tensor[0])
            raw_minus_prior = diff_heatmap(raw_rgb, prior_rgb)
            direct_clean_minus_prior = diff_heatmap(direct_clean_rgb, prior_rgb)
            prediction_lowfreq = cv2.GaussianBlur(raw_rgb.astype(np.float32), (0, 0), 11.0)
            target_rgb = tensor_to_numpy_rgb(batch["target"][0])
            target_lowfreq = cv2.GaussianBlur(target_rgb.astype(np.float32), (0, 0), 11.0)
            mask = tensor_to_numpy_mask(batch["mask"][0])
            glare_mask = tensor_to_numpy_mask(batch["glare_mask"][0])
            reflection_area_mask = tensor_to_numpy_mask(batch["reflection_area_mask"][0])
            scene_reflection_mask = tensor_to_numpy_mask(batch["scene_reflection_mask"][0])
            dark_reflection_mask = tensor_to_numpy_mask(batch["dark_reflection_mask"][0])
            shadow_veil_mask = tensor_to_numpy_mask(batch["shadow_veil_mask"][0])
            lowfreq_reflection_mask = tensor_to_numpy_mask(batch["lowfreq_reflection_mask"][0])
            foreground_structure_guard = tensor_to_numpy_mask(batch["foreground_structure_guard"][0])
            reflection_confidence = tensor_to_numpy_mask(reflection_confidence_tensor[0])
            line_band_confidence = tensor_to_numpy_mask(line_band_confidence_tensor[0])
            text_reflection_confidence = tensor_to_numpy_mask(text_reflection_confidence_tensor[0])
            tint_confidence = tensor_to_numpy_mask(tint_confidence_tensor[0])
            lowfreq_reflection_confidence = tensor_to_numpy_mask(lowfreq_reflection_confidence_tensor[0])
            dark_shadow_confidence = tensor_to_numpy_mask(dark_shadow_confidence_tensor[0])
            signed_delta_np = np.transpose(signed_clean_delta_tensor[0].detach().float().cpu().numpy(), (1, 2, 0))
            signed_delta_effect = np.mean(np.abs(signed_delta_np), axis=2, keepdims=True)
            blend_gate = tensor_to_numpy_mask(blend_gate_tensor[0])
            safe_gate: Optional[np.ndarray] = None
            debug_takeover_masks = compute_visual_takeover_debug_masks(
                reference_rgb,
                prior_rgb,
                raw_rgb,
                mask,
                direct_clean_rgb=direct_clean_rgb,
                reflection_confidence=reflection_confidence,
                line_band_confidence=line_band_confidence,
                text_reflection_confidence=text_reflection_confidence,
                lowfreq_reflection_confidence=lowfreq_reflection_confidence,
                dark_shadow_confidence=dark_shadow_confidence,
                reflection_area_mask=reflection_area_mask,
                lowfreq_reflection_mask=lowfreq_reflection_mask,
            )
            strong_glare_core = debug_takeover_masks["strong_glare_core"]
            takeover_mask = debug_takeover_masks["takeover_mask"]
            visual_rgb = render_prediction_output(
                reference_rgb,
                prior_rgb,
                visual_prior_rgb,
                raw_rgb,
                mask,
                render_mode=args.render_mode,
                direct_clean_rgb=direct_clean_rgb,
                showcase_mode=args.showcase_mode,
                showcase_hard_mode=args.showcase_hard_mode,
                detail_strength=args.showcase_detail_strength,
                smooth_strength=args.showcase_smooth_strength,
                tint_suppress=args.showcase_tint_suppress,
            )
            visual_extreme_rgb = render_prediction_output(
                reference_rgb,
                prior_rgb,
                visual_prior_rgb,
                raw_rgb,
                mask,
                render_mode="visual_extreme",
                direct_clean_rgb=direct_clean_rgb,
                showcase_mode=True,
                showcase_hard_mode=True,
                detail_strength=args.showcase_detail_strength,
                smooth_strength=args.showcase_smooth_strength,
                tint_suppress=args.showcase_tint_suppress,
            )
            if args.render_mode in {"visual_plus", "visual_extreme"}:
                _visual_safe, _visual_candidate, safe_gate = visual_strong_render_components(
                    reference_rgb,
                    prior_rgb,
                    visual_prior_rgb,
                    raw_rgb,
                    mask,
                    direct_clean_rgb=direct_clean_rgb,
                    showcase_mode=args.showcase_mode,
                    showcase_hard_mode=args.showcase_hard_mode,
                    detail_strength=args.showcase_detail_strength,
                    smooth_strength=args.showcase_smooth_strength,
                    tint_suppress=args.showcase_tint_suppress,
                )
            shadow_union_mask = np.clip(
                np.maximum(scene_reflection_mask, np.maximum(dark_reflection_mask, np.maximum(shadow_veil_mask, lowfreq_reflection_mask))),
                0.0,
                1.0,
            )
            lowfreq_reflection_l1 = masked_l1_np(raw_rgb, target_rgb, lowfreq_reflection_mask)
            highlight_region_l1 = masked_l1_np(visual_rgb, target_rgb, strong_glare_core)
            dark_shadow_region_l1 = masked_l1_np(raw_rgb, target_rgb, dark_reflection_mask)
            scene_reflection_l1 = masked_l1_np(raw_rgb, target_rgb, scene_reflection_mask)
            shadow_veil_l1 = masked_l1_np(raw_rgb, target_rgb, shadow_veil_mask)
            line_eval_mask = np.maximum(line_band_confidence, debug_takeover_masks["line_score"])
            glare_suppression_ratio = improvement_ratio_np(
                masked_l1_np(reference_rgb, target_rgb, strong_glare_core),
                highlight_region_l1,
            )
            line_band_suppression_ratio = improvement_ratio_np(
                masked_l1_np(reference_rgb, target_rgb, line_eval_mask),
                masked_l1_np(visual_rgb, target_rgb, line_eval_mask),
            )
            signed_delta_effect_ratio = masked_mean_np(signed_delta_effect, shadow_union_mask) / max(
                masked_l1_np(reference_rgb, target_rgb, shadow_union_mask),
                EPS,
            )
            lowfreq_suppression_ratio = improvement_ratio_np(
                masked_l1_np(prior_rgb, target_rgb, lowfreq_reflection_mask),
                lowfreq_reflection_l1,
            )
            dark_shadow_recovery_ratio = improvement_ratio_np(
                masked_l1_np(prior_rgb, target_rgb, dark_reflection_mask),
                dark_shadow_region_l1,
            )
            foreground_damage_ratio = masked_l1_np(raw_rgb, target_rgb, foreground_structure_guard) / max(
                masked_l1_np(prior_rgb, target_rgb, foreground_structure_guard),
                EPS,
            )
            raw_visualplus_diff_in_shadow = masked_l1_np(raw_rgb, visual_rgb, shadow_union_mask)
            visual_tensor = torch.from_numpy(np.transpose(visual_rgb, (2, 0, 1))).unsqueeze(0).to(device=device, dtype=target.dtype)
            visual_metrics = compute_metrics(visual_tensor, target, mask_tensor)
            visual_plus_minus_prior = diff_heatmap(visual_rgb, prior_rgb)
            direct_vs_prior_focus_improvement = prior_metrics["focus_l1"] - direct_metrics["focus_l1"]
            raw_vs_prior_focus_improvement = prior_metrics["focus_l1"] - raw_metrics["focus_l1"]
            visual_vs_prior_focus_improvement = prior_metrics["focus_l1"] - visual_metrics["focus_l1"]

            totals["reference_l1"] += reference_metrics["l1"]
            totals["reference_focus_l1"] += reference_metrics["focus_l1"]
            totals["reference_psnr"] += reference_metrics["psnr"]
            totals["reference_ssim"] += reference_metrics["ssim"]
            totals["prior_l1"] += prior_metrics["l1"]
            totals["prior_focus_l1"] += prior_metrics["focus_l1"]
            totals["prior_psnr"] += prior_metrics["psnr"]
            totals["prior_ssim"] += prior_metrics["ssim"]
            totals["direct_l1"] += direct_metrics["l1"]
            totals["direct_focus_l1"] += direct_metrics["focus_l1"]
            totals["direct_psnr"] += direct_metrics["psnr"]
            totals["direct_ssim"] += direct_metrics["ssim"]
            totals["raw_l1"] += raw_metrics["l1"]
            totals["raw_focus_l1"] += raw_metrics["focus_l1"]
            totals["raw_psnr"] += raw_metrics["psnr"]
            totals["raw_ssim"] += raw_metrics["ssim"]
            totals["visual_l1"] += visual_metrics["l1"]
            totals["visual_focus_l1"] += visual_metrics["focus_l1"]
            totals["visual_psnr"] += visual_metrics["psnr"]
            totals["visual_ssim"] += visual_metrics["ssim"]
            total_items += 1

            output_info = {
                "result_path": "",
                "raw_result_path": "",
                "comparison_path": "",
                "mask_path": "",
                "mask_gray_path": "",
                "glare_mask_gray_path": "",
                "strong_glare_core_gray_path": "",
                "takeover_mask_path": "",
                "reflection_mask_gray_path": "",
                "debug_mask_panel_path": "",
                "visual_extreme_result_path": "",
                "direct_clean_path": "",
                "raw_minus_prior_path": "",
                "direct_clean_minus_prior_path": "",
                "visual_plus_minus_prior_path": "",
                "blend_gate_path": "",
                "safe_gate_path": "",
                "final_mask_path": "",
                "prediction_lowfreq_path": "",
                "target_lowfreq_path": "",
            }
            if save_images:
                output_info = save_test_sample_outputs(
                    images_dir,
                    scene,
                    group,
                    reference_rgb,
                    prior_rgb,
                    raw_rgb,
                    visual_rgb,
                    target_rgb,
                    mask,
                    glare_mask,
                    reflection_area_mask,
                    blend_gate,
                    safe_gate,
                    direct_clean_rgb,
                    raw_minus_prior,
                    direct_clean_minus_prior,
                    prediction_lowfreq,
                    target_lowfreq,
                    visual_plus_minus_prior,
                    strong_glare_core,
                    takeover_mask,
                    visual_extreme_rgb,
                    scene_reflection_mask,
                    dark_reflection_mask,
                    shadow_veil_mask,
                    lowfreq_reflection_mask,
                    foreground_structure_guard,
                    reflection_confidence,
                    line_band_confidence,
                    text_reflection_confidence,
                    tint_confidence,
                    lowfreq_reflection_confidence,
                    dark_shadow_confidence,
                    args.showcase_mode,
                    args.showcase_hard_mode,
                )
                sample_outputs.append(output_info)
            per_sample_rows.append(
                {
                    "scene": scene,
                    "group": group,
                    "reference_l1": reference_metrics["l1"],
                    "reference_focus_l1": reference_metrics["focus_l1"],
                    "reference_psnr": reference_metrics["psnr"],
                    "reference_ssim": reference_metrics["ssim"],
                    "prior_l1": prior_metrics["l1"],
                    "prior_focus_l1": prior_metrics["focus_l1"],
                    "prior_psnr": prior_metrics["psnr"],
                    "prior_ssim": prior_metrics["ssim"],
                    "direct_l1": direct_metrics["l1"],
                    "direct_focus_l1": direct_metrics["focus_l1"],
                    "direct_psnr": direct_metrics["psnr"],
                    "direct_ssim": direct_metrics["ssim"],
                    "raw_l1": raw_metrics["l1"],
                    "raw_focus_l1": raw_metrics["focus_l1"],
                    "raw_psnr": raw_metrics["psnr"],
                    "raw_ssim": raw_metrics["ssim"],
                    "visual_l1": visual_metrics["l1"],
                    "visual_focus_l1": visual_metrics["focus_l1"],
                    "visual_psnr": visual_metrics["psnr"],
                    "visual_ssim": visual_metrics["ssim"],
                    "direct_vs_prior_focus_improvement": direct_vs_prior_focus_improvement,
                    "raw_vs_prior_focus_improvement": raw_vs_prior_focus_improvement,
                    "visual_vs_prior_focus_improvement": visual_vs_prior_focus_improvement,
                    "highlight_region_l1": highlight_region_l1,
                    "glare_suppression_ratio": glare_suppression_ratio,
                    "line_band_suppression_ratio": line_band_suppression_ratio,
                    "lowfreq_reflection_l1": lowfreq_reflection_l1,
                    "dark_shadow_region_l1": dark_shadow_region_l1,
                    "scene_reflection_l1": scene_reflection_l1,
                    "shadow_veil_l1": shadow_veil_l1,
                    "signed_delta_effect_ratio": signed_delta_effect_ratio,
                    "lowfreq_suppression_ratio": lowfreq_suppression_ratio,
                    "dark_shadow_recovery_ratio": dark_shadow_recovery_ratio,
                    "foreground_damage_ratio": foreground_damage_ratio,
                    "raw_visualplus_diff_in_shadow": raw_visualplus_diff_in_shadow,
                    "comparison_path": output_info.get("comparison_path", ""),
                    "raw_result_path": output_info.get("raw_result_path", ""),
                    "result_path": output_info.get("result_path", ""),
                    "mask_path": output_info.get("mask_path", ""),
                    "mask_gray_path": output_info.get("mask_gray_path", ""),
                    "glare_mask_gray_path": output_info.get("glare_mask_gray_path", ""),
                    "strong_glare_core_gray_path": output_info.get("strong_glare_core_gray_path", ""),
                    "takeover_mask_path": output_info.get("takeover_mask_path", ""),
                    "reflection_mask_gray_path": output_info.get("reflection_mask_gray_path", ""),
                    "debug_mask_panel_path": output_info.get("debug_mask_panel_path", ""),
                    "visual_extreme_result_path": output_info.get("visual_extreme_result_path", ""),
                    "direct_clean_path": output_info.get("direct_clean_path", ""),
                    "raw_minus_prior_path": output_info.get("raw_minus_prior_path", ""),
                    "direct_clean_minus_prior_path": output_info.get("direct_clean_minus_prior_path", ""),
                    "visual_plus_minus_prior_path": output_info.get("visual_plus_minus_prior_path", ""),
                    "blend_gate_path": output_info.get("blend_gate_path", ""),
                    "safe_gate_path": output_info.get("safe_gate_path", ""),
                    "final_mask_path": output_info.get("final_mask_path", ""),
                    "prediction_lowfreq_path": output_info.get("prediction_lowfreq_path", ""),
                    "target_lowfreq_path": output_info.get("target_lowfreq_path", ""),
                }
            )

    divisor = max(total_items, 1)
    metric_paths = write_per_sample_metrics(output_root, per_sample_rows)
    summary: Dict[str, object] = {
        "split": split_name,
        "count": total_items,
        "render_mode": args.render_mode,
        "mask_profile": args.mask_profile,
        "loss_profile": args.loss_profile,
        "prior_profile": args.prior_profile,
        "img_size": args.img_size,
        "reference_l1": totals["reference_l1"] / divisor,
        "reference_focus_l1": totals["reference_focus_l1"] / divisor,
        "reference_psnr": totals["reference_psnr"] / divisor,
        "reference_ssim": totals["reference_ssim"] / divisor,
        "prior_l1": totals["prior_l1"] / divisor,
        "prior_focus_l1": totals["prior_focus_l1"] / divisor,
        "prior_psnr": totals["prior_psnr"] / divisor,
        "prior_ssim": totals["prior_ssim"] / divisor,
        "direct_l1": totals["direct_l1"] / divisor,
        "direct_focus_l1": totals["direct_focus_l1"] / divisor,
        "direct_psnr": totals["direct_psnr"] / divisor,
        "direct_ssim": totals["direct_ssim"] / divisor,
        "raw_l1": totals["raw_l1"] / divisor,
        "raw_focus_l1": totals["raw_focus_l1"] / divisor,
        "raw_psnr": totals["raw_psnr"] / divisor,
        "raw_ssim": totals["raw_ssim"] / divisor,
        "visual_l1": totals["visual_l1"] / divisor,
        "visual_focus_l1": totals["visual_focus_l1"] / divisor,
        "visual_psnr": totals["visual_psnr"] / divisor,
        "visual_ssim": totals["visual_ssim"] / divisor,
        "images_dir": str(images_dir) if save_images else "",
        "sample_outputs": sample_outputs[:20],
        "per_sample_metrics_csv": metric_paths["per_sample_csv"],
        "per_sample_metrics_json": metric_paths["per_sample_json"],
        "worst_cases_csv": metric_paths["worst_cases_csv"],
        "visual_ranked_cases_csv": metric_paths["visual_ranked_cases_csv"],
        "visual_regression_cases_csv": metric_paths["visual_regression_cases_csv"],
        "visual_watch_cases_csv": metric_paths["visual_watch_cases_csv"],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    return {key: float(summary[key]) for key in summary if isinstance(summary[key], (int, float))}
