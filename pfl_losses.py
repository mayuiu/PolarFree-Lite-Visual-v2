from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pfl_model import (
    compute_high_conf_reflection_mask,
    compute_strong_glare_core_tensor,
    decode_reflection_heads,
    visual_strong_alpha_tensor,
)

EPS = 1e-6

def average_pool_ssim(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(prediction, window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)
    sigma_x = F.avg_pool2d(prediction * prediction, window_size, stride=1, padding=window_size // 2) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size // 2) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(prediction * target, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return torch.clamp((numerator / (denominator + EPS)).mean(), 0.0, 1.0)


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def high_frequency_texture_loss(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    pred_low = F.avg_pool2d(prediction, kernel_size=9, stride=1, padding=4)
    target_low = F.avg_pool2d(target, kernel_size=9, stride=1, padding=4)
    pred_high = prediction - pred_low
    target_high = target - target_low
    return weighted_l1(pred_high, target_high, torch.clamp(weights, 0.0, 1.0))


def low_frequency_l1(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, kernel_size: int = 25) -> torch.Tensor:
    padding = kernel_size // 2
    pred_low = F.avg_pool2d(prediction, kernel_size=kernel_size, stride=1, padding=padding)
    target_low = F.avg_pool2d(target, kernel_size=kernel_size, stride=1, padding=padding)
    weight_low = F.avg_pool2d(torch.clamp(weights, 0.0, 1.0), kernel_size=kernel_size, stride=1, padding=padding)
    return weighted_charbonnier(pred_low, target_low, weight_low)


def prior_escape_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    base_prior: Optional[torch.Tensor],
    weights: torch.Tensor,
) -> torch.Tensor:
    if base_prior is None:
        return prediction.new_tensor(0.0)
    pred_error = torch.mean(torch.abs(prediction - target), dim=1, keepdim=True)
    prior_error = torch.mean(torch.abs(base_prior - target), dim=1, keepdim=True)
    improvement_gap = torch.relu(pred_error - 0.92 * prior_error)
    weights = torch.clamp(weights, 0.0, 1.0)
    denominator = torch.sum(weights)
    if float(denominator.detach().cpu().item()) <= EPS:
        return prediction.new_tensor(0.0)
    return torch.sum(improvement_gap * weights) / (denominator + EPS)


def prior_guard_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    base_prior: Optional[torch.Tensor],
    weights: torch.Tensor,
    margin: float = 0.008,
) -> torch.Tensor:
    if base_prior is None:
        return prediction.new_tensor(0.0)
    pred_error = torch.mean(torch.abs(prediction - target), dim=1, keepdim=True)
    prior_error = torch.mean(torch.abs(base_prior - target), dim=1, keepdim=True)
    guard = torch.relu(pred_error - prior_error + margin)
    weights = torch.clamp(weights, 0.0, 1.0)
    denominator = torch.sum(weights)
    if float(denominator.detach().cpu().item()) <= EPS:
        return prediction.new_tensor(0.0)
    return torch.sum(guard * weights) / (denominator + EPS)


def line_reflection_suppression(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    bright_residual = torch.relu(prediction - target)
    weights = torch.clamp(weights, 0.0, 1.0)
    dx = torch.abs(bright_residual[:, :, :, 1:] - bright_residual[:, :, :, :-1])
    dy = torch.abs(bright_residual[:, :, 1:, :] - bright_residual[:, :, :-1, :])
    wx = weights[:, :, :, 1:]
    wy = weights[:, :, 1:, :]
    denominator = (torch.sum(wx) + torch.sum(wy)) * float(prediction.shape[1])
    if float(denominator.detach().cpu().item()) <= EPS:
        return prediction.new_tensor(0.0)
    return (torch.sum(dx * wx) + torch.sum(dy * wy)) / (denominator + EPS)


def charbonnier_error(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target) ** 2 + epsilon * epsilon)


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(charbonnier_error(prediction, target))


def weighted_l1(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    error = torch.abs(prediction - target)
    denominator = torch.sum(weights) * float(prediction.shape[1])
    if float(denominator.detach().cpu().item()) <= EPS:
        return error.new_tensor(0.0)
    return torch.sum(error * weights) / (denominator + EPS)


def weighted_charbonnier(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    error = charbonnier_error(prediction, target)
    denominator = torch.sum(weights) * float(prediction.shape[1])
    if float(denominator.detach().cpu().item()) <= EPS:
        return error.new_tensor(0.0)
    return torch.sum(error * weights) / (denominator + EPS)


def laplacian_filter(image: torch.Tensor) -> torch.Tensor:
    kernel = image.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    kernel = kernel.view(1, 1, 3, 3).expand(image.shape[1], 1, 3, 3)
    return F.conv2d(F.pad(image, (1, 1, 1, 1), mode="reflect"), kernel, groups=image.shape[1])


def compute_laplacian_map(image: torch.Tensor) -> torch.Tensor:
    return laplacian_filter(torch.clamp(image, 0.0, 1.0))


def masked_laplacian_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_lap = compute_laplacian_map(prediction)
    target_lap = compute_laplacian_map(target)
    return weighted_charbonnier(pred_lap, target_lap, torch.clamp(mask, 0.0, 1.0))


def masked_chroma_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_mean = torch.mean(torch.clamp(prediction, 0.0, 1.0), dim=1, keepdim=True)
    target_mean = torch.mean(torch.clamp(target, 0.0, 1.0), dim=1, keepdim=True)
    pred_chroma = prediction - pred_mean
    target_chroma = target - target_mean
    return weighted_charbonnier(pred_chroma, target_chroma, torch.clamp(mask, 0.0, 1.0))


def compute_total_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reflection_mask: torch.Tensor,
    base_prior: Optional[torch.Tensor] = None,
    reference_rgb: Optional[torch.Tensor] = None,
    model_output: Optional[torch.Tensor] = None,
    glare_mask: Optional[torch.Tensor] = None,
    reflection_area_mask: Optional[torch.Tensor] = None,
    scene_reflection_mask: Optional[torch.Tensor] = None,
    dark_reflection_mask: Optional[torch.Tensor] = None,
    shadow_veil_mask: Optional[torch.Tensor] = None,
    lowfreq_reflection_mask: Optional[torch.Tensor] = None,
    foreground_structure_guard: Optional[torch.Tensor] = None,
    hard_case_weight: Optional[torch.Tensor] = None,
    loss_profile: str = "balanced",
    showcase_mode: bool = False,
    visual_strength: float = 1.0,
    showcase_reflection_boost: float = 1.0,
    support_sharpen: float = 1.0,
    perceptual_model: Optional[nn.Module] = None,
    perceptual_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    mask = torch.clamp(reflection_mask, 0.0, 1.0)
    glare = torch.clamp(glare_mask if glare_mask is not None else mask, 0.0, 1.0)
    reflection_area = torch.clamp(reflection_area_mask if reflection_area_mask is not None else mask, 0.0, 1.0)
    scene_reflection = torch.clamp(scene_reflection_mask if scene_reflection_mask is not None else torch.zeros_like(mask), 0.0, 1.0)
    dark_reflection = torch.clamp(dark_reflection_mask if dark_reflection_mask is not None else torch.zeros_like(mask), 0.0, 1.0)
    shadow_veil = torch.clamp(shadow_veil_mask if shadow_veil_mask is not None else torch.zeros_like(mask), 0.0, 1.0)
    lowfreq_reflection = torch.clamp(lowfreq_reflection_mask if lowfreq_reflection_mask is not None else torch.zeros_like(mask), 0.0, 1.0)
    foreground_guard = torch.clamp(foreground_structure_guard if foreground_structure_guard is not None else torch.zeros_like(mask), 0.0, 1.0)
    shadow_region = torch.clamp(torch.maximum(scene_reflection, torch.maximum(dark_reflection, torch.maximum(shadow_veil, lowfreq_reflection))), 0.0, 1.0)
    guard_release = torch.clamp(0.75 * shadow_region + 0.35 * reflection_area + 0.25 * glare, 0.0, 1.0)
    structure_safe = torch.clamp(1.0 - 0.65 * foreground_guard * (1.0 - 0.85 * guard_release), 0.30, 1.0)
    focus_mask = torch.clamp(0.50 * mask + 0.12 * glare + 0.38 * reflection_area, 0.0, 1.0)
    focus_mask = torch.clamp(torch.maximum(focus_mask, 0.55 * shadow_region * structure_safe), 0.0, 1.0)
    core_mask = torch.clamp((reflection_area - 0.45) / 0.55, 0.0, 1.0)
    background_mask = torch.clamp(1.0 - mask, 0.0, 1.0)
    glare_focus_mask = torch.clamp(torch.maximum(glare, core_mask), 0.0, 1.0)
    high_conf_reflection = torch.clamp((reflection_area - 0.36) / 0.42, 0.0, 1.0) * torch.clamp((mask - 0.32) / 0.42, 0.0, 1.0)
    if model_output is not None and model_output.shape[1] >= 3:
        (
            direct_clean,
            predicted_reflection,
            signed_clean_delta,
            residual_support,
            residual_support_hard,
            _learned_gate,
            reflection_confidence,
            line_band_confidence,
            text_reflection_confidence,
            tint_confidence,
            lowfreq_reflection_confidence,
            dark_shadow_confidence,
        ) = decode_reflection_heads(
            model_output,
            reference_rgb,
            mask,
            glare_mask=glare,
            reflection_area_mask=reflection_area,
            visual_strength=visual_strength,
            showcase_reflection_boost=showcase_reflection_boost,
            support_sharpen=support_sharpen,
        )
    else:
        direct_clean = prediction
        predicted_reflection = torch.zeros_like(prediction)
        signed_clean_delta = torch.zeros_like(prediction)
        residual_support = torch.clamp(torch.maximum(mask, torch.maximum(reflection_area, glare)), 0.0, 1.0)
        residual_support_hard = residual_support
        reflection_confidence = residual_support
        line_band_confidence = torch.zeros_like(mask)
        text_reflection_confidence = torch.zeros_like(mask)
        tint_confidence = torch.zeros_like(mask)
        lowfreq_reflection_confidence = torch.zeros_like(mask)
        dark_shadow_confidence = torch.zeros_like(mask)
    if loss_profile == "visual":
        visual_core, visual_boundary = compute_high_conf_reflection_mask(
            reference_rgb,
            base_prior,
            direct_clean,
            mask,
            glare_mask=glare,
            reflection_area_mask=reflection_area,
        )
        focus_mask = torch.clamp(torch.maximum(focus_mask, 0.58 * visual_boundary + 0.42 * reflection_area), 0.0, 1.0)
        core_mask = torch.clamp(torch.maximum(core_mask, visual_core), 0.0, 1.0)
        high_conf_reflection = torch.clamp(torch.maximum(high_conf_reflection, visual_core), 0.0, 1.0)
        glare_focus_mask = torch.clamp(torch.maximum(glare_focus_mask, visual_boundary), 0.0, 1.0)
    strong_glare_core = compute_strong_glare_core_tensor(
        reference_rgb,
        glare,
        mask,
        reflection_area_mask=reflection_area,
        line_confidence=line_band_confidence,
        lowfreq_confidence=lowfreq_reflection_confidence,
    )
    highlight_core_mask = torch.clamp(torch.maximum(glare_focus_mask, strong_glare_core), 0.0, 1.0)
    line_band_loss_mask = torch.clamp(
        torch.maximum(line_band_confidence, torch.maximum(0.55 * strong_glare_core, 0.45 * glare)),
        0.0,
        1.0,
    )
    takeover_loss_mask = torch.clamp(
        torch.maximum(
            strong_glare_core,
            torch.maximum(high_conf_reflection, torch.maximum(lowfreq_reflection, torch.maximum(line_band_loss_mask, shadow_region))),
        ),
        0.0,
        1.0,
    )
    direct_clean_focus_mask = torch.clamp(torch.maximum(focus_mask, reflection_area), 0.0, 1.0)
    direct_clean_core_mask = torch.clamp(torch.maximum(torch.maximum(high_conf_reflection, core_mask), strong_glare_core), 0.0, 1.0)
    direct_clean_background_mask = torch.clamp(1.0 - torch.maximum(mask, reflection_area), 0.0, 1.0)
    if reference_rgb is not None:
        residual_target = torch.relu(reference_rgb - target)
        residual_support3 = residual_support.expand_as(predicted_reflection)
        if model_output is not None and model_output.shape[1] >= 10:
            residual_hard3 = residual_support_hard.expand_as(predicted_reflection)
            residual_strength_tensor = torch.clamp(
                1.20 + 0.95 * residual_hard3 * float(visual_strength) * float(showcase_reflection_boost),
                1.20,
                2.20,
            )
            residual_clean_prediction = torch.clamp(
                reference_rgb - predicted_reflection * residual_hard3 * residual_strength_tensor + signed_clean_delta * residual_support3,
                0.0,
                1.0,
            )
        else:
            residual_strength_tensor = torch.clamp(1.00 + 0.65 * residual_support3, 1.00, 1.65)
            residual_clean_prediction = torch.clamp(
                reference_rgb - predicted_reflection * residual_support3 * residual_strength_tensor,
                0.0,
                1.0,
            )
    else:
        residual_target = torch.zeros_like(predicted_reflection)
        residual_clean_prediction = direct_clean
    residual_target_luma = torch.mean(residual_target, dim=1, keepdim=True)
    gt_residual_mask = torch.clamp((residual_target_luma - 0.01) / 0.12, 0.0, 1.0)
    gt_residual_mask = torch.maximum(
        gt_residual_mask,
        F.avg_pool2d(gt_residual_mask, kernel_size=9, stride=1, padding=4),
    )
    supervised_reflection_mask = torch.clamp(
        torch.maximum(gt_residual_mask, torch.maximum(0.60 * reflection_area, 0.40 * focus_mask)),
        0.0,
        1.0,
    )

    full_l1 = charbonnier_loss(prediction, target)
    focus_l1 = weighted_charbonnier(prediction, target, focus_mask)
    core_l1 = weighted_charbonnier(prediction, target, core_mask)
    background_l1 = weighted_charbonnier(prediction, target, background_mask)
    direct_clean_focus_l1 = weighted_charbonnier(direct_clean, target, direct_clean_focus_mask)
    direct_clean_core_l1 = weighted_charbonnier(direct_clean, target, direct_clean_core_mask)
    direct_clean_background_l1 = weighted_charbonnier(direct_clean, target, direct_clean_background_mask)
    direct_clean_ssim_loss = 1.0 - average_pool_ssim(direct_clean, target)
    direct_clean_grad_loss = gradient_loss(direct_clean, target)
    residual_reflection_l1 = weighted_charbonnier(predicted_reflection, residual_target, supervised_reflection_mask)
    residual_clean_l1 = weighted_charbonnier(residual_clean_prediction, target, supervised_reflection_mask)
    glare_l1 = weighted_charbonnier(prediction, target, glare_focus_mask)
    glare_core_l1 = weighted_charbonnier(prediction, target, strong_glare_core)
    strong_glare_direct_clean_l1 = weighted_charbonnier(direct_clean, target, strong_glare_core)
    highlight_residual_loss = weighted_charbonnier(predicted_reflection, residual_target, highlight_core_mask)
    highlight_clean_loss = weighted_charbonnier(prediction, target, highlight_core_mask)
    highlight_lowfreq_loss = low_frequency_l1(direct_clean, target, highlight_core_mask, kernel_size=31)
    highlight_chroma_loss = masked_chroma_loss(direct_clean, target, highlight_core_mask)
    reflection_focus_l1 = weighted_charbonnier(prediction, target, reflection_area)
    lowfreq_reflection_l1 = low_frequency_l1(prediction, target, torch.clamp(0.55 * reflection_area + 0.45 * high_conf_reflection, 0.0, 1.0))
    prior_escape = prior_escape_loss(prediction, target, base_prior, high_conf_reflection)
    ssim_loss = 1.0 - average_pool_ssim(prediction, target)
    grad = gradient_loss(prediction, target)
    texture = high_frequency_texture_loss(prediction, target, focus_mask)
    high_light_suppression = weighted_l1(torch.relu(prediction - target), torch.zeros_like(prediction), glare_focus_mask)
    line_suppression = line_reflection_suppression(prediction, target, glare_focus_mask)
    line_band_suppression_loss = line_reflection_suppression(direct_clean, target, line_band_loss_mask)
    lowfreq_loss_mask = torch.clamp(lowfreq_reflection * structure_safe, 0.0, 1.0)
    dark_shadow_loss_mask = torch.clamp(dark_reflection * structure_safe, 0.0, 1.0)
    scene_reflection_loss_mask = torch.clamp(scene_reflection * structure_safe, 0.0, 1.0)
    shadow_veil_loss_mask = torch.clamp(shadow_veil * structure_safe, 0.0, 1.0)
    lowfreq_reflection_loss = low_frequency_l1(prediction, target, lowfreq_loss_mask, kernel_size=31)
    dark_shadow_clean_loss = weighted_charbonnier(direct_clean, target, dark_shadow_loss_mask)
    scene_reflection_suppression_loss = weighted_l1(
        torch.relu(prediction - target),
        torch.zeros_like(prediction),
        scene_reflection_loss_mask,
    )
    shadow_veil_consistency_loss = low_frequency_l1(direct_clean, target, shadow_veil_loss_mask, kernel_size=31)
    confidence_target = torch.clamp(torch.maximum(reflection_area, torch.maximum(scene_reflection, shadow_region)), 0.0, 1.0)
    lowfreq_target = torch.clamp(lowfreq_reflection * (1.0 - 0.35 * glare) + 0.25 * shadow_veil + 0.18 * scene_reflection, 0.0, 1.0)
    dark_shadow_target = torch.clamp(dark_reflection * (1.0 - 0.20 * glare) + 0.22 * shadow_veil + 0.14 * scene_reflection, 0.0, 1.0)
    line_target = torch.clamp(glare * torch.clamp((reflection_area - 0.10) / 0.90, 0.0, 1.0) * (1.0 - 0.45 * shadow_veil), 0.0, 1.0)
    text_target = torch.clamp(shadow_veil * torch.clamp((shadow_region - 0.06) / 0.94, 0.0, 1.0) * (1.0 - 0.40 * glare), 0.0, 1.0)
    lowfreq_confidence_loss = weighted_charbonnier(lowfreq_reflection_confidence, lowfreq_target, torch.clamp(lowfreq_target + 0.25, 0.0, 1.0))
    dark_shadow_confidence_loss = weighted_charbonnier(dark_shadow_confidence, dark_shadow_target, torch.clamp(dark_shadow_target + 0.25, 0.0, 1.0))
    reflection_confidence_loss = weighted_charbonnier(reflection_confidence, confidence_target, torch.clamp(confidence_target + 0.20, 0.0, 1.0))
    line_band_confidence_loss = weighted_charbonnier(line_band_confidence, line_target, torch.clamp(line_target + 0.10, 0.0, 1.0))
    text_reflection_confidence_loss = weighted_charbonnier(text_reflection_confidence, text_target, torch.clamp(text_target + 0.10, 0.0, 1.0))
    tint_confidence_loss = weighted_charbonnier(tint_confidence, lowfreq_reflection, torch.clamp(lowfreq_reflection + 0.10, 0.0, 1.0))
    foreground_structure_guard_loss = weighted_charbonnier(prediction, target, torch.clamp(foreground_guard * (1.0 - shadow_region), 0.0, 1.0))
    safe_background_mask = torch.clamp((1.0 - takeover_loss_mask) * (1.0 - reflection_area) * (1.0 - glare), 0.0, 1.0)
    background_protection_loss = weighted_charbonnier(prediction, target, safe_background_mask)
    if base_prior is not None:
        reflection_region = torch.clamp(torch.maximum(high_conf_reflection, torch.maximum(shadow_region, glare_focus_mask)), 0.0, 1.0)
        raw_distance = torch.mean(torch.abs(prediction - base_prior), dim=1, keepdim=True)
        visualplus_should_differ_from_raw_in_reflection_region = weighted_l1(
            torch.relu(0.040 - raw_distance).expand_as(prediction),
            torch.zeros_like(prediction),
            reflection_region,
        )
    else:
        visualplus_should_differ_from_raw_in_reflection_region = prediction.new_tensor(0.0)
    if reference_rgb is not None:
        residual_consistency_target = torch.clamp(reference_rgb - target, -0.7, 0.7)
        residual_consistency_prediction = torch.clamp(predicted_reflection - signed_clean_delta, -0.7, 0.7)
        residual_consistency_loss = weighted_charbonnier(
            residual_consistency_prediction,
            residual_consistency_target,
            torch.clamp(torch.maximum(supervised_reflection_mask, shadow_region), 0.0, 1.0),
        )
    else:
        residual_consistency_loss = prediction.new_tensor(0.0)
    if hard_case_weight is not None:
        sample_weight = torch.clamp(hard_case_weight.view(-1, 1, 1, 1).to(device=prediction.device, dtype=prediction.dtype), 1.0, 6.0)
        hard_focus_mask = torch.clamp(
            (sample_weight - 1.0)
            * torch.maximum(takeover_loss_mask, torch.maximum(highlight_core_mask, torch.maximum(line_band_loss_mask, residual_support_hard))),
            0.0,
            5.0,
        )
        hard_case_weighted_loss = weighted_charbonnier(
            prediction,
            target,
            hard_focus_mask,
        )
        hard_case_direct_clean_loss = weighted_charbonnier(direct_clean, target, hard_focus_mask)
        hard_case_residual_loss = weighted_charbonnier(predicted_reflection, residual_target, hard_focus_mask)
    else:
        hard_case_weighted_loss = prediction.new_tensor(0.0)
        hard_case_direct_clean_loss = prediction.new_tensor(0.0)
        hard_case_residual_loss = prediction.new_tensor(0.0)
    guard_weights = torch.clamp(0.45 * focus_mask + 0.35 * high_conf_reflection + 0.20 * core_mask, 0.0, 1.0)
    raw_prior_guard = prediction.new_tensor(0.0)
    direct_prior_guard = prediction.new_tensor(0.0)
    visual_prior_guard = prediction.new_tensor(0.0)
    if loss_profile == "visual" and base_prior is not None:
        visual_candidate, visual_alpha = visual_strong_alpha_tensor(
            reference_rgb,
            base_prior,
            prediction,
            direct_clean,
            mask,
            glare,
            reflection_area,
        )
        visual_proxy = torch.clamp(base_prior * (1.0 - visual_alpha) + visual_candidate * visual_alpha, 0.0, 1.0)
        raw_prior_guard = prior_guard_loss(prediction, target, base_prior, guard_weights, margin=0.008)
        direct_prior_guard = prior_guard_loss(direct_clean, target, base_prior, guard_weights, margin=0.010)
        visual_prior_guard = prior_guard_loss(visual_proxy, target, base_prior, guard_weights, margin=0.006)
    reflection_outside = prediction.new_tensor(0.0)
    reflection_sparse = prediction.new_tensor(0.0)
    if model_output is not None and model_output.shape[1] >= 7:
        reflection_outside = weighted_l1(predicted_reflection, torch.zeros_like(predicted_reflection), background_mask)
        reflection_sparse = torch.mean(predicted_reflection)
    perceptual_l1 = prediction.new_tensor(0.0)
    if perceptual_model is not None and perceptual_weight > 0.0:
        perceptual_l1 = perceptual_model(prediction, target)
    showcase_loss_mask = torch.clamp(torch.maximum(residual_support_hard, torch.maximum(high_conf_reflection, glare_focus_mask)), 0.0, 1.0)
    direct_laplacian_showcase = masked_laplacian_loss(direct_clean, target, showcase_loss_mask)
    visual_laplacian_showcase = masked_laplacian_loss(prediction, target, showcase_loss_mask)
    visual_chroma_showcase = masked_chroma_loss(prediction, target, showcase_loss_mask)

    profile = loss_profile if loss_profile in {"stable", "balanced", "hard", "visual"} else "balanced"
    weights = {
        "stable": {
            "full_l1": 0.24,
            "focus_l1": 0.23,
            "background_l1": 0.21,
            "direct_clean_focus_l1": 0.090,
            "direct_clean_core_l1": 0.045,
            "direct_clean_background_l1": 0.050,
            "direct_clean_ssim_loss": 0.045,
            "direct_clean_grad_loss": 0.025,
            "residual_reflection_l1": 0.060,
            "residual_clean_l1": 0.080,
            "ssim_loss": 0.10,
            "grad_loss": 0.07,
            "glare_l1": 0.04,
            "reflection_focus_l1": 0.050,
            "core_l1": 0.040,
            "lowfreq_reflection_l1": 0.035,
            "prior_escape": 0.005,
            "texture_loss": 0.03,
            "high_light_suppression": 0.025,
            "line_reflection_suppression": 0.005,
            "raw_prior_guard": 0.0,
            "direct_prior_guard": 0.0,
            "visual_prior_guard": 0.0,
            "reflection_outside": 0.015,
            "reflection_sparse": 0.005,
        },
        "balanced": {
            "full_l1": 0.18,
            "focus_l1": 0.27,
            "background_l1": 0.18,
            "direct_clean_focus_l1": 0.160,
            "direct_clean_core_l1": 0.095,
            "direct_clean_background_l1": 0.040,
            "direct_clean_ssim_loss": 0.020,
            "direct_clean_grad_loss": 0.011,
            "residual_reflection_l1": 0.120,
            "residual_clean_l1": 0.150,
            "ssim_loss": 0.09,
            "grad_loss": 0.06,
            "glare_l1": 0.065,
            "reflection_focus_l1": 0.105,
            "core_l1": 0.090,
            "lowfreq_reflection_l1": 0.060,
            "prior_escape": 0.015,
            "texture_loss": 0.045,
            "high_light_suppression": 0.045,
            "line_reflection_suppression": 0.014,
            "raw_prior_guard": 0.0,
            "direct_prior_guard": 0.0,
            "visual_prior_guard": 0.0,
            "reflection_outside": 0.015,
            "reflection_sparse": 0.005,
        },
        "hard": {
            "full_l1": 0.14,
            "focus_l1": 0.28,
            "background_l1": 0.15,
            "direct_clean_focus_l1": 0.180,
            "direct_clean_core_l1": 0.135,
            "direct_clean_background_l1": 0.030,
            "direct_clean_ssim_loss": 0.040,
            "direct_clean_grad_loss": 0.025,
            "residual_reflection_l1": 0.180,
            "residual_clean_l1": 0.220,
            "ssim_loss": 0.08,
            "grad_loss": 0.06,
            "glare_l1": 0.085,
            "reflection_focus_l1": 0.150,
            "core_l1": 0.135,
            "lowfreq_reflection_l1": 0.095,
            "prior_escape": 0.035,
            "texture_loss": 0.050,
            "high_light_suppression": 0.060,
            "line_reflection_suppression": 0.022,
            "raw_prior_guard": 0.0,
            "direct_prior_guard": 0.0,
            "visual_prior_guard": 0.0,
            "reflection_outside": 0.018,
            "reflection_sparse": 0.006,
        },
        "visual": {
            "full_l1": 0.14,
            "focus_l1": 0.75,
            "background_l1": 0.125,
            "direct_clean_focus_l1": 0.85,
            "direct_clean_core_l1": 0.165,
            "direct_clean_background_l1": 0.040,
            "direct_clean_ssim_loss": 0.028,
            "direct_clean_grad_loss": 0.020,
            "residual_reflection_l1": 0.45,
            "residual_clean_l1": 0.65,
            "ssim_loss": 0.075,
            "grad_loss": 0.060,
            "glare_l1": 0.095,
            "reflection_focus_l1": 0.170,
            "core_l1": 0.145,
            "lowfreq_reflection_l1": 0.125,
            "prior_escape": 0.025,
            "texture_loss": 0.045,
            "high_light_suppression": 0.055,
            "line_reflection_suppression": 0.032,
            "raw_prior_guard": 0.0,
            "direct_prior_guard": 0.0,
            "visual_prior_guard": 0.0,
            "reflection_outside": 0.016,
            "reflection_sparse": 0.006,
            "direct_laplacian_showcase": 0.0,
            "visual_laplacian_showcase": 0.0,
            "visual_chroma_showcase": 0.0,
        },
    }[profile]
    weights.setdefault("direct_laplacian_showcase", 0.0)
    weights.setdefault("visual_laplacian_showcase", 0.0)
    weights.setdefault("visual_chroma_showcase", 0.0)
    weights.setdefault("lowfreq_reflection_loss", 0.42)
    weights.setdefault("dark_shadow_clean_loss", 0.36)
    weights.setdefault("scene_reflection_suppression_loss", 0.32)
    weights.setdefault("shadow_veil_consistency_loss", 0.28)
    weights.setdefault("lowfreq_confidence_loss", 0.12)
    weights.setdefault("dark_shadow_confidence_loss", 0.12)
    weights.setdefault("reflection_confidence_loss", 0.10)
    weights.setdefault("line_band_confidence_loss", 0.05)
    weights.setdefault("text_reflection_confidence_loss", 0.05)
    weights.setdefault("tint_confidence_loss", 0.05)
    weights.setdefault("foreground_structure_guard_loss", 0.12)
    weights.setdefault("visualplus_should_differ_from_raw_in_reflection_region", 0.05)
    weights.setdefault("residual_consistency_loss", 0.24)
    weights.setdefault("hard_case_weighted_loss", 0.20)
    weights.setdefault("highlight_residual_loss", 0.24)
    weights.setdefault("highlight_clean_loss", 0.18)
    weights.setdefault("highlight_lowfreq_loss", 0.20)
    weights.setdefault("highlight_chroma_loss", 0.08)
    weights.setdefault("line_band_suppression_loss", 0.10)
    weights.setdefault("glare_core_l1", 0.12)
    weights.setdefault("strong_glare_direct_clean_l1", 0.22)
    weights.setdefault("background_protection_loss", 0.10)
    weights.setdefault("hard_case_direct_clean_loss", 0.12)
    weights.setdefault("hard_case_residual_loss", 0.10)
    if showcase_mode:
        weights = dict(weights)
        weights.update(
            {
                "background_l1": min(weights["background_l1"], 0.070),
                "direct_clean_background_l1": min(weights["direct_clean_background_l1"], 0.015),
                "residual_reflection_l1": 0.75,
                "residual_clean_l1": 1.00,
                "direct_clean_focus_l1": 1.20,
                "reflection_outside": min(weights["reflection_outside"], 0.008),
                "reflection_sparse": min(weights["reflection_sparse"], 0.003),
                "raw_prior_guard": 0.0,
                "direct_prior_guard": 0.0,
                "visual_prior_guard": 0.0,
                "direct_laplacian_showcase": 0.18,
                "visual_laplacian_showcase": 0.15,
                "visual_chroma_showcase": 0.20,
                "lowfreq_reflection_loss": 0.65,
                "dark_shadow_clean_loss": 0.55,
                "scene_reflection_suppression_loss": 0.55,
                "shadow_veil_consistency_loss": 0.35,
                "lowfreq_confidence_loss": 0.22,
                "dark_shadow_confidence_loss": 0.20,
                "foreground_structure_guard_loss": 0.18,
                "visualplus_should_differ_from_raw_in_reflection_region": 0.25,
                "residual_consistency_loss": 0.35,
                "hard_case_weighted_loss": 0.35,
                "highlight_residual_loss": 0.55,
                "highlight_clean_loss": 0.46,
                "highlight_lowfreq_loss": 0.50,
                "highlight_chroma_loss": 0.16,
                "line_band_suppression_loss": 0.32,
                "glare_core_l1": 0.38,
                "strong_glare_direct_clean_l1": 0.58,
                "background_protection_loss": 0.08,
                "hard_case_direct_clean_loss": 0.30,
                "hard_case_residual_loss": 0.26,
            }
        )
        if profile == "visual":
            weights.update(
                {
                    "focus_l1": 1.00,
                    "reflection_focus_l1": 1.10,
                    "glare_l1": 0.95,
                    "core_l1": 0.50,
                    "lowfreq_reflection_l1": 0.28,
                    "prior_escape": 0.045,
                    "highlight_residual_loss": 0.68,
                    "highlight_clean_loss": 0.58,
                    "highlight_lowfreq_loss": 0.62,
                    "line_band_suppression_loss": 0.42,
                    "glare_core_l1": 0.50,
                    "strong_glare_direct_clean_l1": 0.72,
                }
            )
    total = (
        weights["full_l1"] * full_l1
        + weights["focus_l1"] * focus_l1
        + weights["background_l1"] * background_l1
        + weights["direct_clean_focus_l1"] * direct_clean_focus_l1
        + weights["direct_clean_core_l1"] * direct_clean_core_l1
        + weights["direct_clean_background_l1"] * direct_clean_background_l1
        + weights["direct_clean_ssim_loss"] * direct_clean_ssim_loss
        + weights["direct_clean_grad_loss"] * direct_clean_grad_loss
        + weights["residual_reflection_l1"] * residual_reflection_l1
        + weights["residual_clean_l1"] * residual_clean_l1
        + weights["ssim_loss"] * ssim_loss
        + weights["grad_loss"] * grad
        + weights["glare_l1"] * glare_l1
        + weights["reflection_focus_l1"] * reflection_focus_l1
        + weights["core_l1"] * core_l1
        + weights["lowfreq_reflection_l1"] * lowfreq_reflection_l1
        + weights["prior_escape"] * prior_escape
        + weights["texture_loss"] * texture
        + weights["high_light_suppression"] * high_light_suppression
        + weights["line_reflection_suppression"] * line_suppression
        + weights["highlight_residual_loss"] * highlight_residual_loss
        + weights["highlight_clean_loss"] * highlight_clean_loss
        + weights["highlight_lowfreq_loss"] * highlight_lowfreq_loss
        + weights["highlight_chroma_loss"] * highlight_chroma_loss
        + weights["line_band_suppression_loss"] * line_band_suppression_loss
        + weights["glare_core_l1"] * glare_core_l1
        + weights["strong_glare_direct_clean_l1"] * strong_glare_direct_clean_l1
        + weights["raw_prior_guard"] * raw_prior_guard
        + weights["direct_prior_guard"] * direct_prior_guard
        + weights["visual_prior_guard"] * visual_prior_guard
        + weights["reflection_outside"] * reflection_outside
        + weights["reflection_sparse"] * reflection_sparse
        + weights["direct_laplacian_showcase"] * direct_laplacian_showcase
        + weights["visual_laplacian_showcase"] * visual_laplacian_showcase
        + weights["visual_chroma_showcase"] * visual_chroma_showcase
        + weights["lowfreq_reflection_loss"] * lowfreq_reflection_loss
        + weights["dark_shadow_clean_loss"] * dark_shadow_clean_loss
        + weights["scene_reflection_suppression_loss"] * scene_reflection_suppression_loss
        + weights["shadow_veil_consistency_loss"] * shadow_veil_consistency_loss
        + weights["lowfreq_confidence_loss"] * lowfreq_confidence_loss
        + weights["dark_shadow_confidence_loss"] * dark_shadow_confidence_loss
        + weights["reflection_confidence_loss"] * reflection_confidence_loss
        + weights["line_band_confidence_loss"] * line_band_confidence_loss
        + weights["text_reflection_confidence_loss"] * text_reflection_confidence_loss
        + weights["tint_confidence_loss"] * tint_confidence_loss
        + weights["foreground_structure_guard_loss"] * foreground_structure_guard_loss
        + weights["background_protection_loss"] * background_protection_loss
        + weights["visualplus_should_differ_from_raw_in_reflection_region"] * visualplus_should_differ_from_raw_in_reflection_region
        + weights["residual_consistency_loss"] * residual_consistency_loss
        + weights["hard_case_weighted_loss"] * hard_case_weighted_loss
        + weights["hard_case_direct_clean_loss"] * hard_case_direct_clean_loss
        + weights["hard_case_residual_loss"] * hard_case_residual_loss
        + float(perceptual_weight) * perceptual_l1
    )
    return total, {
        "full_l1": float(full_l1.detach().cpu().item()),
        "focus_l1": float(focus_l1.detach().cpu().item()),
        "core_l1": float(core_l1.detach().cpu().item()),
        "background_l1": float(background_l1.detach().cpu().item()),
        "direct_clean_focus_l1": float(direct_clean_focus_l1.detach().cpu().item()),
        "direct_clean_core_l1": float(direct_clean_core_l1.detach().cpu().item()),
        "direct_clean_background_l1": float(direct_clean_background_l1.detach().cpu().item()),
        "direct_clean_ssim_loss": float(direct_clean_ssim_loss.detach().cpu().item()),
        "direct_clean_grad_loss": float(direct_clean_grad_loss.detach().cpu().item()),
        "residual_reflection_l1": float(residual_reflection_l1.detach().cpu().item()),
        "residual_clean_l1": float(residual_clean_l1.detach().cpu().item()),
        "glare_l1": float(glare_l1.detach().cpu().item()),
        "glare_core_l1": float(glare_core_l1.detach().cpu().item()),
        "strong_glare_direct_clean_l1": float(strong_glare_direct_clean_l1.detach().cpu().item()),
        "highlight_residual_loss": float(highlight_residual_loss.detach().cpu().item()),
        "highlight_clean_loss": float(highlight_clean_loss.detach().cpu().item()),
        "highlight_lowfreq_loss": float(highlight_lowfreq_loss.detach().cpu().item()),
        "highlight_chroma_loss": float(highlight_chroma_loss.detach().cpu().item()),
        "reflection_focus_l1": float(reflection_focus_l1.detach().cpu().item()),
        "lowfreq_reflection_l1": float(lowfreq_reflection_l1.detach().cpu().item()),
        "prior_escape": float(prior_escape.detach().cpu().item()),
        "ssim_loss": float(ssim_loss.detach().cpu().item()),
        "grad_loss": float(grad.detach().cpu().item()),
        "texture_loss": float(texture.detach().cpu().item()),
        "high_light_suppression": float(high_light_suppression.detach().cpu().item()),
        "line_reflection_suppression": float(line_suppression.detach().cpu().item()),
        "line_band_suppression_loss": float(line_band_suppression_loss.detach().cpu().item()),
        "raw_prior_guard": float(raw_prior_guard.detach().cpu().item()),
        "direct_prior_guard": float(direct_prior_guard.detach().cpu().item()),
        "visual_prior_guard": float(visual_prior_guard.detach().cpu().item()),
        "reflection_outside": float(reflection_outside.detach().cpu().item()),
        "reflection_sparse": float(reflection_sparse.detach().cpu().item()),
        "direct_laplacian_showcase": float(direct_laplacian_showcase.detach().cpu().item()),
        "visual_laplacian_showcase": float(visual_laplacian_showcase.detach().cpu().item()),
        "visual_chroma_showcase": float(visual_chroma_showcase.detach().cpu().item()),
        "lowfreq_reflection_loss": float(lowfreq_reflection_loss.detach().cpu().item()),
        "dark_shadow_clean_loss": float(dark_shadow_clean_loss.detach().cpu().item()),
        "scene_reflection_suppression_loss": float(scene_reflection_suppression_loss.detach().cpu().item()),
        "shadow_veil_consistency_loss": float(shadow_veil_consistency_loss.detach().cpu().item()),
        "lowfreq_confidence_loss": float(lowfreq_confidence_loss.detach().cpu().item()),
        "dark_shadow_confidence_loss": float(dark_shadow_confidence_loss.detach().cpu().item()),
        "reflection_confidence_loss": float(reflection_confidence_loss.detach().cpu().item()),
        "line_band_confidence_loss": float(line_band_confidence_loss.detach().cpu().item()),
        "text_reflection_confidence_loss": float(text_reflection_confidence_loss.detach().cpu().item()),
        "tint_confidence_loss": float(tint_confidence_loss.detach().cpu().item()),
        "foreground_structure_guard_loss": float(foreground_structure_guard_loss.detach().cpu().item()),
        "background_protection_loss": float(background_protection_loss.detach().cpu().item()),
        "visualplus_should_differ_from_raw_in_reflection_region": float(
            visualplus_should_differ_from_raw_in_reflection_region.detach().cpu().item()
        ),
        "residual_consistency_loss": float(residual_consistency_loss.detach().cpu().item()),
        "hard_case_weighted_loss": float(hard_case_weighted_loss.detach().cpu().item()),
        "hard_case_direct_clean_loss": float(hard_case_direct_clean_loss.detach().cpu().item()),
        "hard_case_residual_loss": float(hard_case_residual_loss.detach().cpu().item()),
        "perceptual_l1": float(perceptual_l1.detach().cpu().item()),
    }


def compute_direct_pretrain_loss(
    model_output: torch.Tensor,
    target: torch.Tensor,
    reflection_mask: torch.Tensor,
    reference_rgb: torch.Tensor,
    glare_mask: Optional[torch.Tensor] = None,
    reflection_area_mask: Optional[torch.Tensor] = None,
    scene_reflection_mask: Optional[torch.Tensor] = None,
    dark_reflection_mask: Optional[torch.Tensor] = None,
    shadow_veil_mask: Optional[torch.Tensor] = None,
    lowfreq_reflection_mask: Optional[torch.Tensor] = None,
    foreground_structure_guard: Optional[torch.Tensor] = None,
    hard_case_weight: Optional[torch.Tensor] = None,
    visual_strength: float = 1.0,
    showcase_reflection_boost: float = 1.0,
    support_sharpen: float = 1.0,
    perceptual_model: Optional[nn.Module] = None,
    perceptual_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    mask = torch.clamp(reflection_mask, 0.0, 1.0)
    glare = torch.clamp(glare_mask if glare_mask is not None else mask, 0.0, 1.0)
    reflection_area = torch.clamp(reflection_area_mask if reflection_area_mask is not None else mask, 0.0, 1.0)
    scene_reflection = torch.clamp(scene_reflection_mask if scene_reflection_mask is not None else torch.zeros_like(mask), 0.0, 1.0)
    dark_reflection = torch.clamp(dark_reflection_mask if dark_reflection_mask is not None else torch.zeros_like(mask), 0.0, 1.0)
    shadow_veil = torch.clamp(shadow_veil_mask if shadow_veil_mask is not None else torch.zeros_like(mask), 0.0, 1.0)
    lowfreq_reflection = torch.clamp(lowfreq_reflection_mask if lowfreq_reflection_mask is not None else torch.zeros_like(mask), 0.0, 1.0)
    foreground_guard = torch.clamp(foreground_structure_guard if foreground_structure_guard is not None else torch.zeros_like(mask), 0.0, 1.0)
    shadow_region = torch.clamp(torch.maximum(scene_reflection, torch.maximum(dark_reflection, torch.maximum(shadow_veil, lowfreq_reflection))), 0.0, 1.0)
    guard_release = torch.clamp(0.75 * shadow_region + 0.35 * reflection_area + 0.25 * glare, 0.0, 1.0)
    structure_safe = torch.clamp(1.0 - 0.65 * foreground_guard * (1.0 - 0.85 * guard_release), 0.30, 1.0)
    focus_mask = torch.clamp(0.50 * mask + 0.15 * glare + 0.35 * reflection_area, 0.0, 1.0)
    focus_mask = torch.clamp(torch.maximum(focus_mask, 0.55 * shadow_region * structure_safe), 0.0, 1.0)
    (
        direct_clean,
        positive_reflection,
        signed_clean_delta,
        _support_soft,
        _support_hard,
        _learned_gate,
        reflection_confidence,
        line_band_confidence,
        text_reflection_confidence,
        tint_confidence,
        lowfreq_reflection_confidence,
        dark_shadow_confidence,
    ) = decode_reflection_heads(
        model_output,
        reference_rgb,
        mask,
        glare_mask=glare,
        reflection_area_mask=reflection_area,
        visual_strength=visual_strength,
        showcase_reflection_boost=showcase_reflection_boost,
        support_sharpen=support_sharpen,
    )
    residual_target_signed = torch.clamp(reference_rgb - target, -0.6, 0.6)
    residual_target_positive = torch.clamp(reference_rgb - target, 0.0, 1.0)
    gt_residual_luma = torch.mean(torch.abs(reference_rgb - target), dim=1, keepdim=True)
    gt_residual_mask = torch.clamp((gt_residual_luma - 0.006) / 0.09, 0.0, 1.0)
    gt_residual_mask = torch.maximum(gt_residual_mask, F.max_pool2d(gt_residual_mask, kernel_size=5, stride=1, padding=2))
    strong_glare_core = compute_strong_glare_core_tensor(
        reference_rgb,
        glare,
        mask,
        reflection_area_mask=reflection_area,
        line_confidence=line_band_confidence,
        lowfreq_confidence=lowfreq_reflection_confidence,
    )
    hard_focus = torch.clamp(torch.maximum(reflection_area, torch.maximum(shadow_region, 0.65 * focus_mask)), 0.0, 1.0)
    lowfreq_dark_focus = torch.clamp(torch.maximum(lowfreq_reflection, dark_reflection), 0.0, 1.0)
    supervised_mask = torch.clamp(
        torch.maximum(
            gt_residual_mask,
            torch.maximum(0.92 * strong_glare_core, torch.maximum(0.88 * hard_focus, torch.maximum(0.72 * lowfreq_dark_focus, 0.55 * focus_mask))),
        ),
        0.0,
        1.0,
    )
    background_mask = torch.clamp(1.0 - supervised_mask, 0.0, 1.0)
    direct_weights = torch.clamp(0.18 + 0.82 * supervised_mask + 0.20 * shadow_region * structure_safe, 0.0, 1.0)
    direct_l1 = weighted_charbonnier(direct_clean, target, direct_weights)
    direct_lowfreq = low_frequency_l1(direct_clean, target, direct_weights)
    direct_gradient = gradient_loss(direct_clean, target)
    direct_laplacian = masked_laplacian_loss(direct_clean, target, supervised_mask)
    direct_chroma = masked_chroma_loss(direct_clean, target, supervised_mask)
    positive_residual_l1 = weighted_charbonnier(positive_reflection, residual_target_positive, supervised_mask)
    shadow_supervised = torch.clamp(torch.maximum(supervised_mask, shadow_region * structure_safe), 0.0, 1.0)
    signed_delta_l1 = weighted_charbonnier(signed_clean_delta, residual_target_signed, shadow_supervised)
    direct_shadow_l1 = weighted_charbonnier(direct_clean, target, shadow_supervised)
    direct_shadow_lowfreq = low_frequency_l1(direct_clean, target, shadow_supervised, kernel_size=31)
    strong_glare_direct_l1 = weighted_charbonnier(direct_clean, target, strong_glare_core)
    strong_glare_residual_l1 = weighted_charbonnier(positive_reflection, residual_target_positive, strong_glare_core)
    reflection_confidence_target = torch.clamp(torch.maximum(reflection_area, shadow_region), 0.0, 1.0)
    lowfreq_target = torch.clamp(lowfreq_reflection * (1.0 - 0.35 * glare) + 0.25 * shadow_veil + 0.18 * scene_reflection, 0.0, 1.0)
    dark_shadow_target = torch.clamp(dark_reflection * (1.0 - 0.20 * glare) + 0.22 * shadow_veil + 0.14 * scene_reflection, 0.0, 1.0)
    line_target = torch.clamp(glare * torch.clamp((reflection_area - 0.10) / 0.90, 0.0, 1.0) * (1.0 - 0.45 * shadow_veil), 0.0, 1.0)
    text_target = torch.clamp(shadow_veil * torch.clamp((shadow_region - 0.06) / 0.94, 0.0, 1.0) * (1.0 - 0.40 * glare), 0.0, 1.0)
    reflection_confidence_l1 = weighted_charbonnier(
        reflection_confidence,
        reflection_confidence_target,
        torch.clamp(reflection_confidence_target + 0.20, 0.0, 1.0),
    )
    lowfreq_confidence_l1 = weighted_charbonnier(lowfreq_reflection_confidence, lowfreq_target, torch.clamp(lowfreq_target + 0.20, 0.0, 1.0))
    dark_shadow_confidence_l1 = weighted_charbonnier(dark_shadow_confidence, dark_shadow_target, torch.clamp(dark_shadow_target + 0.20, 0.0, 1.0))
    line_confidence_l1 = weighted_charbonnier(line_band_confidence, line_target, torch.clamp(line_target + 0.10, 0.0, 1.0))
    text_confidence_l1 = weighted_charbonnier(text_reflection_confidence, text_target, torch.clamp(text_target + 0.10, 0.0, 1.0))
    tint_confidence_l1 = weighted_charbonnier(tint_confidence, lowfreq_target, torch.clamp(lowfreq_target + 0.10, 0.0, 1.0))
    background_l1 = weighted_charbonnier(direct_clean, target, background_mask)
    if hard_case_weight is not None:
        sample_weight = torch.clamp(hard_case_weight.view(-1, 1, 1, 1).to(device=direct_clean.device, dtype=direct_clean.dtype), 1.0, 6.0)
        hard_case_weighted_loss = weighted_charbonnier(
            direct_clean,
            target,
            torch.clamp((sample_weight - 1.0) * shadow_supervised, 0.0, 5.0),
        )
    else:
        hard_case_weighted_loss = direct_clean.new_tensor(0.0)
    perceptual_l1 = direct_clean.new_tensor(0.0)
    if perceptual_model is not None and perceptual_weight > 0.0:
        perceptual_l1 = perceptual_model(direct_clean, target)
    total = (
        2.35 * direct_l1
        + 1.18 * direct_lowfreq
        + 0.28 * direct_gradient
        + 0.34 * direct_laplacian
        + 0.22 * direct_chroma
        + 1.10 * positive_residual_l1
        + 1.30 * signed_delta_l1
        + 0.92 * direct_shadow_l1
        + 0.78 * direct_shadow_lowfreq
        + 0.52 * strong_glare_direct_l1
        + 0.34 * strong_glare_residual_l1
        + 0.20 * reflection_confidence_l1
        + 0.28 * lowfreq_confidence_l1
        + 0.28 * dark_shadow_confidence_l1
        + 0.16 * line_confidence_l1
        + 0.16 * text_confidence_l1
        + 0.06 * tint_confidence_l1
        + 0.35 * hard_case_weighted_loss
        + 0.20 * background_l1
        + float(perceptual_weight) * perceptual_l1
    )
    return total, {
        "direct_l1": float(direct_l1.detach().cpu().item()),
        "direct_lowfreq": float(direct_lowfreq.detach().cpu().item()),
        "direct_gradient": float(direct_gradient.detach().cpu().item()),
        "direct_laplacian": float(direct_laplacian.detach().cpu().item()),
        "direct_chroma": float(direct_chroma.detach().cpu().item()),
        "positive_residual_l1": float(positive_residual_l1.detach().cpu().item()),
        "signed_delta_l1": float(signed_delta_l1.detach().cpu().item()),
        "direct_shadow_l1": float(direct_shadow_l1.detach().cpu().item()),
        "direct_shadow_lowfreq": float(direct_shadow_lowfreq.detach().cpu().item()),
        "strong_glare_direct_l1": float(strong_glare_direct_l1.detach().cpu().item()),
        "strong_glare_residual_l1": float(strong_glare_residual_l1.detach().cpu().item()),
        "reflection_confidence_l1": float(reflection_confidence_l1.detach().cpu().item()),
        "lowfreq_confidence_l1": float(lowfreq_confidence_l1.detach().cpu().item()),
        "dark_shadow_confidence_l1": float(dark_shadow_confidence_l1.detach().cpu().item()),
        "line_confidence_l1": float(line_confidence_l1.detach().cpu().item()),
        "text_confidence_l1": float(text_confidence_l1.detach().cpu().item()),
        "tint_confidence_l1": float(tint_confidence_l1.detach().cpu().item()),
        "hard_case_weighted_loss": float(hard_case_weighted_loss.detach().cpu().item()),
        "background_l1": float(background_l1.detach().cpu().item()),
        "prior_guard": 0.0,
        "perceptual_l1": float(perceptual_l1.detach().cpu().item()),
    }, direct_clean


class VGGPerceptualLoss(nn.Module):
    def __init__(self, features: nn.Module) -> None:
        super().__init__()
        self.features = features.eval()
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        self.layers = {3, 8, 15}
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = (torch.clamp(prediction.float(), 0.0, 1.0) - self.mean) / self.std
        target = (torch.clamp(target.float(), 0.0, 1.0) - self.mean) / self.std
        loss = prediction.new_tensor(0.0)
        pred_feat = prediction
        target_feat = target
        for index, layer in enumerate(self.features):
            pred_feat = layer(pred_feat)
            target_feat = layer(target_feat)
            if index in self.layers:
                loss = loss + F.l1_loss(pred_feat, target_feat)
        return loss


def build_perceptual_loss_model(device: torch.device) -> Optional[nn.Module]:
    try:
        from torchvision import models
    except Exception:
        return None
    try:
        weights_enum = getattr(models, "VGG16_Weights", None)
        weights = weights_enum.IMAGENET1K_V1 if weights_enum is not None else None
        if weights is None:
            return None
        checkpoint_name = Path(str(weights.url)).name
        checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
        if not checkpoint_path.exists():
            return None
        vgg = models.vgg16(weights=weights, progress=False)
        return VGGPerceptualLoss(vgg.features[:16]).to(device=device).eval()
    except Exception:
        return None


