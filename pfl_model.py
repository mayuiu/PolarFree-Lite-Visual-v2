import argparse
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pfl_config import MODEL_INPUT_CHANNELS, MODEL_OUTPUT_CHANNELS

EPS = 1e-6

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        norm_groups = 8 if out_channels % 8 == 0 else 4
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.layers(x) + self.skip(x))


class PolarFreeLiteUNet(nn.Module):
    def __init__(self, in_channels: int = MODEL_INPUT_CHANNELS, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 6)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ConvBlock(base_channels * 6 + base_channels * 4, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.final = nn.Conv2d(base_channels, MODEL_OUTPUT_CHANNELS, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottleneck = self.enc4(self.pool(e3))
        d3 = self.up(bottleneck)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.final(d1)


def gaussian_blur_tensor(image: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return image
    radius = max(int(math.ceil(3.0 * sigma)), 1)
    kernel_size = radius * 2 + 1
    coords = torch.arange(kernel_size, device=image.device, dtype=image.dtype) - radius
    kernel_1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / torch.sum(kernel_1d)
    channels = image.shape[1]
    kernel_x = kernel_1d.view(1, 1, 1, kernel_size).expand(channels, 1, 1, kernel_size)
    kernel_y = kernel_1d.view(1, 1, kernel_size, 1).expand(channels, 1, kernel_size, 1)
    blurred = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode="reflect"), kernel_x, groups=channels)
    blurred = F.conv2d(F.pad(blurred, (0, 0, radius, radius), mode="reflect"), kernel_y, groups=channels)
    return blurred


# Candidate cleanup: no in-file call sites; retained for now to avoid behavior changes.

def apply_showcase_visual_postprocess(
    showcase_result: torch.Tensor,
    reference_rgb: Optional[torch.Tensor],
    support: torch.Tensor,
) -> torch.Tensor:
    return showcase_visual_polish(showcase_result, reference_rgb, showcase_result, support, support)


def showcase_visual_polish(
    showcase_result: torch.Tensor,
    reference_rgb: Optional[torch.Tensor],
    direct_clean: torch.Tensor,
    support_soft: torch.Tensor,
    support_hard: torch.Tensor,
    tint_confidence: Optional[torch.Tensor] = None,
    lowfreq_confidence: Optional[torch.Tensor] = None,
    dark_shadow_confidence: Optional[torch.Tensor] = None,
    foreground_structure_guard: Optional[torch.Tensor] = None,
    v2_reflection_takeover: Optional[torch.Tensor] = None,
    effective_guard: Optional[torch.Tensor] = None,
    line_band_confidence: Optional[torch.Tensor] = None,
    text_reflection_confidence: Optional[torch.Tensor] = None,
    shadow_veil_mask: Optional[torch.Tensor] = None,
    detail_strength: float = 0.10,
    smooth_strength: float = 0.14,
    tint_suppress: float = 0.12,
) -> torch.Tensor:
    if reference_rgb is None:
        return torch.clamp(showcase_result, 0.0, 1.0)

    support_soft = torch.clamp(support_soft, 0.0, 1.0)
    support_hard = torch.clamp(support_hard, 0.0, 1.0)
    tint_confidence = torch.clamp(tint_confidence if tint_confidence is not None else support_soft, 0.0, 1.0)
    lowfreq_confidence = torch.clamp(
        lowfreq_confidence if lowfreq_confidence is not None else torch.zeros_like(support_soft),
        0.0,
        1.0,
    )
    dark_shadow_confidence = torch.clamp(
        dark_shadow_confidence if dark_shadow_confidence is not None else torch.zeros_like(support_soft),
        0.0,
        1.0,
    )
    line_band_confidence = torch.clamp(
        line_band_confidence if line_band_confidence is not None else torch.zeros_like(support_soft),
        0.0,
        1.0,
    )
    text_reflection_confidence = torch.clamp(
        text_reflection_confidence if text_reflection_confidence is not None else torch.zeros_like(support_soft),
        0.0,
        1.0,
    )
    shadow_veil_mask = torch.clamp(
        shadow_veil_mask if shadow_veil_mask is not None else torch.zeros_like(support_soft),
        0.0,
        1.0,
    )
    foreground_structure_guard = torch.clamp(
        foreground_structure_guard if foreground_structure_guard is not None else torch.zeros_like(support_soft),
        0.0,
        1.0,
    )
    v2_reflection_takeover = torch.clamp(
        v2_reflection_takeover
        if v2_reflection_takeover is not None
        else torch.maximum(support_hard, torch.maximum(lowfreq_confidence, dark_shadow_confidence)),
        0.0,
        1.0,
    )
    effective_guard = torch.clamp(
        effective_guard if effective_guard is not None else foreground_structure_guard * torch.clamp(1.0 - v2_reflection_takeover, 0.0, 1.0),
        0.0,
        1.0,
    )

    hard3 = support_hard.expand_as(showcase_result)
    takeover3 = v2_reflection_takeover.expand_as(showcase_result)
    lowfreq_region = torch.clamp(
        0.45 * lowfreq_confidence
        + 0.35 * dark_shadow_confidence
        + 0.18 * shadow_veil_mask
        + 0.12 * support_hard,
        0.0,
        1.0,
    )
    lowfreq3 = lowfreq_region.expand_as(showcase_result)
    line_text_region = torch.clamp(torch.maximum(line_band_confidence, text_reflection_confidence), 0.0, 1.0).expand_as(showcase_result)
    guard3 = effective_guard.expand_as(showcase_result)

    # Extreme-visual path: the network has already detected difficult reflection regions.
    # In those regions VisualPlus should visibly use DirectClean instead of staying close to Raw/Prior.
    hard_takeover = torch.clamp(
        0.52 * takeover3
        + 0.26 * hard3
        + 0.34 * lowfreq3
        + 0.22 * line_text_region,
        0.0,
        0.92,
    )
    hard_takeover = torch.clamp(hard_takeover * torch.clamp(1.0 - 0.06 * guard3, 0.80, 1.0), 0.0, 0.92)
    showcase_result = torch.clamp(showcase_result * (1.0 - hard_takeover) + direct_clean * hard_takeover, 0.0, 1.0)

    if torch.max(lowfreq_region).detach().cpu().item() > EPS:
        low_showcase = gaussian_blur_tensor(showcase_result, sigma=4.0)
        low_direct = gaussian_blur_tensor(direct_clean, sigma=4.0)
        low_alpha = torch.clamp(
            0.42 * lowfreq_confidence
            + 0.38 * dark_shadow_confidence
            + 0.22 * shadow_veil_mask
            + 0.26 * v2_reflection_takeover,
            0.0,
            0.72,
        )
        low_alpha = torch.clamp(low_alpha * torch.clamp(1.0 - 0.08 * effective_guard, 0.78, 1.0), 0.0, 0.72)
        showcase_result = torch.clamp(showcase_result + low_alpha.expand_as(showcase_result) * (low_direct - low_showcase), 0.0, 1.0)

    tint_region = torch.clamp(
        0.58 * tint_confidence
        + 0.38 * v2_reflection_takeover
        + 0.30 * lowfreq_confidence
        + 0.26 * shadow_veil_mask
        + 0.18 * dark_shadow_confidence,
        0.0,
        1.0,
    )
    channel_mean = torch.mean(showcase_result, dim=1, keepdim=True)
    g_excess = torch.relu(showcase_result[:, 1:2] - channel_mean)
    r_excess = torch.relu(showcase_result[:, 0:1] - channel_mean)
    showcase_result = showcase_result.clone()
    showcase_result[:, 1:2] = showcase_result[:, 1:2] - float(tint_suppress) * g_excess * tint_region
    showcase_result[:, 0:1] = showcase_result[:, 0:1] - 0.45 * float(tint_suppress) * r_excess * tint_region
    showcase_result = torch.clamp(showcase_result, 0.0, 1.0)

    reference_detail = reference_rgb - gaussian_blur_tensor(reference_rgb, sigma=1.2)
    safe_detail_mask = torch.clamp(1.0 - v2_reflection_takeover - 0.50 * lowfreq_region, 0.0, 1.0).expand_as(showcase_result)
    showcase_result = showcase_result + float(detail_strength) * reference_detail * safe_detail_mask

    low_showcase = gaussian_blur_tensor(showcase_result, sigma=1.0)
    smooth_region = torch.clamp(0.55 * support_hard + 0.45 * lowfreq_region, 0.0, 1.0).expand_as(showcase_result)
    smooth_alpha = torch.clamp((float(smooth_strength) * 1.35) * smooth_region, 0.0, 0.58)
    showcase_result = showcase_result * (1.0 - smooth_alpha) + low_showcase * smooth_alpha
    return torch.clamp(showcase_result, 0.0, 1.0)


def visual_strong_alpha_tensor(
    reference_rgb: Optional[torch.Tensor],
    base_prior: torch.Tensor,
    prediction: torch.Tensor,
    direct_clean: torch.Tensor,
    reflection_mask: torch.Tensor,
    glare_mask: Optional[torch.Tensor],
    reflection_area_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    core, boundary = compute_high_conf_reflection_mask(
        reference_rgb,
        base_prior,
        direct_clean,
        reflection_mask,
        glare_mask=glare_mask,
        reflection_area_mask=reflection_area_mask,
    )
    mask = torch.clamp(reflection_mask, 0.0, 1.0)
    glare = torch.clamp(glare_mask if glare_mask is not None else torch.zeros_like(mask), 0.0, 1.0)
    area = torch.clamp(reflection_area_mask if reflection_area_mask is not None else mask, 0.0, 1.0)
    candidate = torch.clamp(0.44 * prediction + 0.46 * direct_clean + 0.10 * base_prior, 0.0, 1.0)
    alpha = torch.clamp(0.04 + 0.18 * boundary + 0.34 * core + 0.07 * glare + 0.06 * area, 0.0, 0.68)
    alpha = torch.where(core > 0.82, torch.clamp(alpha + 0.05 * core, 0.0, 0.72), alpha)

    prior_luma = torch.mean(base_prior, dim=1, keepdim=True)
    cand_luma = torch.mean(candidate, dim=1, keepdim=True)
    ref_luma = torch.mean(reference_rgb, dim=1, keepdim=True) if reference_rgb is not None else prior_luma
    dark_penalty = torch.clamp((prior_luma - cand_luma - 0.10) / 0.22, 0.0, 1.0)
    lowfreq_shift = torch.abs(
        F.avg_pool2d(cand_luma - ref_luma, kernel_size=21, stride=1, padding=10)
        - F.avg_pool2d(prior_luma - ref_luma, kernel_size=21, stride=1, padding=10)
    )
    lowfreq_penalty = torch.clamp((lowfreq_shift - 0.035) / 0.16, 0.0, 1.0)
    cand_center = candidate - cand_luma
    prior_center = base_prior - prior_luma
    color_penalty = torch.clamp((torch.mean(torch.abs(cand_center - prior_center), dim=1, keepdim=True) - 0.035) / 0.16, 0.0, 1.0)

    def grad_mag(image: torch.Tensor) -> torch.Tensor:
        dx = F.pad(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), (0, 1, 0, 0))
        dy = F.pad(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), (0, 0, 0, 1))
        return torch.mean(dx + dy, dim=1, keepdim=True)

    prior_grad = F.avg_pool2d(grad_mag(base_prior), kernel_size=9, stride=1, padding=4)
    cand_grad = F.avg_pool2d(grad_mag(candidate), kernel_size=9, stride=1, padding=4)
    blur_penalty = torch.clamp((prior_grad - cand_grad - 0.018) / 0.08, 0.0, 1.0)
    quality = torch.clamp(
        1.0 - 0.36 * dark_penalty - 0.28 * lowfreq_penalty - 0.20 * color_penalty - 0.22 * blur_penalty,
        0.12,
        1.0,
    )
    outside = torch.clamp(1.0 - torch.maximum(boundary, area), 0.0, 1.0)
    alpha = torch.clamp(alpha * quality * (1.0 - 0.70 * outside), 0.0, 0.72)
    strong_direct = torch.clamp(0.46 * area + 0.42 * core + 0.18 * boundary, 0.0, 0.88)
    candidate = torch.clamp(candidate * (1.0 - strong_direct) + direct_clean * strong_direct, 0.0, 1.0)
    return candidate, alpha


def compute_visual_line_score_tensor(reference_rgb: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if reference_rgb is None:
        return None
    luma = torch.mean(torch.clamp(reference_rgb, 0.0, 1.0), dim=1, keepdim=True)
    local = F.avg_pool2d(luma, kernel_size=9, stride=1, padding=4)
    bright_residual = torch.relu(luma - local)
    horizontal = F.avg_pool2d(bright_residual, kernel_size=(3, 31), stride=1, padding=(1, 15))
    vertical = F.avg_pool2d(bright_residual, kernel_size=(31, 3), stride=1, padding=(15, 1))
    line_score = torch.maximum(horizontal, vertical)
    return torch.clamp((line_score - 0.010) / 0.105, 0.0, 1.0)


def compute_strong_glare_core_tensor(
    reference_rgb: Optional[torch.Tensor],
    glare_mask: torch.Tensor,
    reflection_mask: torch.Tensor,
    reflection_area_mask: Optional[torch.Tensor] = None,
    line_confidence: Optional[torch.Tensor] = None,
    lowfreq_confidence: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    base = torch.clamp(reflection_mask, 0.0, 1.0)
    area = torch.clamp(reflection_area_mask if reflection_area_mask is not None else base, 0.0, 1.0)
    glare = torch.clamp(glare_mask, 0.0, 1.0)
    line = torch.clamp(line_confidence if line_confidence is not None else torch.zeros_like(base), 0.0, 1.0)
    lowfreq = torch.clamp(lowfreq_confidence if lowfreq_confidence is not None else torch.zeros_like(base), 0.0, 1.0)
    visual_line = compute_visual_line_score_tensor(reference_rgb)
    if visual_line is None:
        visual_line = torch.zeros_like(base)

    bright_core = torch.zeros_like(base)
    colored_core = torch.zeros_like(base)
    local_over = torch.zeros_like(base)
    if reference_rgb is not None:
        ref = torch.clamp(reference_rgb, 0.0, 1.0)
        rgb_max = torch.max(ref, dim=1, keepdim=True).values
        rgb_min = torch.min(ref, dim=1, keepdim=True).values
        luma = torch.mean(ref, dim=1, keepdim=True)
        saturation = torch.clamp((rgb_max - rgb_min) / (rgb_max + EPS), 0.0, 1.0)
        bright_core = torch.clamp((rgb_max - 0.70) / 0.24, 0.0, 1.0)
        relaxed_sat = torch.clamp((0.86 - saturation) / 0.86, 0.0, 1.0)
        colored_core = bright_core * torch.clamp((saturation - 0.08) / 0.62, 0.0, 1.0)
        local = F.avg_pool2d(luma, kernel_size=9, stride=1, padding=4)
        local_over = torch.clamp((luma - local - 0.012) / 0.115, 0.0, 1.0)
        bright_core = bright_core * (0.26 + 0.74 * relaxed_sat) + 0.38 * colored_core

    core = torch.clamp(
        0.30 * glare
        + 0.22 * bright_core
        + 0.16 * local_over
        + 0.16 * torch.maximum(line, visual_line)
        + 0.12 * area
        + 0.10 * lowfreq,
        0.0,
        1.0,
    )
    core = torch.maximum(core, torch.clamp((glare - 0.42) / 0.42, 0.0, 1.0) * torch.clamp((area - 0.12) / 0.55, 0.0, 1.0))
    core = torch.maximum(core, 0.72 * torch.maximum(visual_line, line) * torch.clamp((glare + area) * 0.5, 0.0, 1.0))
    core = F.max_pool2d(core, kernel_size=5, stride=1, padding=2)
    core = F.avg_pool2d(torch.clamp(core, 0.0, 1.0), kernel_size=5, stride=1, padding=2)
    return torch.clamp(core, 0.0, 1.0)


def compute_high_conf_reflection_mask(
    reference_rgb: Optional[torch.Tensor],
    base_prior: Optional[torch.Tensor],
    direct_clean: torch.Tensor,
    reflection_mask: torch.Tensor,
    glare_mask: Optional[torch.Tensor] = None,
    reflection_area_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    final_mask = torch.clamp(reflection_mask, 0.0, 1.0)
    reflection_area = torch.clamp(reflection_area_mask if reflection_area_mask is not None else final_mask, 0.0, 1.0)
    glare = torch.clamp(glare_mask if glare_mask is not None else torch.zeros_like(final_mask), 0.0, 1.0)
    prior_diff = torch.zeros_like(final_mask)
    direct_diff = torch.zeros_like(final_mask)
    veil_score = torch.zeros_like(final_mask)
    line_score = torch.zeros_like(final_mask)
    if reference_rgb is not None and base_prior is not None:
        prior_diff = torch.mean(torch.abs(reference_rgb - base_prior), dim=1, keepdim=True)
        veil_source = torch.mean(torch.relu(reference_rgb - base_prior), dim=1, keepdim=True)
        veil_score = F.avg_pool2d(veil_source, kernel_size=25, stride=1, padding=12)
        veil_score = torch.clamp((veil_score - 0.010) / 0.150, 0.0, 1.0)
        prior_diff = torch.clamp((prior_diff - 0.018) / 0.165, 0.0, 1.0)
        line_score = compute_visual_line_score_tensor(reference_rgb)
        if line_score is None:
            line_score = torch.zeros_like(final_mask)
    if base_prior is not None:
        direct_diff = torch.mean(torch.abs(direct_clean - base_prior), dim=1, keepdim=True)
        direct_diff = torch.clamp((direct_diff - 0.016) / 0.155, 0.0, 1.0)
    reflection_conf = torch.clamp((reflection_area - 0.24) / 0.46, 0.0, 1.0)
    final_conf = torch.clamp((final_mask - 0.22) / 0.48, 0.0, 1.0)
    glare_conf = torch.clamp((glare - 0.34) / 0.42, 0.0, 1.0)
    high_conf = torch.clamp(
        0.31 * reflection_conf
        + 0.19 * final_conf
        + 0.15 * prior_diff
        + 0.12 * direct_diff
        + 0.10 * glare_conf
        + 0.08 * line_score
        + 0.05 * veil_score,
        0.0,
        1.0,
    )
    high_conf = torch.maximum(
        high_conf,
        0.55 * torch.maximum(torch.maximum(glare_conf, line_score), torch.maximum(prior_diff, veil_score)),
    )
    high_conf = torch.maximum(high_conf, 0.72 * reflection_conf * torch.maximum(final_conf, prior_diff))
    high_conf = F.avg_pool2d(high_conf, kernel_size=9, stride=1, padding=4)
    boundary = torch.clamp((high_conf - 0.16) / 0.48, 0.0, 1.0)
    core = torch.clamp((high_conf - 0.32) / 0.34, 0.0, 1.0)
    core = torch.maximum(core, torch.clamp((torch.maximum(line_score, glare_conf) - 0.48) / 0.40, 0.0, 1.0) * final_conf)
    core = F.avg_pool2d(torch.clamp(core, 0.0, 1.0), kernel_size=5, stride=1, padding=2)
    boundary = torch.maximum(boundary, core)
    return torch.clamp(core, 0.0, 1.0), torch.clamp(boundary, 0.0, 1.0)


def decode_reflection_heads(
    model_output: torch.Tensor,
    reference_rgb: Optional[torch.Tensor],
    reflection_mask: torch.Tensor,
    glare_mask: Optional[torch.Tensor] = None,
    reflection_area_mask: Optional[torch.Tensor] = None,
    residual_strength: float = 1.0,
    visual_strength: float = 1.0,
    showcase_reflection_boost: float = 1.0,
    support_sharpen: float = 1.0,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    abs_clean = torch.sigmoid(model_output[:, 0:3])
    learned_gate = torch.sigmoid(model_output[:, 3:4]) if model_output.shape[1] >= 4 else torch.ones_like(reflection_mask)
    final_mask = torch.clamp(reflection_mask, 0.0, 1.0)
    reflection_area = torch.clamp(reflection_area_mask if reflection_area_mask is not None else final_mask, 0.0, 1.0)
    glare = torch.clamp(glare_mask if glare_mask is not None else torch.zeros_like(final_mask), 0.0, 1.0)
    zero_conf = torch.zeros_like(final_mask)
    if model_output.shape[1] >= 16:
        reflection_confidence = torch.sigmoid(model_output[:, 10:11])
        line_band_confidence = torch.sigmoid(model_output[:, 11:12])
        text_reflection_confidence = torch.sigmoid(model_output[:, 12:13])
        tint_confidence = torch.sigmoid(model_output[:, 13:14])
        lowfreq_reflection_confidence = torch.sigmoid(model_output[:, 14:15])
        dark_shadow_confidence = torch.sigmoid(model_output[:, 15:16])
    else:
        reflection_confidence = torch.clamp(torch.maximum(final_mask, reflection_area), 0.0, 1.0)
        line_band_confidence = zero_conf
        text_reflection_confidence = zero_conf
        tint_confidence = zero_conf
        lowfreq_reflection_confidence = zero_conf
        dark_shadow_confidence = zero_conf
    learned_support = torch.maximum(
        reflection_confidence,
        torch.maximum(
            line_band_confidence,
            torch.maximum(text_reflection_confidence, torch.maximum(lowfreq_reflection_confidence, dark_shadow_confidence)),
        ),
    )
    strong_glare_core = compute_strong_glare_core_tensor(
        reference_rgb,
        glare,
        final_mask,
        reflection_area_mask=reflection_area,
        line_confidence=line_band_confidence,
        lowfreq_confidence=lowfreq_reflection_confidence,
    )
    learned_support = torch.maximum(learned_support, strong_glare_core)
    support_soft = torch.clamp(torch.maximum(torch.maximum(final_mask, torch.maximum(reflection_area, glare)), learned_support), 0.0, 1.0)
    support_hard = torch.clamp((support_soft - 0.08) / 0.92, 0.0, 1.0)
    if float(support_sharpen) != 1.0:
        support_hard = torch.pow(support_hard, max(0.35, float(support_sharpen)))
    support_hard = F.max_pool2d(support_hard, kernel_size=5, stride=1, padding=2)
    positive_reflection = torch.zeros_like(abs_clean)
    signed_clean_delta = torch.zeros_like(abs_clean)
    if model_output.shape[1] >= 10:
        positive_reflection = torch.clamp(
            torch.sigmoid(model_output[:, 4:7]) * (1.35 + 0.20 * float(showcase_reflection_boost)),
            0.0,
            1.0,
        )
        signed_clean_delta = 0.70 * torch.tanh(model_output[:, 7:10])
        support3 = support_soft.expand_as(reference_rgb if reference_rgb is not None else abs_clean)
        hard3 = support_hard.expand_as(reference_rgb if reference_rgb is not None else abs_clean)
        lowfreq3 = lowfreq_reflection_confidence.expand_as(abs_clean)
        dark3 = dark_shadow_confidence.expand_as(abs_clean)
        line3 = line_band_confidence.expand_as(abs_clean)
        strong3 = strong_glare_core.expand_as(abs_clean)
        residual_strength_tensor = torch.clamp(
            1.15
            + 0.90 * hard3 * float(visual_strength) * float(showcase_reflection_boost)
            + 0.42 * strong3
            + 0.34 * line3
            + 0.30 * lowfreq3
            + 0.22 * dark3,
            1.20,
            2.65,
        )
        delta_strength_tensor = torch.clamp(
            0.80 + 0.45 * support3 + 0.42 * strong3 + 0.75 * dark3 + 0.52 * lowfreq3 + 0.24 * line3,
            0.80,
            2.35,
        )
        if reference_rgb is None:
            residual_clean = abs_clean
        else:
            residual_clean = torch.clamp(
                reference_rgb
                - positive_reflection * hard3 * residual_strength_tensor
                + signed_clean_delta * support3 * delta_strength_tensor,
                0.0,
                1.0,
            )
        residual_mix = torch.clamp(0.92 + 0.06 * strong_glare_core + 0.03 * support_hard, 0.92, 0.995)
        direct_clean = torch.clamp(residual_mix.expand_as(abs_clean) * residual_clean + (1.0 - residual_mix.expand_as(abs_clean)) * abs_clean, 0.0, 1.0)
        return (
            direct_clean,
            positive_reflection,
            signed_clean_delta,
            support_soft,
            support_hard,
            learned_gate,
            reflection_confidence,
            line_band_confidence,
            text_reflection_confidence,
            tint_confidence,
            lowfreq_reflection_confidence,
            dark_shadow_confidence,
        )

    if model_output.shape[1] >= 7:
        positive_reflection = torch.clamp(torch.sigmoid(model_output[:, 4:7]) * 1.25, 0.0, 1.0)
    support3 = support_soft.expand_as(positive_reflection)
    residual_strength_tensor = torch.clamp(learned_gate.new_tensor(float(residual_strength)) + 0.65 * support3, 1.0, 1.65)
    if reference_rgb is None:
        residual_clean = abs_clean
    else:
        residual_clean = torch.clamp(
            reference_rgb - positive_reflection * support3 * residual_strength_tensor,
            0.0,
            1.0,
        )
    direct_clean = torch.clamp(0.82 * residual_clean + 0.18 * abs_clean, 0.0, 1.0)
    return (
        direct_clean,
        positive_reflection,
        signed_clean_delta,
        support_soft,
        support_hard,
        learned_gate,
        reflection_confidence,
        line_band_confidence,
        text_reflection_confidence,
        tint_confidence,
        lowfreq_reflection_confidence,
        dark_shadow_confidence,
    )


def compute_blend_gate(
    model_output: torch.Tensor,
    reflection_mask: torch.Tensor,
    glare_mask: Optional[torch.Tensor] = None,
    reflection_area_mask: Optional[torch.Tensor] = None,
    mask_profile: str = "balanced",
    loss_profile: str = "balanced",
    reference_rgb: Optional[torch.Tensor] = None,
    base_prior: Optional[torch.Tensor] = None,
    showcase_mode: bool = False,
    visual_strength: float = 1.0,
    showcase_reflection_boost: float = 1.0,
    support_sharpen: float = 1.0,
) -> torch.Tensor:
    learned_gate = torch.sigmoid(model_output[:, 3:4]) if model_output.shape[1] >= 4 else torch.ones_like(reflection_mask)
    final_mask = torch.clamp(reflection_mask, 0.0, 1.0)
    glare = torch.clamp(glare_mask if glare_mask is not None else torch.zeros_like(final_mask), 0.0, 1.0)
    strong_glare_core = compute_strong_glare_core_tensor(
        reference_rgb,
        glare,
        final_mask,
        reflection_area_mask=reflection_area,
        line_confidence=line_band_confidence,
        lowfreq_confidence=lowfreq_reflection_confidence,
    )
    reflection_area = torch.clamp(reflection_area_mask if reflection_area_mask is not None else final_mask, 0.0, 1.0)
    profile_gate_max = {"stable": 0.82, "balanced": 0.86, "hard": 0.92, "visual": 0.95 if showcase_mode else 0.92}.get(loss_profile, 0.86)
    blend_gate = torch.clamp(0.10 + 0.56 * final_mask + 0.28 * learned_gate, 0.10, profile_gate_max)
    reflection_signal = torch.clamp(reflection_area - 0.25 * glare, 0.0, 1.0)
    reflection_max = min({"stable": 0.80, "balanced": 0.86, "aggressive": 0.90}.get(mask_profile, 0.86), profile_gate_max)
    reflection_gate = torch.clamp(0.14 + 0.60 * reflection_signal + 0.24 * learned_gate, 0.10, reflection_max)
    blend_gate = torch.where(reflection_signal > 0.22, torch.maximum(blend_gate, reflection_gate), blend_gate)
    reflection_confidence = torch.clamp((reflection_area - 0.36) / 0.42, 0.0, 1.0)
    final_confidence = torch.clamp((final_mask - 0.32) / 0.42, 0.0, 1.0)
    high_confidence_reflection = reflection_confidence * final_confidence
    gate_floor = {"stable": 0.45, "balanced": 0.55, "hard": 0.65, "visual": 0.54}.get(loss_profile, 0.55)
    floor_gate = blend_gate.new_tensor(gate_floor)
    blend_gate = torch.where(high_confidence_reflection > 0.42, torch.maximum(blend_gate, floor_gate), blend_gate)
    glare_confidence = torch.clamp((glare - 0.45) / 0.40, 0.0, 1.0) * torch.clamp((final_mask - 0.25) / 0.45, 0.0, 1.0)
    glare_boost = torch.clamp(blend_gate + 0.08 * glare_confidence, 0.10, profile_gate_max)
    blend_gate = torch.where(glare_confidence > 0.25, torch.maximum(blend_gate, glare_boost), blend_gate)
    if model_output.shape[1] >= 16:
        strong_glare_core = compute_strong_glare_core_tensor(
            reference_rgb,
            glare,
            final_mask,
            reflection_area_mask=reflection_area,
            line_confidence=torch.sigmoid(model_output[:, 11:12]),
            lowfreq_confidence=torch.sigmoid(model_output[:, 14:15]),
        )
        strong_floor = torch.clamp(0.58 + 0.30 * strong_glare_core, 0.58, profile_gate_max)
        blend_gate = torch.where(strong_glare_core > 0.16, torch.maximum(blend_gate, strong_floor), blend_gate)
    if loss_profile == "visual":
        direct_clean, _predicted_reflection, _signed_clean_delta, _support_soft, support_hard, _learned_gate, *_ = decode_reflection_heads(
            model_output,
            reference_rgb,
            final_mask,
            glare_mask=glare,
            reflection_area_mask=reflection_area,
            visual_strength=visual_strength,
            showcase_reflection_boost=showcase_reflection_boost,
            support_sharpen=support_sharpen,
        )
        core, boundary = compute_high_conf_reflection_mask(
            reference_rgb,
            base_prior,
            direct_clean,
            final_mask,
            glare_mask=glare,
            reflection_area_mask=reflection_area,
        )
        boundary_floor = torch.clamp(0.36 + 0.20 * boundary, 0.36, 0.56)
        core_floor = torch.clamp(0.54 + 0.20 * core, 0.54, 0.74)
        visual_gate = torch.clamp(0.30 + 0.22 * boundary + 0.20 * core + 0.08 * learned_gate, 0.08, profile_gate_max)
        blend_gate = torch.where(boundary > 0.14, torch.maximum(blend_gate, boundary_floor), blend_gate)
        blend_gate = torch.where(core > 0.08, torch.maximum(blend_gate, core_floor), blend_gate)
        blend_gate = torch.maximum(blend_gate, visual_gate * torch.clamp(0.55 * boundary + 0.65 * core, 0.0, 1.0))
        outside_guard = torch.clamp(1.0 - torch.maximum(boundary, reflection_area), 0.0, 1.0)
        if showcase_mode:
            outside_guard = outside_guard * torch.clamp(1.0 - support_hard, 0.0, 1.0)
        blend_gate = blend_gate * (1.0 - 0.18 * outside_guard)
        blend_gate = torch.clamp(blend_gate + 0.018 * boundary + 0.022 * core, 0.06, profile_gate_max)
    if showcase_mode:
        _direct_clean, _predicted_reflection, _signed_clean_delta, support_soft, support_hard, _learned_gate, *_ = decode_reflection_heads(
            model_output,
            reference_rgb,
            final_mask,
            glare_mask=glare,
            reflection_area_mask=reflection_area,
            visual_strength=visual_strength,
            showcase_reflection_boost=showcase_reflection_boost,
            support_sharpen=support_sharpen,
        )
        hard_floor = torch.clamp(0.62 + 0.28 * support_hard + 0.08 * support_soft, 0.62, profile_gate_max)
        blend_gate = torch.where(support_hard > 0.18, torch.maximum(blend_gate, hard_floor), blend_gate)
    return torch.clamp(blend_gate, 0.06, profile_gate_max)


def compose_prediction(
    base_prior: torch.Tensor,
    model_output: torch.Tensor,
    reflection_mask: torch.Tensor,
    reference_rgb: Optional[torch.Tensor] = None,
    glare_mask: Optional[torch.Tensor] = None,
    reflection_area_mask: Optional[torch.Tensor] = None,
    scene_reflection_mask: Optional[torch.Tensor] = None,
    dark_reflection_mask: Optional[torch.Tensor] = None,
    shadow_veil_mask: Optional[torch.Tensor] = None,
    lowfreq_reflection_mask: Optional[torch.Tensor] = None,
    foreground_structure_guard: Optional[torch.Tensor] = None,
    mask_profile: str = "balanced",
    loss_profile: str = "balanced",
    visual_strength: float = 1.0,
    showcase_mode: bool = False,
    showcase_hard_mode: bool = False,
    showcase_reflection_boost: float = 1.0,
    support_sharpen: float = 1.0,
    showcase_detail_strength: float = 0.10,
    showcase_smooth_strength: float = 0.14,
    showcase_tint_suppress: float = 0.12,
) -> torch.Tensor:
    (
        direct_clean,
        predicted_reflection,
        signed_clean_delta,
        support_soft,
        support_hard,
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
        reflection_mask,
        glare_mask=glare_mask,
        reflection_area_mask=reflection_area_mask,
        visual_strength=visual_strength,
        showcase_reflection_boost=showcase_reflection_boost,
        support_sharpen=support_sharpen,
    )
    blend_gate = compute_blend_gate(
        model_output,
        reflection_mask,
        glare_mask=glare_mask,
        reflection_area_mask=reflection_area_mask,
        mask_profile=mask_profile,
        loss_profile=loss_profile,
        reference_rgb=reference_rgb,
        base_prior=base_prior,
        showcase_mode=showcase_mode,
        visual_strength=visual_strength,
        showcase_reflection_boost=showcase_reflection_boost,
        support_sharpen=support_sharpen,
    )
    final_mask = torch.clamp(reflection_mask, 0.0, 1.0)
    reflection_area = torch.clamp(reflection_area_mask if reflection_area_mask is not None else final_mask, 0.0, 1.0)
    scene_reflection = torch.clamp(scene_reflection_mask if scene_reflection_mask is not None else torch.zeros_like(final_mask), 0.0, 1.0)
    dark_reflection = torch.clamp(dark_reflection_mask if dark_reflection_mask is not None else torch.zeros_like(final_mask), 0.0, 1.0)
    shadow_veil = torch.clamp(shadow_veil_mask if shadow_veil_mask is not None else torch.zeros_like(final_mask), 0.0, 1.0)
    lowfreq_reflection = torch.clamp(lowfreq_reflection_mask if lowfreq_reflection_mask is not None else torch.zeros_like(final_mask), 0.0, 1.0)
    foreground_guard = torch.clamp(foreground_structure_guard if foreground_structure_guard is not None else torch.zeros_like(final_mask), 0.0, 1.0)
    glare = torch.clamp(glare_mask if glare_mask is not None else torch.zeros_like(final_mask), 0.0, 1.0)

    shadow_support = torch.clamp(
        torch.maximum(scene_reflection, torch.maximum(dark_reflection, torch.maximum(shadow_veil, lowfreq_reflection))),
        0.0,
        1.0,
    )
    confidence_support = torch.clamp(
        torch.maximum(
            reflection_confidence,
            torch.maximum(
                line_band_confidence,
                torch.maximum(
                    text_reflection_confidence,
                    torch.maximum(tint_confidence, torch.maximum(lowfreq_reflection_confidence, dark_shadow_confidence)),
                ),
            ),
        ),
        0.0,
        1.0,
    )
    support_soft = torch.clamp(torch.maximum(support_soft, torch.maximum(strong_glare_core, torch.maximum(shadow_support, confidence_support))), 0.0, 1.0)
    support_hard = torch.clamp(torch.maximum(support_hard, torch.clamp((support_soft - 0.12) / 0.70, 0.0, 1.0)), 0.0, 1.0)
    residual_confidence = torch.clamp((torch.mean(predicted_reflection, dim=1, keepdim=True) - 0.08) / 0.34, 0.0, 1.0)
    high_confidence_reflection = torch.clamp((support_soft - 0.30) / 0.50, 0.0, 1.0) * torch.clamp(
        (torch.maximum(final_mask, reflection_area) - 0.24) / 0.52,
        0.0,
        1.0,
    )
    high_confidence_reflection = torch.clamp(
        torch.maximum(high_confidence_reflection, residual_confidence * support_soft),
        0.0,
        1.0,
    )
    core = torch.zeros_like(final_mask)
    boundary = torch.clamp((support_soft - 0.16) / 0.54, 0.0, 1.0)
    if loss_profile == "visual":
        core, boundary = compute_high_conf_reflection_mask(
            reference_rgb,
            base_prior,
            direct_clean,
            reflection_mask,
            glare_mask=glare_mask,
            reflection_area_mask=reflection_area_mask,
        )
        high_confidence_reflection = torch.clamp(torch.maximum(high_confidence_reflection, core), 0.0, 1.0)
        boundary = torch.clamp(torch.maximum(boundary, torch.clamp(0.55 * boundary + 0.45 * core, 0.0, 1.0)), 0.0, 1.0)

    v2_reflection_takeover = torch.clamp(
        0.35 * support_soft
        + 0.45 * support_hard
        + 0.35 * reflection_confidence
        + 0.32 * line_band_confidence
        + 0.30 * text_reflection_confidence
        + 0.32 * lowfreq_reflection_confidence
        + 0.34 * dark_shadow_confidence
        + 0.36 * strong_glare_core
        + 0.24 * scene_reflection
        + 0.22 * lowfreq_reflection
        + 0.22 * dark_reflection
        + 0.18 * shadow_veil
        + 0.18 * high_confidence_reflection
        + 0.12 * residual_confidence,
        0.0,
        1.0,
    )
    effective_guard = torch.clamp(foreground_guard * torch.clamp(1.0 - v2_reflection_takeover, 0.0, 1.0), 0.0, 1.0)
    safe_release = torch.clamp(
        0.55 * v2_reflection_takeover
        + 0.25 * lowfreq_reflection_confidence
        + 0.25 * dark_shadow_confidence
        + 0.20 * line_band_confidence,
        0.0,
        0.85,
    )
    blend_gate = torch.clamp(
        blend_gate * (1.0 - 0.08 * effective_guard)
        + 0.52 * safe_release
        + 0.30 * v2_reflection_takeover,
        0.08,
        0.995 if showcase_mode else 0.96,
    )

    if reference_rgb is not None:
        protect_mask = reflection_area_mask if reflection_area_mask is not None else reflection_mask
        protect_mask = torch.clamp(protect_mask, 0.0, 1.0)
        protect_source = reference_rgb * (1.0 - protect_mask) + base_prior * protect_mask
    else:
        protect_source = base_prior
    if showcase_mode:
        no_prior_zone = torch.clamp(torch.maximum(torch.maximum(support_hard, high_confidence_reflection), v2_reflection_takeover), 0.0, 1.0)
        protect_source = torch.clamp(protect_source * (1.0 - no_prior_zone) + direct_clean * no_prior_zone, 0.0, 1.0)
    prior_release = torch.clamp((torch.maximum(high_confidence_reflection, v2_reflection_takeover) - 0.10) / 0.48, 0.0, 1.0)
    protect_source = torch.clamp(protect_source * (1.0 - prior_release) + direct_clean * prior_release, 0.0, 1.0)
    prediction = direct_clean * blend_gate + protect_source * (1.0 - blend_gate)

    structure_safe = torch.clamp(1.0 - 0.16 * effective_guard, 0.72, 1.0)
    lowfreq_force = torch.clamp(torch.maximum(lowfreq_reflection, lowfreq_reflection_confidence) * structure_safe, 0.0, 1.0)
    dark_force = torch.clamp(torch.maximum(dark_reflection, dark_shadow_confidence) * structure_safe, 0.0, 1.0)
    text_force = torch.clamp(torch.maximum(text_reflection_confidence, 0.55 * shadow_veil) * torch.clamp(1.0 - 0.18 * effective_guard, 0.55, 1.0), 0.0, 1.0)
    line_force = torch.clamp(line_band_confidence * torch.clamp(1.0 - 0.12 * effective_guard, 0.60, 1.0), 0.0, 1.0)
    tint_force = torch.clamp(tint_confidence, 0.0, 1.0)
    direct_pull = torch.clamp(
        0.26 * support_soft
        + 0.44 * high_confidence_reflection
        + 0.18 * residual_confidence
        + 0.34 * lowfreq_force
        + 0.38 * dark_force
        + 0.28 * text_force
        + 0.30 * line_force
        + 0.36 * strong_glare_core
        + 0.58 * v2_reflection_takeover
        - 0.04 * effective_guard,
        0.0,
        0.985,
    )
    if loss_profile == "visual":
        visual_pull = torch.clamp(0.46 * core + 0.24 * boundary + 0.50 * v2_reflection_takeover + 0.26 * strong_glare_core, 0.0, 0.985)
        direct_pull = torch.maximum(direct_pull, visual_pull)
    prediction = torch.clamp(prediction * (1.0 - direct_pull) + direct_clean * direct_pull, 0.0, 1.0)

    if reference_rgb is not None:
        lowfreq_alpha = torch.clamp(
            0.34 * lowfreq_reflection + 0.42 * lowfreq_reflection_confidence + 0.32 * v2_reflection_takeover,
            0.0,
            0.78,
        )
        lowfreq_alpha = torch.clamp(lowfreq_alpha * torch.clamp(1.0 - 0.08 * effective_guard, 0.78, 1.0), 0.0, 0.78)
        lowfreq_direct = gaussian_blur_tensor(direct_clean, sigma=4.5)
        lowfreq_prediction = gaussian_blur_tensor(prediction, sigma=4.5)
        prediction = torch.clamp(prediction + lowfreq_alpha.expand_as(prediction) * (lowfreq_direct - lowfreq_prediction), 0.0, 1.0)
        dark_alpha = torch.clamp((0.30 * dark_reflection + 0.50 * dark_shadow_confidence + 0.30 * v2_reflection_takeover) * torch.clamp(1.0 - 0.08 * effective_guard, 0.78, 1.0), 0.0, 0.82)
        dark_delta = torch.clamp(signed_clean_delta, -0.42, 0.42)
        prediction = torch.clamp(
            prediction * (1.0 - dark_alpha.expand_as(prediction))
            + torch.clamp(prediction + 0.92 * dark_delta, 0.0, 1.0) * dark_alpha.expand_as(prediction),
            0.0,
            1.0,
        )

    if showcase_mode:
        showcase_core = torch.clamp(torch.maximum(torch.maximum(core, support_hard), torch.maximum(strong_glare_core, torch.maximum(glare, reflection_area))), 0.0, 1.0)
        showcase_core = torch.clamp(torch.maximum(showcase_core, torch.maximum(shadow_support, confidence_support)), 0.0, 1.0)
        showcase_core = torch.clamp(torch.maximum(showcase_core, v2_reflection_takeover), 0.0, 1.0)
        if showcase_hard_mode:
            showcase_core = torch.clamp(torch.maximum(showcase_core, F.max_pool2d(support_hard, kernel_size=7, stride=1, padding=3)), 0.0, 1.0)
        force_direct = torch.clamp(
            0.55 * support_soft
            + 0.80 * showcase_core
            + 0.28 * boundary
            + 0.30 * strong_glare_core
            + 0.20 * glare
            + 0.18 * reflection_area
            + 0.28 * lowfreq_force
            + 0.34 * dark_force
            + 0.22 * line_force
            + 0.22 * text_force
            + 0.76 * v2_reflection_takeover
            + 0.32 * line_band_confidence
            + 0.32 * text_reflection_confidence
            + 0.40 * lowfreq_reflection_confidence
            + 0.42 * dark_shadow_confidence
            - 0.06 * effective_guard,
            0.0,
            0.995,
        )
    else:
        force_direct = torch.clamp(
            0.34 * support_soft + 0.50 * core + 0.18 * boundary + 0.14 * lowfreq_force + 0.18 * dark_force + 0.20 * v2_reflection_takeover,
            0.0,
            0.94,
        )
        force_direct = torch.maximum(force_direct, torch.clamp(0.78 * strong_glare_core + 0.24 * line_force, 0.0, 0.98))
    prediction = torch.clamp(prediction * (1.0 - force_direct) + direct_clean * force_direct, 0.0, 1.0)

    if showcase_mode:
        # VisualPlus is intentionally more aggressive than Raw: in detected reflection regions,
        # preserve DirectClean's advantage instead of averaging it away.
        visual_extra = torch.clamp(
            0.58 * v2_reflection_takeover
            + 0.36 * strong_glare_core
            + 0.40 * lowfreq_reflection_confidence
            + 0.42 * dark_shadow_confidence
            + 0.30 * text_reflection_confidence
            + 0.30 * line_band_confidence
            + 0.18 * tint_confidence,
            0.0,
            0.95,
        )
        visual_extra = torch.clamp(visual_extra * torch.clamp(1.0 - 0.04 * effective_guard, 0.88, 1.0), 0.0, 0.95)
        prediction = torch.clamp(prediction * (1.0 - visual_extra) + direct_clean * visual_extra, 0.0, 1.0)
        direct_low = gaussian_blur_tensor(direct_clean, sigma=1.2)
        lowfreq_mix = torch.clamp(
            0.34 * lowfreq_reflection_confidence
            + 0.34 * dark_shadow_confidence
            + 0.22 * scene_reflection
            + 0.22 * shadow_veil
            + 0.18 * v2_reflection_takeover,
            0.0,
            0.72,
        )
        lowfreq_mix = torch.clamp(lowfreq_mix * torch.clamp(1.0 - 0.06 * effective_guard, 0.82, 1.0), 0.0, 0.72)
        prediction = torch.clamp(prediction * (1.0 - lowfreq_mix.expand_as(prediction)) + direct_low * lowfreq_mix.expand_as(prediction), 0.0, 1.0)
        prediction = showcase_visual_polish(
            prediction,
            reference_rgb,
            direct_clean,
            support_soft,
            support_hard,
            tint_confidence=tint_force,
            lowfreq_confidence=lowfreq_force,
            dark_shadow_confidence=dark_force,
            foreground_structure_guard=foreground_guard,
            v2_reflection_takeover=v2_reflection_takeover,
            effective_guard=effective_guard,
            line_band_confidence=line_force,
            text_reflection_confidence=text_force,
            shadow_veil_mask=shadow_veil,
            detail_strength=showcase_detail_strength,
            smooth_strength=showcase_smooth_strength,
            tint_suppress=showcase_tint_suppress,
        )
    return torch.clamp(prediction, 0.0, 1.0)


def build_model(args: argparse.Namespace) -> nn.Module:
    base_channels = 24 if args.model == "lite" else 32
    return PolarFreeLiteUNet(in_channels=MODEL_INPUT_CHANNELS, base_channels=base_channels)


