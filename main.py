from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

REFERENCE_VIEW = "0deg"
POLARIZATION_ORDER = ("0deg", "45deg", "90deg", "135deg")
EPS = 1e-6

AGGRESSIVE_PERCENTILE = 10.0
VEIL_SIGMA = 13.0
DETAIL_SIGMA = 1.3
MASK_SIGMA = 9.0
GLASS_GHOST_SIGMA = 19.0


def resize_with_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / float(max(height, width)))
    if scale >= 0.999:
        return image

    new_width = max(32, int(round(width * scale)))
    new_height = max(32, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def read_rgb_image(
    path: Path,
    max_side: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if max_side is not None:
        image = resize_with_max_side(image, max_side=max_side)
    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    return image.astype(np.float32) / 255.0


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR))


def ensure_single_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image[..., np.newaxis]
    if image.ndim == 3 and image.shape[2] == 1:
        return image
    if image.ndim == 3:
        return np.mean(image, axis=2, keepdims=True)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def normalize_unit_interval(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image_min = float(np.min(image))
    image_max = float(np.max(image))
    if image_max - image_min < EPS:
        return np.zeros_like(image, dtype=np.float32)
    return (image - image_min) / (image_max - image_min + EPS)


def robust_percentile_normalize(image: np.ndarray, low: float = 5.0, high: float = 95.0) -> np.ndarray:
    image = image.astype(np.float32)
    low_value = float(np.percentile(image, low))
    high_value = float(np.percentile(image, high))
    if high_value - low_value < EPS:
        return normalize_unit_interval(image)
    return np.clip((image - low_value) / (high_value - low_value + EPS), 0.0, 1.0)


def load_polarized_images(
    image_paths: Mapping[str, Path],
    max_side: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, np.ndarray]:
    images: Dict[str, np.ndarray] = {}
    reference_size: Optional[Tuple[int, int]] = None

    for angle in POLARIZATION_ORDER:
        if angle not in image_paths:
            raise KeyError(f"Missing polarized image for angle: {angle}")
        image = read_rgb_image(Path(image_paths[angle]), max_side=max_side, target_size=target_size)
        if reference_size is None:
            reference_size = (image.shape[1], image.shape[0])
        elif (image.shape[1], image.shape[0]) != reference_size:
            image = cv2.resize(image, reference_size, interpolation=cv2.INTER_AREA)
        images[angle] = image
    return images


def compute_polarization_features(images: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    stack = np.stack([images[angle] for angle in POLARIZATION_ORDER], axis=0)
    gray_stack = np.mean(stack, axis=3).astype(np.float32)
    i0_gray, i45_gray, i90_gray, i135_gray = gray_stack
    mean_rgb = np.mean(stack, axis=0)
    mean_gray = np.mean(gray_stack, axis=0)

    s1 = i0_gray - i90_gray
    s2 = i45_gray - i135_gray
    amplitude = np.sqrt(s1 * s1 + s2 * s2)
    dolp = np.clip(amplitude / (mean_gray + EPS), 0.0, 1.0)

    channel_cos = 0.5 * (stack[0] - stack[2])
    channel_sin = 0.5 * (stack[1] - stack[3])
    channel_amplitude = np.sqrt(np.maximum(channel_cos * channel_cos + channel_sin * channel_sin, 0.0))
    continuous_min_rgb = np.clip(mean_rgb - channel_amplitude, 0.0, 1.0)

    return {
        "stack": stack,
        "gray_stack": gray_stack,
        "mean_rgb": mean_rgb,
        "mean_gray": mean_gray,
        "polar_amplitude": amplitude,
        "polar_diff": np.abs(i0_gray - i90_gray),
        "swing": np.max(gray_stack, axis=0) - np.min(gray_stack, axis=0),
        "std": np.std(gray_stack, axis=0),
        "dolp": dolp,
        "aggressive_percentile_rgb": np.percentile(stack, AGGRESSIVE_PERCENTILE, axis=0).astype(np.float32),
        "continuous_min_rgb": continuous_min_rgb.astype(np.float32),
    }


def pick_median_consistent_view(stack: np.ndarray) -> np.ndarray:
    gray_stack = np.mean(stack, axis=3)
    median_luma = np.median(gray_stack, axis=0, keepdims=True)
    deviation = np.abs(gray_stack - median_luma)
    weights = 1.0 / (deviation + 0.02)
    weights = weights / (np.sum(weights, axis=0, keepdims=True) + EPS)
    return np.sum(stack * weights[..., np.newaxis], axis=0)


def average_darkest_views(stack: np.ndarray, num_views: int = 2) -> np.ndarray:
    gray_stack = np.mean(stack, axis=3)
    order = np.argsort(gray_stack, axis=0)
    view_indices = np.repeat(order[:num_views][..., np.newaxis], stack.shape[3], axis=3)
    selected = np.take_along_axis(stack, view_indices, axis=0)
    return np.mean(selected, axis=0)


def build_background_candidate(
    original_rgb: np.ndarray,
    features: Mapping[str, np.ndarray],
    prior_profile: str = "stable",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    low_candidate = features["continuous_min_rgb"]
    aggressive_candidate = features["aggressive_percentile_rgb"]
    darkest_candidate = average_darkest_views(features["stack"], num_views=2)
    median_candidate = pick_median_consistent_view(features["stack"])
    structure_candidate = np.clip(0.58 * darkest_candidate + 0.42 * median_candidate, 0.0, 1.0)

    residual = np.clip(original_rgb - low_candidate, 0.0, 1.0)
    residual_luma = np.mean(residual, axis=2, keepdims=True)
    mix_mask = robust_percentile_normalize(residual_luma, 30, 96)
    structure_residual = np.mean(np.clip(original_rgb - structure_candidate, 0.0, 1.0), axis=2, keepdims=True)
    structure_mask = robust_percentile_normalize(structure_residual, 35, 97)
    mix_mask = np.clip(0.65 * mix_mask + 0.35 * structure_mask, 0.0, 1.0)
    mix_mask = ensure_single_channel(cv2.GaussianBlur(mix_mask.astype(np.float32), (0, 0), MASK_SIGMA))

    detail = original_rgb - cv2.GaussianBlur(original_rgb, (0, 0), DETAIL_SIGMA)
    wide_veil = cv2.GaussianBlur(residual, (0, 0), VEIL_SIGMA)
    corrected = np.clip(original_rgb - 1.06 * wide_veil - 0.30 * residual - 0.12 * structure_residual, 0.0, 1.0)
    base_candidate = np.clip(0.42 * corrected + 0.36 * low_candidate + 0.22 * structure_candidate, 0.0, 1.0)
    aggressive_detail = np.clip(0.62 * aggressive_candidate + 0.38 * low_candidate, 0.0, 1.0)
    masked_candidate = np.clip(0.66 * low_candidate + 0.20 * aggressive_detail + 0.14 * structure_candidate, 0.0, 1.0)
    candidate = np.clip(
        (1.0 - 0.70 * mix_mask) * base_candidate + (0.70 * mix_mask) * masked_candidate,
        0.0,
        1.0,
    )
    if prior_profile == "visual":
        visual_seed = np.maximum(mix_mask, structure_mask).astype(np.float32)
        visual_mask = ensure_single_channel(cv2.GaussianBlur(visual_seed, (0, 0), 5.5))
        visual_core = ensure_single_channel(cv2.GaussianBlur((visual_seed > 0.42).astype(np.float32), (0, 0), 4.0))
        visual_dark_floor = np.clip(
            0.50 * low_candidate
            + 0.34 * aggressive_candidate
            + 0.16 * darkest_candidate,
            0.0,
            1.0,
        )
        visual_core_candidate = np.clip(
            0.58 * low_candidate
            + 0.30 * aggressive_candidate
            + 0.12 * darkest_candidate,
            0.0,
            1.0,
        )
        visual_masked_candidate = np.clip(
            0.64 * visual_dark_floor
            + 0.24 * masked_candidate
            + 0.12 * structure_candidate,
            0.0,
            1.0,
        )
        candidate = np.clip(
            candidate * (1.0 - 0.66 * visual_mask) + visual_masked_candidate * (0.66 * visual_mask),
            0.0,
            1.0,
        )
        candidate = np.clip(
            candidate * (1.0 - 0.34 * visual_core) + visual_core_candidate * (0.34 * visual_core),
            0.0,
            1.0,
        )
    return candidate, low_candidate, mix_mask, detail


def refine_reflection_mask(reflection_score: np.ndarray, prior_profile: str = "stable") -> Tuple[np.ndarray, np.ndarray]:
    score_2d = ensure_single_channel(reflection_score).squeeze(-1).astype(np.float32)
    if prior_profile == "visual":
        low = float(np.percentile(score_2d, 30))
        high = float(np.percentile(score_2d, 63))
        binary_threshold = 0.045
        close_kernel = (27, 19)
        blur_kernel = (31, 31)
        core_threshold = 0.30
    else:
        low = float(np.percentile(score_2d, 38))
        high = float(np.percentile(score_2d, 68))
        binary_threshold = 0.04
        close_kernel = (19, 19)
        blur_kernel = (27, 27)
        core_threshold = 0.28
    soft_mask = np.clip((score_2d - low) / (high - low + EPS), 0.0, 1.0)

    binary_mask = (soft_mask > binary_threshold).astype(np.uint8)
    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel),
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    filtered_mask = np.zeros_like(binary_mask)
    min_area = max(18, int(score_2d.size * 0.00022))

    for label_idx in range(1, num_labels):
        x, y, width, height, area = stats[label_idx]
        component_mask = labels == label_idx
        mean_score = float(np.mean(score_2d[component_mask]))
        touches_border = x == 0 or y == 0 or (x + width) >= score_2d.shape[1] or (y + height) >= score_2d.shape[0]
        density = float(area) / float(max(width * height, 1))
        slender = float(min(width, height)) / float(max(width, height, 1)) < 0.10

        if area < min_area and mean_score < 0.45:
            continue
        if touches_border and density < 0.05 and mean_score < 0.42:
            continue
        if touches_border and slender and area < int(score_2d.size * 0.015):
            continue
        filtered_mask[component_mask] = 1

    if not np.any(filtered_mask):
        filtered_mask = (soft_mask > 0.03).astype(np.uint8)

    soft_mask = cv2.GaussianBlur((soft_mask * filtered_mask).astype(np.float32), blur_kernel, 0)
    soft_mask = np.clip(soft_mask, 0.0, 1.0)
    core_mask = cv2.GaussianBlur(
        (((soft_mask > core_threshold).astype(np.float32)) * filtered_mask).astype(np.float32),
        (17, 17),
        0,
    )
    core_mask = np.clip(core_mask, 0.0, 1.0)
    return soft_mask[..., np.newaxis], core_mask[..., np.newaxis]


def build_reflection_mask(
    original_rgb: np.ndarray,
    low_candidate_rgb: np.ndarray,
    prior_mask: np.ndarray,
    features: Mapping[str, np.ndarray],
    prior_profile: str = "stable",
) -> Tuple[np.ndarray, np.ndarray]:
    original_u8 = (np.clip(original_rgb, 0.0, 1.0) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(original_u8, cv2.COLOR_RGB2HSV)
    value = ensure_single_channel(hsv[..., 2].astype(np.float32) / 255.0)
    saturation = ensure_single_channel(hsv[..., 1].astype(np.float32) / 255.0)

    residual_rgb = np.clip(original_rgb - low_candidate_rgb, 0.0, 1.0)
    residual_luma = np.mean(residual_rgb, axis=2, keepdims=True)
    wide_residual = ensure_single_channel(cv2.GaussianBlur(residual_luma.astype(np.float32), (0, 0), 9.0))
    amplitude_score = robust_percentile_normalize(features["polar_amplitude"], 8, 98)[..., np.newaxis]
    swing_score = robust_percentile_normalize(features["swing"], 12, 98)[..., np.newaxis]
    residual_score = robust_percentile_normalize(residual_luma, 35, 97)
    wide_score = robust_percentile_normalize(wide_residual, 35, 97)
    rgb_max = ensure_single_channel(np.max(original_rgb, axis=2, keepdims=True))
    value_score = robust_percentile_normalize(value, 55, 99)
    rgb_max_score = robust_percentile_normalize(rgb_max, 58, 99)
    relaxed_saturation = np.clip((0.82 - saturation) / 0.82, 0.0, 1.0)
    low_saturation = np.clip((0.62 - saturation) / 0.62, 0.0, 1.0)
    local_over = normalize_unit_interval(np.maximum(value.squeeze(-1) - cv2.GaussianBlur(value.squeeze(-1), (0, 0), 7.0), 0.0))[..., np.newaxis]
    colored_glare = value_score * np.clip((saturation - 0.08) / 0.64, 0.0, 1.0)
    glare_score = np.clip(
        0.42 * value_score * (0.28 + 0.72 * relaxed_saturation)
        + 0.22 * rgb_max_score
        + 0.20 * local_over
        + 0.16 * colored_glare,
        0.0,
        1.0,
    )
    ghost_source = np.mean(
        np.clip(original_rgb - 0.65 * low_candidate_rgb - 0.35 * features["continuous_min_rgb"], 0.0, 1.0),
        axis=2,
        keepdims=True,
    )
    ghost_source = ensure_single_channel(cv2.GaussianBlur(ghost_source.astype(np.float32), (0, 0), GLASS_GHOST_SIGMA))
    ghost_score = robust_percentile_normalize(ghost_source, 30, 96) * (0.30 + 0.70 * relaxed_saturation)

    if prior_profile == "visual":
        line_source = ensure_single_channel(np.maximum(glare_score, 0.55 * residual_score + 0.45 * wide_score)).squeeze(-1)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (37, 3))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 37))
        horizontal_response = cv2.morphologyEx(line_source.astype(np.float32), cv2.MORPH_CLOSE, horizontal_kernel)
        vertical_response = cv2.morphologyEx(line_source.astype(np.float32), cv2.MORPH_CLOSE, vertical_kernel)
        line_response = np.maximum(horizontal_response, vertical_response)
        line_score = robust_percentile_normalize(line_response, 52, 99)[..., np.newaxis]
        veil_source = ensure_single_channel(cv2.GaussianBlur(residual_luma.astype(np.float32), (0, 0), 25.0))
        veil_score = robust_percentile_normalize(veil_source, 28, 96) * (0.35 + 0.65 * relaxed_saturation)
        reflection_score = (
            0.14 * np.clip(prior_mask, 0.0, 1.0)
            + 0.25 * residual_score
            + 0.19 * wide_score
            + 0.09 * amplitude_score
            + 0.06 * swing_score
            + 0.14 * glare_score
            + 0.11 * ghost_score
            + 0.08 * line_score
            + 0.08 * veil_score
        )
        reflection_score = ensure_single_channel(cv2.GaussianBlur(reflection_score.astype(np.float32), (0, 0), 4.2))
    else:
        reflection_score = (
            0.28 * np.clip(prior_mask, 0.0, 1.0)
            + 0.26 * residual_score
            + 0.18 * wide_score
            + 0.07 * amplitude_score
            + 0.05 * swing_score
            + 0.04 * glare_score
            + 0.12 * ghost_score
        )
        reflection_score = ensure_single_channel(cv2.GaussianBlur(reflection_score.astype(np.float32), (0, 0), 5.0))
    return refine_reflection_mask(reflection_score, prior_profile=prior_profile)


def restore_background(
    original_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    low_candidate_rgb: np.ndarray,
    prior_mask: np.ndarray,
    reflection_mask: np.ndarray,
    core_mask: np.ndarray,
    prior_profile: str = "stable",
) -> np.ndarray:
    mask_power = np.clip(np.maximum(reflection_mask, prior_mask), 0.0, 1.0)
    core_power = np.clip(core_mask, 0.0, 1.0)
    if prior_profile == "visual":
        mask_mix = 0.70
        core_mix = 0.58
        glass_base = 0.11
        glass_scale = 0.42
        wide_base = 0.06
        wide_scale = 0.28
    else:
        mask_mix = 0.42
        core_mix = 0.28
        glass_base = 0.08
        glass_scale = 0.24
        wide_base = 0.04
        wide_scale = 0.12
    result = np.clip(candidate_rgb * (1.0 - mask_mix * mask_power) + low_candidate_rgb * (mask_mix * mask_power), 0.0, 1.0)
    result = np.clip(result * (1.0 - core_mix * core_power) + low_candidate_rgb * (core_mix * core_power), 0.0, 1.0)

    glass_veil = cv2.GaussianBlur(np.clip(original_rgb - candidate_rgb, 0.0, 1.0), (0, 0), GLASS_GHOST_SIGMA)
    wide_veil = cv2.GaussianBlur(np.clip(original_rgb - low_candidate_rgb, 0.0, 1.0), (0, 0), 27.0)
    return np.clip(
        result
        - (glass_base + glass_scale * mask_power) * glass_veil
        - (wide_base + wide_scale * core_power) * wide_veil,
        0.0,
        1.0,
    )


def build_polarfree_prior(
    images: Mapping[str, np.ndarray],
    reference_view: str = REFERENCE_VIEW,
    prior_profile: str = "stable",
) -> Dict[str, np.ndarray]:
    if prior_profile not in {"stable", "visual"}:
        raise ValueError(f"Unsupported prior_profile: {prior_profile}")
    original_rgb = images[reference_view]
    features = compute_polarization_features(images)
    candidate_rgb, low_candidate_rgb, prior_mask, _detail = build_background_candidate(original_rgb, features, prior_profile=prior_profile)
    reflection_mask, core_mask = build_reflection_mask(original_rgb, low_candidate_rgb, prior_mask, features, prior_profile=prior_profile)
    prior_rgb = restore_background(
        original_rgb,
        candidate_rgb,
        low_candidate_rgb,
        prior_mask,
        reflection_mask,
        core_mask,
        prior_profile=prior_profile,
    )

    if prior_profile == "visual":
        final_mask = np.clip(0.58 * reflection_mask + 0.42 * core_mask, 0.0, 1.0)
        suppression_mask = ensure_single_channel(
            cv2.GaussianBlur(np.maximum(final_mask, prior_mask).astype(np.float32), (0, 0), 6.8)
        )
        dark_floor_rgb = np.clip(
            0.52 * low_candidate_rgb
            + 0.22 * features["continuous_min_rgb"]
            + 0.26 * features["aggressive_percentile_rgb"],
            0.0,
            1.0,
        )
        wide_veil = cv2.GaussianBlur(np.clip(original_rgb - low_candidate_rgb, 0.0, 1.0), (0, 0), 27.0)
        visual_prior_rgb = np.clip(
            prior_rgb * (1.0 - 0.74 * suppression_mask)
            + dark_floor_rgb * (0.74 * suppression_mask)
            - (0.055 + 0.18 * core_mask) * wide_veil,
            0.0,
            1.0,
        )
    else:
        final_mask = np.clip(0.75 * reflection_mask + 0.25 * core_mask, 0.0, 1.0)
        suppression_mask = ensure_single_channel(
            cv2.GaussianBlur(np.maximum(final_mask, prior_mask).astype(np.float32), (0, 0), 5.0)
        )
        dark_floor_rgb = np.clip(
            0.52 * low_candidate_rgb
            + 0.30 * features["continuous_min_rgb"]
            + 0.18 * features["aggressive_percentile_rgb"],
            0.0,
            1.0,
        )
        wide_veil = cv2.GaussianBlur(np.clip(original_rgb - low_candidate_rgb, 0.0, 1.0), (0, 0), 23.0)
        visual_prior_rgb = np.clip(
            prior_rgb * (1.0 - 0.58 * suppression_mask)
            + dark_floor_rgb * (0.58 * suppression_mask)
            - (0.03 + 0.10 * core_mask) * wide_veil,
            0.0,
            1.0,
        )

    return {
        "reference_rgb": original_rgb.astype(np.float32),
        "prior_rgb": prior_rgb.astype(np.float32),
        "visual_prior_rgb": visual_prior_rgb.astype(np.float32),
        "candidate_rgb": candidate_rgb.astype(np.float32),
        "low_candidate_rgb": low_candidate_rgb.astype(np.float32),
        "reflection_mask": final_mask.astype(np.float32),
        "core_mask": core_mask.astype(np.float32),
        "dolp": ensure_single_channel(features["dolp"]).astype(np.float32),
        "polar_amplitude": ensure_single_channel(features["polar_amplitude"]).astype(np.float32),
        "polar_diff": ensure_single_channel(features["polar_diff"]).astype(np.float32),
        "mean_rgb": features["mean_rgb"].astype(np.float32),
        "continuous_min_rgb": features["continuous_min_rgb"].astype(np.float32),
    }


def save_visual_comparison(
    save_path: Path,
    panels: Tuple[Tuple[str, np.ndarray], ...],
    title: str = "",
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    label_height = 52
    rendered = []
    for label, image in panels:
        image_bgr = cv2.cvtColor((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        panel = np.full((image_bgr.shape[0] + label_height, image_bgr.shape[1], 3), 255, dtype=np.uint8)
        panel[label_height:] = image_bgr
        cv2.putText(panel, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (30, 30, 30), 2, cv2.LINE_AA)
        rendered.append(panel)

    figure = np.hstack(rendered)
    if title:
        cv2.putText(figure, title, (18, figure.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.imwrite(str(save_path), figure)


def save_mask_overlay(path: Path, image: np.ndarray, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_2d = np.clip(ensure_single_channel(mask).squeeze(-1), 0.0, 1.0)
    highlighted = image.copy()
    overlay_color = np.array([1.0, 0.25, 0.12], dtype=np.float32).reshape(1, 1, 3)
    alpha_mask = mask_2d[..., np.newaxis] * 0.24
    highlighted = highlighted * (1.0 - alpha_mask) + overlay_color * alpha_mask

    contours, _ = cv2.findContours((mask_2d * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    highlighted_u8 = (np.clip(highlighted, 0.0, 1.0) * 255).astype(np.uint8)
    highlighted_bgr = cv2.cvtColor(highlighted_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), highlighted_bgr)
