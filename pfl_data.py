import csv
import json
import math
import os
import random
import tempfile
import zipfile
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import main
from pfl_config import (
    CACHE_VERSION_FILE,
    MODEL_INPUT_CHANNELS,
    PACKAGE_CACHE_VERSION,
    PACKAGE_REQUIRED_KEYS,
    PRIOR_CACHE_VERSION,
    SHADOW_MASK_KEYS,
)

EPS = 1e-6
ANGLE_TAG_TO_NAME = {
    "000": "0deg",
    "045": "45deg",
    "090": "90deg",
    "135": "135deg",
}
ANGLE_TAGS = tuple(ANGLE_TAG_TO_NAME.keys())


@dataclass(frozen=True)
class SampleRecord:
    split: str
    subset: str
    scene: str
    group: str
    polar_paths: Dict[str, Path]
    input_rgb_path: Optional[Path]
    gt_group: str
    gt_polar_paths: Dict[str, Path]
    gt_rgb_path: Optional[Path]
    target_kind: str
    hard_case_weight: float = 1.0
    hard_case_failure_types: Tuple[str, ...] = ()
    hard_case_notes: str = ""


@dataclass(frozen=True)
class DatasetPart:
    subset: str
    root: Path
    input_root: Path
    gt_root: Path


def discover_dataset_parts(data_root: Path) -> List[DatasetPart]:
    data_root = Path(data_root)
    parts: List[DatasetPart] = []
    seen: Set[Tuple[Path, Path]] = set()

    def add_part(subset: str, root: Path, input_root: Path, gt_root: Path) -> None:
        key = (input_root.resolve(), gt_root.resolve())
        if key in seen or not input_root.exists() or not gt_root.exists():
            return
        seen.add(key)
        parts.append(DatasetPart(subset=subset, root=root, input_root=input_root, gt_root=gt_root))

    add_part(data_root.name or "root", data_root, data_root / "input", data_root / "gt")
    add_part("test", data_root / "test", data_root / "test" / "input", data_root / "test" / "gt")
    if data_root.exists():
        for child in sorted(path for path in data_root.iterdir() if path.is_dir()):
            add_part(child.name, child, child / "input", child / "gt")

    if not parts:
        raise FileNotFoundError(f"Unable to find input/gt dataset parts under: {data_root}")
    return parts


def discover_group_files(scene_dir: Path, allow_rgb: bool) -> Dict[str, Dict[str, Path]]:
    suffixes = set(ANGLE_TAGS)
    if allow_rgb:
        suffixes.add("rgb")

    groups: Dict[str, Dict[str, Path]] = {}
    for path in sorted(scene_dir.glob("*.png")):
        if "_" not in path.stem:
            continue
        group, suffix = path.stem.split("_", 1)
        suffix = suffix.lower()
        if suffix not in suffixes:
            continue
        groups.setdefault(group, {})[suffix] = path
    return groups


def discover_dataset(data_root: Path, split_name: str) -> List[SampleRecord]:
    records: List[SampleRecord] = []

    for part in discover_dataset_parts(data_root):
        for input_scene in sorted(path for path in part.input_root.iterdir() if path.is_dir()):
            gt_scene = part.gt_root / input_scene.name
            if not gt_scene.exists():
                continue

            input_groups = discover_group_files(input_scene, allow_rgb=True)
            gt_groups = discover_group_files(gt_scene, allow_rgb=True)
            complete_gt_groups = sorted(
                group_id for group_id, files in gt_groups.items() if all(tag in files for tag in ANGLE_TAGS)
            )
            if not complete_gt_groups:
                continue

            # Each scene uses its single clean GT group for every reflected input group.
            preferred_gt_group = "0000" if "0000" in complete_gt_groups else complete_gt_groups[0]
            gt_files = gt_groups[preferred_gt_group]
            gt_polar_paths = {ANGLE_TAG_TO_NAME[tag]: gt_files[tag] for tag in ANGLE_TAGS}
            gt_rgb_path = gt_files.get("rgb")
            target_kind = "gt_rgb" if gt_rgb_path is not None else "gt_0deg"

            for group_id, files in sorted(input_groups.items()):
                if not all(tag in files for tag in ANGLE_TAGS):
                    continue
                records.append(
                    SampleRecord(
                        split=split_name,
                        subset=part.subset,
                        scene=input_scene.name,
                        group=group_id,
                        polar_paths={ANGLE_TAG_TO_NAME[tag]: files[tag] for tag in ANGLE_TAGS},
                        input_rgb_path=files.get("rgb"),
                        gt_group=preferred_gt_group,
                        gt_polar_paths=gt_polar_paths,
                        gt_rgb_path=gt_rgb_path,
                        target_kind=target_kind,
                    )
                )

    if not records:
        raise RuntimeError(f"No complete polarized samples found under: {data_root}")
    return records


def safe_relative_path(path: Optional[Path], root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def serialize_record(record: SampleRecord, data_root: Path) -> Dict[str, object]:
    return {
        "split": record.split,
        "subset": record.subset,
        "scene": record.scene,
        "input_group": record.group,
        "polar_paths": {angle: safe_relative_path(path, data_root) for angle, path in record.polar_paths.items()},
        "input_rgb_path": safe_relative_path(record.input_rgb_path, data_root),
        "gt_group": record.gt_group,
        "gt_polar_paths": {angle: safe_relative_path(path, data_root) for angle, path in record.gt_polar_paths.items()},
        "gt_rgb_path": safe_relative_path(record.gt_rgb_path, data_root),
        "target_kind": record.target_kind,
        "target_fallback": record.target_kind != "gt_rgb",
        "hard_case_weight": record.hard_case_weight,
        "hard_case_failure_types": list(record.hard_case_failure_types),
        "hard_case_notes": record.hard_case_notes,
    }


def write_manifest(records: Sequence[SampleRecord], data_root: Path, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(serialize_record(record, data_root), ensure_ascii=False) + "\n")


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)


def read_target_image(record: SampleRecord, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
    target_path = record.gt_rgb_path or record.gt_polar_paths.get("0deg")
    if target_path is None:
        return None
    return main.read_rgb_image(target_path, target_size=target_size)


def read_input_rgb(
    record: SampleRecord,
    images: Mapping[str, np.ndarray],
    target_size: Tuple[int, int],
) -> np.ndarray:
    if record.input_rgb_path is not None and record.input_rgb_path.exists():
        return main.read_rgb_image(record.input_rgb_path, target_size=target_size).astype(np.float32)
    return np.mean(np.stack([images[angle] for angle in main.POLARIZATION_ORDER], axis=0), axis=0).astype(np.float32)


def resolve_prior_cache_dir(data_root: Path, prior_cache_dir: str) -> Path:
    return Path(prior_cache_dir) if prior_cache_dir else data_root / ".polarfree_cache"


def resolve_package_cache_dir(data_root: Path, package_cache_dir: str) -> Path:
    return Path(package_cache_dir) if package_cache_dir else data_root / ".polarfree_tensor_cache"


def ensure_cache_version(cache_dir: Optional[Path], current_version: str, label: str) -> None:
    if cache_dir is None:
        return
    version_path = cache_dir / CACHE_VERSION_FILE
    try:
        previous_version = version_path.read_text(encoding="utf-8").strip() if version_path.exists() else ""
    except OSError:
        previous_version = ""
    if previous_version == current_version:
        return

    removed = 0
    if cache_dir.exists():
        for cache_path in cache_dir.rglob("*.npz"):
            if not cache_path.is_file():
                continue
            try:
                cache_path.unlink()
                removed += 1
            except OSError:
                pass

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        version_path.write_text(current_version, encoding="utf-8")
    except OSError:
        pass
    if removed:
        print(f"{label} cache version changed: removed {removed} old cache files")


def ensure_runtime_cache_versions(prior_cache_dir: Optional[Path], package_cache_dir: Optional[Path]) -> None:
    ensure_cache_version(prior_cache_dir, PRIOR_CACHE_VERSION, "prior")
    ensure_cache_version(package_cache_dir, PACKAGE_CACHE_VERSION, "package")


def safe_cache_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return cleaned or "root"


def prior_cache_path(cache_dir: Path, img_size: int, record: SampleRecord, prior_profile: str = "stable") -> Path:
    safe_key = (
        f"{safe_cache_component(record.group)}_{img_size}_{PRIOR_CACHE_VERSION}_"
        f"{safe_cache_component(prior_profile)}.npz"
    )
    return (
        cache_dir
        / safe_cache_component(record.split)
        / safe_cache_component(record.subset)
        / safe_cache_component(record.scene)
        / safe_key
    )


def package_cache_path(
    cache_dir: Path,
    img_size: int,
    record: SampleRecord,
    mask_profile: str = "balanced",
    prior_profile: str = "stable",
) -> Path:
    safe_key = (
        f"{safe_cache_component(record.group)}_{img_size}_{PACKAGE_CACHE_VERSION}_"
        f"{safe_cache_component(mask_profile)}_{safe_cache_component(prior_profile)}.npz"
    )
    return (
        cache_dir
        / safe_cache_component(record.split)
        / safe_cache_component(record.subset)
        / safe_cache_component(record.scene)
        / safe_key
    )


def load_prior_from_cache(cache_path: Path, expected_hw: Tuple[int, int]) -> Optional[Dict[str, np.ndarray]]:
    if not cache_path.exists():
        return None
    try:
        with np.load(str(cache_path)) as data:
            prior_rgb = data["prior_rgb"].astype(np.float32)
            if prior_rgb.shape[:2] != expected_hw:
                return None
            return {key: data[key].astype(np.float32) for key in data.files}
    except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile, zlib.error):
        try:
            cache_path.unlink()
        except OSError:
            pass
        return None


def load_package_from_cache(cache_path: Path, img_size: int) -> Optional[Dict[str, np.ndarray]]:
    if not cache_path.exists():
        return None
    try:
        with np.load(str(cache_path)) as data:
            inputs = data["inputs"]
            if inputs.shape != (MODEL_INPUT_CHANNELS, img_size, img_size):
                return None
            return {key: data[key].astype(np.float32) for key in PACKAGE_REQUIRED_KEYS}
    except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile, zlib.error):
        try:
            cache_path.unlink()
        except OSError:
            pass
        return None


def save_prior_to_cache(cache_path: Path, package: Mapping[str, np.ndarray]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = f".{cache_path.stem}.{os.getpid()}.{next(tempfile._get_candidate_names())}.tmp"
    temp_path = cache_path.with_name(temp_name)
    try:
        with temp_path.open("wb") as fp:
            np.savez_compressed(
                fp,
                prior_rgb=package["prior_rgb"],
                visual_prior_rgb=package["visual_prior_rgb"],
                reference_rgb=package["reference_rgb"],
                reflection_mask=package["reflection_mask"],
                dolp=package["dolp"],
                polar_amplitude=package["polar_amplitude"],
            )
        os.replace(str(temp_path), str(cache_path))
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


def save_package_to_cache(cache_path: Path, package: Mapping[str, np.ndarray]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = f".{cache_path.stem}.{os.getpid()}.{next(tempfile._get_candidate_names())}.tmp"
    temp_path = cache_path.with_name(temp_name)
    try:
        with temp_path.open("wb") as fp:
            np.savez(fp, **{key: package[key].astype(np.float16) for key in PACKAGE_REQUIRED_KEYS})
        os.replace(str(temp_path), str(cache_path))
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


def load_or_build_prior_package(
    record: SampleRecord,
    images: Mapping[str, np.ndarray],
    img_size: int,
    cache_dir: Path,
    prior_profile: str = "stable",
) -> Dict[str, np.ndarray]:
    cache_path = prior_cache_path(cache_dir, img_size, record, prior_profile)
    cached = load_prior_from_cache(cache_path, expected_hw=(img_size, img_size))
    if cached is not None:
        return cached

    package = main.build_polarfree_prior(images, reference_view=main.REFERENCE_VIEW, prior_profile=prior_profile)
    save_prior_to_cache(cache_path, package)
    return {
        "prior_rgb": package["prior_rgb"],
        "visual_prior_rgb": package["visual_prior_rgb"],
        "reference_rgb": package["reference_rgb"],
        "reflection_mask": package["reflection_mask"],
        "dolp": package["dolp"],
        "polar_amplitude": package["polar_amplitude"],
    }


def normalize_mask(image: np.ndarray, low: float = 5.0, high: float = 98.0) -> np.ndarray:
    return main.ensure_single_channel(main.robust_percentile_normalize(main.ensure_single_channel(image).squeeze(-1), low, high))


def smooth_mask(mask: np.ndarray, dilate: int, sigma: float) -> np.ndarray:
    mask_2d = main.ensure_single_channel(mask).squeeze(-1).astype(np.float32)
    if dilate > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        mask_2d = cv2.dilate(mask_2d, kernel, iterations=1)
    if sigma > 0:
        mask_2d = cv2.GaussianBlur(mask_2d, (0, 0), sigma)
    return np.clip(mask_2d[..., np.newaxis], 0.0, 1.0).astype(np.float32)


def build_line_glare_response(reference_rgb: np.ndarray) -> np.ndarray:
    gray = rgb_to_gray(reference_rgb)
    gray_norm = np.clip(gray, 0.0, 1.0).astype(np.float32)
    width_kernel = max(9, int(round(gray_norm.shape[1] * 0.045)) | 1)
    height_kernel = max(9, int(round(gray_norm.shape[0] * 0.045)) | 1)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width_kernel, 3))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, height_kernel))
    horizontal = cv2.morphologyEx(gray_norm, cv2.MORPH_TOPHAT, horizontal_kernel)
    vertical = cv2.morphologyEx(gray_norm, cv2.MORPH_TOPHAT, vertical_kernel)
    sobel_x = cv2.Sobel(gray_norm, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_norm, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    local = cv2.GaussianBlur(gray_norm, (0, 0), 5.0)
    bright_residual = np.maximum(gray_norm - local, 0.0)
    horizontal_band = cv2.morphologyEx(bright_residual, cv2.MORPH_CLOSE, horizontal_kernel)
    vertical_band = cv2.morphologyEx(bright_residual, cv2.MORPH_CLOSE, vertical_kernel)
    response = 0.34 * horizontal + 0.28 * vertical + 0.24 * horizontal_band + 0.10 * vertical_band + 0.04 * edge
    return normalize_mask(response, 70, 99)


def build_split_masks(
    reference_rgb: np.ndarray,
    base_prior: np.ndarray,
    prior_reflection_mask: np.ndarray,
    dolp: np.ndarray,
    polar_amplitude: np.ndarray,
    mask_profile: str,
    prior_profile: str = "stable",
) -> Dict[str, np.ndarray]:
    profile = mask_profile if mask_profile in {"stable", "balanced", "aggressive"} else "balanced"
    reference_u8 = (np.clip(reference_rgb, 0.0, 1.0) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(reference_u8, cv2.COLOR_RGB2HSV)
    value = hsv[..., 2].astype(np.float32) / 255.0
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    bright = normalize_mask(value, 62, 99)
    rgb_max = np.max(reference_rgb, axis=2).astype(np.float32)
    rgb_max_score = normalize_mask(rgb_max, 60, 99)
    relaxed_sat = np.clip((0.82 - saturation) / 0.82, 0.0, 1.0)[..., np.newaxis]
    high_pass = normalize_mask(np.clip(value - cv2.GaussianBlur(value, (0, 0), 9.0), 0.0, 1.0), 65, 99)
    local_over = normalize_mask(np.clip(value - cv2.GaussianBlur(value, (0, 0), 5.5), 0.0, 1.0), 58, 99)
    colored_glare = bright * np.clip((saturation - 0.08) / 0.66, 0.0, 1.0)[..., np.newaxis]
    line_response = build_line_glare_response(reference_rgb)
    reference_grad_x = cv2.Sobel(value, cv2.CV_32F, 1, 0, ksize=3)
    reference_grad_y = cv2.Sobel(value, cv2.CV_32F, 0, 1, ksize=3)
    reference_structure = normalize_mask(np.sqrt(reference_grad_x * reference_grad_x + reference_grad_y * reference_grad_y), 55, 99)

    glare_strength = {
        "stable": (0.55, 0.25, 0.20, 5, 2.4),
        "balanced": (0.48, 0.27, 0.25, 7, 3.0),
        "aggressive": (0.42, 0.28, 0.30, 9, 3.4),
    }[profile]
    bright_weight, high_pass_weight, line_weight, glare_dilate, glare_sigma = glare_strength
    glare_mask = (
        bright_weight * bright * (0.24 + 0.76 * relaxed_sat)
        + high_pass_weight * high_pass
        + line_weight * line_response
        + 0.16 * rgb_max_score
        + 0.14 * local_over
        + 0.12 * colored_glare
    )
    glare_mask = smooth_mask(np.clip(glare_mask, 0.0, 1.0), glare_dilate, glare_sigma)

    prior_mask = main.ensure_single_channel(prior_reflection_mask).astype(np.float32)
    reference_luma = rgb_to_gray(reference_rgb)
    prior_luma = rgb_to_gray(base_prior)
    raw_prior_diff = np.abs(reference_luma - prior_luma).astype(np.float32)
    prior_diff = normalize_mask(raw_prior_diff, 35, 98)

    reference_low_mid = cv2.GaussianBlur(reference_luma, (0, 0), 9.0)
    prior_low_mid = cv2.GaussianBlur(prior_luma, (0, 0), 9.0)
    reference_low_wide = cv2.GaussianBlur(reference_luma, (0, 0), 21.0)
    prior_low_wide = cv2.GaussianBlur(prior_luma, (0, 0), 21.0)
    structured_low_diff = normalize_mask(
        0.58 * np.abs(reference_low_mid - prior_low_mid) + 0.42 * np.abs(reference_low_wide - prior_low_wide),
        30,
        97,
    )
    low_color_diff = normalize_mask(
        np.mean(
            np.abs(
                cv2.GaussianBlur(reference_rgb.astype(np.float32), (0, 0), 13.0)
                - cv2.GaussianBlur(base_prior.astype(np.float32), (0, 0), 13.0)
            ),
            axis=2,
            keepdims=True,
        ),
        30,
        97,
    )
    dark_shadow_diff = normalize_mask(np.maximum(prior_low_wide - reference_low_wide, 0.0), 42, 98)
    bright_veil_diff = normalize_mask(np.maximum(reference_low_wide - prior_low_wide, 0.0), 32, 97)
    local_shadow_diff = normalize_mask(
        cv2.GaussianBlur(raw_prior_diff, (0, 0), 13.0) - cv2.GaussianBlur(raw_prior_diff, (0, 0), 39.0),
        35,
        98,
    )
    lowfreq_reflection_mask = smooth_mask(
        np.clip(
            0.34 * structured_low_diff
            + 0.26 * low_color_diff
            + 0.22 * main.ensure_single_channel(bright_veil_diff)
            + 0.18 * main.ensure_single_channel(dark_shadow_diff),
            0.0,
            1.0,
        ),
        17,
        6.5,
    )
    dark_reflection_mask = smooth_mask(
        np.clip(
            0.56 * main.ensure_single_channel(dark_shadow_diff)
            + 0.24 * main.ensure_single_channel(local_shadow_diff)
            + 0.20 * structured_low_diff,
            0.0,
            1.0,
        ),
        15,
        5.8,
    )
    shadow_veil_mask = smooth_mask(
        np.clip(
            0.46 * main.ensure_single_channel(bright_veil_diff)
            + 0.26 * low_color_diff
            + 0.16 * glare_mask
            + 0.12 * line_response,
            0.0,
            1.0,
        ),
        15,
        6.0,
    )
    scene_reflection_mask = smooth_mask(
        np.clip(
            0.34 * lowfreq_reflection_mask
            + 0.24 * dark_reflection_mask
            + 0.22 * shadow_veil_mask
            + 0.20 * main.ensure_single_channel(prior_diff),
            0.0,
            1.0,
        ),
        19,
        7.0,
    )
    foreground_structure_guard = smooth_mask(
        np.clip(
            0.52 * reference_structure
            + 0.28 * normalize_mask(np.maximum(saturation - 0.22, 0.0), 40, 98)
            + 0.20 * normalize_mask(np.abs(reference_luma - prior_luma), 70, 99),
            0.0,
            1.0,
        ),
        5,
        1.6,
    )
    foreground_structure_guard = np.clip(
        foreground_structure_guard * np.clip(1.0 - 0.45 * scene_reflection_mask, 0.25, 1.0),
        0.0,
        1.0,
    ).astype(np.float32)
    position_response = np.maximum.reduce(
        [
            main.ensure_single_channel(prior_diff),
            main.ensure_single_channel(structured_low_diff),
            main.ensure_single_channel(low_color_diff),
            main.ensure_single_channel(dark_shadow_diff),
        ]
    )

    dolp_score = normalize_mask(dolp, 10, 98)
    dolp_low = normalize_mask(cv2.GaussianBlur(main.ensure_single_channel(dolp).squeeze(-1), (0, 0), 15.0), 14, 97)
    amplitude_score = normalize_mask(polar_amplitude, 10, 98)
    amplitude_low = normalize_mask(
        cv2.GaussianBlur(main.ensure_single_channel(polar_amplitude).squeeze(-1), (0, 0), 15.0),
        14,
        97,
    )
    polar_support = np.clip(0.35 * dolp_score + 0.35 * dolp_low + 0.15 * amplitude_score + 0.15 * amplitude_low, 0.0, 1.0)
    reflection_weights = {
        "stable": (0.54, 0.16, 0.13, 0.07, 0.04, 0.06, 7, 3.0, 48),
        "balanced": (0.34, 0.20, 0.20, 0.10, 0.06, 0.10, 9, 3.8, 42),
        "aggressive": (0.26, 0.22, 0.22, 0.12, 0.07, 0.11, 11, 4.5, 36),
    }[profile]
    (
        prior_w,
        position_w,
        structure_w,
        low_color_w,
        shadow_w,
        polar_w,
        reflection_dilate,
        reflection_sigma,
        stretch_low,
    ) = reflection_weights
    reflection_area_mask = (
        prior_w * np.clip(prior_mask, 0.0, 1.0)
        + position_w * position_response
        + structure_w * structured_low_diff
        + low_color_w * low_color_diff
        + shadow_w * np.maximum(dark_shadow_diff, local_shadow_diff)
        + polar_w * polar_support
    )
    reflection_area_mask = np.maximum(
        reflection_area_mask,
        np.clip(
            0.40 * scene_reflection_mask
            + 0.30 * lowfreq_reflection_mask
            + 0.18 * dark_reflection_mask
            + 0.12 * shadow_veil_mask,
            0.0,
            1.0,
        ),
    )
    if prior_profile == "visual":
        veil_response = normalize_mask(
            cv2.GaussianBlur(np.maximum(reference_low_wide - prior_low_wide, 0.0).astype(np.float32), (0, 0), 23.0),
            28,
            96,
        )
        visual_reflection = (
            0.22 * np.clip(prior_mask, 0.0, 1.0)
            + 0.20 * position_response
            + 0.18 * structured_low_diff
            + 0.15 * low_color_diff
            + 0.08 * np.maximum(dark_shadow_diff, local_shadow_diff)
            + 0.09 * polar_support
            + 0.05 * line_response
            + 0.03 * veil_response
        )
        reflection_area_mask = np.maximum(reflection_area_mask, visual_reflection)
    reflection_area_mask = smooth_mask(np.clip(reflection_area_mask, 0.0, 1.0), reflection_dilate, reflection_sigma)
    reflection_area_mask = np.clip(
        0.62 * reflection_area_mask + 0.38 * normalize_mask(reflection_area_mask, stretch_low, 98),
        0.0,
        1.0,
    )

    final_weights = {
        "stable": (0.74, 0.16, 0.28, 7, 3.2),
        "balanced": (0.52, 0.24, 0.44, 11, 5.0),
        "aggressive": (0.42, 0.30, 0.52, 15, 6.2),
    }[profile]
    prior_final_w, glare_final_w, reflection_final_w, final_dilate, final_sigma = final_weights
    final_mask = (
        prior_final_w * np.clip(prior_mask, 0.0, 1.0)
        + glare_final_w * glare_mask
        + reflection_final_w * reflection_area_mask
    )
    if prior_profile == "visual":
        visual_final = (
            0.34 * np.clip(prior_mask, 0.0, 1.0)
            + 0.30 * glare_mask
            + 0.56 * reflection_area_mask
            + 0.08 * line_response
        )
        final_mask = np.maximum(final_mask, visual_final)
    final_mask = np.maximum(
        final_mask,
        np.clip(0.36 * scene_reflection_mask + 0.30 * lowfreq_reflection_mask + 0.22 * dark_reflection_mask, 0.0, 1.0),
    )
    final_mask = smooth_mask(np.clip(final_mask, 0.0, 1.0), final_dilate, final_sigma)
    return {
        "mask": final_mask.astype(np.float32),
        "glare_mask": glare_mask.astype(np.float32),
        "reflection_area_mask": reflection_area_mask.astype(np.float32),
        "scene_reflection_mask": scene_reflection_mask.astype(np.float32),
        "dark_reflection_mask": dark_reflection_mask.astype(np.float32),
        "shadow_veil_mask": shadow_veil_mask.astype(np.float32),
        "lowfreq_reflection_mask": lowfreq_reflection_mask.astype(np.float32),
        "foreground_structure_guard": foreground_structure_guard.astype(np.float32),
    }


def build_model_inputs(
    record: SampleRecord,
    img_size: int,
    prior_cache_dir: Path,
    package_cache_dir: Optional[Path] = None,
    use_package_cache: bool = True,
    mask_profile: str = "balanced",
    prior_profile: str = "stable",
) -> Dict[str, np.ndarray]:
    if use_package_cache and package_cache_dir is not None:
        cached_package = load_package_from_cache(
            package_cache_path(package_cache_dir, img_size, record, mask_profile, prior_profile),
            img_size,
        )
        if cached_package is not None:
            return cached_package

    target_size = (img_size, img_size)
    images = main.load_polarized_images(record.polar_paths, target_size=target_size)
    prior_package = load_or_build_prior_package(
        record,
        images,
        img_size=img_size,
        cache_dir=prior_cache_dir,
        prior_profile=prior_profile,
    )

    reference_rgb = images[main.REFERENCE_VIEW].astype(np.float32)
    input_rgb = read_input_rgb(record, images, target_size=target_size)
    base_prior = prior_package["prior_rgb"].astype(np.float32)
    visual_prior = prior_package["visual_prior_rgb"].astype(np.float32)
    reflection_mask = main.ensure_single_channel(prior_package["reflection_mask"]).astype(np.float32)
    dolp = main.ensure_single_channel(prior_package["dolp"]).astype(np.float32)
    polar_amplitude = main.ensure_single_channel(prior_package["polar_amplitude"]).astype(np.float32)
    masks = build_split_masks(reference_rgb, base_prior, reflection_mask, dolp, polar_amplitude, mask_profile, prior_profile=prior_profile)
    training_mask = masks["mask"]

    polarized_rgb_channels = [images[angle].astype(np.float32) for angle in main.POLARIZATION_ORDER]
    i0_gray = main.ensure_single_channel(np.mean(images["0deg"].astype(np.float32), axis=2, keepdims=True))
    i45_gray = main.ensure_single_channel(np.mean(images["45deg"].astype(np.float32), axis=2, keepdims=True))
    i90_gray = main.ensure_single_channel(np.mean(images["90deg"].astype(np.float32), axis=2, keepdims=True))
    i135_gray = main.ensure_single_channel(np.mean(images["135deg"].astype(np.float32), axis=2, keepdims=True))
    s1 = i0_gray - i90_gray
    s2 = i45_gray - i135_gray
    aop_norm = np.sqrt(s1 * s1 + s2 * s2 + EPS).astype(np.float32)
    aop_cos2 = np.clip(s1 / aop_norm, -1.0, 1.0).astype(np.float32)
    aop_sin2 = np.clip(s2 / aop_norm, -1.0, 1.0).astype(np.float32)
    input_tensor = np.concatenate(
        polarized_rgb_channels
        + [
            input_rgb,
            base_prior,
            visual_prior,
            dolp,
            polar_amplitude,
            aop_cos2,
            aop_sin2,
            training_mask,
            masks["scene_reflection_mask"],
            masks["dark_reflection_mask"],
            masks["shadow_veil_mask"],
            masks["lowfreq_reflection_mask"],
            masks["foreground_structure_guard"],
        ],
        axis=2,
    ).astype(np.float32)

    target_rgb = read_target_image(record, target_size=target_size)
    if target_rgb is None:
        target_rgb = np.zeros_like(reference_rgb, dtype=np.float32)
    else:
        target_rgb = target_rgb.astype(np.float32)

    package = {
        "inputs": np.transpose(input_tensor, (2, 0, 1)).astype(np.float32),
        "base_prior": np.transpose(base_prior, (2, 0, 1)).astype(np.float32),
        "visual_prior": np.transpose(visual_prior, (2, 0, 1)).astype(np.float32),
        "target": np.transpose(target_rgb, (2, 0, 1)).astype(np.float32),
        "reference": np.transpose(reference_rgb, (2, 0, 1)).astype(np.float32),
        "input_rgb": np.transpose(input_rgb, (2, 0, 1)).astype(np.float32),
        "mask": np.transpose(training_mask, (2, 0, 1)).astype(np.float32),
        "glare_mask": np.transpose(masks["glare_mask"], (2, 0, 1)).astype(np.float32),
        "reflection_area_mask": np.transpose(masks["reflection_area_mask"], (2, 0, 1)).astype(np.float32),
        "scene_reflection_mask": np.transpose(masks["scene_reflection_mask"], (2, 0, 1)).astype(np.float32),
        "dark_reflection_mask": np.transpose(masks["dark_reflection_mask"], (2, 0, 1)).astype(np.float32),
        "shadow_veil_mask": np.transpose(masks["shadow_veil_mask"], (2, 0, 1)).astype(np.float32),
        "lowfreq_reflection_mask": np.transpose(masks["lowfreq_reflection_mask"], (2, 0, 1)).astype(np.float32),
        "foreground_structure_guard": np.transpose(masks["foreground_structure_guard"], (2, 0, 1)).astype(np.float32),
    }
    if use_package_cache and package_cache_dir is not None:
        save_package_to_cache(package_cache_path(package_cache_dir, img_size, record, mask_profile, prior_profile), package)
    return package


def apply_geometric_augmentation(package: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    spatial_keys = PACKAGE_REQUIRED_KEYS
    if random.random() < 0.5:
        for key in spatial_keys:
            package[key] = np.flip(package[key], axis=2).copy()
    if random.random() < 0.5:
        for key in spatial_keys:
            package[key] = np.flip(package[key], axis=1).copy()
    if random.random() < 0.25:
        for key in spatial_keys:
            package[key] = np.rot90(package[key], k=1, axes=(1, 2)).copy()
    return package


class PolarFreeLiteDataset(Dataset):
    def __init__(
        self,
        records: Sequence[SampleRecord],
        img_size: int,
        prior_cache_dir: Path,
        package_cache_dir: Optional[Path],
        use_package_cache: bool,
        mask_profile: str = "balanced",
        prior_profile: str = "stable",
        augment: bool = False,
    ) -> None:
        self.records = list(records)
        self.img_size = img_size
        self.prior_cache_dir = prior_cache_dir
        self.package_cache_dir = package_cache_dir
        self.use_package_cache = use_package_cache
        self.mask_profile = mask_profile
        self.prior_profile = prior_profile
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        package = build_model_inputs(
            record,
            img_size=self.img_size,
            prior_cache_dir=self.prior_cache_dir,
            package_cache_dir=self.package_cache_dir,
            use_package_cache=self.use_package_cache,
            mask_profile=self.mask_profile,
            prior_profile=self.prior_profile,
        )
        if self.augment:
            package = apply_geometric_augmentation(package)
        return {
            "inputs": torch.from_numpy(package["inputs"]),
            "base_prior": torch.from_numpy(package["base_prior"]),
            "visual_prior": torch.from_numpy(package["visual_prior"]),
            "target": torch.from_numpy(package["target"]),
            "reference": torch.from_numpy(package["reference"]),
            "input_rgb": torch.from_numpy(package["input_rgb"]),
            "mask": torch.from_numpy(package["mask"]),
            "glare_mask": torch.from_numpy(package["glare_mask"]),
            "reflection_area_mask": torch.from_numpy(package["reflection_area_mask"]),
            "scene_reflection_mask": torch.from_numpy(package["scene_reflection_mask"]),
            "dark_reflection_mask": torch.from_numpy(package["dark_reflection_mask"]),
            "shadow_veil_mask": torch.from_numpy(package["shadow_veil_mask"]),
            "lowfreq_reflection_mask": torch.from_numpy(package["lowfreq_reflection_mask"]),
            "foreground_structure_guard": torch.from_numpy(package["foreground_structure_guard"]),
            "hard_case_weight": torch.tensor(float(record.hard_case_weight), dtype=torch.float32),
            "hard_case_failure_types": ",".join(record.hard_case_failure_types),
            "hard_case_notes": record.hard_case_notes,
            "scene": record.scene,
            "group": record.group,
            "subset": record.subset,
            "target_kind": record.target_kind,
        }


def normalize_failure_types(value: object) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.replace("|", ",").split(",") if item.strip())
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),)


def parse_float(value: object, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: object, default: int = 1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_loss_multipliers(value: object) -> Dict[str, float]:
    if isinstance(value, dict):
        return {str(key): parse_float(multiplier, 1.0) for key, multiplier in value.items()}
    if isinstance(value, str) and value.strip():
        multipliers: Dict[str, float] = {}
        for part in value.replace("|", ",").split(","):
            if "=" not in part:
                continue
            key, raw_value = part.split("=", 1)
            key = key.strip()
            if key:
                multipliers[key] = parse_float(raw_value.strip(), 1.0)
        return multipliers
    return {}


def normalize_hard_case_entry(item: object, index: int = 0, scene_hint: str = "") -> Optional[Dict[str, object]]:
    if isinstance(item, str):
        item = {"scene": item}
    if not isinstance(item, dict):
        return None
    scene = str(item.get("scene") or item.get("path") or item.get("id") or scene_hint).strip()
    if not scene:
        scene = f"case_{index:04d}"
    group = str(item.get("group") or "").strip()
    severity = max(1.0, parse_float(item.get("severity", 1.0), 1.0))
    repeat = max(1, parse_int(item.get("repeat", 1), 1))
    multipliers = normalize_loss_multipliers(item.get("loss_multipliers"))
    return {
        "scene": scene,
        "group": group,
        "failure_types": normalize_failure_types(item.get("failure_types")),
        "severity": severity,
        "repeat": repeat,
        "loss_multipliers": multipliers,
        "notes": str(item.get("notes") or ""),
    }


def load_hard_case_entries(manifest_path: str) -> List[Dict[str, object]]:
    if not manifest_path:
        return []
    path = Path(manifest_path)
    if not path.exists():
        return []
    entries: List[Dict[str, object]] = []
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if isinstance(payload, dict):
            raw_entries = payload.get("cases", payload.get("hard_cases", payload.get("items", [])))
            if not raw_entries and any(key in payload for key in ("scene", "group", "failure_types", "severity", "repeat")):
                raw_entries = [payload]
            if not raw_entries:
                raw_entries = []
                for scene_key, scene_value in payload.items():
                    if isinstance(scene_value, dict):
                        mapped = dict(scene_value)
                        mapped.setdefault("scene", scene_key)
                        raw_entries.append(mapped)
        else:
            raw_entries = payload
        for index, item in enumerate(raw_entries or []):
            entry = normalize_hard_case_entry(item, index=index)
            if entry is not None:
                entries.append(entry)
        return entries

    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None or "scene" not in reader.fieldnames:
            return []
        for index, row in enumerate(reader):
            entry = normalize_hard_case_entry(row, index=index)
            if entry is not None:
                entries.append(entry)
    return entries


# Candidate cleanup: kept as a small compatibility helper for older callers.

def load_hard_case_keys(manifest_path: str) -> Set[Tuple[str, str]]:
    return {
        (str(entry["scene"]), str(entry.get("group", "")))
        for entry in load_hard_case_entries(manifest_path)
        if str(entry.get("scene", "")).strip()
    }


def hard_case_sample_weight(entry: Mapping[str, object]) -> float:
    multipliers = entry.get("loss_multipliers", {})
    multiplier_values = list(multipliers.values()) if isinstance(multipliers, dict) else []
    multiplier_boost = max([1.0] + [parse_float(value, 1.0) for value in multiplier_values])
    severity = parse_float(entry.get("severity", 1.0), 1.0)
    return min(6.0, max(1.0, severity * multiplier_boost))


def append_hard_cases(
    train_records: Sequence[SampleRecord],
    all_train_records: Sequence[SampleRecord],
    manifest_path: str,
    repeat: int,
) -> Tuple[List[SampleRecord], Dict[str, object]]:
    chosen = list(train_records)
    entries = load_hard_case_entries(manifest_path)
    keys = {(str(entry["scene"]), str(entry.get("group", ""))) for entry in entries}
    if not entries or repeat <= 0:
        return chosen, {
            "hard_case_manifest": manifest_path,
            "hard_case_keys": len(keys),
            "hard_case_matches": 0,
            "hard_case_repeat": repeat,
            "train_samples_after_hard_cases": len(chosen),
        }
    matched: List[SampleRecord] = []
    for entry in entries:
        scene = str(entry["scene"])
        group = str(entry.get("group", ""))
        entry_matches = [
            record
            for record in all_train_records
            if record.scene == scene and (not group or record.group == group)
        ]
        entry_repeat = repeat * max(1, parse_int(entry.get("repeat", 1), 1))
        sample_weight = hard_case_sample_weight(entry)
        failure_types = tuple(entry.get("failure_types", ()))
        notes = str(entry.get("notes", ""))
        weighted_matches = [
            replace(
                record,
                hard_case_weight=sample_weight,
                hard_case_failure_types=failure_types,
                hard_case_notes=notes,
            )
            for record in entry_matches
        ]
        chosen = [
            replace(
                record,
                hard_case_weight=sample_weight,
                hard_case_failure_types=failure_types,
                hard_case_notes=notes,
            )
            if record.scene == scene and (not group or record.group == group)
            else record
            for record in chosen
        ]
        matched.extend(weighted_matches)
        for _ in range(entry_repeat):
            chosen.extend(weighted_matches)
    return chosen, {
        "hard_case_manifest": manifest_path,
        "hard_case_keys": len(keys),
        "hard_case_matches": len(matched),
        "hard_case_repeat": repeat,
        "hard_case_json_fields": ["scene", "group", "failure_types", "severity", "repeat", "loss_multipliers", "notes"],
        "hard_case_failure_types": HARD_CASE_FAILURE_TYPES,
        "train_samples_after_hard_cases": len(chosen),
    }


