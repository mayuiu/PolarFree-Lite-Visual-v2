import argparse
import csv
import json
import math
import os
import random
import sys
import tempfile
import zipfile
import zlib
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import cv2
import numpy as np


def _early_arg_value(argv: Sequence[str], name: str, default: str) -> str:
    prefix = f"{name}="
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(prefix):
            return value.split("=", 1)[1]
    return default


def _early_inspect_count(data_root: Path, split_name: str) -> Dict[str, object]:
    angle_tags = ("000", "045", "090", "135")

    def discover_parts(root: Path) -> List[Tuple[str, Path, Path]]:
        parts: List[Tuple[str, Path, Path]] = []
        seen: Set[Tuple[Path, Path]] = set()

        def add(subset: str, input_root: Path, gt_root: Path) -> None:
            if not input_root.exists() or not gt_root.exists():
                return
            key = (input_root.resolve(), gt_root.resolve())
            if key in seen:
                return
            seen.add(key)
            parts.append((subset, input_root, gt_root))

        add(root.name or "root", root / "input", root / "gt")
        add("test", root / "test" / "input", root / "test" / "gt")
        if root.exists():
            for child in sorted(path for path in root.iterdir() if path.is_dir()):
                add(child.name, child / "input", child / "gt")
        return parts

    def group_files(scene_dir: Path, allow_rgb: bool) -> Dict[str, Set[str]]:
        suffixes = set(angle_tags)
        if allow_rgb:
            suffixes.add("rgb")
        groups: Dict[str, Set[str]] = {}
        for path in sorted(scene_dir.glob("*.png")):
            if "_" not in path.stem:
                continue
            group, suffix = path.stem.split("_", 1)
            suffix = suffix.lower()
            if suffix in suffixes:
                groups.setdefault(group, set()).add(suffix)
        return groups

    samples = 0
    scenes: Set[str] = set()
    subset_counts: Dict[str, int] = {}
    target_counts: Dict[str, int] = {}
    for subset, input_root, gt_root in discover_parts(data_root):
        for input_scene in sorted(path for path in input_root.iterdir() if path.is_dir()):
            gt_scene = gt_root / input_scene.name
            if not gt_scene.exists():
                continue
            gt_groups = group_files(gt_scene, allow_rgb=True)
            complete_gt = sorted(group for group, tags in gt_groups.items() if all(tag in tags for tag in angle_tags))
            if not complete_gt:
                continue
            gt_group = "0000" if "0000" in complete_gt else complete_gt[0]
            target_kind = "gt_rgb" if "rgb" in gt_groups[gt_group] else "gt_0deg"
            input_groups = group_files(input_scene, allow_rgb=True)
            for _group, tags in sorted(input_groups.items()):
                if not all(tag in tags for tag in angle_tags):
                    continue
                samples += 1
                scenes.add(f"{subset}/{input_scene.name}")
                subset_counts[subset] = subset_counts.get(subset, 0) + 1
                target_counts[target_kind] = target_counts.get(target_kind, 0) + 1

    return {
        "samples": samples,
        "scenes": len(scenes),
        "subsets": subset_counts,
        "target_kinds": target_counts,
        "target_fallbacks": target_counts.get("gt_0deg", 0),
        "split": split_name,
    }


if _early_arg_value(sys.argv, "--mode", "") == "inspect":
    train_root = Path(_early_arg_value(sys.argv, "--train_root", r"D:\jibi\train"))
    test_root = Path(_early_arg_value(sys.argv, "--test_root", r"D:\jibi\data\test"))
    print(
        json.dumps(
            {
                "train_root": str(train_root),
                "test_root": str(test_root),
                "train": _early_inspect_count(train_root, "train"),
                "test": _early_inspect_count(test_root, "test"),
                "matching_rule": "same scene, one GT group supervises every input group, angle suffixes match",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    raise SystemExit(0)


def _print_help_without_torch() -> None:
    base_dir = Path(__file__).resolve().parent
    default_train_root = Path(r"D:\jibi\train")
    default_test_root = Path(r"D:\jibi\data\test")
    default_save_dir = base_dir / "output" / "polarfree_lite_run"
    parser = argparse.ArgumentParser(description="GPU full-training PolarFree-Lite reflection removal")
    parser.add_argument("--mode", choices=["train", "test", "infer", "inspect", "cache"], required=True)
    parser.add_argument("--train_root", type=str, default=str(default_train_root))
    parser.add_argument("--test_root", type=str, default=str(default_test_root))
    parser.add_argument("--data_root", type=str, default=str(default_test_root))
    parser.add_argument("--save_dir", type=str, default=str(default_save_dir))
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--prior_cache_dir", type=str, default="")
    parser.add_argument("--package_cache_dir", type=str, default="")
    parser.add_argument("--package_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model", choices=["quality", "lite"], default="quality")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", "--grad-accum-steps", dest="grad_accum_steps", type=int, default=2)
    parser.add_argument("--grad_clip", "--grad-clip", dest="grad_clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eval_every", "--eval-every", dest="eval_every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", "--prefetch-factor", dest="prefetch_factor", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--channels_last", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_eval_images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--render-mode", choices=["raw", "hybrid", "visual", "visual_plus"], default="visual")
    parser.add_argument("--direct_pretrain_epochs", type=int, default=0)
    parser.add_argument("--visual_strength", type=float, default=1.0)
    parser.add_argument("--showcase_mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--showcase_hard_mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--showcase_reflection_boost", type=float, default=1.0)
    parser.add_argument("--support_sharpen", type=float, default=1.0)
    parser.add_argument("--showcase_detail_strength", type=float, default=0.10)
    parser.add_argument("--showcase_smooth_strength", type=float, default=0.14)
    parser.add_argument("--showcase_tint_suppress", type=float, default=0.12)
    parser.add_argument("--use_perceptual_loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mask_profile", choices=["stable", "balanced", "aggressive"], default="balanced")
    parser.add_argument("--loss_profile", choices=["stable", "balanced", "hard", "visual"], default="balanced")
    parser.add_argument("--prior_profile", choices=["stable", "visual"], default="stable")
    parser.add_argument("--hard_case_manifest", type=str, default="")
    parser.add_argument("--hard_case_repeat", type=int, default=2)
    parser.add_argument("--preset", choices=["fast_effect", "full_stable", "fast_reflection", "gpu_boost", "visual_showcase", "gpu_boost_visual"], default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--test_limit", "--test-limit", dest="test_limit", type=int, default=0)
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--scene", type=str, default="")
    parser.add_argument("--group", type=str, default="")
    parser.add_argument("--input_000", type=str, default="")
    parser.add_argument("--input_045", type=str, default="")
    parser.add_argument("--input_090", type=str, default="")
    parser.add_argument("--input_135", type=str, default="")
    parser.add_argument("--input_rgb", type=str, default="")
    parser.add_argument("--target_rgb", type=str, default="")
    parser.print_help()


if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    _print_help_without_torch()
    raise SystemExit(0)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import main
from pfl_config import (
    CACHE_VERSION_FILE,
    DEBUG_PANEL_EXTRA_FIELDS,
    HARD_CASE_FAILURE_TYPES,
    METRIC_FIELDS,
    MODEL_INPUT_CHANNELS,
    MODEL_OUTPUT_CHANNELS,
    PACKAGE_CACHE_VERSION,
    PACKAGE_REQUIRED_KEYS,
    PRIOR_CACHE_VERSION,
    SHADOW_MASK_KEYS,
)


ANGLE_TAG_TO_NAME = {
    "000": "0deg",
    "045": "45deg",
    "090": "90deg",
    "135": "135deg",
}
ANGLE_TAGS = tuple(ANGLE_TAG_TO_NAME.keys())

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_ROOT = Path(r"D:\jibi\train")
DEFAULT_TEST_ROOT = Path(r"D:\jibi\data\test")
DEFAULT_DATA_ROOT = DEFAULT_TEST_ROOT
DEFAULT_SAVE_DIR = BASE_DIR / "output" / "polarfree_lite_run"
EPS = 1e-6


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


def prepare_device(require_cuda: bool = False) -> torch.device:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training, but torch.cuda.is_available() is False.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def timestamped_run_name(prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}_gpu_full_visual"


def create_run_dir(base_save_dir: Path, prefix: str = "train") -> Path:
    base_save_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_save_dir / timestamped_run_name(prefix)
    counter = 1
    while run_dir.exists():
        run_dir = base_save_dir / f"{timestamped_run_name(prefix)}_{counter}"
        counter += 1
    run_dir.mkdir(parents=True)
    return run_dir


def write_latest_run_pointer(base_save_dir: Path, run_dir: Path) -> None:
    base_save_dir.mkdir(parents=True, exist_ok=True)
    (base_save_dir / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")


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
    response = 0.45 * horizontal + 0.45 * vertical + 0.10 * edge
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
    low_sat = np.clip((0.62 - saturation) / 0.62, 0.0, 1.0)[..., np.newaxis]
    high_pass = normalize_mask(np.clip(value - cv2.GaussianBlur(value, (0, 0), 9.0), 0.0, 1.0), 65, 99)
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
        bright_weight * bright * (0.35 + 0.65 * low_sat)
        + high_pass_weight * high_pass
        + line_weight * line_response
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
        residual_strength_tensor = torch.clamp(
            1.15
            + 0.90 * hard3 * float(visual_strength) * float(showcase_reflection_boost)
            + 0.26 * line3
            + 0.20 * lowfreq3,
            1.20,
            2.35,
        )
        delta_strength_tensor = torch.clamp(
            0.80 + 0.45 * support3 + 0.75 * dark3 + 0.45 * lowfreq3,
            0.80,
            2.15,
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
        direct_clean = torch.clamp(0.92 * residual_clean + 0.08 * abs_clean, 0.0, 1.0)
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
    support_soft = torch.clamp(torch.maximum(support_soft, torch.maximum(shadow_support, confidence_support)), 0.0, 1.0)
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
        + 0.58 * v2_reflection_takeover
        - 0.04 * effective_guard,
        0.0,
        0.985,
    )
    if loss_profile == "visual":
        visual_pull = torch.clamp(0.42 * core + 0.22 * boundary + 0.46 * v2_reflection_takeover, 0.0, 0.97)
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
        showcase_core = torch.clamp(torch.maximum(torch.maximum(core, support_hard), torch.maximum(glare, reflection_area)), 0.0, 1.0)
        showcase_core = torch.clamp(torch.maximum(showcase_core, torch.maximum(shadow_support, confidence_support)), 0.0, 1.0)
        showcase_core = torch.clamp(torch.maximum(showcase_core, v2_reflection_takeover), 0.0, 1.0)
        if showcase_hard_mode:
            showcase_core = torch.clamp(torch.maximum(showcase_core, F.max_pool2d(support_hard, kernel_size=7, stride=1, padding=3)), 0.0, 1.0)
        force_direct = torch.clamp(
            0.55 * support_soft
            + 0.80 * showcase_core
            + 0.28 * boundary
            + 0.18 * glare
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
    prediction = torch.clamp(prediction * (1.0 - force_direct) + direct_clean * force_direct, 0.0, 1.0)

    if showcase_mode:
        # VisualPlus is intentionally more aggressive than Raw: in detected reflection regions,
        # preserve DirectClean's advantage instead of averaging it away.
        visual_extra = torch.clamp(
            0.58 * v2_reflection_takeover
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
    structure_safe = torch.clamp(1.0 - 0.65 * foreground_guard, 0.15, 1.0)
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
    direct_clean_focus_mask = torch.clamp(torch.maximum(focus_mask, reflection_area), 0.0, 1.0)
    direct_clean_core_mask = torch.clamp(torch.maximum(high_conf_reflection, core_mask), 0.0, 1.0)
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
    reflection_focus_l1 = weighted_charbonnier(prediction, target, reflection_area)
    lowfreq_reflection_l1 = low_frequency_l1(prediction, target, torch.clamp(0.55 * reflection_area + 0.45 * high_conf_reflection, 0.0, 1.0))
    prior_escape = prior_escape_loss(prediction, target, base_prior, high_conf_reflection)
    ssim_loss = 1.0 - average_pool_ssim(prediction, target)
    grad = gradient_loss(prediction, target)
    texture = high_frequency_texture_loss(prediction, target, focus_mask)
    high_light_suppression = weighted_l1(torch.relu(prediction - target), torch.zeros_like(prediction), glare_focus_mask)
    line_suppression = line_reflection_suppression(prediction, target, glare_focus_mask)
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
    lowfreq_confidence_loss = weighted_charbonnier(lowfreq_reflection_confidence, lowfreq_reflection, torch.clamp(lowfreq_reflection + 0.25, 0.0, 1.0))
    dark_shadow_confidence_loss = weighted_charbonnier(dark_shadow_confidence, dark_reflection, torch.clamp(dark_reflection + 0.25, 0.0, 1.0))
    reflection_confidence_loss = weighted_charbonnier(reflection_confidence, confidence_target, torch.clamp(confidence_target + 0.20, 0.0, 1.0))
    line_band_confidence_loss = weighted_charbonnier(line_band_confidence, glare, torch.clamp(glare + 0.10, 0.0, 1.0))
    text_reflection_confidence_loss = weighted_charbonnier(text_reflection_confidence, shadow_veil, torch.clamp(shadow_veil + 0.10, 0.0, 1.0))
    tint_confidence_loss = weighted_charbonnier(tint_confidence, lowfreq_reflection, torch.clamp(lowfreq_reflection + 0.10, 0.0, 1.0))
    foreground_structure_guard_loss = weighted_charbonnier(prediction, target, torch.clamp(foreground_guard * (1.0 - shadow_region), 0.0, 1.0))
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
        hard_case_weighted_loss = weighted_charbonnier(
            prediction,
            target,
            torch.clamp((sample_weight - 1.0) * torch.maximum(focus_mask, shadow_region), 0.0, 5.0),
        )
    else:
        hard_case_weighted_loss = prediction.new_tensor(0.0)
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
        + weights["visualplus_should_differ_from_raw_in_reflection_region"] * visualplus_should_differ_from_raw_in_reflection_region
        + weights["residual_consistency_loss"] * residual_consistency_loss
        + weights["hard_case_weighted_loss"] * hard_case_weighted_loss
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
        "reflection_focus_l1": float(reflection_focus_l1.detach().cpu().item()),
        "lowfreq_reflection_l1": float(lowfreq_reflection_l1.detach().cpu().item()),
        "prior_escape": float(prior_escape.detach().cpu().item()),
        "ssim_loss": float(ssim_loss.detach().cpu().item()),
        "grad_loss": float(grad.detach().cpu().item()),
        "texture_loss": float(texture.detach().cpu().item()),
        "high_light_suppression": float(high_light_suppression.detach().cpu().item()),
        "line_reflection_suppression": float(line_suppression.detach().cpu().item()),
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
        "visualplus_should_differ_from_raw_in_reflection_region": float(
            visualplus_should_differ_from_raw_in_reflection_region.detach().cpu().item()
        ),
        "residual_consistency_loss": float(residual_consistency_loss.detach().cpu().item()),
        "hard_case_weighted_loss": float(hard_case_weighted_loss.detach().cpu().item()),
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
    structure_safe = torch.clamp(1.0 - 0.65 * foreground_guard, 0.15, 1.0)
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
    supervised_mask = torch.clamp(
        torch.maximum(
            gt_residual_mask,
            torch.maximum(0.8 * reflection_area, torch.maximum(0.6 * glare, 0.5 * focus_mask)),
        ),
        0.0,
        1.0,
    )
    background_mask = torch.clamp(1.0 - supervised_mask, 0.0, 1.0)
    direct_weights = torch.clamp(0.35 + 0.65 * supervised_mask, 0.0, 1.0)
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
    reflection_confidence_l1 = weighted_charbonnier(
        reflection_confidence,
        torch.clamp(torch.maximum(reflection_area, shadow_region), 0.0, 1.0),
        torch.clamp(torch.maximum(reflection_area, shadow_region) + 0.20, 0.0, 1.0),
    )
    lowfreq_confidence_l1 = weighted_charbonnier(lowfreq_reflection_confidence, lowfreq_reflection, torch.clamp(lowfreq_reflection + 0.20, 0.0, 1.0))
    dark_shadow_confidence_l1 = weighted_charbonnier(dark_shadow_confidence, dark_reflection, torch.clamp(dark_reflection + 0.20, 0.0, 1.0))
    line_confidence_l1 = weighted_charbonnier(line_band_confidence, glare, torch.clamp(glare + 0.10, 0.0, 1.0))
    text_confidence_l1 = weighted_charbonnier(text_reflection_confidence, shadow_veil, torch.clamp(shadow_veil + 0.10, 0.0, 1.0))
    tint_confidence_l1 = weighted_charbonnier(tint_confidence, lowfreq_reflection, torch.clamp(lowfreq_reflection + 0.10, 0.0, 1.0))
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
        2.0 * direct_l1
        + 1.0 * direct_lowfreq
        + 0.28 * direct_gradient
        + 0.30 * direct_laplacian
        + 0.22 * direct_chroma
        + 0.95 * positive_residual_l1
        + 1.10 * signed_delta_l1
        + 0.70 * direct_shadow_l1
        + 0.55 * direct_shadow_lowfreq
        + 0.20 * reflection_confidence_l1
        + 0.22 * lowfreq_confidence_l1
        + 0.20 * dark_shadow_confidence_l1
        + 0.08 * line_confidence_l1
        + 0.08 * text_confidence_l1
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
    low_sat = np.clip((0.58 - saturation) / 0.58, 0.0, 1.0)
    return blur_mask((bright * low_sat)[..., np.newaxis], sigma=5.0)


def gradient_strength_np(image: np.ndarray) -> np.ndarray:
    image = np.clip(image.astype(np.float32), 0.0, 1.0)
    gray = np.mean(image, axis=2).astype(np.float32)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(dx * dx + dy * dy)[..., np.newaxis]


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

    if render_mode not in {"visual", "visual_plus"}:
        raise ValueError(f"Unsupported render mode: {render_mode}")

    if render_mode == "visual_plus":
        visual_plus, _candidate, _alpha = visual_strong_render_components(
            reference_rgb,
            prior_rgb,
            visual_prior_rgb,
            raw_prediction_rgb,
            reflection_mask,
            direct_clean_rgb=direct_clean_rgb,
            showcase_mode=showcase_mode,
            showcase_hard_mode=showcase_hard_mode,
            detail_strength=detail_strength,
            smooth_strength=smooth_strength,
            tint_suppress=tint_suppress,
        )
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


def save_history(history: Sequence[Dict[str, float]], training_dir: Path) -> None:
    training_dir.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    csv_path = training_dir / "history.csv"
    json_path = training_dir / "history.json"
    fieldnames = sorted({key for entry in history for key in entry.keys()})
    if "epoch" in fieldnames:
        fieldnames.remove("epoch")
        fieldnames.insert(0, "epoch")
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(list(history), fp, ensure_ascii=False, indent=2)
    save_loss_curve(history, training_dir / "loss_curve.png")


def save_loss_curve(history: Sequence[Dict[str, float]], save_path: Path) -> None:
    width, height = 960, 540
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    margin = 60
    epochs = [entry["epoch"] for entry in history]
    train_losses = [entry["train_loss"] for entry in history]
    test_pairs = [(index, entry["test_loss"]) for index, entry in enumerate(history) if "test_loss" in entry]
    test_losses = [value for _index, value in test_pairs]
    max_loss = max(train_losses + test_losses + [1e-4])
    min_loss = min(train_losses + test_losses + [0.0])
    span = max(max_loss - min_loss, 1e-6)

    cv2.rectangle(canvas, (margin, margin), (width - margin, height - margin), (30, 30, 30), 2)
    title = "Training Loss" if not test_losses else "Training / Test Loss"
    cv2.putText(canvas, title, (margin, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (20, 20, 20), 2, cv2.LINE_AA)

    def project(index: int, value: float) -> Tuple[int, int]:
        x = int(margin + (width - 2 * margin) * (index / max(len(epochs) - 1, 1)))
        y_ratio = (value - min_loss) / span
        y = int((height - margin) - (height - 2 * margin) * y_ratio)
        return x, y

    train_points = [project(index, value) for index, value in enumerate(train_losses)]
    test_points = [project(index, value) for index, value in test_pairs]
    if len(train_points) > 1:
        cv2.polylines(canvas, [np.array(train_points, dtype=np.int32)], False, (56, 105, 233), 3)
    if len(test_points) > 1:
        cv2.polylines(canvas, [np.array(test_points, dtype=np.int32)], False, (66, 165, 66), 3)

    cv2.putText(canvas, "train", (width - 205, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (56, 105, 233), 2, cv2.LINE_AA)
    if test_losses:
        cv2.putText(canvas, "test", (width - 105, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (66, 165, 66), 2, cv2.LINE_AA)
    cv2.imwrite(str(save_path), canvas)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    history: Sequence[Dict[str, float]],
    epoch: int,
    save_path: Path,
    args: argparse.Namespace,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": list(history),
            "img_size": args.img_size,
            "render_mode": args.render_mode,
            "mask_profile": args.mask_profile,
            "loss_profile": args.loss_profile,
            "prior_profile": args.prior_profile,
            "preset": args.preset,
            "model_input_channels": MODEL_INPUT_CHANNELS,
        },
        str(save_path),
    )


def load_weights(model: nn.Module, weights_path: Path, device: torch.device) -> Dict[str, object]:
    checkpoint = torch.load(str(weights_path), map_location=device)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    grad_accum_steps: int = 1,
    grad_clip: float = 0.0,
    loss_profile: str = "balanced",
    mask_profile: str = "balanced",
    amp_enabled: bool = True,
    channels_last: bool = False,
    direct_pretrain: bool = False,
    showcase_mode: bool = False,
    showcase_hard_mode: bool = False,
    visual_strength: float = 1.0,
    showcase_reflection_boost: float = 1.0,
    support_sharpen: float = 1.0,
    showcase_detail_strength: float = 0.10,
    showcase_smooth_strength: float = 0.14,
    showcase_tint_suppress: float = 0.12,
    perceptual_model: Optional[nn.Module] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "l1": 0.0,
        "focus_l1": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }
    total_items = 0

    if training:
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)

    total_batches = len(loader)
    for batch_index, batch in enumerate(loader, start=1):
        batch = batch_to_device(batch, device=device, channels_last=channels_last)
        inputs = batch["inputs"]
        base_prior = batch["base_prior"]
        reference = batch["reference"]
        target = batch["target"]
        mask = batch["mask"]
        glare_mask = batch["glare_mask"]
        reflection_area_mask = batch["reflection_area_mask"]
        scene_reflection_mask = batch["scene_reflection_mask"]
        dark_reflection_mask = batch["dark_reflection_mask"]
        shadow_veil_mask = batch["shadow_veil_mask"]
        lowfreq_reflection_mask = batch["lowfreq_reflection_mask"]
        foreground_structure_guard = batch["foreground_structure_guard"]
        hard_case_weight = batch.get("hard_case_weight")

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            model_output = model(inputs)
            if direct_pretrain:
                loss, loss_parts, prediction = compute_direct_pretrain_loss(
                    model_output,
                    target,
                    mask,
                    reference,
                    glare_mask=glare_mask,
                    reflection_area_mask=reflection_area_mask,
                    scene_reflection_mask=scene_reflection_mask,
                    dark_reflection_mask=dark_reflection_mask,
                    shadow_veil_mask=shadow_veil_mask,
                    lowfreq_reflection_mask=lowfreq_reflection_mask,
                    foreground_structure_guard=foreground_structure_guard,
                    hard_case_weight=hard_case_weight,
                    visual_strength=visual_strength,
                    showcase_reflection_boost=showcase_reflection_boost,
                    support_sharpen=support_sharpen,
                    perceptual_model=perceptual_model,
                    perceptual_weight=0.10 if perceptual_model is not None else 0.0,
                )
            else:
                prediction = compose_prediction(
                    base_prior,
                    model_output,
                    mask,
                    reference_rgb=reference,
                    glare_mask=glare_mask,
                    reflection_area_mask=reflection_area_mask,
                    scene_reflection_mask=scene_reflection_mask,
                    dark_reflection_mask=dark_reflection_mask,
                    shadow_veil_mask=shadow_veil_mask,
                    lowfreq_reflection_mask=lowfreq_reflection_mask,
                    foreground_structure_guard=foreground_structure_guard,
                    mask_profile=mask_profile,
                    loss_profile=loss_profile,
                    visual_strength=visual_strength,
                    showcase_mode=showcase_mode,
                    showcase_hard_mode=showcase_hard_mode,
                    showcase_reflection_boost=showcase_reflection_boost,
                    support_sharpen=support_sharpen,
                    showcase_detail_strength=showcase_detail_strength,
                    showcase_smooth_strength=showcase_smooth_strength,
                    showcase_tint_suppress=showcase_tint_suppress,
                )
                loss, loss_parts = compute_total_loss(
                    prediction,
                    target,
                    mask,
                    base_prior=base_prior,
                    reference_rgb=reference,
                    model_output=model_output,
                    glare_mask=glare_mask,
                    reflection_area_mask=reflection_area_mask,
                    scene_reflection_mask=scene_reflection_mask,
                    dark_reflection_mask=dark_reflection_mask,
                    shadow_veil_mask=shadow_veil_mask,
                    lowfreq_reflection_mask=lowfreq_reflection_mask,
                    foreground_structure_guard=foreground_structure_guard,
                    hard_case_weight=hard_case_weight,
                    loss_profile=loss_profile,
                    showcase_mode=showcase_mode,
                    visual_strength=visual_strength,
                    showcase_reflection_boost=showcase_reflection_boost,
                    support_sharpen=support_sharpen,
                    perceptual_model=perceptual_model,
                    perceptual_weight=0.06 if perceptual_model is not None else 0.0,
                )

        if training:
            assert optimizer is not None
            assert scaler is not None
            scaled_loss = loss / max(grad_accum_steps, 1)
            scaler.scale(scaled_loss).backward()
            if batch_index % max(grad_accum_steps, 1) == 0 or batch_index == total_batches:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        metrics = compute_metrics(prediction.detach(), target, mask)
        batch_size = inputs.shape[0]
        totals["loss"] += float(loss.detach().cpu().item()) * batch_size
        totals["l1"] += metrics["l1"] * batch_size
        totals["focus_l1"] += metrics["focus_l1"] * batch_size
        totals["psnr"] += metrics["psnr"] * batch_size
        totals["ssim"] += metrics["ssim"] * batch_size
        for key, value in loss_parts.items():
            totals[f"loss_{key}"] = totals.get(f"loss_{key}", 0.0) + value * batch_size
        total_items += batch_size

    divisor = max(total_items, 1)
    return {key: value / divisor for key, value in totals.items()}


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
    reflection_mask_gray_path = output_dir / f"{prefix}_reflection_mask_gray.png"
    debug_mask_panel_path = output_dir / f"{debug_prefix}_debug_mask_panel.png"
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
        "reflection_mask_gray_path": str(reflection_mask_gray_path) if reflection_area_mask is not None else "",
        "debug_mask_panel_path": debug_panel_output,
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
            if args.render_mode == "visual_plus":
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
            dark_shadow_region_l1 = masked_l1_np(raw_rgb, target_rgb, dark_reflection_mask)
            scene_reflection_l1 = masked_l1_np(raw_rgb, target_rgb, scene_reflection_mask)
            shadow_veil_l1 = masked_l1_np(raw_rgb, target_rgb, shadow_veil_mask)
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
                "reflection_mask_gray_path": "",
                "debug_mask_panel_path": "",
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
                    "reflection_mask_gray_path": output_info.get("reflection_mask_gray_path", ""),
                    "debug_mask_panel_path": output_info.get("debug_mask_panel_path", ""),
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


def limit_records(records: Sequence[SampleRecord], limit: int) -> List[SampleRecord]:
    chosen = list(records)
    return chosen[:limit] if limit > 0 else chosen


def summarize_records(records: Sequence[SampleRecord]) -> Dict[str, object]:
    subset_counts: Dict[str, int] = {}
    target_counts: Dict[str, int] = {}
    scenes = set()
    for record in records:
        scenes.add(f"{record.subset}/{record.scene}")
        subset_counts[record.subset] = subset_counts.get(record.subset, 0) + 1
        target_counts[record.target_kind] = target_counts.get(record.target_kind, 0) + 1
    return {
        "samples": len(records),
        "scenes": len(scenes),
        "subsets": subset_counts,
        "target_kinds": target_counts,
        "target_fallbacks": sum(1 for record in records if record.target_kind != "gt_rgb"),
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


def build_model(args: argparse.Namespace) -> nn.Module:
    base_channels = 24 if args.model == "lite" else 32
    return PolarFreeLiteUNet(in_channels=MODEL_INPUT_CHANNELS, base_channels=base_channels)


def train_pipeline(args: argparse.Namespace) -> None:
    device = prepare_device(require_cuda=True)
    train_root = Path(args.train_root)
    test_root = Path(args.test_root)
    train_prior_cache_dir = resolve_prior_cache_dir(train_root, args.prior_cache_dir)
    test_prior_cache_dir = resolve_prior_cache_dir(test_root, args.prior_cache_dir)
    train_package_cache_dir = resolve_package_cache_dir(train_root, args.package_cache_dir)
    test_package_cache_dir = resolve_package_cache_dir(test_root, args.package_cache_dir)
    ensure_runtime_cache_versions(train_prior_cache_dir, train_package_cache_dir)
    ensure_runtime_cache_versions(test_prior_cache_dir, test_package_cache_dir)
    base_save_dir = Path(args.save_dir)
    run_dir = create_run_dir(base_save_dir, prefix="train")
    write_latest_run_pointer(base_save_dir, run_dir)

    all_train_records = discover_dataset(train_root, split_name="train")
    train_records = limit_records(all_train_records, args.limit)
    train_records, hard_case_info = append_hard_cases(
        train_records,
        all_train_records,
        args.hard_case_manifest,
        args.hard_case_repeat,
    )
    test_limit = args.test_limit if args.test_limit > 0 else args.limit
    test_records = limit_records(discover_dataset(test_root, split_name="test"), test_limit)
    if not train_records:
        raise RuntimeError(f"No train records found under: {train_root}")
    if not test_records:
        raise RuntimeError(f"No test records found under: {test_root}")

    splits_dir = run_dir / "splits"
    write_manifest(train_records, train_root, splits_dir / "train_manifest.jsonl")
    write_manifest(test_records, test_root, splits_dir / "test_manifest.jsonl")

    train_dataset = PolarFreeLiteDataset(
        train_records,
        img_size=args.img_size,
        prior_cache_dir=train_prior_cache_dir,
        package_cache_dir=train_package_cache_dir,
        use_package_cache=args.package_cache,
        mask_profile=args.mask_profile,
        prior_profile=args.prior_profile,
        augment=True,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=device.type == "cuda",
    )
    test_loader: Optional[DataLoader] = None
    if args.eval_every > 0:
        test_dataset = PolarFreeLiteDataset(
            test_records,
            img_size=args.img_size,
            prior_cache_dir=test_prior_cache_dir,
            package_cache_dir=test_package_cache_dir,
            use_package_cache=args.package_cache,
            mask_profile=args.mask_profile,
            prior_profile=args.prior_profile,
            augment=False,
        )
        test_loader = make_loader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            prefetch_factor=args.prefetch_factor,
            pin_memory=device.type == "cuda",
        )

    model = build_model(args).to(device=device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    warm_start_path = ""
    warm_start_epoch = 0
    if args.weights:
        warm_start_path = str(Path(args.weights))
        try:
            warm_start_checkpoint = load_weights(model, Path(args.weights), device)
            warm_start_epoch = int(warm_start_checkpoint.get("epoch", 0) or 0) if isinstance(warm_start_checkpoint, dict) else 0
        except RuntimeError as exc:
            raise RuntimeError(
                "Warm start checkpoint is incompatible with the current model. "
                f"Expected MODEL_INPUT_CHANNELS={MODEL_INPUT_CHANNELS}; weights={args.weights}. "
                "Use a Showcase-Hard v2 31-channel / 16-head polarfree_lite_glass_v2 checkpoint."
            ) from exc
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    perceptual_model = build_perceptual_loss_model(device) if args.use_perceptual_loss else None
    history: List[Dict[str, float]] = []
    training_dir = run_dir / "training"
    checkpoints_dir = run_dir / "checkpoints"
    best_checkpoint_path = checkpoints_dir / "best_model.pth"
    best_raw_focus_checkpoint_path = checkpoints_dir / "best_raw_focus_model.pth"
    best_raw_psnr_checkpoint_path = checkpoints_dir / "best_raw_psnr_model.pth"
    best_raw_ssim_checkpoint_path = checkpoints_dir / "best_raw_ssim_model.pth"
    final_checkpoint_path = checkpoints_dir / "final_model.pth"
    best_validation_focus_l1: Optional[float] = None
    best_validation_epoch = 0
    best_raw_focus_l1: Optional[float] = None
    best_raw_focus_epoch = 0
    best_raw_psnr: Optional[float] = None
    best_raw_psnr_epoch = 0
    best_raw_ssim: Optional[float] = None
    best_raw_ssim_epoch = 0

    print(f"device={device} name={torch.cuda.get_device_name(0)}")
    print(f"run_dir={run_dir}")
    print(f"train_root={train_root}")
    print(f"test_root={test_root}")
    print(f"train_samples={len(train_records)} test_samples={len(test_records)}")
    print(
        f"preset={args.preset or 'none'} mask_profile={args.mask_profile} "
        f"loss_profile={args.loss_profile} prior_profile={args.prior_profile} "
        f"hard_case_manifest={args.hard_case_manifest or 'none'} "
        f"hard_case_matches={hard_case_info['hard_case_matches']} "
        f"hard_case_repeat={hard_case_info['hard_case_repeat']} "
        f"train_samples_after_hard_cases={hard_case_info['train_samples_after_hard_cases']}"
    )
    print(
        f"model={args.model} img_size={args.img_size} batch_size={args.batch_size} "
        f"grad_accum_steps={args.grad_accum_steps} epochs={args.epochs} render_mode={args.render_mode} "
        f"package_cache={args.package_cache}"
    )
    train_loader_config = loader_runtime_config(args.num_workers, args.prefetch_factor, args.batch_size, device.type == "cuda")
    eval_loader_config = loader_runtime_config(0, args.prefetch_factor, 1, device.type == "cuda")
    print(f"train_loader={train_loader_config}")
    print(f"eval_loader={eval_loader_config}")
    print(f"amp_enabled={amp_enabled} channels_last={args.channels_last} save_eval_images={args.save_eval_images}")
    print(
        f"showcase_mode={args.showcase_mode} visual_strength={args.visual_strength} "
        f"showcase_hard_mode={args.showcase_hard_mode} showcase_reflection_boost={args.showcase_reflection_boost} "
        f"support_sharpen={args.support_sharpen} "
        f"showcase_detail_strength={args.showcase_detail_strength} "
        f"showcase_smooth_strength={args.showcase_smooth_strength} "
        f"showcase_tint_suppress={args.showcase_tint_suppress} "
        f"direct_pretrain_epochs={args.direct_pretrain_epochs} "
        f"use_perceptual_loss={args.use_perceptual_loss} perceptual_active={perceptual_model is not None}"
    )
    print(f"warm_start={bool(warm_start_path)} weights={warm_start_path or 'none'} checkpoint_epoch={warm_start_epoch}")

    for epoch in range(1, args.epochs + 1):
        direct_pretrain = epoch <= max(args.direct_pretrain_epochs, 0)
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scaler=scaler,
            grad_accum_steps=args.grad_accum_steps,
            grad_clip=args.grad_clip,
            loss_profile=args.loss_profile,
            mask_profile=args.mask_profile,
            amp_enabled=amp_enabled,
            channels_last=args.channels_last,
            direct_pretrain=direct_pretrain,
            showcase_mode=args.showcase_mode,
            showcase_hard_mode=args.showcase_hard_mode,
            visual_strength=args.visual_strength,
            showcase_reflection_boost=args.showcase_reflection_boost,
            support_sharpen=args.support_sharpen,
            showcase_detail_strength=args.showcase_detail_strength,
            showcase_smooth_strength=args.showcase_smooth_strength,
            showcase_tint_suppress=args.showcase_tint_suppress,
            perceptual_model=perceptual_model,
        )
        entry = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_l1": train_metrics["l1"],
            "train_focus_l1": train_metrics["focus_l1"],
            "train_psnr": train_metrics["psnr"],
            "train_ssim": train_metrics["ssim"],
        }
        for key, value in sorted(train_metrics.items()):
            if key not in {"loss", "l1", "focus_l1", "psnr", "ssim"}:
                entry[f"train_{key}"] = value
        if test_loader is not None and epoch % args.eval_every == 0:
            test_metrics = run_epoch(
                model,
                test_loader,
                device,
                optimizer=None,
                scaler=None,
                loss_profile=args.loss_profile,
                mask_profile=args.mask_profile,
                amp_enabled=amp_enabled,
                channels_last=args.channels_last,
                direct_pretrain=False,
                showcase_mode=args.showcase_mode,
                showcase_hard_mode=args.showcase_hard_mode,
                visual_strength=args.visual_strength,
                showcase_reflection_boost=args.showcase_reflection_boost,
                support_sharpen=args.support_sharpen,
                showcase_detail_strength=args.showcase_detail_strength,
                showcase_smooth_strength=args.showcase_smooth_strength,
                showcase_tint_suppress=args.showcase_tint_suppress,
                perceptual_model=perceptual_model,
            )
            entry.update(
                {
                    "test_loss": test_metrics["loss"],
                    "test_l1": test_metrics["l1"],
                    "test_focus_l1": test_metrics["focus_l1"],
                    "test_psnr": test_metrics["psnr"],
                    "test_ssim": test_metrics["ssim"],
                }
            )
            for key, value in sorted(test_metrics.items()):
                if key not in {"loss", "l1", "focus_l1", "psnr", "ssim"}:
                    entry[f"test_{key}"] = value

        history.append(entry)
        save_history(history, training_dir)
        if "test_focus_l1" in entry and (
            best_validation_focus_l1 is None or entry["test_focus_l1"] < best_validation_focus_l1
        ):
            best_validation_focus_l1 = entry["test_focus_l1"]
            best_validation_epoch = epoch
            save_checkpoint(model, optimizer, history, epoch, best_checkpoint_path, args)
        if "test_focus_l1" in entry and (best_raw_focus_l1 is None or entry["test_focus_l1"] < best_raw_focus_l1):
            best_raw_focus_l1 = entry["test_focus_l1"]
            best_raw_focus_epoch = epoch
            save_checkpoint(model, optimizer, history, epoch, best_raw_focus_checkpoint_path, args)
        if "test_psnr" in entry and (best_raw_psnr is None or entry["test_psnr"] > best_raw_psnr):
            best_raw_psnr = entry["test_psnr"]
            best_raw_psnr_epoch = epoch
            save_checkpoint(model, optimizer, history, epoch, best_raw_psnr_checkpoint_path, args)
        if "test_ssim" in entry and (best_raw_ssim is None or entry["test_ssim"] > best_raw_ssim):
            best_raw_ssim = entry["test_ssim"]
            best_raw_ssim_epoch = epoch
            save_checkpoint(model, optimizer, history, epoch, best_raw_ssim_checkpoint_path, args)

        test_note = f" test_loss={entry['test_loss']:.4f}" if "test_loss" in entry else ""
        print(
            f"epoch={epoch:03d}/{args.epochs:03d} "
            f"phase={'direct_pretrain' if direct_pretrain else 'visual'} "
            f"train_loss={entry['train_loss']:.4f}{test_note} "
            f"train_psnr={entry['train_psnr']:.2f}"
        )

    save_checkpoint(model, optimizer, history, args.epochs, final_checkpoint_path, args)

    test_output_root = run_dir / "test_results"
    final_test_summary = evaluate_records(
        model=model,
        records=test_records,
        args=args,
        output_root=test_output_root,
        prior_cache_dir=test_prior_cache_dir,
        package_cache_dir=test_package_cache_dir,
        split_name="test",
        save_images=args.save_eval_images,
    )

    report = {
        "run_dir": str(run_dir),
        "train_root": str(train_root),
        "test_root": str(test_root),
        "train_prior_cache_dir": str(train_prior_cache_dir),
        "test_prior_cache_dir": str(test_prior_cache_dir),
        "train_package_cache_dir": str(train_package_cache_dir),
        "test_package_cache_dir": str(test_package_cache_dir),
        "train_summary": summarize_records(train_records),
        "test_summary": summarize_records(test_records),
        "mask_profile": args.mask_profile,
        "loss_profile": args.loss_profile,
        "prior_profile": args.prior_profile,
        "preset": args.preset,
        "amp_enabled": amp_enabled,
        "channels_last": args.channels_last,
        "save_eval_images": args.save_eval_images,
        "showcase_mode": args.showcase_mode,
        "showcase_hard_mode": args.showcase_hard_mode,
        "visual_strength": args.visual_strength,
        "showcase_reflection_boost": args.showcase_reflection_boost,
        "support_sharpen": args.support_sharpen,
        "showcase_detail_strength": args.showcase_detail_strength,
        "showcase_smooth_strength": args.showcase_smooth_strength,
        "showcase_tint_suppress": args.showcase_tint_suppress,
        "direct_pretrain_epochs": args.direct_pretrain_epochs,
        "use_perceptual_loss": args.use_perceptual_loss,
        "perceptual_active": perceptual_model is not None,
        "warm_start": bool(warm_start_path),
        "warm_start_weights": warm_start_path,
        "warm_start_epoch": warm_start_epoch,
        "hard_case_info": hard_case_info,
        "epochs_completed": args.epochs,
        "best_model": str(best_checkpoint_path),
        "best_model_selection": "lowest_test_focus_l1",
        "best_validation_focus_l1": best_validation_focus_l1,
        "best_validation_epoch": best_validation_epoch,
        "best_raw_focus_model": str(best_raw_focus_checkpoint_path),
        "best_raw_focus_epoch": best_raw_focus_epoch,
        "best_raw_focus_l1": best_raw_focus_l1,
        "best_raw_psnr_model": str(best_raw_psnr_checkpoint_path),
        "best_raw_psnr_epoch": best_raw_psnr_epoch,
        "best_raw_psnr": best_raw_psnr,
        "best_raw_ssim_model": str(best_raw_ssim_checkpoint_path),
        "best_raw_ssim_epoch": best_raw_ssim_epoch,
        "best_raw_ssim": best_raw_ssim,
        "final_model": str(final_checkpoint_path),
        "history_csv": str(training_dir / "history.csv"),
        "loss_curve": str(training_dir / "loss_curve.png"),
        "test_results": str(test_output_root),
        "final_test_summary": final_test_summary,
    }
    with (run_dir / "run_report.json").open("w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def choose_standalone_test_output_dir(args: argparse.Namespace, weights_path: Path) -> Path:
    save_dir = Path(args.save_dir)
    default_save = DEFAULT_SAVE_DIR.resolve()
    try:
        if save_dir.resolve() == default_save and weights_path.parent.name == "checkpoints":
            return weights_path.parent.parent / "test_results"
    except OSError:
        pass
    return create_run_dir(save_dir, prefix="test") / "test_results"


def resolve_weights_path(args: argparse.Namespace) -> Path:
    if args.weights:
        return Path(args.weights)
    latest_path = Path(args.save_dir) / "latest_run.txt"
    if latest_path.exists():
        run_dir = Path(latest_path.read_text(encoding="utf-8").strip())
        for name in ("best_model.pth", "final_model.pth"):
            candidate = run_dir / "checkpoints" / name
            if candidate.exists():
                return candidate
    raise ValueError("--weights is required when no latest_run.txt best_model.pth can be found.")


def test_pipeline(args: argparse.Namespace) -> None:
    device = prepare_device(require_cuda=False)
    weights_path = resolve_weights_path(args)
    if args.split == "train":
        data_root = Path(args.train_root)
        records = discover_dataset(data_root, split_name="train")
        split_name = "train"
    elif args.split == "all":
        data_root = Path(args.test_root)
        records = discover_dataset(Path(args.train_root), split_name="train") + discover_dataset(data_root, split_name="test")
        split_name = "all"
    else:
        data_root = Path(args.test_root)
        records = discover_dataset(data_root, split_name="test")
        split_name = "test"
    chosen = limit_records(records, args.limit)
    if not chosen:
        raise RuntimeError("No records matched the requested test split.")

    prior_cache_dir = resolve_prior_cache_dir(data_root, args.prior_cache_dir)
    package_cache_dir = resolve_package_cache_dir(data_root, args.package_cache_dir)
    ensure_runtime_cache_versions(prior_cache_dir, package_cache_dir)
    model = build_model(args).to(device=device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    checkpoint = load_weights(model, weights_path, device)
    output_root = choose_standalone_test_output_dir(args, weights_path)
    summary = evaluate_records(
        model=model,
        records=chosen,
        args=args,
        output_root=output_root,
        prior_cache_dir=prior_cache_dir,
        package_cache_dir=package_cache_dir,
        split_name=split_name,
        save_images=True,
    )
    summary["checkpoint_epoch"] = float(checkpoint.get("epoch", 0) or 0)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def infer_pipeline(args: argparse.Namespace) -> None:
    weights_path = resolve_weights_path(args)
    required = [args.input_000, args.input_045, args.input_090, args.input_135]
    if not all(required):
        raise ValueError("Infer mode requires explicit --input_000 --input_045 --input_090 --input_135 paths.")

    device = prepare_device(require_cuda=False)
    data_root = Path(args.data_root)
    prior_cache_dir = resolve_prior_cache_dir(data_root, args.prior_cache_dir)
    package_cache_dir = resolve_package_cache_dir(data_root, args.package_cache_dir)
    ensure_runtime_cache_versions(prior_cache_dir, package_cache_dir)
    sample = SampleRecord(
        split="infer",
        subset=args.scene or "manual",
        scene=args.scene or "manual",
        group=args.group or "sample",
        polar_paths={
            "0deg": Path(args.input_000),
            "45deg": Path(args.input_045),
            "90deg": Path(args.input_090),
            "135deg": Path(args.input_135),
        },
        input_rgb_path=Path(args.input_rgb) if args.input_rgb else None,
        gt_group="",
        gt_polar_paths={},
        gt_rgb_path=Path(args.target_rgb) if args.target_rgb else None,
        target_kind="target_rgb" if args.target_rgb else "none",
    )
    package = build_model_inputs(
        sample,
        img_size=args.img_size,
        prior_cache_dir=prior_cache_dir,
        package_cache_dir=package_cache_dir,
        use_package_cache=args.package_cache,
        mask_profile=args.mask_profile,
        prior_profile=args.prior_profile,
    )
    model = build_model(args).to(device=device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    load_weights(model, weights_path, device)
    model.eval()

    amp_enabled = bool(args.amp and device.type == "cuda")
    with torch.no_grad():
        infer_batch = batch_to_device(
            {
                "inputs": torch.from_numpy(package["inputs"]).unsqueeze(0),
                "base_prior": torch.from_numpy(package["base_prior"]).unsqueeze(0),
                "reference": torch.from_numpy(package["reference"]).unsqueeze(0),
                "mask": torch.from_numpy(package["mask"]).unsqueeze(0),
                "glare_mask": torch.from_numpy(package["glare_mask"]).unsqueeze(0),
                "reflection_area_mask": torch.from_numpy(package["reflection_area_mask"]).unsqueeze(0),
                "scene_reflection_mask": torch.from_numpy(package["scene_reflection_mask"]).unsqueeze(0),
                "dark_reflection_mask": torch.from_numpy(package["dark_reflection_mask"]).unsqueeze(0),
                "shadow_veil_mask": torch.from_numpy(package["shadow_veil_mask"]).unsqueeze(0),
                "lowfreq_reflection_mask": torch.from_numpy(package["lowfreq_reflection_mask"]).unsqueeze(0),
                "foreground_structure_guard": torch.from_numpy(package["foreground_structure_guard"]).unsqueeze(0),
            },
            device=device,
            channels_last=args.channels_last,
        )
        inputs = infer_batch["inputs"]
        base_prior = infer_batch["base_prior"]
        reference_tensor = infer_batch["reference"]
        mask_tensor = infer_batch["mask"]
        glare_mask_tensor = infer_batch["glare_mask"]
        reflection_area_mask_tensor = infer_batch["reflection_area_mask"]
        scene_reflection_mask_tensor = infer_batch["scene_reflection_mask"]
        dark_reflection_mask_tensor = infer_batch["dark_reflection_mask"]
        shadow_veil_mask_tensor = infer_batch["shadow_veil_mask"]
        lowfreq_reflection_mask_tensor = infer_batch["lowfreq_reflection_mask"]
        foreground_structure_guard_tensor = infer_batch["foreground_structure_guard"]
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            model_output = model(inputs)
            (
                direct_clean_tensor,
                _predicted_reflection,
                _signed_clean_delta,
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
                model_output,
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
                model_output,
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

    reference_rgb = tensor_to_numpy_rgb(torch.from_numpy(package["reference"]))
    prior_rgb = tensor_to_numpy_rgb(torch.from_numpy(package["base_prior"]))
    visual_prior_rgb = tensor_to_numpy_rgb(torch.from_numpy(package["visual_prior"]))
    raw_rgb = tensor_to_numpy_rgb(raw_prediction[0])
    direct_clean_rgb = tensor_to_numpy_rgb(direct_clean_tensor[0])
    target_rgb = tensor_to_numpy_rgb(torch.from_numpy(package["target"])) if args.target_rgb else None
    mask = tensor_to_numpy_mask(torch.from_numpy(package["mask"]))
    glare_mask = tensor_to_numpy_mask(torch.from_numpy(package["glare_mask"]))
    reflection_area_mask = tensor_to_numpy_mask(torch.from_numpy(package["reflection_area_mask"]))
    scene_reflection_mask = tensor_to_numpy_mask(torch.from_numpy(package["scene_reflection_mask"]))
    dark_reflection_mask = tensor_to_numpy_mask(torch.from_numpy(package["dark_reflection_mask"]))
    shadow_veil_mask = tensor_to_numpy_mask(torch.from_numpy(package["shadow_veil_mask"]))
    lowfreq_reflection_mask = tensor_to_numpy_mask(torch.from_numpy(package["lowfreq_reflection_mask"]))
    foreground_structure_guard = tensor_to_numpy_mask(torch.from_numpy(package["foreground_structure_guard"]))
    reflection_confidence = tensor_to_numpy_mask(reflection_confidence_tensor[0])
    line_band_confidence = tensor_to_numpy_mask(line_band_confidence_tensor[0])
    text_reflection_confidence = tensor_to_numpy_mask(text_reflection_confidence_tensor[0])
    tint_confidence = tensor_to_numpy_mask(tint_confidence_tensor[0])
    lowfreq_reflection_confidence = tensor_to_numpy_mask(lowfreq_reflection_confidence_tensor[0])
    dark_shadow_confidence = tensor_to_numpy_mask(dark_shadow_confidence_tensor[0])
    visual_rgb = render_prediction_output(
        reference_rgb,
        prior_rgb,
        visual_prior_rgb,
        raw_rgb,
        mask,
        args.render_mode,
        direct_clean_rgb=direct_clean_rgb,
        showcase_mode=args.showcase_mode,
        showcase_hard_mode=args.showcase_hard_mode,
        detail_strength=args.showcase_detail_strength,
        smooth_strength=args.showcase_smooth_strength,
        tint_suppress=args.showcase_tint_suppress,
    )
    raw_minus_prior = diff_heatmap(raw_rgb, prior_rgb)
    direct_clean_minus_prior = diff_heatmap(direct_clean_rgb, prior_rgb)
    visual_plus_minus_prior = diff_heatmap(visual_rgb, prior_rgb)

    infer_dir = create_run_dir(Path(args.save_dir), prefix="infer")
    output_info = save_test_sample_outputs(
        infer_dir,
        sample.scene,
        sample.group,
        reference_rgb,
        prior_rgb,
        raw_rgb,
        visual_rgb,
        target_rgb,
        mask,
        glare_mask,
        reflection_area_mask,
        direct_clean_rgb=direct_clean_rgb,
        raw_minus_prior=raw_minus_prior,
        direct_clean_minus_prior=direct_clean_minus_prior,
        visual_plus_minus_prior=visual_plus_minus_prior,
        scene_reflection_mask=scene_reflection_mask,
        dark_reflection_mask=dark_reflection_mask,
        shadow_veil_mask=shadow_veil_mask,
        lowfreq_reflection_mask=lowfreq_reflection_mask,
        foreground_structure_guard=foreground_structure_guard,
        reflection_confidence=reflection_confidence,
        line_band_confidence=line_band_confidence,
        text_reflection_confidence=text_reflection_confidence,
        tint_confidence=tint_confidence,
        lowfreq_reflection_confidence=lowfreq_reflection_confidence,
        dark_shadow_confidence=dark_shadow_confidence,
        showcase_mode=args.showcase_mode,
        showcase_hard_mode=args.showcase_hard_mode,
    )
    print(json.dumps(output_info, ensure_ascii=False, indent=2))


def inspect_pipeline(args: argparse.Namespace) -> None:
    train_records = discover_dataset(Path(args.train_root), split_name="train")
    test_records = discover_dataset(Path(args.test_root), split_name="test")
    summary = {
        "train_root": args.train_root,
        "test_root": args.test_root,
        "train": summarize_records(train_records),
        "test": summarize_records(test_records),
        "matching_rule": "same scene, one GT group supervises every input group, angle suffixes match",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def warm_cache_records(
    records: Sequence[SampleRecord],
    img_size: int,
    prior_cache_dir: Path,
    package_cache_dir: Optional[Path],
    use_package_cache: bool,
    mask_profile: str,
    prior_profile: str,
    num_workers: int,
    prefetch_factor: int,
    label: str,
) -> None:
    if not records:
        print(f"cache {label}: no records")
        return
    dataset = PolarFreeLiteDataset(
        records,
        img_size=img_size,
        prior_cache_dir=prior_cache_dir,
        package_cache_dir=package_cache_dir,
        use_package_cache=use_package_cache,
        mask_profile=mask_profile,
        prior_profile=prior_profile,
        augment=False,
    )
    loader = make_loader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=False,
    )
    started = datetime.now()
    print(
        f"cache {label}: records={len(records)} img_size={img_size} "
        f"mask_profile={mask_profile} prior_profile={prior_profile} "
        f"loader={loader_runtime_config(num_workers, prefetch_factor, 1, False)}"
    )
    for index, _batch in enumerate(loader, start=1):
        if index == 1 or index % 50 == 0 or index == len(records):
            elapsed = (datetime.now() - started).total_seconds()
            rate = index / max(elapsed, 1e-6)
            remaining = (len(records) - index) / max(rate, 1e-6)
            print(
                f"cache {label}: {index}/{len(records)} "
                f"elapsed={elapsed/60.0:.1f}m eta={remaining/60.0:.1f}m"
            )
    elapsed = (datetime.now() - started).total_seconds()
    print(f"cache {label}: done records={len(records)} elapsed={elapsed/60.0:.1f}m")


def cache_pipeline(args: argparse.Namespace) -> None:
    train_root = Path(args.train_root)
    test_root = Path(args.test_root)
    train_prior_cache_dir = resolve_prior_cache_dir(train_root, args.prior_cache_dir)
    test_prior_cache_dir = resolve_prior_cache_dir(test_root, args.prior_cache_dir)
    train_package_cache_dir = resolve_package_cache_dir(train_root, args.package_cache_dir)
    test_package_cache_dir = resolve_package_cache_dir(test_root, args.package_cache_dir)
    ensure_runtime_cache_versions(train_prior_cache_dir, train_package_cache_dir)
    ensure_runtime_cache_versions(test_prior_cache_dir, test_package_cache_dir)
    started = datetime.now()

    train_records = limit_records(discover_dataset(train_root, split_name="train"), args.limit)
    test_limit = args.test_limit if args.test_limit > 0 else args.limit
    test_records = limit_records(discover_dataset(test_root, split_name="test"), test_limit)
    print(
        f"cache_only train_root={train_root} test_root={test_root} "
        f"train_records={len(train_records)} test_records={len(test_records)} "
        f"package_cache={args.package_cache}"
    )
    warm_cache_records(
        train_records,
        img_size=args.img_size,
        prior_cache_dir=train_prior_cache_dir,
        package_cache_dir=train_package_cache_dir,
        use_package_cache=args.package_cache,
        mask_profile=args.mask_profile,
        prior_profile=args.prior_profile,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        label="train",
    )
    warm_cache_records(
        test_records,
        img_size=args.img_size,
        prior_cache_dir=test_prior_cache_dir,
        package_cache_dir=test_package_cache_dir,
        use_package_cache=args.package_cache,
        mask_profile=args.mask_profile,
        prior_profile=args.prior_profile,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        label="test",
    )
    elapsed = (datetime.now() - started).total_seconds()
    print(f"cache_only done elapsed={elapsed/60.0:.1f}m")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPU full-training PolarFree-Lite reflection removal")
    parser.add_argument("--mode", choices=["train", "test", "infer", "inspect", "cache"], required=True)
    parser.add_argument("--train_root", type=str, default=str(DEFAULT_TRAIN_ROOT))
    parser.add_argument("--test_root", type=str, default=str(DEFAULT_TEST_ROOT))
    parser.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--save_dir", type=str, default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--prior_cache_dir", type=str, default="")
    parser.add_argument("--package_cache_dir", type=str, default="")
    parser.add_argument("--package_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model", choices=["quality", "lite"], default="quality")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", "--grad-accum-steps", dest="grad_accum_steps", type=int, default=2)
    parser.add_argument("--grad_clip", "--grad-clip", dest="grad_clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eval_every", "--eval-every", dest="eval_every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", "--prefetch-factor", dest="prefetch_factor", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--channels_last", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_eval_images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--render-mode", choices=["raw", "hybrid", "visual", "visual_plus"], default="visual")
    parser.add_argument("--direct_pretrain_epochs", type=int, default=0)
    parser.add_argument("--visual_strength", type=float, default=1.0)
    parser.add_argument("--showcase_mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--showcase_hard_mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--showcase_reflection_boost", type=float, default=1.0)
    parser.add_argument("--support_sharpen", type=float, default=1.0)
    parser.add_argument("--showcase_detail_strength", type=float, default=0.10)
    parser.add_argument("--showcase_smooth_strength", type=float, default=0.14)
    parser.add_argument("--showcase_tint_suppress", type=float, default=0.12)
    parser.add_argument("--use_perceptual_loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mask_profile", choices=["stable", "balanced", "aggressive"], default="balanced")
    parser.add_argument("--loss_profile", choices=["stable", "balanced", "hard", "visual"], default="balanced")
    parser.add_argument("--prior_profile", choices=["stable", "visual"], default="stable")
    parser.add_argument("--hard_case_manifest", type=str, default="")
    parser.add_argument("--hard_case_repeat", type=int, default=2)
    parser.add_argument("--preset", choices=["fast_effect", "full_stable", "fast_reflection", "gpu_boost", "visual_showcase", "gpu_boost_visual"], default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--test_limit", "--test-limit", dest="test_limit", type=int, default=0)
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--scene", type=str, default="")
    parser.add_argument("--group", type=str, default="")
    parser.add_argument("--input_000", type=str, default="")
    parser.add_argument("--input_045", type=str, default="")
    parser.add_argument("--input_090", type=str, default="")
    parser.add_argument("--input_135", type=str, default="")
    parser.add_argument("--input_rgb", type=str, default="")
    parser.add_argument("--target_rgb", type=str, default="")
    return parser


def arg_was_provided(argv: Sequence[str], option_names: Sequence[str]) -> bool:
    for value in argv:
        for option_name in option_names:
            if value == option_name or value.startswith(f"{option_name}="):
                return True
    return False


def apply_preset(args: argparse.Namespace, argv: Sequence[str]) -> None:
    presets: Dict[str, Dict[str, object]] = {
        "fast_effect": {
            "img_size": 384,
            "batch_size": 1,
            "grad_accum_steps": 4,
            "epochs": 15,
            "lr": 8e-5,
            "eval_every": 3,
            "test_limit": 20,
            "num_workers": 2,
            "render_mode": "raw",
            "model": "quality",
            "mask_profile": "balanced",
            "loss_profile": "balanced",
        },
        "full_stable": {
            "img_size": 384,
            "batch_size": 1,
            "grad_accum_steps": 4,
            "epochs": 30,
            "lr": 6e-5,
            "eval_every": 5,
            "test_limit": 30,
            "num_workers": 2,
            "render_mode": "raw",
            "model": "quality",
            "mask_profile": "stable",
            "loss_profile": "stable",
        },
        "fast_reflection": {
            "img_size": 384,
            "batch_size": 1,
            "grad_accum_steps": 4,
            "epochs": 10,
            "lr": 8e-5,
            "eval_every": 3,
            "test_limit": 20,
            "num_workers": 0,
            "render_mode": "raw",
            "model": "quality",
            "mask_profile": "balanced",
            "loss_profile": "balanced",
        },
        "gpu_boost": {
            "img_size": 384,
            "batch_size": 2,
            "grad_accum_steps": 2,
            "epochs": 10,
            "lr": 8e-5,
            "eval_every": 5,
            "test_limit": 10,
            "num_workers": 2,
            "prefetch_factor": 2,
            "render_mode": "raw",
            "model": "quality",
            "mask_profile": "balanced",
            "loss_profile": "balanced",
        },
        "visual_showcase": {
            "img_size": 384,
            "batch_size": 1,
            "grad_accum_steps": 4,
            "epochs": 10,
            "lr": 4e-5,
            "eval_every": 2,
            "test_limit": 20,
            "num_workers": 0,
            "render_mode": "visual_plus",
            "model": "quality",
            "mask_profile": "balanced",
            "loss_profile": "visual",
            "prior_profile": "visual",
            "showcase_mode": True,
            "visual_strength": 1.35,
        },
        "gpu_boost_visual": {
            "img_size": 384,
            "batch_size": 2,
            "grad_accum_steps": 2,
            "epochs": 10,
            "lr": 4e-5,
            "eval_every": 2,
            "test_limit": 20,
            "num_workers": 2,
            "prefetch_factor": 2,
            "render_mode": "visual_plus",
            "model": "quality",
            "mask_profile": "balanced",
            "loss_profile": "visual",
            "prior_profile": "visual",
            "channels_last": True,
            "amp": True,
            "save_eval_images": False,
            "showcase_mode": True,
            "visual_strength": 1.35,
        },
    }
    if not args.preset:
        return
    preset_values = presets.get(args.preset)
    if preset_values is None:
        return
    aliases: Dict[str, Tuple[str, ...]] = {
        "img_size": ("--img_size",),
        "batch_size": ("--batch_size",),
        "grad_accum_steps": ("--grad_accum_steps", "--grad-accum-steps"),
        "epochs": ("--epochs",),
        "lr": ("--lr",),
        "eval_every": ("--eval_every", "--eval-every"),
        "test_limit": ("--test_limit", "--test-limit"),
        "num_workers": ("--num_workers",),
        "prefetch_factor": ("--prefetch_factor", "--prefetch-factor"),
        "render_mode": ("--render-mode",),
        "model": ("--model",),
        "mask_profile": ("--mask_profile",),
        "loss_profile": ("--loss_profile",),
        "prior_profile": ("--prior_profile",),
        "direct_pretrain_epochs": ("--direct_pretrain_epochs",),
        "visual_strength": ("--visual_strength",),
        "showcase_mode": ("--showcase_mode", "--no-showcase_mode"),
        "showcase_hard_mode": ("--showcase_hard_mode", "--no-showcase_hard_mode"),
        "showcase_reflection_boost": ("--showcase_reflection_boost",),
        "support_sharpen": ("--support_sharpen",),
        "showcase_detail_strength": ("--showcase_detail_strength",),
        "showcase_smooth_strength": ("--showcase_smooth_strength",),
        "showcase_tint_suppress": ("--showcase_tint_suppress",),
        "use_perceptual_loss": ("--use_perceptual_loss", "--no-use_perceptual_loss"),
        "channels_last": ("--channels_last", "--no-channels_last"),
        "amp": ("--amp", "--no-amp"),
        "save_eval_images": ("--save_eval_images", "--no-save_eval_images"),
    }
    for key, value in preset_values.items():
        if not arg_was_provided(argv, aliases.get(key, (f"--{key}",))):
            setattr(args, key, value)


def main_cli() -> None:
    parser = build_parser()
    args = parser.parse_args()
    apply_preset(args, sys.argv[1:])
    set_seed(args.seed)

    if args.mode == "train":
        train_pipeline(args)
    elif args.mode == "test":
        test_pipeline(args)
    elif args.mode == "cache":
        cache_pipeline(args)
    elif args.mode == "inspect":
        inspect_pipeline(args)
    else:
        infer_pipeline(args)


if __name__ == "__main__":
    main_cli()
