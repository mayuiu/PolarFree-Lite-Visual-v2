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
from dataclasses import dataclass
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
    & "E:\ana\shell\condabin\conda-hook.ps1"
    conda
    activate
    pytorch
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import main


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
PRIOR_CACHE_VERSION = "quality_v2"
PACKAGE_CACHE_VERSION = "quality_tensor_v1"
MODEL_INPUT_CHANNELS = 24
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


def safe_cache_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return cleaned or "root"


def prior_cache_path(cache_dir: Path, img_size: int, record: SampleRecord) -> Path:
    safe_key = f"{safe_cache_component(record.group)}_{img_size}_{PRIOR_CACHE_VERSION}.npz"
    return (
        cache_dir
        / safe_cache_component(record.split)
        / safe_cache_component(record.subset)
        / safe_cache_component(record.scene)
        / safe_key
    )


def package_cache_path(cache_dir: Path, img_size: int, record: SampleRecord) -> Path:
    safe_key = f"{safe_cache_component(record.group)}_{img_size}_{PACKAGE_CACHE_VERSION}.npz"
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
            required = ("inputs", "base_prior", "visual_prior", "target", "reference", "input_rgb", "mask")
            return {key: data[key].astype(np.float32) for key in required}
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
            np.savez(
                fp,
                inputs=package["inputs"].astype(np.float16),
                base_prior=package["base_prior"].astype(np.float16),
                visual_prior=package["visual_prior"].astype(np.float16),
                target=package["target"].astype(np.float16),
                reference=package["reference"].astype(np.float16),
                input_rgb=package["input_rgb"].astype(np.float16),
                mask=package["mask"].astype(np.float16),
            )
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
) -> Dict[str, np.ndarray]:
    cache_path = prior_cache_path(cache_dir, img_size, record)
    cached = load_prior_from_cache(cache_path, expected_hw=(img_size, img_size))
    if cached is not None:
        return cached

    package = main.build_polarfree_prior(images, reference_view=main.REFERENCE_VIEW)
    save_prior_to_cache(cache_path, package)
    return {
        "prior_rgb": package["prior_rgb"],
        "visual_prior_rgb": package["visual_prior_rgb"],
        "reference_rgb": package["reference_rgb"],
        "reflection_mask": package["reflection_mask"],
        "dolp": package["dolp"],
        "polar_amplitude": package["polar_amplitude"],
    }


def build_model_inputs(
    record: SampleRecord,
    img_size: int,
    prior_cache_dir: Path,
    package_cache_dir: Optional[Path] = None,
    use_package_cache: bool = True,
) -> Dict[str, np.ndarray]:
    if use_package_cache and package_cache_dir is not None:
        cached_package = load_package_from_cache(package_cache_path(package_cache_dir, img_size, record), img_size)
        if cached_package is not None:
            return cached_package

    target_size = (img_size, img_size)
    images = main.load_polarized_images(record.polar_paths, target_size=target_size)
    prior_package = load_or_build_prior_package(record, images, img_size=img_size, cache_dir=prior_cache_dir)

    reference_rgb = images[main.REFERENCE_VIEW].astype(np.float32)
    input_rgb = read_input_rgb(record, images, target_size=target_size)
    base_prior = prior_package["prior_rgb"].astype(np.float32)
    visual_prior = prior_package["visual_prior_rgb"].astype(np.float32)
    reflection_mask = main.ensure_single_channel(prior_package["reflection_mask"]).astype(np.float32)
    dolp = main.ensure_single_channel(prior_package["dolp"]).astype(np.float32)
    polar_amplitude = main.ensure_single_channel(prior_package["polar_amplitude"]).astype(np.float32)

    polarized_rgb_channels = [images[angle].astype(np.float32) for angle in main.POLARIZATION_ORDER]
    input_tensor = np.concatenate(
        polarized_rgb_channels + [input_rgb, base_prior, visual_prior, dolp, polar_amplitude, reflection_mask],
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
        "mask": np.transpose(reflection_mask, (2, 0, 1)).astype(np.float32),
    }
    if use_package_cache and package_cache_dir is not None:
        save_package_to_cache(package_cache_path(package_cache_dir, img_size, record), package)
    return package


def apply_geometric_augmentation(package: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if random.random() < 0.5:
        for key in ("inputs", "base_prior", "visual_prior", "target", "reference", "input_rgb", "mask"):
            package[key] = np.flip(package[key], axis=2).copy()
    if random.random() < 0.5:
        for key in ("inputs", "base_prior", "visual_prior", "target", "reference", "input_rgb", "mask"):
            package[key] = np.flip(package[key], axis=1).copy()
    if random.random() < 0.25:
        for key in ("inputs", "base_prior", "visual_prior", "target", "reference", "input_rgb", "mask"):
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
        augment: bool = False,
    ) -> None:
        self.records = list(records)
        self.img_size = img_size
        self.prior_cache_dir = prior_cache_dir
        self.package_cache_dir = package_cache_dir
        self.use_package_cache = use_package_cache
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
        self.final = nn.Conv2d(base_channels, 4, kernel_size=1)

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


def compose_prediction(base_prior: torch.Tensor, model_output: torch.Tensor, reflection_mask: torch.Tensor) -> torch.Tensor:
    residual = torch.tanh(model_output[:, :3]) * 0.65
    learned_gate = torch.sigmoid(model_output[:, 3:4]) if model_output.shape[1] >= 4 else torch.ones_like(reflection_mask)
    prior_gate = 0.15 + 0.85 * torch.clamp(reflection_mask, 0.0, 1.0)
    residual_gate = torch.clamp(0.58 * prior_gate + 0.42 * learned_gate, 0.0, 1.0)
    return torch.clamp(base_prior + residual * residual_gate, 0.0, 1.0)


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


def compute_total_loss(prediction: torch.Tensor, target: torch.Tensor, reflection_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    full_l1 = charbonnier_loss(prediction, target)
    focus_l1 = weighted_charbonnier(prediction, target, torch.clamp(reflection_mask, 0.0, 1.0))
    ssim_loss = 1.0 - average_pool_ssim(prediction, target)
    grad = gradient_loss(prediction, target)
    texture = high_frequency_texture_loss(prediction, target, reflection_mask)
    bright_ghost = weighted_l1(torch.relu(prediction - target), torch.zeros_like(prediction), reflection_mask)
    total = 0.26 * full_l1 + 0.34 * focus_l1 + 0.12 * ssim_loss + 0.07 * grad + 0.13 * texture + 0.08 * bright_ghost
    return total, {
        "l1": float(full_l1.detach().cpu().item()),
        "focus_l1": float(focus_l1.detach().cpu().item()),
        "ssim_loss": float(ssim_loss.detach().cpu().item()),
        "grad_loss": float(grad.detach().cpu().item()),
        "texture_loss": float(texture.detach().cpu().item()),
        "bright_ghost": float(bright_ghost.detach().cpu().item()),
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
    low_sat = np.clip((0.58 - saturation) / 0.58, 0.0, 1.0)
    return blur_mask((bright * low_sat)[..., np.newaxis], sigma=5.0)


def render_prediction_output(
    reference_rgb: np.ndarray,
    prior_rgb: np.ndarray,
    visual_prior_rgb: np.ndarray,
    raw_prediction_rgb: np.ndarray,
    reflection_mask: np.ndarray,
    render_mode: str,
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

    if render_mode != "visual":
        raise ValueError(f"Unsupported render mode: {render_mode}")

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


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, prefetch_factor: int = 4) -> DataLoader:
    kwargs: Dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


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
        inputs = batch["inputs"].to(device, non_blocking=True, memory_format=torch.channels_last)
        base_prior = batch["base_prior"].to(device, non_blocking=True, memory_format=torch.channels_last)
        target = batch["target"].to(device, non_blocking=True, memory_format=torch.channels_last)
        mask = batch["mask"].to(device, non_blocking=True, memory_format=torch.channels_last)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            model_output = model(inputs)
            prediction = compose_prediction(base_prior, model_output, mask)
            loss, _loss_parts = compute_total_loss(prediction, target, mask)

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
        total_items += batch_size

    divisor = max(total_items, 1)
    return {key: value / divisor for key, value in totals.items()}


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
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{scene}_{group}"
    result_path = output_dir / f"{prefix}_result.png"
    raw_path = output_dir / f"{prefix}_raw_result.png"
    comparison_path = output_dir / f"{prefix}_comparison.png"
    mask_path = output_dir / f"{prefix}_mask.png"

    main.save_rgb_image(result_path, visual_rgb)
    main.save_rgb_image(raw_path, raw_rgb)
    main.save_mask_overlay(mask_path, reference_rgb, mask)

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
        title=f"scene={scene} group={group}",
    )
    return {
        "result_path": str(result_path),
        "raw_result_path": str(raw_path),
        "comparison_path": str(comparison_path),
        "mask_path": str(mask_path),
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
        augment=False,
    )
    loader = make_loader(dataset, batch_size=1, shuffle=False, num_workers=0, prefetch_factor=args.prefetch_factor)
    images_dir = output_root / "images"
    summary_path = output_root / "all_summary.json"

    totals: Dict[str, float] = {
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

    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device, non_blocking=True, memory_format=torch.channels_last)
            base_prior = batch["base_prior"].to(device, non_blocking=True, memory_format=torch.channels_last)
            target = batch["target"].to(device, non_blocking=True, memory_format=torch.channels_last)
            mask_tensor = batch["mask"].to(device, non_blocking=True, memory_format=torch.channels_last)

            residual = model(inputs)
            raw_prediction = compose_prediction(base_prior, residual, mask_tensor)
            raw_metrics = compute_metrics(raw_prediction, target, mask_tensor)

            scene = str(batch["scene"][0])
            group = str(batch["group"][0])
            reference_rgb = tensor_to_numpy_rgb(batch["reference"][0])
            prior_rgb = tensor_to_numpy_rgb(batch["base_prior"][0])
            visual_prior_rgb = tensor_to_numpy_rgb(batch["visual_prior"][0])
            raw_rgb = tensor_to_numpy_rgb(raw_prediction[0])
            target_rgb = tensor_to_numpy_rgb(batch["target"][0])
            mask = tensor_to_numpy_mask(batch["mask"][0])
            visual_rgb = render_prediction_output(
                reference_rgb,
                prior_rgb,
                visual_prior_rgb,
                raw_rgb,
                mask,
                render_mode=args.render_mode,
            )
            visual_tensor = torch.from_numpy(np.transpose(visual_rgb, (2, 0, 1))).unsqueeze(0).to(device=device, dtype=target.dtype)
            visual_metrics = compute_metrics(visual_tensor, target, mask_tensor)

            totals["raw_l1"] += raw_metrics["l1"]
            totals["raw_focus_l1"] += raw_metrics["focus_l1"]
            totals["raw_psnr"] += raw_metrics["psnr"]
            totals["raw_ssim"] += raw_metrics["ssim"]
            totals["visual_l1"] += visual_metrics["l1"]
            totals["visual_focus_l1"] += visual_metrics["focus_l1"]
            totals["visual_psnr"] += visual_metrics["psnr"]
            totals["visual_ssim"] += visual_metrics["ssim"]
            total_items += 1

            if save_images:
                sample_outputs.append(
                    save_test_sample_outputs(
                        images_dir,
                        scene,
                        group,
                        reference_rgb,
                        prior_rgb,
                        raw_rgb,
                        visual_rgb,
                        target_rgb,
                        mask,
                    )
                )

    divisor = max(total_items, 1)
    summary: Dict[str, object] = {
        "split": split_name,
        "count": total_items,
        "render_mode": args.render_mode,
        "img_size": args.img_size,
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
    base_save_dir = Path(args.save_dir)
    run_dir = create_run_dir(base_save_dir, prefix="train")
    write_latest_run_pointer(base_save_dir, run_dir)

    train_records = limit_records(discover_dataset(train_root, split_name="train"), args.limit)
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
        augment=True,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    test_loader: Optional[DataLoader] = None
    if args.eval_every > 0:
        test_dataset = PolarFreeLiteDataset(
            test_records,
            img_size=args.img_size,
            prior_cache_dir=test_prior_cache_dir,
            package_cache_dir=test_package_cache_dir,
            use_package_cache=args.package_cache,
            augment=False,
        )
        test_loader = make_loader(test_dataset, batch_size=1, shuffle=False, num_workers=0, prefetch_factor=args.prefetch_factor)

    model = build_model(args).to(device=device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history: List[Dict[str, float]] = []
    training_dir = run_dir / "training"
    checkpoints_dir = run_dir / "checkpoints"
    best_checkpoint_path = checkpoints_dir / "best_model.pth"
    final_checkpoint_path = checkpoints_dir / "final_model.pth"
    best_train_loss = float("inf")

    print(f"device={device} name={torch.cuda.get_device_name(0)}")
    print(f"run_dir={run_dir}")
    print(f"train_root={train_root}")
    print(f"test_root={test_root}")
    print(f"train_samples={len(train_records)} test_samples={len(test_records)}")
    print(
        f"model={args.model} img_size={args.img_size} batch_size={args.batch_size} "
        f"grad_accum_steps={args.grad_accum_steps} epochs={args.epochs} render_mode={args.render_mode} "
        f"package_cache={args.package_cache}"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scaler=scaler,
            grad_accum_steps=args.grad_accum_steps,
            grad_clip=args.grad_clip,
        )
        entry = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_l1": train_metrics["l1"],
            "train_focus_l1": train_metrics["focus_l1"],
            "train_psnr": train_metrics["psnr"],
            "train_ssim": train_metrics["ssim"],
        }
        if test_loader is not None and epoch % args.eval_every == 0:
            test_metrics = run_epoch(model, test_loader, device, optimizer=None, scaler=None)
            entry.update(
                {
                    "test_loss": test_metrics["loss"],
                    "test_l1": test_metrics["l1"],
                    "test_focus_l1": test_metrics["focus_l1"],
                    "test_psnr": test_metrics["psnr"],
                    "test_ssim": test_metrics["ssim"],
                }
            )

        history.append(entry)
        save_history(history, training_dir)
        if train_metrics["loss"] < best_train_loss:
            best_train_loss = train_metrics["loss"]
            save_checkpoint(model, optimizer, history, epoch, best_checkpoint_path, args)

        test_note = f" test_loss={entry['test_loss']:.4f}" if "test_loss" in entry else ""
        print(
            f"epoch={epoch:03d}/{args.epochs:03d} "
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
        save_images=True,
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
        "epochs_completed": args.epochs,
        "best_model": str(best_checkpoint_path),
        "best_model_selection": "lowest_train_loss",
        "best_train_loss": best_train_loss,
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
    model = build_model(args).to(device=device, memory_format=torch.channels_last)
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
    )
    model = build_model(args).to(device=device, memory_format=torch.channels_last)
    load_weights(model, weights_path, device)
    model.eval()

    with torch.no_grad():
        inputs = torch.from_numpy(package["inputs"]).unsqueeze(0).to(device, memory_format=torch.channels_last)
        base_prior = torch.from_numpy(package["base_prior"]).unsqueeze(0).to(device, memory_format=torch.channels_last)
        mask_tensor = torch.from_numpy(package["mask"]).unsqueeze(0).to(device, memory_format=torch.channels_last)
        raw_prediction = compose_prediction(base_prior, model(inputs), mask_tensor)

    reference_rgb = tensor_to_numpy_rgb(torch.from_numpy(package["reference"]))
    prior_rgb = tensor_to_numpy_rgb(torch.from_numpy(package["base_prior"]))
    visual_prior_rgb = tensor_to_numpy_rgb(torch.from_numpy(package["visual_prior"]))
    raw_rgb = tensor_to_numpy_rgb(raw_prediction[0])
    target_rgb = tensor_to_numpy_rgb(torch.from_numpy(package["target"])) if args.target_rgb else None
    mask = tensor_to_numpy_mask(torch.from_numpy(package["mask"]))
    visual_rgb = render_prediction_output(reference_rgb, prior_rgb, visual_prior_rgb, raw_rgb, mask, args.render_mode)

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPU full-training PolarFree-Lite reflection removal")
    parser.add_argument("--mode", choices=["train", "test", "infer", "inspect"], required=True)
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
    parser.add_argument("--render-mode", choices=["raw", "hybrid", "visual"], default="visual")
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


def main_cli() -> None:
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    if args.mode == "train":
        train_pipeline(args)
    elif args.mode == "test":
        test_pipeline(args)
    elif args.mode == "inspect":
        inspect_pipeline(args)
    else:
        infer_pipeline(args)


if __name__ == "__main__":
    main_cli()
