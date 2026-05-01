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
    parser.add_argument("--render-mode", choices=["raw", "hybrid", "visual", "visual_plus", "visual_extreme"], default="visual")
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

from pfl_data import (
    DatasetPart,
    PolarFreeLiteDataset,
    SampleRecord,
    append_hard_cases,
    apply_geometric_augmentation,
    build_line_glare_response,
    build_model_inputs,
    build_split_masks,
    discover_dataset,
    discover_dataset_parts,
    discover_group_files,
    ensure_cache_version,
    ensure_runtime_cache_versions,
    load_hard_case_entries,
    load_hard_case_keys,
    load_package_from_cache,
    load_prior_from_cache,
    load_or_build_prior_package,
    normalize_failure_types,
    normalize_hard_case_entry,
    normalize_loss_multipliers,
    normalize_mask,
    package_cache_path,
    prior_cache_path,
    read_input_rgb,
    read_target_image,
    resolve_package_cache_dir,
    resolve_prior_cache_dir,
    rgb_to_gray,
    safe_cache_component,
    safe_relative_path,
    save_package_to_cache,
    save_prior_to_cache,
    serialize_record,
    smooth_mask,
    write_manifest,
)

from pfl_model import (
    ConvBlock,
    PolarFreeLiteUNet,
    apply_showcase_visual_postprocess,
    build_model,
    compose_prediction,
    compute_blend_gate,
    compute_high_conf_reflection_mask,
    compute_visual_line_score_tensor,
    decode_reflection_heads,
    gaussian_blur_tensor,
    showcase_visual_polish,
    visual_strong_alpha_tensor,
)

from pfl_losses import (
    VGGPerceptualLoss,
    average_pool_ssim,
    build_perceptual_loss_model,
    charbonnier_error,
    charbonnier_loss,
    compute_direct_pretrain_loss,
    compute_laplacian_map,
    compute_total_loss,
    gradient_loss,
    high_frequency_texture_loss,
    laplacian_filter,
    line_reflection_suppression,
    low_frequency_l1,
    masked_chroma_loss,
    masked_laplacian_loss,
    prior_escape_loss,
    prior_guard_loss,
    weighted_charbonnier,
    weighted_l1,
)

from pfl_eval import (
    batch_to_device,
    blur_mask,
    build_glare_mask,
    compute_metrics,
    diff_heatmap,
    evaluate_records,
    gradient_strength_np,
    improvement_ratio_np,
    loader_runtime_config,
    make_loader,
    mask_to_rgb,
    masked_l1_np,
    masked_mean_np,
    render_prediction_output,
    save_debug_mask_panel,
    save_mask_gray,
    save_test_sample_outputs,
    tensor_to_numpy_mask,
    tensor_to_numpy_rgb,
    visual_strong_render_components,
    write_per_sample_metrics,
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
    parser.add_argument("--render-mode", choices=["raw", "hybrid", "visual", "visual_plus", "visual_extreme"], default="visual")
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
