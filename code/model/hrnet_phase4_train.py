from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn.functional as F
import torchvision
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from hrnet_overfit32 import HRNetTipHeatmap, gaussian_heatmap, round_half_up


CHECKPOINT_WARNING = "PHASE 4 FORMAL BASELINE — model selection uses held-out Val mean pixel error"
PHASE3_WARNING = "Phase 3 overfit checkpoint was NOT loaded and is forbidden as initialization"


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path)
    config["_workspace_root"] = str(config_path.parent.parent)
    return config


def resolve_path(config: dict[str, Any], value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(config["_workspace_root"]) / path
    return path.resolve()


def config_for_save(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result.pop("_config_path", None)
    result.pop("_workspace_root", None)
    return result


def build_logger(path: Path, append: bool) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("hrnet_phase4")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, mode="a" if append else "w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)
    return logger


def configure_reproducibility(seed: int, deterministic: bool, benchmark: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = bool(benchmark)
    torch.backends.cudnn.deterministic = bool(deterministic)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_integrity(
    manifest: pd.DataFrame,
    config: dict[str, Any],
    manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train = manifest.loc[manifest["split"] == "train"].copy().reset_index(drop=True)
    val = manifest.loc[manifest["split"] == "val"].copy().reset_index(drop=True)
    exp = config["experiment"]
    actual_train = {
        "frames": int(len(train)),
        "cases": int(train["case_identifier"].nunique()),
        "series": int(train["series_uid"].nunique()),
    }
    actual_val = {
        "frames": int(len(val)),
        "cases": int(val["case_identifier"].nunique()),
        "series": int(val["series_uid"].nunique()),
    }
    expected_train = {
        "frames": int(exp["expected_train_frames"]),
        "cases": int(exp["expected_train_cases"]),
        "series": int(exp["expected_train_series"]),
    }
    expected_val = {
        "frames": int(exp["expected_val_frames"]),
        "cases": int(exp["expected_val_cases"]),
        "series": int(exp["expected_val_series"]),
    }
    val_cases = sorted(val["case_code"].astype(str).unique().tolist())
    expected_val_cases = sorted(str(value) for value in exp["expected_val_case_codes"])
    intersections = {}
    for column in ["case_identifier", "case_code", "series_uid", "dicom_absolute_path"]:
        left = set(train[column].astype(str))
        right = set(val[column].astype(str))
        intersections[column] = sorted(left & right)
    coordinate_finite = bool(
        np.isfinite(manifest[["true_tip_col", "true_tip_row"]].to_numpy(dtype=np.float64)).all()
    )
    coordinate_in_bounds = bool(
        manifest["true_tip_col"].between(0, 1023, inclusive="both").all()
        and manifest["true_tip_row"].between(0, 1023, inclusive="both").all()
    )
    duplicate_paths = int(manifest["dicom_absolute_path"].astype(str).duplicated().sum())
    passed = (
        actual_train == expected_train
        and actual_val == expected_val
        and val_cases == expected_val_cases
        and all(len(values) == 0 for values in intersections.values())
        and coordinate_finite
        and coordinate_in_bounds
        and duplicate_paths == 0
    )
    report = {
        "passed": passed,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "train": actual_train,
        "val": actual_val,
        "expected_train": expected_train,
        "expected_val": expected_val,
        "val_case_codes": val_cases,
        "expected_val_case_codes": expected_val_cases,
        "intersections": intersections,
        "duplicate_dicom_paths": duplicate_paths,
        "coordinates_finite": coordinate_finite,
        "coordinates_in_1024_bounds": coordinate_in_bounds,
    }
    if not passed:
        raise RuntimeError(f"Split integrity check failed:\n{json.dumps(report, indent=2)}")
    return train, val, report


class XAHeatmapDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        split_name: str,
        input_hw: tuple[int, int],
        heatmap_hw: tuple[int, int],
        sigma: float,
        channels: int,
        preload: bool,
        logger: logging.Logger,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True).copy()
        self.split_name = split_name
        self.input_hw = input_hw
        self.heatmap_hw = heatmap_hw
        self.sigma = float(sigma)
        self.channels = int(channels)
        if self.channels != 3:
            raise ValueError("HRNet baseline requires three replicated grayscale channels")
        self.cache: list[dict[str, Any] | None] = [None] * len(self.manifest)
        if preload:
            logger.info(f"[Dataset] Preloading {split_name}: {len(self.manifest)} DICOM frames...")
            for index in tqdm(
                range(len(self.manifest)),
                desc=f"[{split_name}] DICOM",
                unit="frame",
                dynamic_ncols=True,
            ):
                self.cache[index] = self._load_sample(index)

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_sample(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        path = str(row["dicom_absolute_path"])
        try:
            ds = pydicom.dcmread(path)
            pixels = np.asarray(ds.pixel_array)
        except Exception as exc:
            raise RuntimeError(f"Failed to decode DICOM: {path}\n{exc}") from exc
        if pixels.shape != (1024, 1024):
            raise ValueError(f"Expected 1024x1024 single-frame DICOM, got {pixels.shape}: {path}")
        if str(getattr(ds, "PhotometricInterpretation", "")) != "MONOCHROME2":
            raise ValueError(f"Expected MONOCHROME2: {path}")
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        if slope != 1.0 or intercept != 0.0:
            raise ValueError(
                f"Stored-pixel minmax refuses non-identity rescale: slope={slope}, "
                f"intercept={intercept}, path={path}"
            )
        image = torch.from_numpy(pixels.astype(np.float32, copy=False)).unsqueeze(0)
        image_min = float(image.min())
        image_max = float(image.max())
        if image_max <= image_min:
            raise ValueError(f"Constant image cannot be min-max normalized: {path}")
        image = (image - image_min) / (image_max - image_min)
        resized = F.interpolate(
            image.unsqueeze(0), size=self.input_hw, mode="bilinear", align_corners=False
        ).squeeze(0).contiguous()

        original_h, original_w = pixels.shape
        heatmap_h, heatmap_w = self.heatmap_hw
        original_x = float(row["true_tip_col"])
        original_y = float(row["true_tip_row"])
        heatmap_x = original_x * heatmap_w / original_w
        heatmap_y = original_y * heatmap_h / original_h
        quant_x = min(max(round_half_up(heatmap_x), 0), heatmap_w - 1)
        quant_y = min(max(round_half_up(heatmap_y), 0), heatmap_h - 1)
        target = gaussian_heatmap(
            heatmap_h, heatmap_w, heatmap_x, heatmap_y, self.sigma
        ).contiguous()
        return {
            "image_single": resized,
            "target_heatmap": target,
            "original_xy": torch.tensor([original_x, original_y], dtype=torch.float32),
            "heatmap_xy": torch.tensor([heatmap_x, heatmap_y], dtype=torch.float32),
            "gt_quant_xy": torch.tensor([quant_x, quant_y], dtype=torch.long),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "dicom_path": path,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.cache[index] if self.cache[index] is not None else self._load_sample(index)
        assert sample is not None
        result = dict(sample)
        result["image"] = sample["image_single"].repeat(self.channels, 1, 1)
        return result


def build_model(config: dict[str, Any], device: torch.device) -> nn.Module:
    model_cfg = config["model"]
    if bool(model_cfg["pretrained"]):
        raise ValueError("Phase 4 baseline is frozen to pretrained=False")
    if str(model_cfg["initialization"]) != "fresh_random":
        raise ValueError("Phase 4 must start from fresh_random initialization")
    return HRNetTipHeatmap(
        variant=str(model_cfg["variant"]),
        pretrained=False,
        output_channels=int(model_cfg["output_channels"]),
        expected_output_hw=tuple(int(v) for v in config["data"]["heatmap_size"]),
    ).to(device)


def create_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    training = config["training"]
    if str(training["optimizer"]).lower() != "adam":
        raise ValueError("Phase 4 baseline is frozen to Adam")
    return torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )


def create_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> Any:
    training = config["training"]
    if str(training["scheduler"]) != "MultiStepLR":
        raise ValueError("Phase 4 baseline is frozen to MultiStepLR")
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(v) for v in training["scheduler_milestones"]],
        gamma=float(training["scheduler_gamma"]),
    )


def build_criterion(config: dict[str, Any]) -> nn.Module:
    """Default plain MSE; Phase 5C replaces only this loss factory."""
    return nn.MSELoss()


def gradient_l2_norm(model: nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            grad = parameter.grad.detach()
            total += float(torch.sum(grad * grad))
    return math.sqrt(total)


def batch_predictions(
    outputs: torch.Tensor,
    original_xy: torch.Tensor,
    original_hw: tuple[int, int],
    heatmap_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch_size = outputs.shape[0]
    flat = outputs.detach().reshape(batch_size, -1)
    top_values, top_indices = torch.topk(flat, k=2, dim=1)
    pred_y = torch.div(top_indices[:, 0], heatmap_hw[1], rounding_mode="floor")
    pred_x = top_indices[:, 0] % heatmap_hw[1]
    pred_original_x = pred_x.float() * original_hw[1] / heatmap_hw[1]
    pred_original_y = pred_y.float() * original_hw[0] / heatmap_hw[0]
    errors = torch.sqrt(
        (pred_original_x - original_xy[:, 0]) ** 2
        + (pred_original_y - original_xy[:, 1]) ** 2
    )
    return (
        errors.cpu().numpy(),
        pred_x.cpu().numpy(),
        pred_y.cpu().numpy(),
        top_values[:, 0].cpu().numpy(),
        top_values[:, 1].cpu().numpy(),
    )


def summarize_errors(errors: list[float]) -> dict[str, float]:
    values = np.asarray(errors, dtype=np.float64)
    return {
        "mean_error_px": float(values.mean()),
        "median_error_px": float(np.median(values)),
        "p95_error_px": float(np.percentile(values, 95)),
        "max_error_px": float(values.max()),
        "success_5px": float(100.0 * np.mean(values < 5.0)),
        "success_10px": float(100.0 * np.mean(values < 10.0)),
        "success_20px": float(100.0 * np.mean(values < 20.0)),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    original_hw: tuple[int, int],
    heatmap_hw: tuple[int, int],
    epoch: int,
    max_epochs: int,
    gradient_explosion_norm: float,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    loss_weights: list[int] = []
    errors: list[float] = []
    gradient_norms: list[float] = []
    bar = tqdm(loader, desc=f"[Train {epoch:03d}/{max_epochs:03d}]", unit="batch", dynamic_ncols=True)
    for batch in bar:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target_heatmap"].to(device, non_blocking=True)
        originals = batch["original_xy"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if hasattr(criterion, "current_dicom_path"):
            criterion.current_dicom_path = str(batch["dicom_path"][0])
        outputs = model(images)
        loss = criterion(outputs, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite train loss at epoch {epoch}: {batch['dicom_path'][0]}"
            )
        loss.backward()
        grad_norm = gradient_l2_norm(model)
        if not math.isfinite(grad_norm):
            raise FloatingPointError(
                f"Non-finite gradient norm at epoch {epoch}: {batch['dicom_path'][0]}"
            )
        if grad_norm > gradient_explosion_norm:
            raise FloatingPointError(
                f"Gradient explosion at epoch {epoch}: norm={grad_norm:.3f}, "
                f"path={batch['dicom_path'][0]}"
            )
        optimizer.step()
        batch_errors, _, _, _, _ = batch_predictions(
            outputs, originals, original_hw, heatmap_hw
        )
        losses.append(float(loss.detach()))
        loss_weights.append(int(images.shape[0]))
        errors.extend(batch_errors.astype(float).tolist())
        gradient_norms.append(grad_norm)
        bar.set_postfix(
            loss=f"{losses[-1]:.3g}",
            mean_px=f"{np.mean(errors):.1f}",
            grad=f"{grad_norm:.2g}",
        )
    summary = summarize_errors(errors)
    summary.update(
        {
            "loss": float(np.average(losses, weights=loss_weights)),
            "gradient_norm": float(np.mean(gradient_norms)),
            "gradient_norm_max": float(np.max(gradient_norms)),
        }
    )
    return summary


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    manifest: pd.DataFrame,
    original_hw: tuple[int, int],
    heatmap_hw: tuple[int, int],
    epoch_label: str,
    collect_predictions: bool,
) -> tuple[dict[str, float], pd.DataFrame | None]:
    model.eval()
    losses: list[float] = []
    loss_weights: list[int] = []
    errors: list[float] = []
    peaks: list[float] = []
    second_peaks: list[float] = []
    output_means: list[float] = []
    output_stds: list[float] = []
    rows: list[dict[str, Any]] = []
    bar = tqdm(loader, desc=f"[Val {epoch_label}]", unit="batch", dynamic_ncols=True)
    with torch.no_grad():
        for batch in bar:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target_heatmap"].to(device, non_blocking=True)
            originals = batch["original_xy"].to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            if not torch.isfinite(loss) or not torch.isfinite(outputs).all():
                raise FloatingPointError(
                    f"Non-finite validation output/loss: {batch['dicom_path'][0]}"
                )
            batch_errors, pred_x, pred_y, peak, second = batch_predictions(
                outputs, originals, original_hw, heatmap_hw
            )
            losses.append(float(loss))
            loss_weights.append(int(images.shape[0]))
            errors.extend(batch_errors.astype(float).tolist())
            peaks.extend(peak.astype(float).tolist())
            second_peaks.extend(second.astype(float).tolist())
            output_means.extend(outputs.mean(dim=(1, 2, 3)).cpu().numpy().astype(float).tolist())
            output_stds.extend(outputs.std(dim=(1, 2, 3)).cpu().numpy().astype(float).tolist())
            if collect_predictions:
                sample_indices = batch["sample_index"].cpu().numpy().astype(int)
                originals_cpu = batch["original_xy"].cpu().numpy()
                for position, sample_index in enumerate(sample_indices):
                    source = manifest.iloc[sample_index]
                    px_h = int(pred_x[position])
                    py_h = int(pred_y[position])
                    rows.append(
                        {
                            "case_id": str(source["case_code"]),
                            "case_identifier": str(source["case_identifier"]),
                            "series_id": str(source["series_uid"]),
                            "sequence_identifier": str(source["sequence_identifier"]),
                            "dicom_path": str(source["dicom_absolute_path"]),
                            "frame_index": int(source["frame_index"]),
                            "gt_x": float(originals_cpu[position, 0]),
                            "gt_y": float(originals_cpu[position, 1]),
                            "pred_x_heatmap": px_h,
                            "pred_y_heatmap": py_h,
                            "pred_x_original": float(px_h * original_hw[1] / heatmap_hw[1]),
                            "pred_y_original": float(py_h * original_hw[0] / heatmap_hw[0]),
                            "error_px": float(batch_errors[position]),
                            "heatmap_peak_value": float(peak[position]),
                            "second_peak_value": float(second[position]),
                            "peak_margin": float(peak[position] - second[position]),
                        }
                    )
            bar.set_postfix(loss=f"{losses[-1]:.3g}", mean_px=f"{np.mean(errors):.1f}")
    summary = summarize_errors(errors)
    summary.update(
        {
            "loss": float(np.average(losses, weights=loss_weights)),
            "pred_heatmap_max_mean": float(np.mean(peaks)),
            "pred_heatmap_max_median": float(np.median(peaks)),
            "pred_heatmap_mean": float(np.mean(output_means)),
            "pred_heatmap_std": float(np.mean(output_stds)),
            "second_peak_mean": float(np.mean(second_peaks)),
            "peak_margin_mean": float(np.mean(np.asarray(peaks) - np.asarray(second_peaks))),
        }
    )
    return summary, pd.DataFrame.from_records(rows) if collect_predictions else None


def plumbing_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    logger.info("[Sanity] Running one Train batch forward/loss/backward/optimizer step...")
    batch = next(iter(loader))
    images = batch["image"].to(device)
    targets = batch["target_heatmap"].to(device)
    optimizer = create_optimizer(model, config)
    criterion = build_criterion(config)
    if hasattr(criterion, "current_dicom_path"):
        criterion.current_dicom_path = str(batch["dicom_path"][0])
    before = next(model.parameters()).detach().clone()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(images)
    loss = criterion(outputs, targets)
    finite = bool(torch.isfinite(outputs).all() and torch.isfinite(loss))
    if not finite:
        raise FloatingPointError(f"Sanity forward/loss is non-finite: {batch['dicom_path'][0]}")
    loss.backward()
    grad_norm = gradient_l2_norm(model)
    if not math.isfinite(grad_norm) or grad_norm <= 0:
        raise FloatingPointError(f"Sanity gradient invalid: {grad_norm}")
    optimizer.step()
    delta = float(torch.max(torch.abs(next(model.parameters()).detach() - before)))
    if delta <= 0:
        raise RuntimeError("Sanity optimizer step did not change parameters")
    report = {
        "image_shape": list(images.shape),
        "target_shape": list(targets.shape),
        "output_shape": list(outputs.shape),
        "image_min": float(images.min()),
        "image_max": float(images.max()),
        "target_min": float(targets.min()),
        "target_max": float(targets.max()),
        "output_min": float(outputs.min()),
        "output_max": float(outputs.max()),
        "loss": float(loss),
        "nan": False,
        "inf": False,
        "gradient_available": True,
        "gradient_l2_norm": grad_norm,
        "optimizer_parameter_delta": delta,
        "first_dicom_path": str(batch["dicom_path"][0]),
    }
    for key in [
        "image_shape", "target_shape", "output_shape", "image_min", "image_max",
        "target_min", "target_max", "output_min", "output_max", "loss",
        "gradient_l2_norm", "optimizer_parameter_delta",
    ]:
        logger.info(f"[Sanity] {key}: {report[key]}")
    logger.info("[PASS] Forward / finite Loss / Backward / Optimizer step")
    return report


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    record: dict[str, Any],
    config: dict[str, Any],
    state: dict[str, Any],
    generator: torch.Generator,
    kind: str,
) -> None:
    torch.save(
        {
            "warning": CHECKPOINT_WARNING,
            "phase3_checkpoint_loaded": False,
            "kind": kind,
            "epoch": int(epoch),
            "model_variant": config["model"]["variant"],
            "pretrained": False,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": record,
            "training_state": state,
            "train_generator_state": generator.get_state(),
            "config": config_for_save(config),
        },
        path,
    )


def aggregate_predictions(predictions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        errors = group["error_px"].to_numpy(dtype=np.float64)
        row = {column: value for column, value in zip(group_columns, keys)}
        row.update(
            {
                "n": int(len(group)),
                "mean_error_px": float(errors.mean()),
                "median_error_px": float(np.median(errors)),
                "p95_error_px": float(np.percentile(errors, 95)),
                "max_error_px": float(errors.max()),
                "success_5px": float(100.0 * np.mean(errors < 5.0)),
                "success_10px": float(100.0 * np.mean(errors < 10.0)),
                "success_20px": float(100.0 * np.mean(errors < 20.0)),
                "heatmap_peak_mean": float(group["heatmap_peak_value"].mean()),
                "peak_margin_mean": float(group["peak_margin"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def environment_text(device: torch.device) -> str:
    lines = [
        f"Python: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        f"PyTorch: {torch.__version__}",
        f"torchvision: {torchvision.__version__}",
        f"timm: {timm.__version__}",
        f"pydicom: {pydicom.__version__}",
        f"CUDA runtime reported by PyTorch: {torch.version.cuda}",
        f"Device selected: {device}",
    ]
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        lines.extend(
            [
                f"CUDA device name: {torch.cuda.get_device_name(0)}",
                f"CUDA device capability: {torch.cuda.get_device_capability(0)}",
                f"CUDA memory total MiB: {total / 1024**2:.1f}",
                f"CUDA memory available MiB at startup: {free / 1024**2:.1f}",
            ]
        )
    lines.extend([PHASE3_WARNING, "Mixed precision: disabled (continuity with Phase 3)"])
    return "\n".join(lines) + "\n"


def save_model_summary(path: Path, model: nn.Module, config: dict[str, Any], device: torch.device) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    text = "\n".join(
        [
            "Phase 4 HRNet formal baseline model summary",
            "=" * 72,
            CHECKPOINT_WARNING,
            PHASE3_WARNING,
            f"Variant: {config['model']['variant']}",
            "Pretrained: False",
            "Initialization: fresh random",
            f"Device: {device}",
            f"Trainable parameters: {trainable:,}",
            f"Total parameters: {total:,}",
            "Input: [B, 3, 512, 512]",
            "Output: [B, 1, 128, 128]",
            "",
            repr(model),
        ]
    )
    path.write_text(text, encoding="utf-8")


def print_header(
    logger: logging.Logger,
    config: dict[str, Any],
    device: torch.device,
    train: pd.DataFrame,
    val: pd.DataFrame,
    train_iterations: int,
    val_iterations: int,
) -> None:
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    logger.info("=" * 62)
    logger.info("Phase 4 HRNet Formal Baseline")
    logger.info("=" * 62)
    logger.info(f"Device: {device_name}")
    logger.info(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        logger.info(f"GPU memory: available {free / 1024**2:.0f} / total {total / 1024**2:.0f} MiB")
    logger.info(f"Model: {config['model']['variant']} | Pretrained: False | fresh random init")
    logger.info(
        f"Train: {len(train)} frames / {train['case_identifier'].nunique()} cases / "
        f"{train['series_uid'].nunique()} Series"
    )
    logger.info(
        f"Val: {len(val)} frames / {val['case_identifier'].nunique()} cases / "
        f"{val['series_uid'].nunique()} Series"
    )
    logger.info("Input: 512x512 | Heatmap: 128x128 | sigma=2 | channels=3")
    logger.info(f"Batch size: {config['data']['batch_size']}")
    logger.info(f"Train iterations/epoch: {train_iterations}")
    logger.info(f"Val iterations/epoch: {val_iterations}")
    logger.info(
        f"Optimizer: Adam | LR={config['training']['learning_rate']} | "
        f"weight_decay={config['training']['weight_decay']} | Loss=MSE"
    )
    logger.info(
        f"Scheduler: MultiStepLR milestones={config['training']['scheduler_milestones']} "
        f"gamma={config['training']['scheduler_gamma']}"
    )
    logger.info(
        f"Early stopping: Val mean error, patience="
        f"{config['training']['early_stopping']['patience']}, "
        f"min_delta={config['training']['early_stopping']['min_delta_px']}px"
    )
    logger.info(PHASE3_WARNING)
    logger.info("=" * 62)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 formal HRNet Train/Val baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = resolve_path(config, config["project"]["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = output_dir / "training_log.csv"
    if training_log_path.exists() and not args.resume:
        raise RuntimeError(
            f"Existing Phase 4 training log found: {training_log_path}. "
            "Use --resume to continue; refusing to overwrite a formal run."
        )
    logger = build_logger(log_dir / "phase4_run.log", append=args.resume)

    seed = int(config["experiment"]["seed"])
    configure_reproducibility(
        seed,
        bool(config["training"]["deterministic"]),
        bool(config["training"]["cudnn_benchmark"]),
    )
    device = select_device()
    if device.type != "cuda":
        logger.warning(f"[Device] CUDA unavailable; selected {device}")
    manifest_path = resolve_path(config, config["project"]["manifest"])
    manifest = pd.read_csv(
        manifest_path,
        dtype={"case_code": str, "case_identifier": str},
        low_memory=False,
    )
    train_frames, val_frames, integrity = split_integrity(manifest, config, manifest_path)
    (output_dir / "split_integrity.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("[Integrity] PASS: Train/Val case, Series, and DICOM path intersections are empty.")
    logger.info(f"[Integrity] Val cases: {', '.join(integrity['val_case_codes'])}")

    config_saved = config_for_save(config)
    with (output_dir / "config_used.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_saved, handle, sort_keys=False, allow_unicode=True)
    (output_dir / "environment.txt").write_text(environment_text(device), encoding="utf-8")

    input_hw = tuple(int(v) for v in config["data"]["input_size"])
    heatmap_hw = tuple(int(v) for v in config["data"]["heatmap_size"])
    original_hw = tuple(int(v) for v in config["data"]["original_size"])
    train_dataset = XAHeatmapDataset(
        train_frames,
        "Train",
        input_hw,
        heatmap_hw,
        float(config["data"]["sigma"]),
        int(config["data"]["grayscale_channels"]),
        bool(config["data"]["preload_images"]),
        logger,
    )
    val_dataset = XAHeatmapDataset(
        val_frames,
        "Val",
        input_hw,
        heatmap_hw,
        float(config["data"]["sigma"]),
        int(config["data"]["grayscale_channels"]),
        bool(config["data"]["preload_images"]),
        logger,
    )
    batch_size = int(config["data"]["batch_size"])
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": int(config["data"]["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=train_generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    print_header(
        logger,
        config,
        device,
        train_frames,
        val_frames,
        len(train_loader),
        len(val_loader),
    )
    if len(train_loader) != math.ceil(len(train_frames) / batch_size):
        raise RuntimeError("Unexpected Train iteration count")

    if not args.resume:
        probe = build_model(config, device)
        sanity = plumbing_test(probe, train_loader, device, config, logger)
        (output_dir / "batch_sanity.json").write_text(
            json.dumps(sanity, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        save_model_summary(output_dir / "model_summary.txt", probe, config, device)
        del probe
        if device.type == "cuda":
            torch.cuda.empty_cache()
        configure_reproducibility(
            seed,
            bool(config["training"]["deterministic"]),
            bool(config["training"]["cudnn_benchmark"]),
        )
        train_generator.manual_seed(seed)

    model = build_model(config, device)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    criterion = build_criterion(config)
    max_epochs = int(config["training"]["max_epochs"])
    early_cfg = config["training"]["early_stopping"]
    collapse_cfg = config["training"]["collapse_monitor"]
    state: dict[str, Any] = {
        "best_val_mean": float("inf"),
        "best_val_mean_epoch": 0,
        "best_val_median": float("inf"),
        "best_val_median_epoch": 0,
        "best_val_mse": float("inf"),
        "best_val_mse_epoch": 0,
        "early_reference": float("inf"),
        "early_counter": 0,
        "collapse_counter": 0,
    }
    log_rows: list[dict[str, Any]] = []
    start_epoch = 1
    if args.resume:
        last_path = checkpoint_dir / "last.pth"
        if not last_path.exists():
            raise FileNotFoundError(f"Cannot resume; missing {last_path}")
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if bool(checkpoint.get("phase3_checkpoint_loaded", True)):
            raise RuntimeError("Resume checkpoint provenance is invalid")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        state.update(checkpoint["training_state"])
        # A checkpoint loaded with map_location=CUDA moves this CPU DataLoader
        # generator state to CUDA as well.  Move only the RNG-state tensor back
        # to CPU; the byte values (and therefore the sample order) are unchanged.
        train_generator.set_state(checkpoint["train_generator_state"].cpu())
        start_epoch = int(checkpoint["epoch"]) + 1
        log_rows = pd.read_csv(training_log_path).to_dict("records")
        logger.info(f"[Resume] Continuing from epoch {start_epoch}; Phase 3 checkpoint remains unused.")

    stop_reason = "max_epochs_completed"
    run_started = time.time()
    for epoch in range(start_epoch, max_epochs + 1):
        epoch_started = time.time()
        lr = float(optimizer.param_groups[0]["lr"])
        logger.info("")
        logger.info("-" * 62)
        logger.info(f"Epoch {epoch:03d} / {max_epochs:03d} | LR={lr:.3e}")
        try:
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                original_hw,
                heatmap_hw,
                epoch,
                max_epochs,
                float(config["training"]["gradient_explosion_norm"]),
            )
            val_metrics, _ = validate(
                model,
                val_loader,
                criterion,
                device,
                val_frames,
                original_hw,
                heatmap_hw,
                f"{epoch:03d}/{max_epochs:03d}",
                False,
            )
        except Exception as exc:
            diagnostic = {
                "epoch": epoch,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "phase3_checkpoint_loaded": False,
            }
            (output_dir / "training_failure.json").write_text(
                json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.exception("[STOP] Numerical/data failure; diagnostics saved.")
            raise

        is_best = val_metrics["mean_error_px"] < float(state["best_val_mean"])
        is_best_median = val_metrics["median_error_px"] < float(state["best_val_median"])
        is_best_mse = val_metrics["loss"] < float(state["best_val_mse"])
        if is_best:
            state["best_val_mean"] = val_metrics["mean_error_px"]
            state["best_val_mean_epoch"] = epoch
        if is_best_median:
            state["best_val_median"] = val_metrics["median_error_px"]
            state["best_val_median_epoch"] = epoch
        if is_best_mse:
            state["best_val_mse"] = val_metrics["loss"]
            state["best_val_mse_epoch"] = epoch

        meaningful_improvement = (
            val_metrics["mean_error_px"]
            < float(state["early_reference"]) - float(early_cfg["min_delta_px"])
        )
        if meaningful_improvement:
            state["early_reference"] = val_metrics["mean_error_px"]
            state["early_counter"] = 0
        else:
            state["early_counter"] = int(state["early_counter"]) + 1

        collapsed = (
            val_metrics["pred_heatmap_max_mean"] < float(collapse_cfg["max_mean_below"])
            and val_metrics["pred_heatmap_std"] < float(collapse_cfg["heatmap_std_below"])
        )
        state["collapse_counter"] = int(state["collapse_counter"]) + 1 if collapsed else 0
        record = {
            "epoch": epoch,
            "learning_rate": lr,
            "train_loss": train_metrics["loss"],
            "train_mean_error_px": train_metrics["mean_error_px"],
            "train_median_error_px": train_metrics["median_error_px"],
            "train_p95_error_px": train_metrics["p95_error_px"],
            "val_loss": val_metrics["loss"],
            "val_mean_error_px": val_metrics["mean_error_px"],
            "val_median_error_px": val_metrics["median_error_px"],
            "val_p95_error_px": val_metrics["p95_error_px"],
            "val_max_error_px": val_metrics["max_error_px"],
            "val_success_5px": val_metrics["success_5px"],
            "val_success_10px": val_metrics["success_10px"],
            "val_success_20px": val_metrics["success_20px"],
            "pred_heatmap_max_mean": val_metrics["pred_heatmap_max_mean"],
            "pred_heatmap_max_median": val_metrics["pred_heatmap_max_median"],
            "pred_heatmap_mean": val_metrics["pred_heatmap_mean"],
            "pred_heatmap_std": val_metrics["pred_heatmap_std"],
            "peak_margin_mean": val_metrics["peak_margin_mean"],
            "gradient_norm": train_metrics["gradient_norm"],
            "gradient_norm_max": train_metrics["gradient_norm_max"],
            "is_best": bool(is_best),
            "early_stop_counter": int(state["early_counter"]),
            "collapse_counter": int(state["collapse_counter"]),
            "epoch_seconds": float(time.time() - epoch_started),
        }
        # Optional detached diagnostics. The plain-MSE Phase 4/5A path returns
        # no extras and is unchanged; Phase 5C records both loss scales and
        # foreground/background contributions without altering model selection.
        record.update(train_metrics.get("_epoch_extra", {}))
        record.update(val_metrics.get("_epoch_extra", {}))
        log_rows.append(record)
        pd.DataFrame.from_records(log_rows).to_csv(
            training_log_path, index=False, encoding="utf-8-sig"
        )
        scheduler.step()

        if is_best:
            save_checkpoint(
                checkpoint_dir / "best_by_val_mean_error.pth",
                epoch,
                model,
                optimizer,
                scheduler,
                record,
                config,
                state,
                train_generator,
                "BEST_BY_VAL_MEAN_ERROR",
            )
        if is_best_median and bool(config["checkpointing"]["save_best_by_val_median"]):
            save_checkpoint(
                checkpoint_dir / "best_by_val_median.pth",
                epoch,
                model,
                optimizer,
                scheduler,
                record,
                config,
                state,
                train_generator,
                "BEST_BY_VAL_MEDIAN",
            )
        if is_best_mse:
            save_checkpoint(
                checkpoint_dir / "best_by_val_mse.pth",
                epoch,
                model,
                optimizer,
                scheduler,
                record,
                config,
                state,
                train_generator,
                "BEST_BY_VAL_MSE",
            )
        save_checkpoint(
            checkpoint_dir / "last.pth",
            epoch,
            model,
            optimizer,
            scheduler,
            record,
            config,
            state,
            train_generator,
            "LAST",
        )

        logger.info(
            f"TRAIN | MSE={train_metrics['loss']:.8f} | mean={train_metrics['mean_error_px']:.3f}px | "
            f"median={train_metrics['median_error_px']:.3f}px | grad={train_metrics['gradient_norm']:.4f}"
        )
        logger.info(
            f"VAL   | MSE={val_metrics['loss']:.8f} | mean={val_metrics['mean_error_px']:.3f}px | "
            f"median={val_metrics['median_error_px']:.3f}px | P95={val_metrics['p95_error_px']:.3f}px | "
            f"max={val_metrics['max_error_px']:.3f}px"
        )
        logger.info(
            f"VAL   | <5px={val_metrics['success_5px']:.2f}% | <10px={val_metrics['success_10px']:.2f}% | "
            f"<20px={val_metrics['success_20px']:.2f}% | peak_mean={val_metrics['pred_heatmap_max_mean']:.4f} | "
            f"map_std={val_metrics['pred_heatmap_std']:.5f}"
        )
        logger.info(
            f"BEST  | Val mean={state['best_val_mean']:.3f}px @ epoch {state['best_val_mean_epoch']} | "
            f"early_stop={state['early_counter']}/{early_cfg['patience']} | "
            f"collapse={state['collapse_counter']}/{collapse_cfg['patience_epochs']} | "
            f"epoch_time={record['epoch_seconds']:.1f}s"
        )

        if int(state["collapse_counter"]) >= int(collapse_cfg["patience_epochs"]):
            stop_reason = "near_zero_heatmap_collapse"
            logger.error("[STOP] Near-zero heatmap collapse persisted for configured patience.")
            break
        if int(state["early_counter"]) >= int(early_cfg["patience"]):
            stop_reason = "early_stopping_val_mean_error"
            logger.info("[Early Stop] Val mean error did not improve meaningfully within patience.")
            break

    if not training_log_path.exists():
        raise RuntimeError("Training ended without any completed epoch")
    training_log = pd.read_csv(training_log_path)
    best_path = checkpoint_dir / "best_by_val_mean_error.pth"
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    if bool(best_checkpoint.get("phase3_checkpoint_loaded", True)):
        raise RuntimeError("Best checkpoint provenance check failed")
    model.load_state_dict(best_checkpoint["model_state_dict"])
    best_val_metrics, val_predictions = validate(
        model,
        val_loader,
        criterion,
        device,
        val_frames,
        original_hw,
        heatmap_hw,
        f"best@{best_checkpoint['epoch']}",
        True,
    )
    assert val_predictions is not None
    val_predictions.insert(0, "best_epoch", int(best_checkpoint["epoch"]))
    val_predictions.to_csv(
        output_dir / "val_predictions_best.csv", index=False, encoding="utf-8-sig"
    )
    by_case = aggregate_predictions(val_predictions, ["case_id"])
    by_series = aggregate_predictions(
        val_predictions, ["case_id", "series_id", "sequence_identifier"]
    )
    by_case.to_csv(output_dir / "val_metrics_by_case.csv", index=False, encoding="utf-8-sig")
    by_series.to_csv(output_dir / "val_metrics_by_series.csv", index=False, encoding="utf-8-sig")
    worst_count = int(config["visualization"]["worst_count"])
    worst = val_predictions.nlargest(worst_count, "error_px").copy().reset_index(drop=True)
    worst.insert(0, "worst_rank", np.arange(1, len(worst) + 1))
    worst["objective_flag"] = np.where(
        worst["peak_margin"] < 0.01,
        "low_peak_margin",
        np.where(worst["heatmap_peak_value"] > 0.5, "confident_large_error", "unclear"),
    )
    worst["manual_visual_assessment"] = "unclear"
    worst.to_csv(output_dir / "worst_20_predictions.csv", index=False, encoding="utf-8-sig")

    best_row = training_log.loc[
        training_log["epoch"] == int(best_checkpoint["epoch"])
    ].iloc[0]
    summary = {
        "status": "completed" if stop_reason != "near_zero_heatmap_collapse" else "stopped_diagnostic",
        "stop_reason": stop_reason,
        "epochs_completed": int(training_log["epoch"].max()),
        "best_epoch": int(best_checkpoint["epoch"]),
        "primary_checkpoint": str(best_path),
        "phase3_checkpoint_loaded": False,
        "formal_training_initialization": "fresh_random",
        "train_at_best": {
            "loss": float(best_row["train_loss"]),
            "mean_error_px": float(best_row["train_mean_error_px"]),
            "median_error_px": float(best_row["train_median_error_px"]),
        },
        "val_at_best": best_val_metrics,
        "best_checkpoint_record": best_checkpoint["metrics"],
        "elapsed_seconds_this_process": float(time.time() - run_started),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("")
    logger.info("[Training Complete]")
    logger.info(f"Stop reason: {stop_reason}")
    logger.info(f"Epochs completed: {summary['epochs_completed']}")
    logger.info(f"Primary checkpoint: BEST_BY_VAL_MEAN_ERROR @ epoch {summary['best_epoch']}")
    logger.info(
        f"Best Val: mean={best_val_metrics['mean_error_px']:.3f}px | "
        f"median={best_val_metrics['median_error_px']:.3f}px | "
        f"P95={best_val_metrics['p95_error_px']:.3f}px | max={best_val_metrics['max_error_px']:.3f}px"
    )
    logger.info(PHASE3_WARNING)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[Stopped] User interrupted Phase 4. Resume with --resume.", file=sys.stderr)
        raise
