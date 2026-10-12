from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DEBUG_CHECKPOINT_WARNING = "NOT FOR FORMAL EVALUATION — 32-image debug/sanity overfit checkpoint"


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


def build_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("hrnet_phase3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)
    return logger


def configure_reproducibility(seed: int, deterministic: bool, cudnn_benchmark: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    torch.backends.cudnn.deterministic = bool(deterministic)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_or_load_subset(
    train_frames: pd.DataFrame,
    manifest_path: Path,
    subset_size: int,
    seed: int,
) -> pd.DataFrame:
    if manifest_path.exists():
        subset = pd.read_csv(manifest_path, dtype={"case_code": str}, low_memory=False)
        if len(subset) != subset_size:
            raise RuntimeError(f"Existing overfit manifest has {len(subset)} rows, expected {subset_size}")
        known_paths = set(train_frames["dicom_absolute_path"].astype(str))
        unknown = [path for path in subset["dicom_absolute_path"].astype(str) if path not in known_paths]
        if unknown:
            raise RuntimeError(f"Existing overfit manifest contains a non-Train path: {unknown[0]}")
        return subset.sort_values("overfit_sample_index").reset_index(drop=True)

    if subset_size < 2 * train_frames["series_uid"].nunique():
        raise ValueError("subset_size must allow at least two samples per Train Series")
    rng = np.random.default_rng(seed)
    selected: dict[str, dict[str, Any]] = {}
    grouped = list(train_frames.groupby("series_uid", sort=True))
    for series_uid, group in grouped:
        group = group.sort_values("frame_index").reset_index(drop=True)
        for fraction, reason in ((0.20, "series_temporal_20pct"), (0.80, "series_temporal_80pct")):
            index = int(round(fraction * (len(group) - 1)))
            record = group.iloc[index].to_dict()
            record["selection_reason"] = reason
            selected[str(record["dicom_absolute_path"])] = record

    remaining = subset_size - len(selected)
    series_order = np.arange(len(grouped))
    rng.shuffle(series_order)
    extra_series = series_order[:remaining]
    for series_index in extra_series:
        _, group = grouped[int(series_index)]
        group = group.sort_values("frame_index").reset_index(drop=True)
        candidates = group[~group["dicom_absolute_path"].astype(str).isin(selected)].copy()
        target = int(rng.integers(0, len(candidates)))
        record = candidates.iloc[target].to_dict()
        record["selection_reason"] = "seeded_extra_series_diversity"
        selected[str(record["dicom_absolute_path"])] = record

    subset = pd.DataFrame.from_records(list(selected.values()))
    if len(subset) != subset_size:
        raise RuntimeError(f"Sampling produced {len(subset)} rows, expected {subset_size}")
    subset = subset.sort_values(["case_code", "sequence_identifier", "frame_index"]).reset_index(drop=True)
    subset.insert(0, "overfit_sample_index", np.arange(len(subset), dtype=int))
    subset["selection_seed"] = seed
    subset["purpose"] = "32-image plumbing/overfit sanity only; not generalization evaluation"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return subset


def gaussian_heatmap(height: int, width: int, center_x: float, center_y: float, sigma: float) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    return torch.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma**2)).unsqueeze(0)


def round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


class XAOverfitDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        input_hw: tuple[int, int],
        heatmap_hw: tuple[int, int],
        sigma: float,
        channels: int,
        preload: bool,
        logger: logging.Logger,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True).copy()
        self.input_hw = input_hw
        self.heatmap_hw = heatmap_hw
        self.sigma = float(sigma)
        self.channels = int(channels)
        if self.channels != 3:
            raise ValueError("Phase 3 HRNet input is fixed to three replicated grayscale channels")
        self.cache: list[dict[str, Any] | None] = [None] * len(self.manifest)
        if preload:
            logger.info(f"[Dataset] Preloading and preprocessing {len(self.manifest)} DICOM frames...")
            for index in tqdm(range(len(self.manifest)), desc="[Dataset] DICOM", unit="frame", dynamic_ncols=True):
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
                f"Phase 3 stored-pixel minmax refuses non-identity rescale: "
                f"slope={slope}, intercept={intercept}, path={path}"
            )
        image = torch.from_numpy(pixels.astype(np.float32, copy=False)).unsqueeze(0)
        image_min = float(torch.min(image))
        image_max = float(torch.max(image))
        if image_max <= image_min:
            raise ValueError(f"Constant image cannot be min-max normalized: {path}")
        image = (image - image_min) / (image_max - image_min)
        resized = F.interpolate(
            image.unsqueeze(0), size=self.input_hw, mode="bilinear", align_corners=False
        ).squeeze(0)

        original_h, original_w = pixels.shape
        input_h, input_w = self.input_hw
        heatmap_h, heatmap_w = self.heatmap_hw
        original_x = float(row["true_tip_col"])
        original_y = float(row["true_tip_row"])
        input_x = original_x * input_w / original_w
        input_y = original_y * input_h / original_h
        heatmap_x = original_x * heatmap_w / original_w
        heatmap_y = original_y * heatmap_h / original_h
        quant_x = min(max(round_half_up(heatmap_x), 0), heatmap_w - 1)
        quant_y = min(max(round_half_up(heatmap_y), 0), heatmap_h - 1)
        heatmap = gaussian_heatmap(heatmap_h, heatmap_w, heatmap_x, heatmap_y, self.sigma)
        return {
            "image_single": resized.contiguous(),
            "target_heatmap": heatmap.contiguous(),
            "original_xy": torch.tensor([original_x, original_y], dtype=torch.float32),
            "input_xy": torch.tensor([input_x, input_y], dtype=torch.float32),
            "heatmap_xy": torch.tensor([heatmap_x, heatmap_y], dtype=torch.float32),
            "gt_quant_xy": torch.tensor([quant_x, quant_y], dtype=torch.long),
            "image_min_stored": torch.tensor(image_min, dtype=torch.float32),
            "image_max_stored": torch.tensor(image_max, dtype=torch.float32),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "dicom_path": path,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.cache[index] if self.cache[index] is not None else self._load_sample(index)
        assert sample is not None
        result = dict(sample)
        result["image"] = sample["image_single"].repeat(self.channels, 1, 1)
        return result


class HRNetTipHeatmap(nn.Module):
    def __init__(self, variant: str, pretrained: bool, output_channels: int, expected_output_hw: tuple[int, int]) -> None:
        super().__init__()
        self.variant = variant
        self.expected_output_hw = expected_output_hw
        self.backbone = timm.create_model(variant, pretrained=pretrained)
        high_resolution_channels = int(self.backbone.stage4[-1].num_in_chs[0])
        # Remove timm's classification-only aggregation modules. The full HRNet
        # stage1→stage4 multi-resolution fusion remains intact.
        self.backbone.incre_modules = None
        self.backbone.downsamp_modules = None
        self.backbone.final_layer = None
        self.backbone.global_pool = nn.Identity()
        self.backbone.classifier = nn.Identity()
        self.heatmap_head = nn.Conv2d(high_resolution_channels, output_channels, kernel_size=1)
        nn.init.normal_(self.heatmap_head.weight, std=0.001)
        nn.init.zeros_(self.heatmap_head.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.backbone.act1(self.backbone.bn1(self.backbone.conv1(image)))
        x = self.backbone.act2(self.backbone.bn2(self.backbone.conv2(x)))
        branches = self.backbone.stages(x)
        output = self.heatmap_head(branches[0])
        if tuple(output.shape[-2:]) != tuple(self.expected_output_hw):
            raise RuntimeError(
                f"HRNet output shape {tuple(output.shape)} does not match expected heatmap "
                f"size {self.expected_output_hw}"
            )
        return output


def create_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    training = config["training"]
    name = str(training["optimizer"]).lower()
    if name != "adam":
        raise ValueError("Phase 3 configuration currently supports the requested Adam optimizer only")
    return torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )


def create_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> Any:
    training = config["training"]
    if str(training["scheduler"]) != "MultiStepLR":
        raise ValueError("Phase 3 configuration currently supports MultiStepLR only")
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(value) for value in training["scheduler_milestones"]],
        gamma=float(training["scheduler_gamma"]),
    )


def build_model(config: dict[str, Any], device: torch.device) -> HRNetTipHeatmap:
    model_config = config["model"]
    heatmap_hw = tuple(int(value) for value in config["data"]["heatmap_size"])
    model = HRNetTipHeatmap(
        variant=str(model_config["variant"]),
        pretrained=bool(model_config["pretrained"]),
        output_channels=int(model_config["output_channels"]),
        expected_output_hw=heatmap_hw,
    )
    return model.to(device)


def plumbing_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    logger.info("")
    logger.info("[Plumbing] Running one-batch forward/loss/backward/optimizer step...")
    model.train()
    optimizer = create_optimizer(model, config)
    criterion = nn.MSELoss()
    batch = next(iter(loader))
    images = batch["image"].to(device, non_blocking=True)
    targets = batch["target_heatmap"].to(device, non_blocking=True)
    gt_coordinates = batch["original_xy"]
    logger.info(f"image tensor shape : {list(images.shape)}")
    logger.info(f"image min/max      : {float(images.min()):.6f} / {float(images.max()):.6f}")
    logger.info(f"GT coordinate[0]   : x={float(gt_coordinates[0,0]):.4f}, y={float(gt_coordinates[0,1]):.4f}")
    logger.info(f"GT heatmap shape   : {list(targets.shape)}")
    logger.info(f"GT heatmap min/max : {float(targets.min()):.6f} / {float(targets.max()):.6f}")

    optimizer.zero_grad(set_to_none=True)
    outputs = model(images)
    if tuple(outputs.shape) != tuple(targets.shape) or not torch.isfinite(outputs).all():
        raise RuntimeError(f"Forward failed: output={tuple(outputs.shape)}, target={tuple(targets.shape)}")
    logger.info(f"model output shape : {list(outputs.shape)}")
    logger.info(f"model output min/max: {float(outputs.min()):.6f} / {float(outputs.max()):.6f}")
    logger.info("[PASS] Forward")

    loss = criterion(outputs, targets)
    logger.info(f"loss                : {float(loss):.8f}")
    if not torch.isfinite(loss):
        raise RuntimeError("Loss is NaN or Inf")
    logger.info("[PASS] Loss")

    tracked_parameter = model.heatmap_head.weight
    before = tracked_parameter.detach().clone()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
    if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("Backward produced missing, NaN or Inf gradients")
    grad_norm = math.sqrt(sum(float(torch.sum(gradient.detach() ** 2)) for gradient in gradients))
    if grad_norm <= 0:
        raise RuntimeError("Gradient norm is zero")
    logger.info(f"gradient L2 norm    : {grad_norm:.8f}")
    logger.info("[PASS] Backward")

    optimizer.step()
    parameter_delta = float(torch.max(torch.abs(tracked_parameter.detach() - before)))
    if parameter_delta <= 0:
        raise RuntimeError("Optimizer did not update the tracked parameter")
    logger.info(f"max parameter delta : {parameter_delta:.10f}")
    logger.info("[PASS] Optimizer step")
    return {
        "image_shape": list(images.shape),
        "target_shape": list(targets.shape),
        "output_shape": list(outputs.shape),
        "loss": float(loss.detach()),
        "gradient_l2_norm": grad_norm,
        "max_parameter_delta": parameter_delta,
        "passed": True,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    manifest: pd.DataFrame,
    original_hw: tuple[int, int],
    heatmap_hw: tuple[int, int],
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    model.eval()
    criterion = nn.MSELoss(reduction="mean")
    weighted_loss = 0.0
    total = 0
    predictions_by_index: dict[int, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    original_h, original_w = original_hw
    heatmap_h, heatmap_w = heatmap_hw
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target_heatmap"].to(device, non_blocking=True)
        outputs = model(images)
        if not torch.isfinite(outputs).all():
            raise RuntimeError("Evaluation output contains NaN or Inf")
        batch_size = images.shape[0]
        loss = criterion(outputs, targets)
        weighted_loss += float(loss) * batch_size
        total += batch_size
        flat_indices = outputs.flatten(1).argmax(dim=1)
        pred_y = torch.div(flat_indices, heatmap_w, rounding_mode="floor").cpu().numpy()
        pred_x = (flat_indices % heatmap_w).cpu().numpy()
        original_xy = batch["original_xy"].cpu().numpy()
        gt_quant_xy = batch["gt_quant_xy"].cpu().numpy()
        sample_indices = batch["sample_index"].cpu().numpy()
        output_cpu = outputs.detach().cpu().numpy()
        for item in range(batch_size):
            sample_index = int(sample_indices[item])
            px_h = int(pred_x[item])
            py_h = int(pred_y[item])
            px_original = px_h * original_w / heatmap_w
            py_original = py_h * original_h / heatmap_h
            gt_x, gt_y = map(float, original_xy[item])
            original_error = float(np.hypot(px_original - gt_x, py_original - gt_y))
            quant_error = float(
                np.hypot(px_h - int(gt_quant_xy[item, 0]), py_h - int(gt_quant_xy[item, 1]))
            )
            frame = manifest.iloc[sample_index]
            rows.append(
                {
                    "overfit_sample_index": sample_index,
                    "case_code": frame["case_code"],
                    "series_uid": frame["series_uid"],
                    "sequence_identifier": frame["sequence_identifier"],
                    "frame_index": int(frame["frame_index"]),
                    "dicom_absolute_path": frame["dicom_absolute_path"],
                    "gt_original_x": gt_x,
                    "gt_original_y": gt_y,
                    "gt_heatmap_x_float": float(batch["heatmap_xy"][item, 0]),
                    "gt_heatmap_y_float": float(batch["heatmap_xy"][item, 1]),
                    "gt_quantized_heatmap_x": int(gt_quant_xy[item, 0]),
                    "gt_quantized_heatmap_y": int(gt_quant_xy[item, 1]),
                    "pred_heatmap_x": px_h,
                    "pred_heatmap_y": py_h,
                    "pred_original_x": px_original,
                    "pred_original_y": py_original,
                    "original_error_px": original_error,
                    "heatmap_quantized_error_px": quant_error,
                    "pred_heatmap_min": float(output_cpu[item].min()),
                    "pred_heatmap_max": float(output_cpu[item].max()),
                    "pred_peak_value": float(output_cpu[item, 0, py_h, px_h]),
                }
            )
            predictions_by_index[sample_index] = output_cpu[item, 0]
    per_frame = pd.DataFrame.from_records(rows).sort_values("overfit_sample_index").reset_index(drop=True)
    predictions = np.stack([predictions_by_index[index] for index in range(len(manifest))])
    original_errors = per_frame["original_error_px"].to_numpy(dtype=np.float64)
    quant_errors = per_frame["heatmap_quantized_error_px"].to_numpy(dtype=np.float64)
    metrics = {
        "loss": weighted_loss / total,
        "mean_original_error_px": float(np.mean(original_errors)),
        "median_original_error_px": float(np.median(original_errors)),
        "p95_original_error_px": float(np.percentile(original_errors, 95)),
        "max_original_error_px": float(np.max(original_errors)),
        "mean_heatmap_quantized_error": float(np.mean(quant_errors)),
        "median_heatmap_quantized_error": float(np.median(quant_errors)),
        "p95_heatmap_quantized_error": float(np.percentile(quant_errors, 95)),
        "max_heatmap_quantized_error": float(np.max(quant_errors)),
    }
    return metrics, per_frame, predictions


def selected_indices_by_series(manifest: pd.DataFrame, count: int) -> list[int]:
    selected: list[int] = []
    grouped = list(manifest.groupby("series_uid", sort=True))
    while len(selected) < count:
        added = False
        for _, group in grouped:
            candidate_position = len(selected) // max(1, len(grouped))
            candidate_position = min(candidate_position, len(group) - 1)
            index = int(group.iloc[candidate_position]["overfit_sample_index"])
            if index not in selected:
                selected.append(index)
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
    return selected


def save_model_summary(
    path: Path,
    model: nn.Module,
    config: dict[str, Any],
    device: torch.device,
    plumbing: dict[str, Any],
) -> None:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    text = "\n".join(
        [
            "Phase 3 HRNet model summary",
            "=" * 72,
            DEBUG_CHECKPOINT_WARNING,
            f"Python: {platform.python_version()}",
            f"PyTorch: {torch.__version__}",
            f"torchvision: {__import__('torchvision').__version__}",
            f"timm: {timm.__version__}",
            f"Device: {device} ({device_name})",
            f"Variant: {config['model']['variant']}",
            f"Pretrained: {config['model']['pretrained']}",
            f"Trainable parameters: {trainable:,}",
            f"Total parameters: {total:,}",
            f"Plumbing input shape: {plumbing['image_shape']}",
            f"Plumbing target shape: {plumbing['target_shape']}",
            f"Plumbing output shape: {plumbing['output_shape']}",
            "",
            repr(model),
        ]
    )
    path.write_text(text, encoding="utf-8")


def plot_curves(training_log: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(training_log["epoch"], training_log["loss"], label="Evaluation MSE on fixed 32")
    ax.plot(training_log["epoch"], training_log["train_batch_loss"], label="Mean batch loss", alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (log scale)")
    ax.set_title("HRNet 32-image overfit loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "train_mse_loss.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    axes[0].plot(training_log["epoch"], training_log["mean_original_error_px"], label="Mean")
    axes[0].plot(training_log["epoch"], training_log["median_original_error_px"], label="Median")
    axes[0].set_ylabel("Original-space error (pixels)")
    axes[0].set_title("Original 1024-space localization error")
    axes[1].plot(training_log["epoch"], training_log["mean_heatmap_quantized_error"], label="Mean")
    axes[1].plot(training_log["epoch"], training_log["median_heatmap_quantized_error"], label="Median")
    axes[1].set_ylabel("Error (128 heatmap pixels)")
    axes[1].set_title("Quantized-target heatmap error")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "train_localization_error.png", dpi=180)
    plt.close(fig)


def plot_progression(
    dataset: XAOverfitDataset,
    indices: list[int],
    progression: dict[int, np.ndarray],
    figures_dir: Path,
) -> None:
    epochs = sorted(progression)
    fig, axes = plt.subplots(len(indices), len(epochs) + 1, figsize=(3.0 * (len(epochs) + 1), 2.8 * len(indices)), squeeze=False)
    for row, index in enumerate(indices):
        sample = dataset[index]
        gt = sample["target_heatmap"][0].numpy()
        axes[row, 0].imshow(gt, cmap="magma", vmin=0, vmax=1)
        axes[row, 0].set_title("GT" if row == 0 else "")
        gt_quant = sample["gt_quant_xy"].numpy()
        axes[row, 0].scatter([gt_quant[0]], [gt_quant[1]], s=30, facecolors="none", edgecolors="cyan")
        for column, epoch in enumerate(epochs, 1):
            pred = progression[epoch][row]
            py, px = np.unravel_index(int(np.argmax(pred)), pred.shape)
            axes[row, column].imshow(pred, cmap="magma", vmin=0, vmax=1)
            axes[row, column].scatter([px], [py], s=30, facecolors="none", edgecolors="lime")
            axes[row, column].set_title(f"Epoch {epoch}" if row == 0 else "")
        frame = dataset.manifest.iloc[index]
        axes[row, 0].set_ylabel(f"{frame['case_code']}/{frame['sequence_identifier']}\nf={int(frame['frame_index'])}")
        for column in range(len(epochs) + 1):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    fig.suptitle("Predicted heatmap learning progression", y=0.999)
    fig.tight_layout()
    fig.savefig(figures_dir / "heatmap_learning_progression.png", dpi=180)
    plt.close(fig)


def plot_final_overlays(
    dataset: XAOverfitDataset,
    indices: list[int],
    final_metrics: pd.DataFrame,
    figures_dir: Path,
) -> None:
    ncols = 4
    nrows = math.ceil(len(indices) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows), squeeze=False)
    metric_by_index = final_metrics.set_index("overfit_sample_index")
    heatmap_h, heatmap_w = dataset.heatmap_hw
    input_h, input_w = dataset.input_hw
    for panel, index in enumerate(indices):
        row, column = divmod(panel, ncols)
        sample = dataset[index]
        metric = metric_by_index.loc[index]
        image = sample["image_single"][0].numpy()
        gt_x, gt_y = sample["input_xy"].numpy()
        pred_x = float(metric["pred_heatmap_x"]) * input_w / heatmap_w
        pred_y = float(metric["pred_heatmap_y"]) * input_h / heatmap_h
        axes[row, column].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row, column].scatter([gt_x], [gt_y], s=70, facecolors="none", edgecolors="red", linewidths=1.4, label="GT")
        axes[row, column].scatter([pred_x], [pred_y], s=42, facecolors="none", edgecolors="lime", linewidths=1.3, label="Pred")
        frame = dataset.manifest.iloc[index]
        axes[row, column].set_title(
            f"{frame['case_code']}/{frame['sequence_identifier']} f={int(frame['frame_index'])}\n"
            f"orig err={float(metric['original_error_px']):.2f}px, hm err={float(metric['heatmap_quantized_error_px']):.2f}"
        )
        axes[row, column].set_axis_off()
    for panel in range(len(indices), nrows * ncols):
        row, column = divmod(panel, ncols)
        axes[row, column].set_axis_off()
    handles = [
        plt.Line2D([], [], color="red", marker="o", linestyle="None", markerfacecolor="none", label="GT"),
        plt.Line2D([], [], color="lime", marker="o", linestyle="None", markerfacecolor="none", label="Prediction"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2)
    fig.suptitle("Final debug-overfit overlays — GT red, prediction green", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(figures_dir / "final_overfit_overlays.png", dpi=180)
    plt.close(fig)


def plot_heatmap_pairs(
    dataset: XAOverfitDataset,
    indices: list[int],
    predictions: np.ndarray,
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(len(indices), 2, figsize=(6.5, 2.8 * len(indices)), squeeze=False)
    for row, index in enumerate(indices):
        gt = dataset[index]["target_heatmap"][0].numpy()
        pred = predictions[index]
        axes[row, 0].imshow(gt, cmap="magma", vmin=0, vmax=1)
        axes[row, 1].imshow(pred, cmap="magma", vmin=0, vmax=1)
        frame = dataset.manifest.iloc[index]
        axes[row, 0].set_ylabel(f"{frame['case_code']}/{frame['sequence_identifier']}\nf={int(frame['frame_index'])}")
        if row == 0:
            axes[row, 0].set_title("GT Gaussian heatmap")
            axes[row, 1].set_title("Predicted heatmap")
        for column in range(2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    fig.suptitle("GT vs predicted heatmaps — best debug checkpoint", y=0.999)
    fig.tight_layout()
    fig.savefig(figures_dir / "gt_vs_pred_heatmaps.png", dpi=180)
    plt.close(fig)


def theoretical_floor_for_subset(dataset: XAOverfitDataset) -> dict[str, float]:
    errors: list[float] = []
    original_h, original_w = (1024, 1024)
    heatmap_h, heatmap_w = dataset.heatmap_hw
    for index in range(len(dataset)):
        sample = dataset[index]
        gt_x, gt_y = map(float, sample["original_xy"])
        quant_x, quant_y = map(int, sample["gt_quant_xy"])
        mapped_x = quant_x * original_w / heatmap_w
        mapped_y = quant_y * original_h / heatmap_h
        errors.append(float(np.hypot(mapped_x - gt_x, mapped_y - gt_y)))
    array = np.asarray(errors)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def write_readme(
    path: Path,
    config: dict[str, Any],
    device: torch.device,
    subset: pd.DataFrame,
    training_log: pd.DataFrame,
    best_epoch: int,
    best_metrics: dict[str, float],
    final_metrics: pd.DataFrame,
    floor: dict[str, float],
    plumbing: dict[str, Any],
) -> None:
    initial = training_log.iloc[0]
    last = training_log.iloc[-1]
    loss_reduction = 100.0 * (1.0 - best_metrics["loss"] / float(initial["loss"]))
    exact_quantized = int((final_metrics["heatmap_quantized_error_px"] == 0).sum())
    anomalous = final_metrics.sort_values(
        ["heatmap_quantized_error_px", "original_error_px"], ascending=False
    ).head(5)
    anomaly_lines = "\n".join(
        f"- {row.case_code}/{row.sequence_identifier} frame {int(row.frame_index)}: "
        f"heatmap error={row.heatmap_quantized_error_px:.3f}, original error={row.original_error_px:.3f}px"
        for row in anomalous.itertuples()
    )
    clear_overfit = (
        best_metrics["loss"] < float(initial["loss"])
        and best_metrics["median_heatmap_quantized_error"] == 0.0
        and best_metrics["mean_original_error_px"] <= floor["mean"] + 1.0
    )
    assessment = (
        "显示出明确的小集记忆/overfit：loss 持续下降，quantized-target median error 为 0，"
        "original-space error 已接近该 32 帧自身的理论量化下限。"
        if clear_overfit
        else "只达到部分 overfit；请结合曲线和逐帧异常决定是否需要继续调试，不能仅凭 loss 宣布通过。"
    )
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    text = f"""# Phase 3 — HRNet Plumbing + 32-image Overfit Test

> **{DEBUG_CHECKPOINT_WARNING}**

本阶段只使用固定 Train subset，没有读取 Val、没有修改 Train/Val split、没有进行 2,784 帧正式训练，也没有评估泛化能力。

## 固定设置

- Preprocessing：No-clipping per-frame min-max `[0,1]`
- 原图 / 输入 / Heatmap：1024×1024 / 512×512 / 128×128
- GT：`x=true_tip_col, y=true_tip_row`，左上角原点
- Gaussian sigma：{float(config['data']['sigma']):g}
- Model：timm `{config['model']['variant']}`，pretrained={config['model']['pretrained']}
- Input / output：{plumbing['image_shape']} / {plumbing['output_shape']}
- Device：{device} ({device_name})
- Optimizer：{config['training']['optimizer']}，initial LR={float(config['training']['learning_rate']):g}
- Loss：MSE

## Dataset

固定 32 帧来自 {subset['series_uid'].nunique()} 个 Train Series、{subset['case_identifier'].nunique()} 个 case。每个 Series 至少两帧，不是连续取 32 帧；manifest 固定在 `overfit_32_manifest.csv`，seed={config['experiment']['seed']}。

## Plumbing test

- Forward：PASS
- Loss finite：PASS
- Backward gradient：PASS，L2 norm={plumbing['gradient_l2_norm']:.6f}
- Optimizer parameter update：PASS，max delta={plumbing['max_parameter_delta']:.8f}

单 batch plumbing 通过后重新以同一 seed 初始化模型，再开始 overfit，避免 plumbing 的一次更新污染 epoch 0 基线。

## Overfit result

最佳 debug checkpoint：epoch {best_epoch}。

| 指标 | Epoch 0 | Last epoch {int(last['epoch'])} | Best checkpoint |
|---|---:|---:|---:|
| MSE | {float(initial['loss']):.8f} | {float(last['loss']):.8f} | {best_metrics['loss']:.8f} |
| Mean original error | {float(initial['mean_original_error_px']):.3f}px | {float(last['mean_original_error_px']):.3f}px | {best_metrics['mean_original_error_px']:.3f}px |
| Median original error | {float(initial['median_original_error_px']):.3f}px | {float(last['median_original_error_px']):.3f}px | {best_metrics['median_original_error_px']:.3f}px |
| Mean quantized-target error | {float(initial['mean_heatmap_quantized_error']):.3f} hm px | {float(last['mean_heatmap_quantized_error']):.3f} hm px | {best_metrics['mean_heatmap_quantized_error']:.3f} hm px |
| Median quantized-target error | {float(initial['median_heatmap_quantized_error']):.3f} hm px | {float(last['median_heatmap_quantized_error']):.3f} hm px | {best_metrics['median_heatmap_quantized_error']:.3f} hm px |

最佳 MSE 相对 epoch 0 下降 {loss_reduction:.2f}%。最佳 checkpoint 有 {exact_quantized}/32 个样本的 integer argmax 与最近 GT heatmap pixel 完全一致。

结论：{assessment}

## Quantization floor

该固定 32 帧自己的 128×128 integer-argmax 理论下限：mean={floor['mean']:.4f}px、median={floor['median']:.4f}px、P95={floor['p95']:.4f}px、max={floor['max']:.4f}px。它与 Phase 2 全 Train 统计的 mean 3.0713px / median 3.1889px / P95 4.8303px / max 5.6569px 含义相同，但样本集合不同。

Original-space error 不应被要求趋近 0；当 quantized-target error 为 0 时，剩余 original error 就是上述网格离散化误差，而不是模型定位误差。

## Failed / anomalous samples

以下列出最佳 checkpoint 误差最大的 5 帧，不自动把它们称为数据错误：

{anomaly_lines}

## 关键输出

- `training_log.csv`：epoch 0 到最终 epoch 的 MSE、original error、quantized-target error 和 LR。
- `final_per_frame_metrics.csv`：最佳 checkpoint 的 32 帧逐帧坐标与误差。
- `best_overfit_checkpoint.pth`：仅用于 debug/sanity，**NOT FOR FORMAL EVALUATION**。
- `figures/train_mse_loss.png`
- `figures/train_localization_error.png`
- `figures/heatmap_learning_progression.png`
- `figures/final_overfit_overlays.png`
- `figures/gt_vs_pred_heatmaps.png`

## 是否具备进入 full Train/Val 的条件

只有在 plumbing 全部 PASS 且上述小集呈现清楚的记忆趋势时，才能说明代码链路具备下一阶段资格；它仍不能说明模型能泛化。下一阶段若启动正式训练，应保持独立 Val、重新定义正式 checkpoint/early stopping 规则，并明确这份 32-image debug checkpoint 不可复用为正式评估结果。
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 HRNet plumbing and 32-image overfit sanity test")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--defer-report",
        action="store_true",
        help="Stop after numeric/checkpoint outputs; a separate no-torch process will render reports.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = resolve_path(config, config["project"]["output_dir"])
    figures_dir = output_dir / "figures"
    logs_dir = output_dir / "logs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(logs_dir / "phase3_run.log")

    seed = int(config["experiment"]["seed"])
    configure_reproducibility(
        seed,
        bool(config["training"]["deterministic"]),
        bool(config["training"]["cudnn_benchmark"]),
    )
    device = select_device()
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    logger.info("[Phase 3] Loading Train manifest...")
    logger.info(f"[Device] Device selected: {device} ({device_name})")
    logger.info(f"[Environment] torch={torch.__version__} timm={timm.__version__}")

    manifest_path = resolve_path(config, config["project"]["train_manifest"])
    all_frames = pd.read_csv(manifest_path, dtype={"case_code": str}, low_memory=False)
    split = str(config["experiment"]["split"])
    train_frames = all_frames[all_frames["split"] == split].copy()
    actual = (
        train_frames["case_identifier"].nunique(),
        train_frames["series_uid"].nunique(),
        len(train_frames),
    )
    expected = (
        int(config["experiment"]["expected_train_cases"]),
        int(config["experiment"]["expected_train_series"]),
        int(config["experiment"]["expected_train_frames"]),
    )
    if actual != expected:
        raise RuntimeError(f"Frozen Train split mismatch: expected {expected}, got {actual}")
    val_paths = set(all_frames.loc[all_frames["split"] != split, "dicom_absolute_path"].astype(str))
    logger.info(f"[Phase 3] Frozen Train: cases={actual[0]} Series={actual[1]} frames={actual[2]}")
    logger.info("[Phase 3] Val is excluded from subset creation, training and model selection.")

    subset_path = output_dir / "overfit_32_manifest.csv"
    subset = create_or_load_subset(
        train_frames,
        subset_path,
        int(config["experiment"]["subset_size"]),
        seed,
    )
    overlap = set(subset["dicom_absolute_path"].astype(str)) & val_paths
    if overlap:
        raise RuntimeError(f"Overfit subset contains a Val path: {next(iter(overlap))}")
    logger.info(
        f"[Phase 3] Fixed overfit subset: {len(subset)} frames from "
        f"{subset['series_uid'].nunique()} Series / {subset['case_identifier'].nunique()} cases"
    )

    config_to_save = copy.deepcopy(config)
    config_to_save.pop("_config_path", None)
    config_to_save.pop("_workspace_root", None)
    with (output_dir / "config_used.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_save, handle, sort_keys=False, allow_unicode=True)

    input_hw = tuple(int(value) for value in config["data"]["input_size"])
    heatmap_hw = tuple(int(value) for value in config["data"]["heatmap_size"])
    original_hw = tuple(int(value) for value in config["data"]["original_size"])
    dataset = XAOverfitDataset(
        subset,
        input_hw,
        heatmap_hw,
        float(config["data"]["sigma"]),
        int(config["data"]["grayscale_channels"]),
        bool(config["data"]["preload_images"]),
        logger,
    )
    loader_kwargs = {
        "batch_size": int(config["data"]["batch_size"]),
        "num_workers": int(config["data"]["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(dataset, shuffle=True, generator=generator, **loader_kwargs)
    eval_loader = DataLoader(dataset, shuffle=False, **loader_kwargs)

    logger.info(f"[Model] {config['model']['variant']} | pretrained={config['model']['pretrained']}")
    probe_model = build_model(config, device)
    plumbing = plumbing_test(probe_model, eval_loader, device, config, logger)
    (output_dir / "plumbing_result.json").write_text(
        json.dumps(plumbing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_model_summary(output_dir / "model_summary.txt", probe_model, config, device, plumbing)
    del probe_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Fresh deterministic initialization: the plumbing optimizer step is not part
    # of the overfit training baseline.
    configure_reproducibility(
        seed,
        bool(config["training"]["deterministic"]),
        bool(config["training"]["cudnn_benchmark"]),
    )
    model = build_model(config, device)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    criterion = nn.MSELoss()

    logger.info("")
    logger.info("[Overfit Test] Evaluating untrained epoch 0 baseline...")
    initial_metrics, _, initial_predictions = evaluate(
        model, eval_loader, device, subset, original_hw, heatmap_hw
    )
    progression_epochs = {int(value) for value in config["training"]["progression_epochs"]}
    progression_indices = selected_indices_by_series(
        subset, int(config["visualization"]["progression_samples"])
    )
    progression: dict[int, np.ndarray] = {
        0: initial_predictions[progression_indices].copy()
    }
    progression_cache_path = output_dir / "heatmap_progression_cache.npz"
    np.savez_compressed(
        progression_cache_path,
        **{f"epoch_{key}": value for key, value in progression.items()},
    )
    log_rows: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "loss": initial_metrics["loss"],
            "train_batch_loss": float("nan"),
            **{key: value for key, value in initial_metrics.items() if key != "loss"},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
    ]
    logger.info(
        f"Epoch 000/{int(config['training']['epochs'])} | Loss={initial_metrics['loss']:.8f} | "
        f"Mean Error={initial_metrics['mean_original_error_px']:.3f}px | "
        f"Median Error={initial_metrics['median_original_error_px']:.3f}px | "
        f"Mean HM Error={initial_metrics['mean_heatmap_quantized_error']:.3f}"
    )

    best_key = (initial_metrics["mean_heatmap_quantized_error"], initial_metrics["loss"])
    best_epoch = 0
    best_metrics: dict[str, float] = dict(initial_metrics)
    checkpoint_path = output_dir / "best_overfit_checkpoint.pth"
    torch.save(
        {
            "warning": DEBUG_CHECKPOINT_WARNING,
            "epoch": 0,
            "model_variant": config["model"]["variant"],
            "pretrained": config["model"]["pretrained"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": initial_metrics,
            "config": config_to_save,
            "subset_manifest": "overfit_32_manifest.csv",
        },
        checkpoint_path,
    )
    epochs = int(config["training"]["epochs"])
    epoch_bar = tqdm(range(1, epochs + 1), desc="[Overfit] Epochs", unit="epoch", dynamic_ncols=True)
    for epoch in epoch_bar:
        model.train()
        batch_losses: list[float] = []
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target_heatmap"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            if not torch.isfinite(loss):
                path = batch["dicom_path"][0]
                raise RuntimeError(f"NaN/Inf training loss at epoch {epoch}; first batch DICOM={path}")
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))

        metrics, per_frame, predictions = evaluate(
            model, eval_loader, device, subset, original_hw, heatmap_hw
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        record = {
            "epoch": epoch,
            "loss": metrics["loss"],
            "train_batch_loss": float(np.mean(batch_losses)),
            **{key: value for key, value in metrics.items() if key != "loss"},
            "learning_rate": current_lr,
        }
        log_rows.append(record)
        pd.DataFrame.from_records(log_rows).to_csv(
            output_dir / "training_log.csv", index=False, encoding="utf-8-sig"
        )

        current_key = (metrics["mean_heatmap_quantized_error"], metrics["loss"])
        if current_key < best_key:
            best_key = current_key
            best_epoch = epoch
            best_metrics = dict(metrics)
            checkpoint = {
                "warning": DEBUG_CHECKPOINT_WARNING,
                "epoch": epoch,
                "model_variant": config["model"]["variant"],
                "pretrained": config["model"]["pretrained"],
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "config": config_to_save,
                "subset_manifest": "overfit_32_manifest.csv",
            }
            torch.save(checkpoint, checkpoint_path)

        if epoch in progression_epochs:
            progression[epoch] = predictions[progression_indices].copy()
            # Persist the requested snapshots immediately.  They are diagnostic
            # artifacts and should survive an interruption during later plotting.
            np.savez_compressed(
                progression_cache_path,
                **{f"epoch_{key}": value for key, value in progression.items()},
            )
        epoch_bar.set_postfix(
            loss=f"{metrics['loss']:.6g}",
            mean_px=f"{metrics['mean_original_error_px']:.2f}",
            median_px=f"{metrics['median_original_error_px']:.2f}",
            hm=f"{metrics['mean_heatmap_quantized_error']:.2f}",
            lr=f"{current_lr:.1e}",
        )
        logger.info(
            f"Epoch {epoch:03d}/{epochs} | Loss={metrics['loss']:.8f} | "
            f"Mean Error={metrics['mean_original_error_px']:.3f}px | "
            f"Median Error={metrics['median_original_error_px']:.3f}px | "
            f"Mean HM Error={metrics['mean_heatmap_quantized_error']:.3f} | LR={current_lr:.2e}"
        )
        scheduler.step()
    epoch_bar.close()

    training_log = pd.DataFrame.from_records(log_rows)
    training_log.to_csv(output_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_metrics, best_per_frame, best_predictions = evaluate(
        model, eval_loader, device, subset, original_hw, heatmap_hw
    )
    best_per_frame.insert(1, "checkpoint_epoch", best_epoch)
    best_per_frame.to_csv(output_dir / "final_per_frame_metrics.csv", index=False, encoding="utf-8-sig")

    # Matplotlib does not need the training optimizer/model on the GPU.  Releasing
    # them here keeps report generation independent of the peak training footprint.
    del checkpoint, optimizer, scheduler
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.defer_report:
        logger.info(
            "[Training Done] Numeric outputs and progression cache are complete; "
            "report rendering is delegated to a separate process."
        )
        return 0

    logger.info("[Figures] Generating training curves and coordinate/heatmap visualizations...")
    plot_curves(training_log, figures_dir)
    if epochs not in progression:
        progression[epochs] = predictions[progression_indices].copy()
    plot_progression(dataset, progression_indices, progression, figures_dir)
    overlay_indices = selected_indices_by_series(
        subset, int(config["visualization"]["final_overlay_samples"])
    )
    plot_final_overlays(dataset, overlay_indices, best_per_frame, figures_dir)
    pair_indices = selected_indices_by_series(
        subset, int(config["visualization"]["heatmap_pair_samples"])
    )
    plot_heatmap_pairs(dataset, pair_indices, best_predictions, figures_dir)

    floor = theoretical_floor_for_subset(dataset)
    write_readme(
        output_dir / "README.md",
        config,
        device,
        subset,
        training_log,
        best_epoch,
        best_metrics,
        best_per_frame,
        floor,
        plumbing,
    )
    run_summary = {
        "warning": DEBUG_CHECKPOINT_WARNING,
        "device": str(device),
        "device_name": device_name,
        "model_variant": config["model"]["variant"],
        "pretrained": config["model"]["pretrained"],
        "subset_frames": len(subset),
        "subset_series": int(subset["series_uid"].nunique()),
        "best_epoch": best_epoch,
        "epoch0": initial_metrics,
        "last_epoch": training_log.iloc[-1].to_dict(),
        "best": best_metrics,
        "subset_quantization_floor": floor,
        "plumbing": plumbing,
        "model_training_scope": "32-image deliberate overfit/debug only",
        "formal_training_performed": False,
        "val_used": False,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("")
    logger.info("[Done] HRNet 32-image overfit sanity completed.")
    logger.info(f"[Done] Best epoch: {best_epoch}")
    logger.info(
        f"[Done] MSE {initial_metrics['loss']:.8f} -> {best_metrics['loss']:.8f} | "
        f"Mean original error {initial_metrics['mean_original_error_px']:.3f} -> "
        f"{best_metrics['mean_original_error_px']:.3f}px | "
        f"Mean quantized HM error {initial_metrics['mean_heatmap_quantized_error']:.3f} -> "
        f"{best_metrics['mean_heatmap_quantized_error']:.3f}"
    )
    logger.info(f"[Done] {DEBUG_CHECKPOINT_WARNING}")
    logger.info(f"[Done] Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[Stopped] User interrupted Phase 3. No source data were modified.", file=sys.stderr)
        raise
