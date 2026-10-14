from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "code" / "model"
sys.path.insert(0, str(MODEL_DIR))

import hrnet_overfit32  # noqa: E402
import hrnet_phase5a_train  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P6B = load_module("phase10_p6b", MODEL_DIR / "phase6b_local_refinement_256_baseline.py")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def crop_fixed(image: np.ndarray, x0: int, y0: int, side: int = 256) -> np.ndarray:
    out = np.zeros((side, side), dtype=np.float32)
    h, w = image.shape
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, w), min(y0 + side, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return out


def read_image(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path)
    pixels = np.asarray(ds.pixel_array)
    if pixels.shape != (1024, 1024):
        raise ValueError(f"Expected 1024x1024 DICOM: {path}")
    if str(getattr(ds, "PhotometricInterpretation", "")) != "MONOCHROME2":
        raise ValueError(f"Expected MONOCHROME2: {path}")
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    if slope != 1 or intercept != 0:
        raise ValueError(f"Unexpected rescale {slope}/{intercept}: {path}")
    image = pixels.astype(np.float32, copy=False)
    low, high = float(image.min()), float(image.max())
    if not np.isfinite(low + high) or high <= low:
        raise ValueError(f"Invalid image range: {path}")
    return (image - low) / (high - low)


def heatmap(cx: float, cy: float, size: int = 64, sigma: float = 2.0) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma)).astype(np.float32)


def blur(image: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3 * sigma)))
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(coords * coords) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    x = torch.from_numpy(image[None, None])
    k = torch.from_numpy(kernel)
    x = F.conv2d(x, k.view(1, 1, 1, -1), padding=(0, radius))
    x = F.conv2d(x, k.view(1, 1, -1, 1), padding=(radius, 0))
    return x[0, 0].numpy()


class LocalAugmentationDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, train: bool, condition: str, seed: int, cfg: dict):
        self.manifest = manifest.reset_index(drop=True).copy()
        self.train = train
        self.condition = condition
        self.seed = int(seed)
        self.cfg = cfg["augmentation"]
        self.cache: dict[int, np.ndarray] = {}
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.manifest)

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + 1000003 * (self.epoch + 1) + 7919 * (index + 1))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _jitter_px(self) -> int:
        if not self.condition.startswith("A2_jitter"):
            return 0
        suffix = self.condition.removeprefix("A2_jitter")
        if suffix.isdigit():
            return int(suffix)
        return int(self.cfg["jitter_px"])

    def _load(self, index: int) -> np.ndarray:
        if index not in self.cache:
            self.cache[index] = read_image(str(self.manifest.iloc[index].dicom_absolute_path))
        return self.cache[index]

    def _sample_geometry(self, row, rng: np.random.Generator) -> tuple[int, int, float, float]:
        dx = dy = 0
        jitter = self._jitter_px()
        if self.train and jitter > 0:
            dx = int(rng.integers(-jitter, jitter + 1))
            dy = int(rng.integers(-jitter, jitter + 1))
        x0 = math.floor(float(row.raw_col) + dx - 128)
        y0 = math.floor(float(row.raw_row) + dy - 128)
        local_x = float(row.true_tip_col) - x0
        local_y = float(row.true_tip_row) - y0
        return x0, y0, local_x, local_y

    def _intensity(self, crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.train or self.condition != "A1_intensity":
            return crop
        if rng.random() > float(self.cfg["intensity_probability"]):
            return crop
        contrast = rng.uniform(*self.cfg["contrast_range"])
        gamma = rng.uniform(*self.cfg["gamma_range"])
        brightness = rng.uniform(*self.cfg["brightness_range"])
        out = np.clip((crop - 0.5) * contrast + 0.5 + brightness, 0, 1)
        out = np.clip(out, 0, 1) ** gamma
        if rng.random() < float(self.cfg["blur_probability"]):
            out = blur(out, float(rng.uniform(*self.cfg["blur_sigma_range"])))
        noise_sigma = float(rng.uniform(*self.cfg["noise_sigma_range"]))
        out = np.clip(out + rng.normal(0, noise_sigma, size=out.shape).astype(np.float32), 0, 1)
        return out.astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        rng = self._rng(index) if self.train else np.random.default_rng(0)
        x0, y0, local_x, local_y = self._sample_geometry(row, rng)
        if not (0 <= local_x < 256 and 0 <= local_y < 256):
            raise ValueError(f"GT outside crop at index {index}: ({local_x}, {local_y})")
        crop = crop_fixed(self._load(index), x0, y0)
        crop = self._intensity(crop, rng)
        target = heatmap(local_x / 4.0, local_y / 4.0)
        return {
            "image": torch.from_numpy(np.repeat(crop[None], 3, axis=0).copy()),
            "target_heatmap": torch.from_numpy(target[None].copy()),
            "original_xy": torch.tensor([local_x, local_y], dtype=torch.float32),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "crop_origin": torch.tensor([x0, y0], dtype=torch.float32),
            "dicom_path": str(row.dicom_absolute_path),
        }


class WeightedMSE(nn.Module):
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if predicted.shape != target.shape or predicted.shape[1:] != (1, 64, 64):
            raise ValueError(f"Unexpected heatmap shapes: {predicted.shape} / {target.shape}")
        return ((1.0 + 9.0 * target) * (predicted - target).square()).mean()


def build_model(device: torch.device, seed: int) -> nn.Module:
    seed_all(seed)
    model = hrnet_phase5a_train.build_pretrained_model(
        {
            "model": {
                "variant": "hrnet_w18_small_v2",
                "pretrained": True,
                "output_channels": 1,
                "pretrained_hf_hub_id": "timm/hrnet_w18_small_v2.gluon_in1k",
                "initialization": "pretrained_backbone_fresh_heatmap_head",
            },
            "data": {"heatmap_size": [64, 64]},
        },
        device,
    )
    return model


def train_model(dataset: Dataset, indices: np.ndarray, seed: int, device: torch.device) -> tuple[nn.Module, list[dict]]:
    model = build_model(device, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    criterion = WeightedMSE()
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    logs = []
    for epoch in range(1, 4):
        if isinstance(dataset, LocalAugmentationDataset):
            dataset.set_epoch(epoch)
        model.train()
        losses = []
        for batch in tqdm(loader, desc=f"train seed={seed} epoch={epoch}/3", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target_heatmap"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = criterion(output, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite augmentation training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        logs.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "seed": seed})
    return model, logs


def infer(model: nn.Module, dataset: Dataset, indices: np.ndarray, device: torch.device) -> pd.DataFrame:
    model.eval()
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="OOF inference", leave=False):
            output = model(batch["image"].to(device, non_blocking=True))[:, 0].cpu().numpy()
            ids = batch["sample_index"].numpy()
            for heat, index in zip(output, ids):
                row = dataset.manifest.iloc[int(index)]
                hy, hx = np.unravel_index(int(np.argmax(heat)), heat.shape)
                x0, y0 = map(float, batch["crop_origin"][list(ids).index(index)].numpy())
                candidate_x = x0 + 4.0 * hx
                candidate_y = y0 + 4.0 * hy
                raw_x, raw_y = float(row.raw_col), float(row.raw_row)
                gt_x, gt_y = float(row.true_tip_col), float(row.true_tip_row)
                final_x = raw_x + 0.75 * (candidate_x - raw_x)
                final_y = raw_y + 0.75 * (candidate_y - raw_y)
                raw_error = math.hypot(raw_x - gt_x, raw_y - gt_y)
                final_error = math.hypot(final_x - gt_x, final_y - gt_y)
                rows.append(
                    {
                        "case_id": str(row.case_id),
                        "series_id": str(row.series_id),
                        "frame_index": int(row.frame_index),
                        "dicom_path": str(row.dicom_absolute_path),
                        "raw_error_px": raw_error,
                        "final_error_px": final_error,
                        "gain_px": raw_error - final_error,
                        "pred_x": candidate_x,
                        "pred_y": candidate_y,
                        "gt_x": gt_x,
                        "gt_y": gt_y,
                    }
                )
    return pd.DataFrame(rows)


def run_sanity(manifest: pd.DataFrame, cfg: dict, out: Path) -> None:
    train = manifest[manifest.split.eq("train")].reset_index(drop=True)
    selected = np.linspace(0, len(train) - 1, num=min(64, len(train)), dtype=int)
    rows = []
    for condition in cfg["conditions"]:
        dataset = LocalAugmentationDataset(train, True, condition, int(cfg["seeds"][0]), cfg)
        for epoch in (1, 2):
            dataset.set_epoch(epoch)
            for index in selected:
                sample = dataset[int(index)]
                peak = torch.nonzero(
                    sample["target_heatmap"][0] == sample["target_heatmap"][0].max(),
                    as_tuple=False,
                )[0]
                peak_y, peak_x = int(peak[0]), int(peak[1])
                local_x, local_y = sample["original_xy"].numpy()
                target_x, target_y = peak_x * 4.0, peak_y * 4.0
                rows.append(
                    {
                        "condition": condition,
                        "epoch": epoch,
                        "sample_index": int(index),
                        "crop_x0": float(sample["crop_origin"][0]),
                        "crop_y0": float(sample["crop_origin"][1]),
                        "gt_local_x": float(local_x),
                        "gt_local_y": float(local_y),
                        "heatmap_peak_x": peak_x,
                        "heatmap_peak_y": peak_y,
                        "heatmap_quantization_error_px": float(
                            math.hypot(target_x - local_x, target_y - local_y)
                        ),
                        "image_min": float(sample["image"].min()),
                        "image_max": float(sample["image"].max()),
                        "image_mean": float(sample["image"].mean()),
                        "image_std": float(sample["image"].std()),
                    }
                )
    report = pd.DataFrame(rows)
    report.to_csv(out / "augmentation_sanity_samples.csv", index=False)
    by_condition = (
        report.groupby(["condition", "epoch"])
        .agg(
            samples=("sample_index", "size"),
            max_quantization_error_px=("heatmap_quantization_error_px", "max"),
            mean_image=("image_mean", "mean"),
            mean_image_std=("image_std", "mean"),
            crop_x0_unique=("crop_x0", "nunique"),
            crop_y0_unique=("crop_y0", "nunique"),
        )
        .reset_index()
    )
    by_condition.to_csv(out / "augmentation_sanity_summary.csv", index=False)
    a0 = report[(report.condition.eq("A0_none")) & (report.epoch.eq(1))].set_index("sample_index")
    has_intensity = "A1_intensity" in cfg["conditions"]
    a1 = (
        report[(report.condition.eq("A1_intensity")) & (report.epoch.eq(1))].set_index("sample_index")
        if has_intensity
        else None
    )
    jitter_conditions = [
        condition for condition in cfg["conditions"] if condition.startswith("A2_jitter")
    ]
    jitter_condition = jitter_conditions[0] if jitter_conditions else None
    a2_e1 = (
        report[(report.condition.eq(jitter_condition)) & (report.epoch.eq(1))].set_index("sample_index")
        if jitter_condition
        else None
    )
    a2_e2 = (
        report[(report.condition.eq(jitter_condition)) & (report.epoch.eq(2))].set_index("sample_index")
        if jitter_condition
        else None
    )
    checks = {
        "a0_coordinates_stable": bool(a0.heatmap_quantization_error_px.max() <= 2.83),
        "a1_geometry_matches_a0": bool(
            not has_intensity
            or np.allclose(a0[["crop_x0", "crop_y0"]], a1[["crop_x0", "crop_y0"]])
        ),
        "a2_jitter_changes_some_crop_origins": bool(
            jitter_condition
            and (
                np.any(a0.crop_x0.to_numpy() != a2_e1.crop_x0.to_numpy())
                or np.any(a0.crop_y0.to_numpy() != a2_e1.crop_y0.to_numpy())
            )
        ),
        "a2_epoch_resampling_changes_some_crop_origins": bool(
            jitter_condition
            and (
                np.any(a2_e1.crop_x0.to_numpy() != a2_e2.crop_x0.to_numpy())
                or np.any(a2_e1.crop_y0.to_numpy() != a2_e2.crop_y0.to_numpy())
            )
        ),
        "all_gt_inside_crop": bool(
            ((report.gt_local_x >= 0) & (report.gt_local_x < 256)
             & (report.gt_local_y >= 0) & (report.gt_local_y < 256)).all()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Augmentation sanity failed: {checks}")
    (out / "augmentation_sanity.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print("[Phase10] Augmentation sanity passed.")


def run_visualization(manifest: pd.DataFrame, cfg: dict, out: Path, samples_per_condition: int = 8) -> None:
    """Save human-readable before/after crop and target-heatmap panels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = manifest[manifest.split.eq("train")].reset_index(drop=True)
    selected = np.linspace(
        0, len(train) - 1, num=min(samples_per_condition, len(train)), dtype=int
    )
    vis_dir = out / "augmentation_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for condition in cfg["conditions"]:
        dataset = LocalAugmentationDataset(train, True, condition, int(cfg["seeds"][0]), cfg)
        clean_dataset = LocalAugmentationDataset(train, True, "A0_none", int(cfg["seeds"][0]), cfg)
        for epoch in (1, 2):
            dataset.set_epoch(epoch)
            clean_dataset.set_epoch(epoch)
            for index in selected:
                row = train.iloc[int(index)]
                image = dataset._load(int(index))
                clean_x0, clean_y0, clean_local_x, clean_local_y = clean_dataset._sample_geometry(
                    row, np.random.default_rng(0)
                )
                clean_crop = crop_fixed(image, clean_x0, clean_y0)
                sample = dataset[int(index)]
                aug_crop = sample["image"][0].numpy()
                aug_x0, aug_y0 = sample["crop_origin"].numpy().tolist()
                gt_x, gt_y = sample["original_xy"].numpy().tolist()
                raw_x = float(row.raw_col) - aug_x0
                raw_y = float(row.raw_row) - aug_y0
                target = sample["target_heatmap"][0].numpy()
                jitter_dx = aug_x0 - clean_x0
                jitter_dy = aug_y0 - clean_y0

                fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
                axes[0].imshow(clean_crop, cmap="gray", vmin=0, vmax=1)
                axes[0].scatter(
                    [clean_local_x], [clean_local_y], c="lime", s=42, marker="x", label="GT"
                )
                axes[0].set_title("Clean crop")
                axes[0].legend(loc="upper right", fontsize=8)

                axes[1].imshow(aug_crop, cmap="gray", vmin=0, vmax=1)
                axes[1].scatter([raw_x], [raw_y], c="cyan", s=42, marker="+", label="Raw")
                axes[1].scatter([gt_x], [gt_y], c="red", s=42, marker="x", label="GT")
                axes[1].set_title(
                    f"Augmented crop\norigin shift=({jitter_dx:+.0f}, {jitter_dy:+.0f}) px"
                )
                axes[1].legend(loc="upper right", fontsize=8)

                axes[2].imshow(target, cmap="hot", vmin=0, vmax=1)
                axes[2].scatter([gt_x / 4.0], [gt_y / 4.0], c="cyan", s=35, marker="x")
                axes[2].set_title("Target heatmap (64 x 64)")

                for axis in axes:
                    axis.set_xlim(-1, 256 if axis is not axes[2] else 64)
                    axis.set_ylim(256 if axis is not axes[2] else 64, -1)
                    axis.set_aspect("equal")
                    axis.set_xlabel("x")
                    axis.set_ylabel("y")
                fig.suptitle(
                    f"{condition} | epoch={epoch} | sample={int(index)} | "
                    f"case={row.case_id} frame={int(row.frame_index)}"
                )
                filename = f"{condition}_epoch{epoch}_sample{int(index):04d}.png"
                fig.savefig(vis_dir / filename, dpi=140)
                plt.close(fig)
                rows.append(
                    {
                        "condition": condition,
                        "epoch": epoch,
                        "sample_index": int(index),
                        "case_id": str(row.case_id),
                        "frame_index": int(row.frame_index),
                        "filename": filename,
                        "clean_x0": clean_x0,
                        "clean_y0": clean_y0,
                        "aug_x0": aug_x0,
                        "aug_y0": aug_y0,
                        "jitter_dx": jitter_dx,
                        "jitter_dy": jitter_dy,
                        "raw_local_x": raw_x,
                        "raw_local_y": raw_y,
                        "gt_local_x": gt_x,
                        "gt_local_y": gt_y,
                    }
                )

    pd.DataFrame(rows).to_csv(out / "augmentation_visualizations.csv", index=False)
    print(f"[Phase10] Saved {len(rows)} visualization panels to {vis_dir}")


def metrics(pred: pd.DataFrame) -> dict[str, float]:
    e = pred.final_error_px.to_numpy(float)
    gain = pred.gain_px.to_numpy(float)
    return {
        "n": len(pred),
        "mean_px": float(e.mean()),
        "case_macro_mean_px": float(pred.groupby("case_id").final_error_px.mean().mean()),
        "median_px": float(np.median(e)),
        "p95_px": float(np.percentile(e, 95)),
        "p99_px": float(np.percentile(e, 99)),
        "harmful_gt5_pct": float(np.mean(gain < -5) * 100),
        "harmful_gt10_pct": float(np.mean(gain < -10) * 100),
        "improved_pct": float(np.mean(gain > 0) * 100),
        "raw_lt10_harmful_gt10_pct": float(
            np.mean(pred.loc[pred.raw_error_px < 10, "gain_px"].to_numpy(float) < -10) * 100
        ),
    }


def train_final_candidate(manifest: pd.DataFrame, cfg: dict, out: Path, device: torch.device) -> None:
    final_cfg = cfg.get("final_training", {})
    condition = str(final_cfg.get("condition", "A2_jitter8"))
    seed = int(final_cfg.get("seed", cfg["seeds"][0]))
    if condition not in cfg["conditions"]:
        raise ValueError(f"Final condition {condition} is not listed in config conditions")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for final candidate training")

    train = manifest[manifest.split.eq("train")].reset_index(drop=True)
    if train.empty:
        raise RuntimeError("No training rows available for final candidate training")
    if set(train.case_id) & {"908", "771"}:
        raise RuntimeError("Final Test cases are present in final candidate training data")

    print(
        f"[Phase10] Final candidate training: condition={condition} seed={seed} "
        f"frames={len(train)} cases={train.case_id.nunique()}"
    )
    dataset = LocalAugmentationDataset(train, True, condition, seed, cfg)
    model, logs = train_model(dataset, np.arange(len(train)), seed, device)

    checkpoints = out / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints / f"phase10_{condition}_all_train_seed{seed}_epoch3.pth"
    payload = {
        "format": "phase10_final_candidate_v1",
        "model_state_dict": model.state_dict(),
        "model": {
            "variant": cfg["model_variant"],
            "pretrained_backbone": bool(cfg["pretrained"]),
            "input_size": int(cfg["input_size"]),
            "heatmap_size": int(cfg["heatmap_size"]),
            "output_channels": 1,
        },
        "training": {
            "condition": condition,
            "seed": seed,
            "epochs": 3,
            "batch_size": int(cfg["batch_size"]),
            "learning_rate": float(cfg["learning_rate"]),
            "weight_decay": float(cfg["weight_decay"]),
            "loss": "weighted_mse_1_plus_9_gt",
            "sigma": float(cfg["sigma"]),
        },
        "inference": {
            "decoder": "integer_argmax",
            "alpha": float(cfg["alpha"]),
        },
        "data": {
            "train_frames": int(len(train)),
            "train_cases": int(train.case_id.nunique()),
            "train_series": int(train.series_id.nunique()),
            "manifest_hash": canonical_hash(train.to_dict(orient="records")),
            "frozen_val_used_for_training": False,
            "final_test_used_for_training": False,
        },
    }
    torch.save(payload, checkpoint_path)
    pd.DataFrame(logs).to_csv(out / "final_candidate_training_logs.csv", index=False)
    metadata = {
        "status": "complete",
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "condition": condition,
        "seed": seed,
        "train_frames": int(len(train)),
        "train_cases": int(train.case_id.nunique()),
        "frozen_val_used": False,
        "final_test_used": False,
    }
    (out / "final_candidate_status.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    del model
    torch.cuda.empty_cache()
    print(f"[Phase10] Final candidate checkpoint saved: {checkpoint_path}")


def run_frozen_val_confirmation(
    manifest: pd.DataFrame, cfg: dict, out: Path, device: torch.device
) -> None:
    """Compare the saved jitter8 candidate with a matched freshly trained A0 baseline."""
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Frozen Val confirmation")
    final_cfg = cfg.get("final_training", {})
    seed = int(final_cfg.get("seed", cfg["seeds"][0]))
    checkpoints = out / "checkpoints"
    jitter_checkpoint = checkpoints / f"phase10_A2_jitter8_all_train_seed{seed}_epoch3.pth"
    if not jitter_checkpoint.exists():
        raise FileNotFoundError(f"Saved jitter8 candidate is missing: {jitter_checkpoint}")

    train = manifest[manifest.split.eq("train")].reset_index(drop=True)
    val = manifest[manifest.split.eq("val")].reset_index(drop=True)
    if val.empty:
        raise RuntimeError("Frozen Val is absent from manifest")
    if set(val.case_id) & {"908", "771"}:
        raise RuntimeError("Final Test cases are present in Frozen Val")

    print(
        f"[Phase10] Frozen Val confirmation: train_frames={len(train)} "
        f"val_frames={len(val)} val_cases={val.case_id.nunique()}"
    )
    jitter_payload = torch.load(jitter_checkpoint, map_location=device, weights_only=False)
    jitter_model = build_model(device, seed)
    jitter_model.load_state_dict(jitter_payload["model_state_dict"], strict=True)

    baseline_dataset = LocalAugmentationDataset(train, True, "A0_none", seed, cfg)
    baseline_model, baseline_logs = train_model(
        baseline_dataset, np.arange(len(train)), seed, device
    )
    val_clean = LocalAugmentationDataset(val, False, "A0_none", seed, cfg)
    indices = np.arange(len(val))
    jitter_pred = infer(jitter_model, val_clean, indices, device)
    baseline_pred = infer(baseline_model, val_clean, indices, device)
    jitter_pred["condition"] = "A2_jitter8"
    jitter_pred["seed"] = seed
    baseline_pred["condition"] = "A0_none"
    baseline_pred["seed"] = seed
    predictions = pd.concat([baseline_pred, jitter_pred], ignore_index=True)

    confirmation_dir = out / "frozen_val_confirmation"
    confirmation_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(confirmation_dir / "frozen_val_predictions.csv", index=False)
    metric_rows = [
        {"condition": condition, "seed": seed, **metrics(group)}
        for condition, group in predictions.groupby("condition")
    ]
    summary = pd.DataFrame(metric_rows).sort_values("condition").reset_index(drop=True)
    summary.to_csv(confirmation_dir / "frozen_val_metrics.csv", index=False)
    pd.DataFrame(baseline_logs).to_csv(
        confirmation_dir / "baseline_a0_training_logs.csv", index=False
    )
    status = {
        "status": "complete",
        "frozen_val_used": True,
        "final_test_used": False,
        "seed": seed,
        "saved_jitter_checkpoint": str(jitter_checkpoint.relative_to(ROOT)),
        "baseline_training_condition": "A0_none",
        "train_frames": int(len(train)),
        "val_frames": int(len(val)),
        "val_cases": int(val.case_id.nunique()),
    }
    (confirmation_dir / "run_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    del jitter_model, baseline_model
    torch.cuda.empty_cache()
    print(summary.to_string(index=False))
    print(f"[Phase10] Frozen Val confirmation saved: {confirmation_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/phase10_augmentation_final.json",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--visualize-only", action="store_true")
    parser.add_argument("--visualization-samples", type=int, default=8)
    parser.add_argument("--train-final-candidate", action="store_true")
    parser.add_argument("--confirm-frozen-val", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_path = ROOT / cfg["manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run phase10_prepare_augmentation_data.py first: {manifest_path}")
    manifest = pd.read_csv(manifest_path, dtype={"case_id": str, "series_id": str})
    if set(manifest.loc[manifest.split.eq("train"), "case_id"]) & {"908", "771"}:
        raise RuntimeError("Final Test cases are present in training manifest")
    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if args.prepare_only:
        print("[Phase10] Manifest/config verified; no training started.")
        return
    if args.sanity_only:
        run_sanity(manifest, cfg, out)
        return
    if args.visualize_only:
        run_visualization(manifest, cfg, out, max(1, args.visualization_samples))
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.train_final_candidate:
        train_final_candidate(manifest, cfg, out, device)
        return
    if args.confirm_frozen_val:
        run_frozen_val_confirmation(manifest, cfg, out, device)
        return
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the formal augmentation run")
    train = manifest[manifest.split.eq("train")].reset_index(drop=True)
    old_cases = sorted(train.loc[train.split_group.eq("ORIGINAL_TRAIN"), "case_id"].unique())
    rows = []
    logs = []
    for condition in cfg["conditions"]:
        for seed in cfg["seeds"]:
            for fold, held_case in enumerate(old_cases, start=1):
                print(f"[Phase10] condition={condition} seed={seed} held_case={held_case}")
                fit_df = train[train.case_id.ne(held_case)].reset_index(drop=True)
                hold_df = train[(train.case_id.eq(held_case)) & (train.frame_index >= 2)].reset_index(drop=True)
                fit_aug = LocalAugmentationDataset(fit_df, True, condition, seed + fold, cfg)
                hold_clean = LocalAugmentationDataset(hold_df, False, "A0_none", seed, cfg)
                model, train_logs = train_model(fit_aug, np.arange(len(fit_df)), seed + fold, device)
                pred = infer(model, hold_clean, np.arange(len(hold_df)), device)
                pred["condition"] = condition
                pred["seed"] = seed
                pred["held_case"] = held_case
                pred["fold"] = fold
                rows.append(pred)
                for record in train_logs:
                    logs.append({**record, "condition": condition, "held_case": held_case, "fold": fold})
                del model
                torch.cuda.empty_cache()
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_csv(out / "oof_predictions.csv", index=False)
    pd.DataFrame(logs).to_csv(out / "training_logs.csv", index=False)
    metric_rows = []
    for (condition, seed), group in predictions.groupby(["condition", "seed"]):
        metric_rows.append({"condition": condition, "seed": seed, **metrics(group)})
    metric_table = pd.DataFrame(metric_rows)
    metric_table.to_csv(out / "oof_metrics_by_condition.csv", index=False)
    summary = (
        metric_table.groupby("condition")
        .agg(
            case_macro_mean_px=("case_macro_mean_px", "mean"),
            mean_px=("mean_px", "mean"),
            p95_px=("p95_px", "mean"),
            harmful_gt10_pct=("harmful_gt10_pct", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out / "oof_condition_summary.csv", index=False)
    (out / "run_status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "conditions": cfg["conditions"],
                "seeds": cfg["seeds"],
                "test_used": False,
                "final_test_cases": ["908", "771"],
                "oof_rows": len(predictions),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print("[Phase10] OOF complete. Frozen Val and Final Test were not read.")


if __name__ == "__main__":
    main()
