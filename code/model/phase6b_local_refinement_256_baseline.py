"""Phase 6B: frozen raw-centered 256 px local refinement, then paired audit.

Run from the repository root with the matching VS Code PowerShell entry point.
No previous tip checkpoint is used for initialization. Val is used only for
checkpoint/early-stop selection and frozen hold-out reporting.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import hrnet_phase4_train as phase4
import hrnet_phase5a_train as phase5a
import hrnet_phase5c_weighted_mse as phase5c
from hrnet_overfit32 import gaussian_heatmap, round_half_up


OUT: Path | None = None
DATASETS: dict[str, "LocalTipDataset"] = {}
SOURCE_BUILD_LOGGER = phase4.build_logger
SOURCE_ENVIRONMENT = phase4.environment_text
SOURCE_PLUMBING = phase4.plumbing_test
SOURCE_TRAIN_EPOCH = phase4.train_one_epoch
SOURCE_VALIDATE = phase4.validate


def say(message: str) -> None:
    print(message, flush=True)


def crop_origin(raw_x: float, raw_y: float, side: int = 256) -> tuple[int, int]:
    return math.floor(raw_x - side / 2), math.floor(raw_y - side / 2)


def crop_fixed(image: np.ndarray, x0: int, y0: int, side: int = 256) -> np.ndarray:
    """Zero pad out-of-bounds area; never shift the raw-centered crop."""
    cropped = np.zeros((side, side), dtype=np.float32)
    h, w = image.shape
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, w), min(y0 + side, h)
    if sx1 > sx0 and sy1 > sy0:
        cropped[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = image[sy0:sy1, sx0:sx1]
    return cropped


def read_full_minmax(path: str) -> np.ndarray:
    try:
        dicom = pydicom.dcmread(path)
        pixels = np.asarray(dicom.pixel_array)
    except Exception as exc:
        raise RuntimeError(f"DICOM decoding failed: {path}\n{exc}") from exc
    if pixels.shape != (1024, 1024):
        raise ValueError(f"Expected one 1024x1024 frame, got {pixels.shape}: {path}")
    if str(getattr(dicom, "PhotometricInterpretation", "")) != "MONOCHROME2":
        raise ValueError(f"PhotometricInterpretation is not MONOCHROME2: {path}")
    slope = float(getattr(dicom, "RescaleSlope", 1))
    intercept = float(getattr(dicom, "RescaleIntercept", 0))
    if slope != 1 or intercept != 0:
        raise ValueError(f"Unexpected rescale slope/intercept {slope}/{intercept}: {path}")
    image = pixels.astype(np.float32, copy=False)
    low, high = float(image.min()), float(image.max())
    if not np.isfinite(low + high) or high <= low:
        raise ValueError(f"Non-finite or constant image: {path}")
    return (image - low) / (high - low)


class LocalTipDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, split_name: str,
                 input_hw: tuple[int, int], heatmap_hw: tuple[int, int],
                 sigma: float, channels: int, preload: bool,
                 logger: logging.Logger) -> None:
        if input_hw != (256, 256) or heatmap_hw != (64, 64) or channels != 3 or sigma != 2:
            raise ValueError("Phase 6B geometry is frozen to 256 input / 64 heatmap / sigma 2 / 3 channels")
        self.manifest = manifest.reset_index(drop=True).copy()
        self.split_name = split_name
        self.cache: list[dict[str, Any] | None] = [None] * len(self.manifest)
        DATASETS[split_name] = self
        if preload:
            logger.info(f"[Dataset] Preloading {split_name}: {len(self.manifest)} raw-centered crops")
            for i in tqdm(range(len(self)), desc=f"[{split_name}] DICOM", unit="frame", dynamic_ncols=True):
                self.cache[i] = self._load_sample(i)

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_sample(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        path = str(row.dicom_absolute_path)
        image = read_full_minmax(path)
        x0, y0 = crop_origin(float(row.raw_col), float(row.raw_row))
        crop = crop_fixed(image, x0, y0)
        local_x = float(row.true_tip_col) - x0
        local_y = float(row.true_tip_row) - y0
        if not (0 <= local_x < 256 and 0 <= local_y < 256):
            raise ValueError(f"GT outside raw-centered crop: local=({local_x},{local_y}), DICOM={path}")
        hm_x, hm_y = local_x / 4, local_y / 4
        target = gaussian_heatmap(64, 64, hm_x, hm_y, 2.0).contiguous()
        return {
            "image_single": torch.from_numpy(crop).unsqueeze(0).contiguous(),
            "target_heatmap": target,
            "original_xy": torch.tensor([local_x, local_y], dtype=torch.float32),
            "heatmap_xy": torch.tensor([hm_x, hm_y], dtype=torch.float32),
            "gt_quant_xy": torch.tensor([min(max(round_half_up(hm_x), 0), 63),
                                         min(max(round_half_up(hm_y), 0), 63)], dtype=torch.long),
            "crop_origin": torch.tensor([x0, y0], dtype=torch.float32),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "dicom_path": path,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.cache[index] if self.cache[index] is not None else self._load_sample(index)
        assert sample is not None
        return {**sample, "image": sample["image_single"].repeat(3, 1, 1)}


class LocalWeightedMSE(phase5c.WeightedMSE):
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if predicted.shape != target.shape or predicted.ndim != 4 or predicted.shape[1:] != (1, 64, 64):
            raise ValueError(f"Phase 6B requires [B,1,64,64], got {tuple(predicted.shape)} vs {tuple(target.shape)}: {self.current_dicom_path}")
        if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
            raise FloatingPointError(f"Non-finite heatmap: {self.current_dicom_path}")
        minimum, maximum = float(predicted.detach().min()), float(predicted.detach().max())
        if max(abs(minimum), abs(maximum)) > self.absolute_limit:
            raise FloatingPointError(f"Predicted heatmap outside ±{self.absolute_limit}: {self.current_dicom_path}")
        error = (predicted - target).square()
        weighted = (1 + self.alpha * target) * error
        with torch.no_grad():
            foreground = target > self.fg_threshold
            stats = self.stats
            stats["plain_sse"] += float(error.detach().sum())
            stats["weighted_sse"] += float(weighted.detach().sum())
            stats["fg_plain_sse"] += float(error.detach().masked_select(foreground).sum())
            stats["bg_plain_sse"] += float(error.detach().masked_select(~foreground).sum())
            stats["fg_weighted_sse"] += float(weighted.detach().masked_select(foreground).sum())
            stats["bg_weighted_sse"] += float(weighted.detach().masked_select(~foreground).sum())
            stats["pixels"] += float(target.numel())
            stats["fg_pixels"] += float(foreground.sum())
            self.pred_min = min(self.pred_min, minimum)
            self.pred_max = max(self.pred_max, maximum)
        return weighted.mean()


def build_criterion(config: dict[str, Any]) -> LocalWeightedMSE:
    if config["training"]["loss"] != "heatmap_weighted_mse":
        raise ValueError("Phase 6B loss changed")
    opts = config["weighted_loss"]
    return LocalWeightedMSE(float(opts["alpha"]), float(opts["foreground_diagnostic_threshold"]),
                            float(opts["prediction_abs_safety_stop"]))


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: Any, criterion: LocalWeightedMSE,
                *args: Any, **kwargs: Any) -> dict[str, Any]:
    criterion.reset()
    metrics = SOURCE_TRAIN_EPOCH(model, loader, optimizer, criterion, *args, **kwargs)
    metrics["_epoch_extra"] = criterion.diagnostic("train")
    return metrics


def validate_epoch(model: nn.Module, loader: DataLoader, criterion: LocalWeightedMSE,
                   *args: Any, **kwargs: Any) -> tuple[dict[str, Any], pd.DataFrame | None]:
    criterion.reset()
    metrics, predictions = SOURCE_VALIDATE(model, loader, criterion, *args, **kwargs)
    metrics["_epoch_extra"] = criterion.diagnostic("val")
    return metrics, predictions


def model_summary(path: Path, model: nn.Module, config: dict[str, Any], device: torch.device) -> None:
    total = sum(p.numel() for p in model.parameters())
    path.write_text(
        "Phase 6B raw-centered local tip refinement\n"
        f"Model: {config['model']['variant']} | ImageNet pretrained backbone, fresh 1-channel head\n"
        "Input [B,3,256,256] -> output [B,1,64,64]\n"
        "Weighted MSE = mean((1+9*GT)*(Pred-GT)^2)\n"
        f"Device: {device} | parameters: {total:,}\n\n{model!r}\n", encoding="utf-8")


def phase6b_header(logger: logging.Logger, config: dict[str, Any], device: torch.device,
                   train: pd.DataFrame, val: pd.DataFrame, train_iterations: int,
                   val_iterations: int) -> None:
    logger.info("=" * 66)
    logger.info("Phase 6B | Raw-centered 256x256 Local Tip Refinement Baseline")
    logger.info("=" * 66)
    logger.info(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else device}; torch={torch.__version__}; CUDA={torch.version.cuda}")
    logger.info(f"Train {len(train)} frames / {train.case_identifier.nunique()} cases / {train.series_uid.nunique()} Series | {train_iterations} batches/epoch")
    logger.info(f"Val {len(val)} frames / {val.case_identifier.nunique()} cases / {val.series_uid.nunique()} Series | {val_iterations} batches/epoch")
    logger.info("Crop center=raw only; input=256; heatmap=64; sigma=2; padding=zero; augmentation=none")
    logger.info("ImageNet pretrained HRNet backbone + fresh heatmap head; no prior tip checkpoint")
    logger.info(f"Adam LR={config['training']['learning_rate']} WD={config['training']['weight_decay']} | batch={config['data']['batch_size']} | weighted MSE alpha=9")
    logger.info(f"Max epochs={config['training']['max_epochs']}; Early stop Val mean, patience={config['training']['early_stopping']['patience']}, min delta={config['training']['early_stopping']['min_delta_px']} px")
    logger.info("=" * 66)


def save_checkpoint(path: Path, epoch: int, model: Any, optimizer: Any, scheduler: Any,
                    record: dict[str, Any], config: dict[str, Any], state: dict[str, Any],
                    generator: torch.Generator, kind: str) -> None:
    phase5c.save_checkpoint(path, epoch, model, optimizer, scheduler, record, config, state, generator, kind)


def roundtrip_test(manifest: pd.DataFrame, output: Path) -> None:
    rng = np.random.default_rng(20260918)
    indices = rng.choice(len(manifest), size=100, replace=False)
    rows = []
    for index in indices:
        row = manifest.iloc[int(index)]
        x0, y0 = crop_origin(float(row.raw_col), float(row.raw_row))
        local_x = float(row.true_tip_col) - x0
        local_y = float(row.true_tip_row) - y0
        restored_x, restored_y = local_x + x0, local_y + y0
        error = math.hypot(restored_x-float(row.true_tip_col), restored_y-float(row.true_tip_row))
        rows.append({"dicom_path": str(row.dicom_absolute_path), "crop_x0": x0, "crop_y0": y0,
                     "gt_local_x": local_x, "gt_local_y": local_y, "roundtrip_error_px": error})
    result = pd.DataFrame(rows)
    result.to_csv(output / "coordinate_roundtrip_100.csv", index=False, encoding="utf-8-sig")
    if float(result.roundtrip_error_px.max()) >= 1e-6:
        raise RuntimeError("Coordinate round-trip failed")
    say(f"[Geometry] 100/100 coordinate round trips PASS; max={result.roundtrip_error_px.max():.3e} px")


THRESHOLDS = (2, 3, 5, 10, 20, 40, 50)


def error_metrics(errors: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=np.float64)
    if not len(values):
        return {"n": 0}
    result: dict[str, float | int] = {
        "n": int(len(values)), "mean_px": float(values.mean()),
        "median_px": float(np.median(values)), "p95_px": float(np.percentile(values, 95)),
        "p99_px": float(np.percentile(values, 99)), "max_px": float(values.max()),
    }
    result.update({f"lt_{t}px_pct": float(100*np.mean(values < t)) for t in THRESHOLDS})
    return result


def base_predictions(manifest: pd.DataFrame, correction: tuple[float, float]) -> pd.DataFrame:
    result = manifest[["split", "case_code", "series_uid", "sequence_identifier", "frame_index",
                       "dicom_absolute_path", "raw_col", "raw_row", "true_tip_col", "true_tip_row"]].copy()
    result = result.rename(columns={"case_code": "case_id", "series_uid": "series_id",
                                    "dicom_absolute_path": "dicom_path", "raw_col": "raw_x", "raw_row": "raw_y",
                                    "true_tip_col": "gt_x", "true_tip_row": "gt_y"})
    result["raw_error_px"] = np.hypot(result.raw_x-result.gt_x, result.raw_y-result.gt_y)
    result["global_corrected_x"] = result.raw_x + correction[0]
    result["global_corrected_y"] = result.raw_y + correction[1]
    result["global_correction_error_px"] = np.hypot(result.global_corrected_x-result.gt_x,
                                                      result.global_corrected_y-result.gt_y)
    return result


def save_baselines(train: pd.DataFrame, val: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, tuple[float, float]]:
    dx = float((train.true_tip_col - train.raw_col).mean())
    dy = float((train.true_tip_row - train.raw_row).mean())
    tr, va = base_predictions(train, (dx,dy)), base_predictions(val, (dx,dy))
    raw_rows = []
    global_rows = []
    for split, frame in [("train", tr), ("val", va)]:
        raw_rows.append({"split": split, "baseline": "system_raw", **error_metrics(frame.raw_error_px.to_numpy())})
        global_rows.append({"split": split, "baseline": "train_global_mean_dxdy", "train_mean_dx_px": dx,
                            "train_mean_dy_px": dy, **error_metrics(frame.global_correction_error_px.to_numpy())})
    pd.DataFrame(raw_rows).to_csv(output / "raw_baseline_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(global_rows).to_csv(output / "global_mean_correction_baseline.csv", index=False, encoding="utf-8-sig")
    say(f"[Baseline] Train-only global correction dx={dx:.3f}, dy={dy:.3f} px")
    say(f"[Baseline] Val raw mean={raw_rows[1]['mean_px']:.3f}, median={raw_rows[1]['median_px']:.3f} px; global-mean-corrected mean={global_rows[1]['mean_px']:.3f} px")
    return tr, va, (dx,dy)


def independent_peak(heatmap: np.ndarray, radius: int) -> tuple[float, float, float, int, int]:
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    first = float(heatmap[y, x])
    second_map = heatmap.copy()
    yy, xx = np.ogrid[:heatmap.shape[0], :heatmap.shape[1]]
    second_map[(xx-x)**2+(yy-y)**2 <= radius**2] = -np.inf
    second = float(np.max(second_map))
    return first, second, first-second, int(x), int(y)


def inference(model: nn.Module, dataset: LocalTipDataset, device: torch.device,
              split: str, baseline: pd.DataFrame, radius: int) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=device.type=="cuda")
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[{split}] Best checkpoint inference", unit="batch", dynamic_ncols=True):
            image = batch["image"].to(device, non_blocking=True)
            pred = model(image)
            if pred.shape[1:] != (1,64,64) or not torch.isfinite(pred).all():
                raise RuntimeError(f"Invalid inference heatmap {tuple(pred.shape)}: {batch['dicom_path'][0]}")
            maps = pred[:,0].detach().cpu().numpy()
            indices = batch["sample_index"].cpu().numpy().astype(int)
            origins = batch["crop_origin"].cpu().numpy()
            for j, index in enumerate(indices):
                row = baseline.iloc[int(index)]
                x0, y0 = float(origins[j,0]), float(origins[j,1])
                peak, second, margin, hx, hy = independent_peak(maps[j], radius)
                px_local, py_local = hx*4.0, hy*4.0
                px, py = x0+px_local, y0+py_local
                err = math.hypot(px-float(row.gt_x), py-float(row.gt_y))
                z = maps[j].reshape(-1).astype(np.float64)
                z = np.exp(z-z.max()); z /= z.sum()
                entropy = float(-(z*np.log(z+1e-30)).sum()/math.log(z.size))
                records.append({
                    "split": split, "case_id": str(row.case_id), "series_id": str(row.series_id),
                    "sequence_identifier": str(row.sequence_identifier), "frame_index": int(row.frame_index),
                    "dicom_path": str(row.dicom_path), "raw_x": float(row.raw_x), "raw_y": float(row.raw_y),
                    "gt_x": float(row.gt_x), "gt_y": float(row.gt_y), "crop_x0": x0, "crop_y0": y0,
                    "gt_local_x": float(row.gt_x)-x0, "gt_local_y": float(row.gt_y)-y0,
                    "pred_x_heatmap": hx, "pred_y_heatmap": hy,
                    "pred_x_local": px_local, "pred_y_local": py_local,
                    "pred_x_original": px, "pred_y_original": py,
                    "raw_error_px": float(row.raw_error_px), "model_error_px": err,
                    "global_correction_error_px": float(row.global_correction_error_px),
                    "gain_px": float(row.raw_error_px)-err,
                    "heatmap_peak_value": peak, "second_independent_peak_value": second,
                    "peak_margin": margin, "heatmap_entropy": entropy,
                })
    result = pd.DataFrame(records)
    if len(result) != len(dataset) or not np.isfinite(result.model_error_px).all():
        raise RuntimeError(f"Incomplete or nonfinite {split} inference")
    if not np.allclose(result.gt_x.to_numpy(), baseline.gt_x.to_numpy()) or not np.allclose(result.gt_y.to_numpy(), baseline.gt_y.to_numpy()):
        raise RuntimeError(f"Inference manifest row order mismatch: {split}")
    return result


def gain_metrics(df: pd.DataFrame) -> dict[str, Any]:
    gain = df.gain_px.to_numpy()
    return {
        "n": len(df), "mean_gain_px": float(gain.mean()), "median_gain_px": float(np.median(gain)),
        "improved_pct": float(100*np.mean(gain >= 1)), "worsened_pct": float(100*np.mean(gain <= -1)),
        "unchanged_abs_gain_lt_1_pct": float(100*np.mean(np.abs(gain) < 1)),
        **{f"gain_gt_{t}px_pct": float(100*np.mean(gain > t)) for t in (5,10,20)},
        **{f"loss_gt_{t}px_pct": float(100*np.mean(gain < -t)) for t in (5,10,20)},
    }


def grouped_metrics(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple): keys = (keys,)
        row = dict(zip(group_columns, keys))
        row.update({"n": len(group), "raw_mean_px": float(group.raw_error_px.mean()),
                    "raw_median_px": float(group.raw_error_px.median()),
                    "raw_p95_px": float(np.percentile(group.raw_error_px,95)),
                    "model_mean_px": float(group.model_error_px.mean()),
                    "model_median_px": float(group.model_error_px.median()),
                    "model_p95_px": float(np.percentile(group.model_error_px,95)),
                    "mean_gain_px": float(group.gain_px.mean()),
                    "improved_pct": float(100*np.mean(group.gain_px >= 1)),
                    "worsened_pct": float(100*np.mean(group.gain_px <= -1))})
        rows.append(row)
    return pd.DataFrame(rows)


def spearman_no_blas(left: pd.Series, right: pd.Series) -> float:
    """Average-tie Spearman using scalar sums; avoids this env's duplicate OpenMP runtime."""
    a=left.rank(method="average").tolist()
    b=right.rank(method="average").tolist()
    ma=math.fsum(a)/len(a); mb=math.fsum(b)/len(b)
    cov=math.fsum((x-ma)*(y-mb) for x,y in zip(a,b))
    va=math.fsum((x-ma)**2 for x in a)
    vb=math.fsum((y-mb)**2 for y in b)
    return cov/math.sqrt(va*vb) if va>0 and vb>0 else float("nan")


def quantization_floor(manifest: pd.DataFrame, output: Path) -> dict[str, Any]:
    rows = []
    for row in manifest.itertuples(index=False):
        x0, y0 = crop_origin(float(row.raw_col), float(row.raw_row))
        lx, ly = float(row.true_tip_col)-x0, float(row.true_tip_row)-y0
        hx = min(max(round_half_up(lx/4),0),63)
        hy = min(max(round_half_up(ly/4),0),63)
        qx, qy = x0+4*hx, y0+4*hy
        rows.append({"split": row.split, "case_id": row.case_code, "series_id": row.series_uid,
                     "dicom_path": row.dicom_absolute_path, "gt_x": row.true_tip_col, "gt_y": row.true_tip_row,
                     "quant_x": qx, "quant_y": qy,
                     "quantization_error_px": math.hypot(qx-float(row.true_tip_col),qy-float(row.true_tip_row))})
    qf = pd.DataFrame(rows)
    qf.to_csv(output / "heatmap_quantization_floor.csv", index=False, encoding="utf-8-sig")
    return {split: error_metrics(g.quantization_error_px.to_numpy()) for split,g in qf.groupby("split")}


def save_metric_tables(train: pd.DataFrame, val: pd.DataFrame, output: Path) -> dict[str, Any]:
    pd.DataFrame([{"split":"train", "model":"local_hrnet_256", **error_metrics(train.model_error_px.to_numpy())}]).to_csv(
        output/"train_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"split":"val", "model":"local_hrnet_256", **error_metrics(val.model_error_px.to_numpy())}]).to_csv(
        output/"val_metrics.csv",index=False,encoding="utf-8-sig")
    train.to_csv(output/"train_predictions.csv",index=False,encoding="utf-8-sig")
    val.to_csv(output/"val_predictions.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"split":split, **gain_metrics(df)} for split,df in [("train",train),("val",val)]]).to_csv(
        output/"refinement_gain_summary.csv",index=False,encoding="utf-8-sig")
    bins = [0,5,10,20,40,60,np.inf]
    labels = ["[0,5)","[5,10)","[10,20)","[20,40)","[40,60)","[60,+)"]
    binned = val.copy()
    binned["raw_error_bin"] = pd.cut(binned.raw_error_px,bins=bins,labels=labels,right=False)
    bin_rows = []
    for label in labels:
        group = binned.loc[binned.raw_error_bin == label]
        if not len(group):
            bin_rows.append({"raw_error_bin":label,"n":0})
            continue
        bin_rows.append({"raw_error_bin":label,"n":len(group),
                         "raw_mean_px":float(group.raw_error_px.mean()),"raw_median_px":float(group.raw_error_px.median()),
                         "model_mean_px":float(group.model_error_px.mean()),"model_median_px":float(group.model_error_px.median()),
                         "gain_mean_px":float(group.gain_px.mean()),"gain_median_px":float(group.gain_px.median()),
                         "improved_pct":float(100*np.mean(group.gain_px>=1)),
                         "worsened_pct":float(100*np.mean(group.gain_px<=-1)),
                         "model_lt_5px_pct":float(100*np.mean(group.model_error_px<5)),
                         "model_lt_10px_pct":float(100*np.mean(group.model_error_px<10))})
    pd.DataFrame(bin_rows).to_csv(output/"refinement_gain_by_raw_bin.csv",index=False,encoding="utf-8-sig")
    grouped_metrics(val,["case_id"]).to_csv(output/"metrics_by_case.csv",index=False,encoding="utf-8-sig")
    grouped_metrics(val,["case_id","series_id","sequence_identifier"]).to_csv(
        output/"metrics_by_series.csv",index=False,encoding="utf-8-sig")
    all_preds = pd.concat([train,val],ignore_index=True)
    confidence_rows = []
    for label,group in [("train",train),("val",val)]+[(f"val_{case}",g) for case,g in val.groupby("case_id")]:
        confidence_rows.append({"group":label,"n":len(group),
            "peak_mean":float(group.heatmap_peak_value.mean()),"peak_median":float(group.heatmap_peak_value.median()),
            "margin_mean":float(group.peak_margin.mean()),"margin_median":float(group.peak_margin.median()),
            "entropy_mean":float(group.heatmap_entropy.mean()),"entropy_median":float(group.heatmap_entropy.median()),
            "spearman_error_peak":spearman_no_blas(group.model_error_px,group.heatmap_peak_value),
            "spearman_error_margin":spearman_no_blas(group.model_error_px,group.peak_margin),
            "spearman_gain_margin":spearman_no_blas(group.gain_px,group.peak_margin)})
    pd.DataFrame(confidence_rows).to_csv(output/"confidence_analysis.csv",index=False,encoding="utf-8-sig")
    return {"train_model":error_metrics(train.model_error_px.to_numpy()),
            "val_model":error_metrics(val.model_error_px.to_numpy()),
            "val_gain":gain_metrics(val),
            "val_raw":error_metrics(val.raw_error_px.to_numpy()),
            "val_global":error_metrics(val.global_correction_error_px.to_numpy())}


def chart_save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def montage(rows: pd.DataFrame, title: str, destination: Path) -> None:
    n = min(len(rows), 12)
    fig, axes = plt.subplots(3,4,figsize=(15,12))
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, rows.head(n).itertuples(index=False)):
        crop = crop_fixed(read_full_minmax(str(row.dicom_path)), int(row.crop_x0), int(row.crop_y0))
        ax.imshow(crop,cmap="gray",vmin=0,vmax=1)
        ax.scatter(float(row.raw_x)-float(row.crop_x0),float(row.raw_y)-float(row.crop_y0),
                   s=42,marker="o",edgecolors="cyan",facecolors="none",linewidths=1.2,label="raw")
        ax.scatter(float(row.gt_local_x),float(row.gt_local_y),
                   s=42,marker="o",edgecolors="red",facecolors="none",linewidths=1.2,label="GT")
        ax.scatter(float(row.pred_x_local),float(row.pred_y_local),
                   s=45,marker="x",c="lime",linewidths=1.3,label="model")
        ax.set_title(f"{row.case_id} / {row.sequence_identifier} / f{row.frame_index}\n"
                     f"raw {row.raw_error_px:.1f}px → model {row.model_error_px:.1f}px",fontsize=9)
        ax.set_xlim(0,255); ax.set_ylim(255,0)
        ax.axis("off")
    fig.suptitle(title+" | raw cyan, GT red, model green",fontsize=14)
    chart_save(fig,destination)


def generate_figures(train: pd.DataFrame, val: pd.DataFrame, log: pd.DataFrame, output: Path) -> None:
    figdir = output/"figures"
    figdir.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(log.epoch,log.train_loss,label="Train weighted MSE")
    ax.plot(log.epoch,log.val_loss,label="Val weighted MSE")
    ax.set(xlabel="Epoch",ylabel="Weighted MSE",title="Training loss")
    ax.grid(alpha=.2);ax.legend();chart_save(fig,figdir/"training_loss.png")

    fig,ax=plt.subplots(figsize=(8,5))
    for name, values in [("System raw",val.raw_error_px),("Train-only global correction",val.global_correction_error_px),
                         ("Local HRNet",val.model_error_px)]:
        sorted_values=np.sort(values.to_numpy()); ax.plot(sorted_values,np.arange(1,len(sorted_values)+1)/len(sorted_values),label=name)
    ax.set(xlabel="Original-space localization error (px)",ylabel="Cumulative fraction",title="Val error CDF")
    ax.grid(alpha=.2);ax.legend();chart_save(fig,figdir/"raw_vs_model_cdf.png")

    fig,ax=plt.subplots(figsize=(7,7))
    for case,g in val.groupby("case_id"):
        ax.scatter(g.raw_error_px,g.model_error_px,s=8,alpha=.45,label=case)
    limit=max(60,float(np.percentile(np.r_[val.raw_error_px,val.model_error_px],99)))
    ax.plot([0,limit],[0,limit],"k--",lw=1,label="Equal error")
    ax.set(xlim=(0,limit),ylim=(0,limit),xlabel="Raw error (px)",ylabel="Model error (px)",title="Val paired errors")
    ax.grid(alpha=.2);ax.legend();chart_save(fig,figdir/"raw_vs_model_scatter.png")

    fig,ax=plt.subplots(figsize=(8,5))
    ax.hist(val.gain_px,bins=60,color="steelblue",edgecolor="white")
    ax.axvline(0,color="black",ls="--");ax.set(xlabel="Gain = raw error − model error (px)",ylabel="Val frames",title="Refinement gain")
    chart_save(fig,figdir/"gain_histogram.png")

    fig,ax=plt.subplots(figsize=(8,5))
    for case,g in val.groupby("case_id"):
        ax.scatter(g.raw_error_px,g.gain_px,s=9,alpha=.45,label=case)
    ax.axhline(0,color="black",ls="--");ax.set(xlabel="Raw error (px)",ylabel="Gain (px)",title="Does refinement rescue larger raw errors?")
    ax.grid(alpha=.2);ax.legend();chart_save(fig,figdir/"gain_vs_raw_error.png")

    for grouping,filename in [(["case_id"],"metrics_by_case.png"),(["case_id","sequence_identifier"],"metrics_by_series.png")]:
        metrics=grouped_metrics(val,grouping)
        labels=metrics[grouping].astype(str).agg("/".join,axis=1)
        x=np.arange(len(metrics)); fig,ax=plt.subplots(figsize=(max(8,len(metrics)*2),5))
        ax.bar(x-.2,metrics.raw_mean_px,width=.4,label="Raw mean")
        ax.bar(x+.2,metrics.model_mean_px,width=.4,label="Model mean")
        ax.set_xticks(x,labels,rotation=25,ha="right")
        ax.set(ylabel="Mean error (px)",title="Val by "+" / ".join(grouping))
        ax.grid(axis="y",alpha=.2);ax.legend();chart_save(fig,figdir/filename)

    rescues=val.sort_values("gain_px",ascending=False).head(12)
    regressions=val.loc[val.raw_error_px<10].sort_values("gain_px").head(12)
    if len(regressions)<12:
        regressions=pd.concat([regressions,val.sort_values("gain_px").head(12-len(regressions))]).drop_duplicates("dicom_path")
    good=val.loc[(val.raw_error_px<10)&(val.model_error_px<10)].sort_values("model_error_px").head(12)
    fail=val.loc[(val.raw_error_px>=20)&(val.model_error_px>=20)].sort_values("model_error_px",ascending=False).head(12)
    for name,frame in [("best_rescues",rescues),("worst_regressions",regressions),
                       ("both_good",good),("both_fail",fail)]:
        montage(frame,name.replace("_"," ").title(),figdir/f"{name}.png")

    for (case,series),group in val.groupby(["case_id","sequence_identifier"],sort=True):
        g=group.sort_values("frame_index")
        frame=g.frame_index.to_numpy();fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True)
        for ax,axis in zip(axes[:2],["x","y"]):
            ax.plot(frame,g[f"gt_{axis}"],lw=1,label="GT")
            ax.plot(frame,g[f"raw_{axis}"],lw=1,label="Raw")
            ax.plot(frame,g[f"pred_{axis}_original"],lw=1,label="Model")
            ax.set_ylabel(f"{axis} (original px)");ax.grid(alpha=.2);ax.legend(ncol=3)
        axes[2].plot(frame,g.raw_error_px,lw=1,label="Raw error")
        axes[2].plot(frame,g.model_error_px,lw=1,label="Model error")
        axes[2].set(xlabel="Frame index",ylabel="Error (px)");axes[2].grid(alpha=.2);axes[2].legend()
        fig.suptitle(f"Val trajectory {case} / {series} (unsmoothed)")
        safe="".join(c if c.isalnum() or c in "-_" else "_" for c in str(series))
        chart_save(fig,figdir/f"trajectory_{case}_{safe}.png")
    say(f"[Figures] Saved {len(list(figdir.glob('*.png')))} PNG figures in {figdir}")


def fmt_metrics(name: str, values: dict[str, Any]) -> str:
    return (f"| {name} | {values['mean_px']:.3f} | {values['median_px']:.3f} | "
            f"{values['p95_px']:.3f} | {values['p99_px']:.3f} | {values['max_px']:.3f} | "
            f"{values['lt_5px_pct']:.2f}% | {values['lt_10px_pct']:.2f}% | {values['lt_20px_pct']:.2f}% |")


def write_readme(config: dict[str, Any], summary: dict[str, Any], train: pd.DataFrame,
                 val: pd.DataFrame, qfloor: dict[str, Any], output: Path) -> None:
    log=pd.read_csv(output/"training_log.csv")
    run=json.loads((output/"run_summary.json").read_text(encoding="utf-8"))
    best=int(run["best_epoch"])
    bycase=grouped_metrics(val,["case_id"])
    byseries=grouped_metrics(val,["case_id","series_id","sequence_identifier"])
    bins=pd.read_csv(output/"refinement_gain_by_raw_bin.csv")
    easy5=val[val.raw_error_px<5];easy10=val[val.raw_error_px<10]
    vals=[
        "# Phase 6B — raw-centered 256×256 local tip refinement baseline",
        "",
        "This is a completed single-variable local-refinement experiment, not a deployment or clinical-validation claim. The frozen Model-clean v1 split is 2,784 Train frames (8 cases, 12 Series) and 928 Val frames (3 held-out cases, 4 Series). No Test set or prior tip-localization checkpoint was used.",
        "",
        "## Frozen setup and provenance",
        "",
        "- `raw_col/raw_row` is the product's automatic coarse tip, available at inference; it alone sets the crop center. `true_tip_col/true_tip_row` is final corrected GT, used only for supervision and evaluation.",
        "- First normalize the entire 1024×1024 DICOM frame by per-frame min–max, then take a fixed 256×256 window with `(x0,y0) = floor(raw_x−128, raw_y−128)` and zero padding outside the source. No crop-level re-stretch, augmentation, coordinate jitter, image noise, or ImageNet mean/std normalization.",
        "- HRNet-W18-small-v2 via timm, ImageNet-1K pretrained backbone and fresh 1-channel heatmap head; input `[B,3,256,256]`, output `[B,1,64,64]`; Gaussian sigma=2; integer heatmap argmax, multiplied by 4 and translated by crop origin back to original pixels.",
        "- Weighted MSE is exactly `mean((1+9×GT_heatmap)×(Pred−GT_heatmap)^2)`, without mean-one weight renormalization. Adam LR=0.001, weight decay=0, batch=16, MultiStepLR milestones 120/160 and gamma=0.2, seed 20260918; up to 120 epochs, Val-mean early stopping patience=15 and min delta=0.1 px. These optimize settings match Phase 5C; the key change is raw-centered 256 ROI/64 heatmap rather than 512 full frame/128 heatmap.",
        f"- Best checkpoint: `checkpoints/best_by_val_mean_error.pth` at epoch {best}; {run['epochs_completed']} epochs completed; stop reason `{run['stop_reason']}`. Its selection uses held-out Val mean error, so this Val is not an untouched final Test set.",
        "- `split_integrity.json`, `pretrained_loading_report.json`, `batch_sanity.json`, `coordinate_roundtrip_100.csv`, and `config.json` document the checks. The 100-frame coordinate round trip passed at <1e-6 px. All GT points lie inside the observed 256 crop; this does not guarantee coverage of unseen larger raw errors.",
        "",
        "## Paired Val comparison (original 1024-pixel coordinates)",
        "",
        "| Method | Mean px | Median px | P95 px | P99 px | Max px | <5 px | <10 px | <20 px |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        fmt_metrics("System raw",summary["val_raw"]),
        fmt_metrics("Train-only global mean correction",summary["val_global"]),
        fmt_metrics("Local HRNet",summary["val_model"]),
        "",
        f"Train model mean error at best checkpoint: {summary['train_model']['mean_px']:.3f} px; Val model mean error: {summary['val_model']['mean_px']:.3f} px. The difference is descriptive and not a clinical generalization estimate.",
        f"Mean paired gain (raw−model): {summary['val_gain']['mean_gain_px']:.3f} px; median {summary['val_gain']['median_gain_px']:.3f} px; improved (≥1 px) {summary['val_gain']['improved_pct']:.2f}%; worsened (≤−1 px) {summary['val_gain']['worsened_pct']:.2f}%; unchanged (<1 px absolute gain) {summary['val_gain']['unchanged_abs_gain_lt_1_pct']:.2f}%.",
        "",
        "## Where refinement helps or hurts",
        "",
        "| Raw error bin (px) | N | Raw mean | Model mean | Mean gain | Improved | Worsened | Model <5 px | Model <10 px |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bins.itertuples(index=False):
        if int(row.n):
            vals.append(f"| {row.raw_error_bin} | {row.n} | {row.raw_mean_px:.2f} | {row.model_mean_px:.2f} | {row.gain_mean_px:.2f} | {row.improved_pct:.1f}% | {row.worsened_pct:.1f}% | {row.model_lt_5px_pct:.1f}% | {row.model_lt_10px_pct:.1f}% |")
        else:
            vals.append(f"| {row.raw_error_bin} | 0 | — | — | — | — | — | — | — |")
    vals.extend([
        "",
        f"Easy raw frames: raw<5 px N={len(easy5)}, model mean={easy5.model_error_px.mean() if len(easy5) else float('nan'):.2f} px, worsened ≥1 px={100*np.mean(easy5.gain_px<=-1) if len(easy5) else float('nan'):.1f}%; raw<10 px N={len(easy10)}, model mean={easy10.model_error_px.mean() if len(easy10) else float('nan'):.2f} px, worsened ≥1 px={100*np.mean(easy10.gain_px<=-1) if len(easy10) else float('nan'):.1f}%.",
        f"Large gains >5/10/20 px: {summary['val_gain']['gain_gt_5px_pct']:.1f}% / {summary['val_gain']['gain_gt_10px_pct']:.1f}% / {summary['val_gain']['gain_gt_20px_pct']:.1f}%; large regressions <−5/−10/−20 px: {summary['val_gain']['loss_gt_5px_pct']:.1f}% / {summary['val_gain']['loss_gt_10px_pct']:.1f}% / {summary['val_gain']['loss_gt_20px_pct']:.1f}%.",
        "",
        "## Held-out cases and Series",
        "",
        "| Case | N | Raw mean | Model mean | Model median | Model P95 | Mean gain | Improved | Worsened |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in bycase.itertuples(index=False):
        vals.append(f"| {row.case_id} | {row.n} | {row.raw_mean_px:.2f} | {row.model_mean_px:.2f} | {row.model_median_px:.2f} | {row.model_p95_px:.2f} | {row.mean_gain_px:.2f} | {row.improved_pct:.1f}% | {row.worsened_pct:.1f}% |")
    vals.extend(["", "The complete per-Series table is `metrics_by_series.csv`; individual coordinates and paired errors are in `val_predictions.csv`.", ""])
    for row in byseries.itertuples(index=False):
        vals.append(f"- {row.case_id}/{row.sequence_identifier}: N={row.n}, raw mean {row.raw_mean_px:.2f} px, model mean {row.model_mean_px:.2f} px, mean gain {row.mean_gain_px:.2f} px.")
    conf=pd.read_csv(output/"confidence_analysis.csv").set_index("group")
    worst=val.nsmallest(20,"gain_px")
    vals.extend([
        "", "## Representation, confidence, and failure cautions", "",
        f"64×64 integer argmax quantization floor (Train): mean {qfloor['train']['mean_px']:.3f}, median {qfloor['train']['median_px']:.3f}, P95 {qfloor['train']['p95_px']:.3f}, max {qfloor['train']['max_px']:.3f} px. Val floor: mean {qfloor['val']['mean_px']:.3f}, P95 {qfloor['val']['p95_px']:.3f} px. This is coordinate discretization, not model error.",
        f"Peak/margin/entropy medians: Train {conf.loc['train','peak_median']:.3f}/{conf.loc['train','margin_median']:.3f}/{conf.loc['train','entropy_median']:.3f}; Val {conf.loc['val','peak_median']:.3f}/{conf.loc['val','margin_median']:.3f}/{conf.loc['val','entropy_median']:.3f}. Val Spearman(error, margin)={conf.loc['val','spearman_error_margin']:.3f}; this is exploratory, not calibrated uncertainty.",
        f"Worst 20 negative-gain frames by case: {worst.case_id.value_counts().to_dict()}. These are descriptive examples; montage markers show whether prediction appears to land on a catheter-like or other structure, but no anatomical cause is asserted automatically.",
        "A 256 ROI had 100% observed Train/Val GT coverage in Phase 6A, yet synthetic extra coarse-error tails can move GT outside it; this training result does not establish robustness to those tails. Val has only three held-out cases with correlated adjacent frames.",
        "",
        "## Files and interpretation", "",
        "All CSV coordinates and errors use original-image pixels unless explicitly suffixed `_local` or `_heatmap`. `raw_baseline_metrics.csv`, `global_mean_correction_baseline.csv`, `train_metrics.csv`, `val_metrics.csv`, `refinement_gain_summary.csv`, `refinement_gain_by_raw_bin.csv`, `metrics_by_case.csv`, `metrics_by_series.csv`, `heatmap_quantization_floor.csv`, and `confidence_analysis.csv` are separate auditable summaries. `training_log.csv` records every completed epoch; `train_predictions.csv` and `val_predictions.csv` record every frame. The figures include paired CDF/scatter/gain, case/Series comparisons, four crop montage categories, and unsmoothed per-Val-Series trajectories. Outputs are confined to this Phase 6B folder; original DICOM/CSV/JSON and frozen split were not modified.",
    ])
    successful=(summary["val_gain"]["improved_pct"]>50 and summary["val_model"]["mean_px"]<summary["val_raw"]["mean_px"])
    vals.extend(["", "## Evidence-based judgment", "",
        ("This local baseline improves a majority of Val frames and lowers overall mean error versus raw, but easy-frame regressions, per-case consistency, and tail failures still determine whether it is suitable as a product refinement." if successful else
         "This local baseline does not meet the simple majority-improved and lower-mean-than-raw criterion simultaneously. It should not replace the system raw coordinate without a separate safeguard or further experiment."),
        "Next questions only (not executed): (1) whether a raw-confidence gate prevents worsening already-accurate frames; (2) whether a single bounded coordinate-jitter experiment helps rare coarse-error tails; (3) whether a 384 ROI improves robustness without losing local precision. Each should change one main variable and retain paired raw-vs-model, case/Series, and tail metrics.",
        "", "**STOP:** No Phase 6C training, Test evaluation, new loss, or automatic parameter search was started."])
    (output/"README.md").write_text("\n".join(vals)+"\n",encoding="utf-8")


def validate_frozen_config(config: dict[str, Any], workspace: Path) -> None:
    import yaml
    previous=yaml.safe_load((workspace/"configs/hrnet_phase5c_weighted_mse.yaml").read_text(encoding="utf-8"))
    for section,keys in {"experiment":["seed","expected_train_frames","expected_train_cases","expected_train_series","expected_val_frames","expected_val_cases","expected_val_series","expected_val_case_codes"],
                         "model":["implementation","variant","pretrained","pretrained_tag","pretrained_hf_hub_id","output_channels","initialization"],
                         "training":["max_epochs","optimizer","learning_rate","weight_decay","loss","scheduler","scheduler_milestones","scheduler_gamma","deterministic","cudnn_benchmark","early_stopping","collapse_monitor","gradient_explosion_norm"],
                         "weighted_loss":["equation","alpha","normalize_weight_to_mean_one","foreground_diagnostic_threshold","prediction_abs_safety_stop"]}.items():
        for key in keys:
            if config[section][key] != previous[section][key]:
                raise RuntimeError(f"Frozen Phase 5C setting changed: {section}.{key}")
    if config["data"]["augmentation"] != "none" or config["data"]["roi_center"] != "raw":
        raise RuntimeError("Phase 6B must be raw-centered with no augmentation")
    if config["data"]["input_size"] != [256,256] or config["data"]["heatmap_size"] != [64,64]:
        raise RuntimeError("Phase 6B geometry differs from 256 -> 64")


def main() -> int:
    parser=argparse.ArgumentParser(description="Phase 6B raw-centered 256 local refinement")
    parser.add_argument("--config",required=True)
    parser.add_argument("--finalize-only",action="store_true",help="Reuse already completed best checkpoint for reporting; never retrain")
    args=parser.parse_args()
    config=phase4.load_config(args.config)
    workspace=Path(config["_workspace_root"])
    output=phase4.resolve_path(config,config["project"]["output_dir"])
    global OUT
    OUT=output
    output.mkdir(parents=True,exist_ok=True)
    (output/"figures").mkdir(exist_ok=True)
    (output/"logs").mkdir(exist_ok=True)
    try:
        say("[1/7] Frozen split, configuration, and raw-centered coordinate checks")
        validate_frozen_config(config,workspace)
        (output/"config.json").write_text(json.dumps(phase4.config_for_save(config),indent=2,ensure_ascii=False),encoding="utf-8")
        manifest_path=phase4.resolve_path(config,config["project"]["manifest"])
        manifest=pd.read_csv(manifest_path,dtype={"case_code":str,"case_identifier":str},low_memory=False)
        train,val,integrity=phase4.split_integrity(manifest,config,manifest_path)
        if not np.isfinite(manifest[["raw_col","raw_row","true_tip_col","true_tip_row"]].to_numpy(dtype=float)).all():
            raise RuntimeError("Missing/nonfinite raw or GT coordinates")
        (output/"split_integrity.json").write_text(json.dumps(integrity,indent=2,ensure_ascii=False),encoding="utf-8")
        roundtrip_test(pd.concat([train,val],ignore_index=True),output)
        say("[Integrity] Train/Val case, Series, DICOM intersections: 0 / 0 / 0")
        say("[2/7] System raw and Train-only global-mean-correction baselines")
        base_train,base_val,_=save_baselines(train,val,output)
        if not args.finalize_only:
            if (output/"training_log.csv").exists():
                raise RuntimeError("Training log already exists; refusing to overwrite. Use --finalize-only only if best checkpoint completed.")
            say("[3/7] Verify ImageNet pretrained backbone, 256->64 model, and one-batch gradient")
            phase4.configure_reproducibility(int(config["experiment"]["seed"]),True,False)
            report=phase5a.verify_pretrained_loading(config,output)
            say(f"[Pretrained] PASS; {report['parameters_different_from_same_seed_random']:,} retained backbone parameters differ from same-seed random initialization")
            phase4.CHECKPOINT_WARNING="PHASE 6B: raw-centered 256 local refinement; best by held-out Val mean original-space error"
            phase4.PHASE3_WARNING="No prior tip-localization checkpoint loaded; ImageNet pretrained backbone + fresh heatmap head"
            phase4.XAHeatmapDataset=LocalTipDataset
            phase4.build_model=phase5a.build_pretrained_model
            phase4.build_criterion=build_criterion
            phase4.train_one_epoch=train_epoch
            phase4.validate=validate_epoch
            phase4.save_checkpoint=save_checkpoint
            phase4.save_model_summary=model_summary
            phase4.print_header=phase6b_header
            phase4.environment_text=lambda device: SOURCE_ENVIRONMENT(device)+"\nPhase 6B raw-centered 256 crop; pretrained backbone; fresh head; weighted MSE; no augmentation or prior tip checkpoint.\n"
            say("[4/7] Formal Train/Val training; progress, Val mean, best epoch, patience below")
            original_argv=sys.argv
            try:
                sys.argv=[original_argv[0],"--config",args.config]
                phase4.main()
            finally:
                sys.argv=original_argv
            run_path=output/"run_summary.json"
            run=json.loads(run_path.read_text(encoding="utf-8"))
            run.update({"formal_training_initialization":"ImageNet pretrained backbone + fresh heatmap head",
                        "phase4_checkpoint_loaded":False,"phase5a_checkpoint_loaded":False,
                        "phase5b_checkpoint_loaded":False,"phase5c_checkpoint_loaded":False,
                        "crop_center":"raw_col/raw_row","input_size":[256,256],"heatmap_size":[64,64]})
            run_path.write_text(json.dumps(run,indent=2,ensure_ascii=False),encoding="utf-8")
            # The generic Phase 4 engine names local-coordinate predictions *original*.
            # Preserve them explicitly as local diagnostics, never as the final prediction file.
            legacy=output/"val_predictions_best.csv"
            if legacy.exists(): legacy.rename(output/"val_predictions_best_local_internal.csv")
        else:
            if not (output/"checkpoints/best_by_val_mean_error.pth").exists():
                raise FileNotFoundError("--finalize-only requires completed best checkpoint")
            say("[Resume reporting] Reusing completed best checkpoint; no training")
        say("[5/7] Best-checkpoint full Train/Val inference and confidence")
        logger=logging.getLogger("phase6b_finalize")
        if "Train" not in DATASETS:
            DATASETS["Train"]=LocalTipDataset(train,"Train",(256,256),(64,64),2,3,True,logger)
        if "Val" not in DATASETS:
            DATASETS["Val"]=LocalTipDataset(val,"Val",(256,256),(64,64),2,3,True,logger)
        device=phase4.select_device()
        if device.type!="cuda": raise RuntimeError("Phase 6B formal inference requires CUDA")
        model=phase4.HRNetTipHeatmap("hrnet_w18_small_v2",False,1,(64,64)).to(device)
        checkpoint=torch.load(output/"checkpoints/best_by_val_mean_error.pth",map_location=device,weights_only=False)
        if checkpoint.get("phase5c_checkpoint_loaded",False): raise RuntimeError("Unexpected Phase 5C checkpoint provenance")
        model.load_state_dict(checkpoint["model_state_dict"],strict=True)
        radius=int(config["heatmap_confidence"]["independent_peak_suppression_radius"])
        train_pred=inference(model,DATASETS["Train"],device,"train",base_train,radius)
        val_pred=inference(model,DATASETS["Val"],device,"val",base_val,radius)
        del model
        say("[6/7] Paired gains, raw-error bins, quantization floor, and figures")
        summary=save_metric_tables(train_pred,val_pred,output)
        qfloor=quantization_floor(pd.concat([train,val],ignore_index=True),output)
        training_log=pd.read_csv(output/"training_log.csv")
        subprocess.run([sys.executable,str(Path(__file__).with_name("phase6b_render_figures.py")),
                        "--output",str(output)],check=True)
        say("[7/7] Report and terminal summary")
        write_readme(config,summary,train_pred,val_pred,qfloor,output)
        run=json.loads((output/"run_summary.json").read_text(encoding="utf-8"))
        say("="*66)
        say(f"Best epoch {run['best_epoch']} / completed {run['epochs_completed']} | {run['stop_reason']}")
        say(f"Val raw: mean {summary['val_raw']['mean_px']:.3f}, median {summary['val_raw']['median_px']:.3f}, P95 {summary['val_raw']['p95_px']:.3f} px")
        say(f"Val Train-only correction: mean {summary['val_global']['mean_px']:.3f} px")
        say(f"Val local HRNet: mean {summary['val_model']['mean_px']:.3f}, median {summary['val_model']['median_px']:.3f}, P95 {summary['val_model']['p95_px']:.3f}, P99 {summary['val_model']['p99_px']:.3f} px")
        say(f"Paired gain: mean {summary['val_gain']['mean_gain_px']:.3f} px; improved {summary['val_gain']['improved_pct']:.1f}%, worsened {summary['val_gain']['worsened_pct']:.1f}%")
        for row in grouped_metrics(val_pred,["case_id"]).itertuples(index=False):
            say(f"Val case {row.case_id}: raw {row.raw_mean_px:.2f} -> model {row.model_mean_px:.2f} px; gain {row.mean_gain_px:.2f}; improved {row.improved_pct:.1f}%")
        say(f"64x64 integer-argmax floor: Train mean {qfloor['train']['mean_px']:.3f}, P95 {qfloor['train']['p95_px']:.3f} px")
        say(f"Saved all records and figures to {output}")
        say("[STOP] Phase 6B finished. No next experiment or Test evaluation started.")
        return 0
    except BaseException as exc:
        (output/"phase6b_failure.json").write_text(json.dumps({"type":type(exc).__name__,"message":str(exc),
            "traceback":traceback.format_exc()},ensure_ascii=False,indent=2),encoding="utf-8")
        raise


if __name__=="__main__":
    raise SystemExit(main())
