from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
import yaml
from PIL import Image


METHOD_LABELS = {
    "minmax": "No clipping min-max",
    "p0_1_p99_9": "P0.1-P99.9",
    "p0_5_p99_5": "P0.5-P99.5",
    "p1_p99_5": "P1-P99.5",
    "dicom_window": "DICOM WC/WW",
}

METHOD_THRESHOLDS = {
    "minmax": ("pixel_min", "pixel_max"),
    "p0_1_p99_9": ("p0_1", "p99_9"),
    "p0_5_p99_5": ("p0_5", "p99_5"),
    "p1_p99_5": ("p1", "p99_5"),
}


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path)
    config["_workspace_root"] = str(path.parent.parent)
    return config


def resolve_project_path(config: dict[str, Any], value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(config["_workspace_root"]) / path
    return path.resolve()


def build_logger(log_path: Path, *, append: bool = False) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"phase2.{log_path.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="a" if append else "w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)
    return logger


def first_numeric(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (list, tuple)):
        return first_numeric(value[0]) if value else float("nan")
    try:
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            return first_numeric(list(value)[0])
    except TypeError:
        pass
    text = str(value).strip().strip("[]")
    for separator in ("\\", ","):
        if separator in text:
            text = text.split(separator, 1)[0]
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def read_dicom_stored(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    dicom_path = Path(path)
    try:
        ds = pydicom.dcmread(dicom_path)
        array = ds.pixel_array
    except Exception as exc:  # pragma: no cover - error path must retain filename
        raise RuntimeError(f"Failed to decode DICOM: {dicom_path}\n{exc}") from exc
    if array.ndim != 2:
        raise ValueError(f"Expected a single-frame 2-D DICOM, got {array.shape}: {dicom_path}")
    photometric = str(getattr(ds, "PhotometricInterpretation", ""))
    if photometric != "MONOCHROME2":
        raise ValueError(f"Expected MONOCHROME2, got {photometric!r}: {dicom_path}")
    metadata = {
        "BitsAllocated": int(getattr(ds, "BitsAllocated", 0)),
        "BitsStored": int(getattr(ds, "BitsStored", 0)),
        "PixelRepresentation": int(getattr(ds, "PixelRepresentation", 0)),
        "PhotometricInterpretation": photometric,
        "RescaleSlope": first_numeric(getattr(ds, "RescaleSlope", 1.0)),
        "RescaleIntercept": first_numeric(getattr(ds, "RescaleIntercept", 0.0)),
        "WindowCenter": first_numeric(getattr(ds, "WindowCenter", None)),
        "WindowWidth": first_numeric(getattr(ds, "WindowWidth", None)),
    }
    return np.asarray(array), metadata


def round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def centered_crop(array: np.ndarray, y: float, x: float, size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    cy = round_half_up(y)
    cx = round_half_up(x)
    before = size // 2
    after = size - before
    y0 = max(0, cy - before)
    y1 = min(array.shape[0], cy + after)
    x0 = max(0, cx - before)
    x1 = min(array.shape[1], cx + after)
    return array[y0:y1, x0:x1], (y0, y1, x0, x1)


def method_limits(method: str, frame_stats: pd.Series | dict[str, Any], metadata: dict[str, Any] | None = None) -> tuple[float, float]:
    if method in METHOD_THRESHOLDS:
        low_key, high_key = METHOD_THRESHOLDS[method]
        return float(frame_stats[low_key]), float(frame_stats[high_key])
    if method == "dicom_window":
        if metadata is None:
            raise ValueError("DICOM metadata are required for dicom_window")
        center = float(metadata["WindowCenter"])
        width = float(metadata["WindowWidth"])
        if not np.isfinite(center) or not np.isfinite(width) or width <= 1:
            raise ValueError(f"Invalid DICOM window center/width: center={center}, width={width}")
        # DICOM PS3.3 C.11.2 linear window definition.
        return center - 0.5 - (width - 1.0) / 2.0, center - 0.5 + (width - 1.0) / 2.0
    raise KeyError(f"Unknown preprocessing method: {method}")


def apply_preprocessing(
    image: np.ndarray,
    method: str,
    frame_stats: pd.Series | dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> np.ndarray:
    low, high = method_limits(method, frame_stats, metadata)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(image.shape, dtype=np.float32)
    result = np.clip(image.astype(np.float32), low, high)
    result = (result - np.float32(low)) / np.float32(high - low)
    return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)


def resize_float(image: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    output_h, output_w = output_hw
    pil_image = Image.fromarray(image.astype(np.float32), mode="F")
    resized = pil_image.resize((output_w, output_h), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def gaussian_heatmap(height: int, width: int, center_x: float, center_y: float, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    exponent = -((xx - np.float32(center_x)) ** 2 + (yy - np.float32(center_y)) ** 2) / np.float32(2.0 * sigma**2)
    return np.exp(exponent).astype(np.float32)


class XATipHeatmapDataset:
    """Configurable, read-only XA → normalized image + point heatmap pipeline."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        frame_stats: pd.DataFrame,
        method: str,
        input_hw: tuple[int, int],
        heatmap_hw: tuple[int, int],
        sigma: float,
    ) -> None:
        if method not in METHOD_LABELS:
            raise ValueError(f"Unsupported method {method!r}; choose from {sorted(METHOD_LABELS)}")
        self.manifest = manifest.reset_index(drop=True).copy()
        self.stats_by_path = frame_stats.set_index("dicom_absolute_path", drop=False)
        self.method = method
        self.input_hw = input_hw
        self.heatmap_hw = heatmap_hw
        self.sigma = float(sigma)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        path = str(row["dicom_absolute_path"])
        image, metadata = read_dicom_stored(path)
        stats = self.stats_by_path.loc[path]
        normalized = apply_preprocessing(image, self.method, stats, metadata)
        resized = resize_float(normalized, self.input_hw)

        original_h, original_w = image.shape
        input_h, input_w = self.input_hw
        heatmap_h, heatmap_w = self.heatmap_hw
        original_x = float(row["true_tip_col"])
        original_y = float(row["true_tip_row"])
        input_x = original_x * input_w / original_w
        input_y = original_y * input_h / original_h
        heatmap_x = original_x * heatmap_w / original_w
        heatmap_y = original_y * heatmap_h / original_h
        heatmap = gaussian_heatmap(heatmap_h, heatmap_w, heatmap_x, heatmap_y, self.sigma)
        return {
            "row": row,
            "metadata": metadata,
            "original": image,
            "normalized": normalized,
            "input": resized,
            "heatmap": heatmap,
            "coordinates": {
                "original_x": original_x,
                "original_y": original_y,
                "input_x": input_x,
                "input_y": input_y,
                "heatmap_x": heatmap_x,
                "heatmap_y": heatmap_y,
            },
        }
