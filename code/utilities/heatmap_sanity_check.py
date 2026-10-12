from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from phase2_common import (
    METHOD_LABELS,
    XATipHeatmapDataset,
    build_logger,
    load_config,
    resolve_project_path,
    round_half_up,
)


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def quantization_table(
    frames: pd.DataFrame,
    original_hw: tuple[int, int],
    heatmap_hw: tuple[int, int],
) -> pd.DataFrame:
    original_h, original_w = original_hw
    heatmap_h, heatmap_w = heatmap_hw
    rows: list[dict[str, Any]] = []
    for _, frame in frames.iterrows():
        x = float(frame["true_tip_col"])
        y = float(frame["true_tip_row"])
        hx = x * heatmap_w / original_w
        hy = y * heatmap_h / original_h
        hx_round_unclipped = round_half_up(hx)
        hy_round_unclipped = round_half_up(hy)
        hx_round = min(max(hx_round_unclipped, 0), heatmap_w - 1)
        hy_round = min(max(hy_round_unclipped, 0), heatmap_h - 1)
        x_back = hx_round * original_w / heatmap_w
        y_back = hy_round * original_h / heatmap_h
        dx = x_back - x
        dy = y_back - y
        rows.append(
            {
                "split": frame["split"],
                "case_code": frame["case_code"],
                "series_uid": frame["series_uid"],
                "sequence_identifier": frame["sequence_identifier"],
                "frame_index": int(frame["frame_index"]),
                "dicom_absolute_path": frame["dicom_absolute_path"],
                "original_x": x,
                "original_y": y,
                "heatmap_x_float": hx,
                "heatmap_y_float": hy,
                "heatmap_x_rounded_unclipped": hx_round_unclipped,
                "heatmap_y_rounded_unclipped": hy_round_unclipped,
                "heatmap_x_rounded": hx_round,
                "heatmap_y_rounded": hy_round,
                "round_was_boundary_clipped": bool(
                    hx_round != hx_round_unclipped or hy_round != hy_round_unclipped
                ),
                "mapped_back_x": x_back,
                "mapped_back_y": y_back,
                "error_dx_pixels": dx,
                "error_dy_pixels": dy,
                "error_euclidean_pixels": float(np.hypot(dx, dy)),
            }
        )
    return pd.DataFrame.from_records(rows)


def upsample_heatmap(heatmap: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    output_h, output_w = output_hw
    image = Image.fromarray(heatmap.astype(np.float32), mode="F")
    return np.asarray(
        image.resize((output_w, output_h), resample=Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def update_readme(readme_path: Path, quant: pd.DataFrame, config: dict[str, Any], method: str, sanity_passed: int, sanity_total: int) -> None:
    errors = quant["error_euclidean_pixels"].to_numpy(dtype=np.float64)
    boundary_count = int(quant["round_was_boundary_clipped"].sum())
    heatmap = config["heatmap"]
    section = f"""<!-- HEATMAP_SECTION_START -->
## 8. 128×128 Heatmap theoretical quantization error

全部 {len(quant)} 个 Train GT 坐标按配置执行：原图 {heatmap['original_size'][1]}×{heatmap['original_size'][0]} → 输入 {heatmap['input_size'][1]}×{heatmap['input_size'][0]} → Heatmap {heatmap['heatmap_size'][1]}×{heatmap['heatmap_size'][0]}。坐标保持 float 生成 Gaussian（sigma={float(heatmap['sigma']):g}），只在模拟未来 integer argmax 时 round 到最近 heatmap pixel，再映射回原图。

| 指标 | 原图像素 |
|---|---:|
| Mean | {float(np.mean(errors)):.4f} |
| Median | {float(np.median(errors)):.4f} |
| P95 | {percentile(errors, 95):.4f} |
| Max | {float(np.max(errors)):.4f} |

发生 heatmap 边界 clip 的 round 坐标：{boundary_count}/{len(quant)}。坐标可视化 sanity check：{sanity_passed}/{sanity_total} 通过。可视化使用配置中的 **{METHOD_LABELS[method]}**，仅用于 pipeline 几何检查，不代表它已被选为最终 preprocessing。

**这不是模型误差。** 它是 128×128 heatmap 配合整数 argmax 本身可能带来的理论坐标离散化误差。当前只报告结果，没有擅自改成 256×256，也没有实现 soft-argmax、DARK/UDP 或 subpixel refinement。
<!-- HEATMAP_SECTION_END -->"""
    text = readme_path.read_text(encoding="utf-8")
    start_marker = "<!-- HEATMAP_SECTION_START -->"
    end_marker = "<!-- HEATMAP_SECTION_END -->"
    start = text.index(start_marker)
    end = text.index(end_marker) + len(end_marker)
    readme_path.write_text(text[:start] + section + text[end:], encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="XA point heatmap and coordinate sanity check")
    parser.add_argument("--config", required=True, help="Path to preprocessing_phase2.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = resolve_project_path(config, config["project"]["output_dir"])
    figures_dir = output_dir / "figures"
    log_path = output_dir / "logs" / "phase2_run.log"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(log_path, append=True)

    logger.info("")
    logger.info("[Heatmap] Loading Train manifest and preprocessing cache...")
    manifest = pd.read_csv(resolve_project_path(config, config["project"]["model_clean_frames"]), low_memory=False)
    cached = pd.read_csv(resolve_project_path(config, config["project"]["cached_intensity_stats"]), low_memory=False)
    split = config["data"]["split"]
    frames = manifest[manifest["split"] == split].copy()
    stats = cached[cached["split"] == split].copy()
    frames["dicom_absolute_path"] = frames["dicom_absolute_path"].astype(str)
    stats["dicom_absolute_path"] = stats["dicom_absolute_path"].astype(str)
    frames = frames.sort_values(["case_code", "sequence_identifier", "frame_index"]).reset_index(drop=True)
    if len(frames) != int(config["data"]["expected_frames"]):
        raise RuntimeError(f"Expected {config['data']['expected_frames']} Train frames, found {len(frames)}")

    heatmap_config = config["heatmap"]
    method = str(heatmap_config["preprocessing_method"])
    if method == "dicom_window":
        summary_path = output_dir / "preprocessing_run_metadata.json"
        if summary_path.exists():
            metadata = json.loads(summary_path.read_text(encoding="utf-8"))
            if not metadata.get("dicom_window_eligible", False):
                raise RuntimeError(
                    "heatmap.preprocessing_method=dicom_window, but Method E was marked ineligible: "
                    + str(metadata.get("dicom_window_reason", "unknown reason"))
                )

    original_hw = tuple(int(value) for value in heatmap_config["original_size"])
    input_hw = tuple(int(value) for value in heatmap_config["input_size"])
    heatmap_hw = tuple(int(value) for value in heatmap_config["heatmap_size"])
    sigma = float(heatmap_config["sigma"])
    logger.info(
        f"[Heatmap] Method={method} | original={original_hw} | input={input_hw} | "
        f"heatmap={heatmap_hw} | sigma={sigma:g}"
    )

    logger.info("[Heatmap] Computing integer-argmax theoretical quantization error for all Train GT...")
    quant = quantization_table(frames, original_hw, heatmap_hw)
    quant.to_csv(output_dir / "heatmap_quantization_error.csv", index=False, encoding="utf-8-sig")

    sampling_path = output_dir / "sampling_manifest.csv"
    if not sampling_path.exists():
        raise RuntimeError("sampling_manifest.csv is missing; run preprocessing_compare.py first")
    sampling = pd.read_csv(sampling_path, low_memory=False)
    montage = sampling[sampling["selected_for_montage"].astype(str).str.lower().isin({"true", "1"})].copy()
    # Older/generated sampling manifests may intentionally contain only identity
    # fields. Reattach the authoritative CSV GT from the read-only Train manifest.
    coordinate_columns = ["dicom_absolute_path", "raw_row", "raw_col", "true_tip_row", "true_tip_col"]
    missing_coordinates = [column for column in coordinate_columns[1:] if column not in montage.columns]
    if missing_coordinates:
        montage = montage.merge(
            frames[coordinate_columns],
            on="dicom_absolute_path",
            how="left",
            validate="one_to_one",
        )
    if montage[["true_tip_row", "true_tip_col"]].isna().any().any():
        bad_path = montage.loc[
            montage[["true_tip_row", "true_tip_col"]].isna().any(axis=1), "dicom_absolute_path"
        ].iloc[0]
        raise RuntimeError(f"Missing Train CSV GT coordinates for sanity sample: {bad_path}")
    requested = int(heatmap_config["sanity_samples"])
    if len(montage) < requested:
        raise RuntimeError(f"Need {requested} sanity samples, found {len(montage)}")
    montage = montage.sort_values(["case_code", "sequence_identifier"]).head(requested).reset_index(drop=True)
    dataset = XATipHeatmapDataset(montage, stats, method, input_hw, heatmap_hw, sigma)

    logger.info("[Heatmap] Checking coordinate transforms and generating sanity montage...")
    samples: list[dict[str, Any]] = []
    sanity_passed = 0
    for index in tqdm(range(len(dataset)), desc="[Heatmap] Sanity samples", unit="sample", dynamic_ncols=True):
        try:
            sample = dataset[index]
        except Exception:
            path = str(montage.iloc[index]["dicom_absolute_path"])
            logger.exception(f"[ERROR] Heatmap sanity DICOM: {path}")
            raise
        coordinates = sample["coordinates"]
        expected_input_x = coordinates["original_x"] * input_hw[1] / original_hw[1]
        expected_input_y = coordinates["original_y"] * input_hw[0] / original_hw[0]
        max_y, max_x = np.unravel_index(int(np.argmax(sample["heatmap"])), sample["heatmap"].shape)
        expected_max_x = min(max(round_half_up(coordinates["heatmap_x"]), 0), heatmap_hw[1] - 1)
        expected_max_y = min(max(round_half_up(coordinates["heatmap_y"]), 0), heatmap_hw[0] - 1)
        passed = (
            np.isclose(coordinates["input_x"], expected_input_x)
            and np.isclose(coordinates["input_y"], expected_input_y)
            and max_x == expected_max_x
            and max_y == expected_max_y
        )
        if passed:
            sanity_passed += 1
        else:
            logger.error(
                f"[Heatmap] Coordinate check failed: {sample['row']['dicom_absolute_path']} | "
                f"argmax=({max_x},{max_y}), expected=({expected_max_x},{expected_max_y})"
            )
        sample["sanity_passed"] = passed
        samples.append(sample)

    fig, axes = plt.subplots(len(samples), 4, figsize=(12, 2.9 * len(samples)), squeeze=False)
    for row_index, sample in enumerate(samples):
        row = sample["row"]
        coords = sample["coordinates"]
        bits_stored = int(sample["metadata"]["BitsStored"])
        original_display = sample["original"].astype(np.float32) / np.float32((1 << bits_stored) - 1)
        axes[row_index, 0].imshow(original_display, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 0].scatter(
            [coords["original_x"]], [coords["original_y"]], s=42, facecolors="none", edgecolors="#00ff33", linewidths=1.2
        )
        axes[row_index, 1].imshow(sample["input"], cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].scatter(
            [coords["input_x"]], [coords["input_y"]], s=42, facecolors="none", edgecolors="#00ff33", linewidths=1.2
        )
        axes[row_index, 2].imshow(sample["heatmap"], cmap="magma", vmin=0, vmax=1)
        axes[row_index, 2].scatter(
            [coords["heatmap_x"]], [coords["heatmap_y"]], s=35, facecolors="none", edgecolors="#00ff33", linewidths=1.0
        )
        heatmap_up = upsample_heatmap(sample["heatmap"], input_hw)
        axes[row_index, 3].imshow(sample["input"], cmap="gray", vmin=0, vmax=1)
        axes[row_index, 3].imshow(heatmap_up, cmap="magma", vmin=0, vmax=1, alpha=0.45)
        axes[row_index, 3].scatter(
            [coords["input_x"]], [coords["input_y"]], s=42, facecolors="none", edgecolors="#00ff33", linewidths=1.2
        )
        for column in range(4):
            axes[row_index, column].set_axis_off()
        axes[row_index, 0].text(
            4,
            1015,
            f"{row['case_code']}/{row['sequence_identifier']} f={int(row['frame_index'])}",
            color="yellow",
            fontsize=7,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1},
        )
        if row_index == 0:
            axes[row_index, 0].set_title("A. Original 1024 + GT")
            axes[row_index, 1].set_title(f"B. {METHOD_LABELS[method]} 512 + GT")
            axes[row_index, 2].set_title("C. 128 Gaussian heatmap")
            axes[row_index, 3].set_title("D. Heatmap overlay")
    fig.suptitle("Heatmap coordinate sanity — Train only; floating-point Gaussian centers", y=0.999)
    fig.tight_layout()
    figure_path = figures_dir / "heatmap_coordinate_sanity.png"
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)

    if sanity_passed != len(samples):
        raise RuntimeError(f"Heatmap coordinate sanity failed for {len(samples) - sanity_passed}/{len(samples)} samples")
    logger.info(f"[Heatmap] {sanity_passed}/{len(samples)} sanity samples passed.")
    logger.info(f"[Figures] Saved {figure_path.name}")

    update_readme(output_dir / "README.md", quant, config, method, sanity_passed, len(samples))
    errors = quant["error_euclidean_pixels"].to_numpy(dtype=np.float64)
    summary = {
        "n_train_gt": int(len(quant)),
        "mean_original_pixels": float(np.mean(errors)),
        "median_original_pixels": float(np.median(errors)),
        "p95_original_pixels": percentile(errors, 95),
        "max_original_pixels": float(np.max(errors)),
        "boundary_clipped_count": int(quant["round_was_boundary_clipped"].sum()),
        "sanity_samples_passed": sanity_passed,
        "sanity_samples_total": len(samples),
        "model_training_performed": False,
    }
    (output_dir / "heatmap_quantization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("")
    logger.info("[Summary] 128x128 integer-argmax quantization error in original pixels")
    logger.info(
        f"  mean={summary['mean_original_pixels']:.4f} | median={summary['median_original_pixels']:.4f} | "
        f"P95={summary['p95_original_pixels']:.4f} | max={summary['max_original_pixels']:.4f}"
    )
    logger.info("[Done] Heatmap coordinate sanity complete. No model training was performed.")
    logger.info(f"[Done] Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[Stopped] Interrupted by user; no source data were modified.", file=sys.stderr)
        raise
