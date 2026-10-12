from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import timm
import torch
import yaml
from torch import nn
from tqdm import tqdm

import hrnet_phase4_train as phase4
from hrnet_overfit32 import HRNetTipHeatmap


CHECKPOINT_WARNING = (
    "PHASE 5A SINGLE-VARIABLE EXPERIMENT — pretrained backbone; "
    "model selection uses held-out Val mean pixel error"
)
INITIALIZATION_NOTE = (
    "ImageNet-1K pretrained HRNet backbone + freshly initialized 1-channel heatmap head; "
    "no Phase 3 or Phase 4 checkpoint was loaded"
)

_ACTIVE_CONFIG: dict[str, Any] | None = None
_ACTIVE_OUTPUT_DIR: Path | None = None
_SUPPRESSION_RADIUS = 6
_BASE_BUILD_LOGGER = phase4.build_logger
_BASE_ENVIRONMENT_TEXT = phase4.environment_text


def build_logger(path: Path, append: bool) -> logging.Logger:
    return _BASE_BUILD_LOGGER(path.parent / "phase5a_run.log", append)


def build_pretrained_model(config: dict[str, Any], device: torch.device) -> nn.Module:
    model_cfg = config["model"]
    if not bool(model_cfg["pretrained"]):
        raise ValueError("Phase 5A is frozen to pretrained=True")
    if str(model_cfg["initialization"]) != "pretrained_backbone_fresh_heatmap_head":
        raise ValueError("Phase 5A initialization provenance is invalid")
    return HRNetTipHeatmap(
        variant=str(model_cfg["variant"]),
        pretrained=True,
        output_channels=int(model_cfg["output_channels"]),
        expected_output_hw=tuple(int(v) for v in config["data"]["heatmap_size"]),
    ).to(device)


def _tensor_stats(tensor: torch.Tensor) -> dict[str, float | list[int]]:
    value = tensor.detach().float().cpu()
    return {
        "shape": list(value.shape),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "l2_norm": float(torch.linalg.vector_norm(value)),
        "min": float(value.min()),
        "max": float(value.max()),
    }


def verify_pretrained_loading(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Fail closed unless timm weights load and retained backbone tensors differ from random."""
    seed = int(config["experiment"]["seed"])
    variant = str(config["model"]["variant"])
    torch.manual_seed(seed)
    random_model = HRNetTipHeatmap(
        variant=variant,
        pretrained=False,
        output_channels=int(config["model"]["output_channels"]),
        expected_output_hw=tuple(int(v) for v in config["data"]["heatmap_size"]),
    ).cpu()
    torch.manual_seed(seed)
    try:
        pretrained_model = HRNetTipHeatmap(
            variant=variant,
            pretrained=True,
            output_channels=int(config["model"]["output_channels"]),
            expected_output_hw=tuple(int(v) for v in config["data"]["heatmap_size"]),
        ).cpu()
    except Exception as exc:
        failure = {
            "passed": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "action": "STOPPED before sanity check or formal training",
        }
        (output_dir / "pretrained_loading_report.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        raise RuntimeError("Pretrained loading failed; Phase 5A must not fall back to random") from exc

    random_params = dict(random_model.backbone.named_parameters())
    pretrained_params = dict(pretrained_model.backbone.named_parameters())
    missing_keys = sorted(set(random_params) - set(pretrained_params))
    unexpected_keys = sorted(set(pretrained_params) - set(random_params))
    shape_mismatches: list[str] = []
    changed_names: list[str] = []
    unchanged_names: list[str] = []
    changed_numel = 0
    retained_numel = 0
    for name in sorted(set(random_params) & set(pretrained_params)):
        left = random_params[name].detach().cpu()
        right = pretrained_params[name].detach().cpu()
        if left.shape != right.shape:
            shape_mismatches.append(name)
            continue
        retained_numel += right.numel()
        if torch.equal(left, right):
            unchanged_names.append(name)
        else:
            changed_names.append(name)
            changed_numel += right.numel()

    random_head = random_model.heatmap_head.weight.detach().cpu()
    pretrained_head = pretrained_model.heatmap_head.weight.detach().cpu()
    fresh_head_equal_same_seed = bool(torch.equal(random_head, pretrained_head))
    first_key = "conv1.weight"
    if first_key not in random_params:
        first_key = next(iter(random_params))
    first_random = random_params[first_key].detach().cpu()
    first_pretrained = pretrained_params[first_key].detach().cpu()
    first_difference = first_pretrained.float() - first_random.float()
    pretrained_cfg = timm.get_pretrained_cfg(variant).to_dict()
    report = {
        "passed": bool(
            not missing_keys
            and not unexpected_keys
            and not shape_mismatches
            and changed_numel > 0
            and fresh_head_equal_same_seed
        ),
        "model": variant,
        "pretrained": True,
        "source": {
            "provider": "timm / Hugging Face Hub cache",
            "tag": str(config["model"]["pretrained_tag"]),
            "hf_hub_id": str(config["model"]["pretrained_hf_hub_id"]),
            "dataset": "ImageNet-1K",
            "timm_pretrained_cfg": pretrained_cfg,
        },
        "timm_loader_result": "completed without exception",
        "retained_backbone_parameter_tensors": len(pretrained_params),
        "retained_backbone_parameter_count": int(retained_numel),
        "tensors_different_from_same_seed_random": len(changed_names),
        "parameters_different_from_same_seed_random": int(changed_numel),
        "tensors_identical_to_same_seed_random": len(unchanged_names),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatches": shape_mismatches,
        "first_retained_convolution": first_key,
        "first_convolution_random_stats": _tensor_stats(first_random),
        "first_convolution_pretrained_stats": _tensor_stats(first_pretrained),
        "first_convolution_difference": {
            "max_abs": float(first_difference.abs().max()),
            "l2_norm": float(torch.linalg.vector_norm(first_difference)),
        },
        "task_heatmap_head": {
            "fresh_initialization": True,
            "equal_to_same_seed_random_model_head": fresh_head_equal_same_seed,
            "weight_stats": _tensor_stats(pretrained_head),
            "bias_stats": _tensor_stats(pretrained_model.heatmap_head.bias),
        },
        "phase3_checkpoint_loaded": False,
        "phase4_checkpoint_loaded": False,
        "imagenet_mean_std_applied": False,
        "input_preprocessing": "Phase 4 per-frame no-clipping min-max only",
    }
    if not report["passed"]:
        (output_dir / "pretrained_loading_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        raise RuntimeError("Pretrained verification failed; refusing to start formal training")
    (output_dir / "pretrained_loading_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "Phase 5A pretrained loading report",
        "=" * 72,
        "PASS: timm pretrained loading completed; no random fallback occurred.",
        f"Model: {variant}",
        f"Source: {report['source']['hf_hub_id']}",
        "Pretraining dataset: ImageNet-1K",
        f"Retained backbone tensors: {len(pretrained_params):,}",
        f"Retained backbone parameters: {retained_numel:,}",
        f"Changed vs same-seed random: {len(changed_names):,} tensors / {changed_numel:,} parameters",
        f"Missing keys: {missing_keys}",
        f"Unexpected keys: {unexpected_keys}",
        f"Shape mismatches: {shape_mismatches}",
        f"First conv random: {_tensor_stats(first_random)}",
        f"First conv pretrained: {_tensor_stats(first_pretrained)}",
        f"First conv difference: {report['first_convolution_difference']}",
        f"Fresh task head verified under same seed: {fresh_head_equal_same_seed}",
        "ImageNet mean/std normalization applied: False (keeps Phase 4 input pipeline fixed)",
        "Phase 3 checkpoint loaded: False",
        "Phase 4 checkpoint loaded: False",
    ]
    (output_dir / "pretrained_loading_report.txt").write_text("\n".join(lines), encoding="utf-8")
    del random_model, pretrained_model
    gc.collect()
    return report


def _independent_peaks(
    outputs: torch.Tensor, radius: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    maps = outputs[:, 0]
    batch, height, width = maps.shape
    flat = maps.reshape(batch, -1)
    peak, index = flat.max(dim=1)
    y1 = torch.div(index, width, rounding_mode="floor")
    x1 = index % width
    suppressed = maps.clone()
    yy = torch.arange(height, device=maps.device).view(1, height, 1)
    xx = torch.arange(width, device=maps.device).view(1, 1, width)
    mask = (yy - y1.view(-1, 1, 1)) ** 2 + (xx - x1.view(-1, 1, 1)) ** 2 <= radius**2
    suppressed[mask] = -torch.inf
    second, index2 = suppressed.reshape(batch, -1).max(dim=1)
    y2 = torch.div(index2, width, rounding_mode="floor")
    x2 = index2 % width
    return peak, index, second, index2, x1, y1


def validate_phase5a(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
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
    weights: list[int] = []
    errors: list[float] = []
    peaks: list[float] = []
    seconds: list[float] = []
    margins: list[float] = []
    ratios: list[float] = []
    entropies: list[float] = []
    distances: list[float] = []
    output_means: list[float] = []
    output_stds: list[float] = []
    rows: list[dict[str, Any]] = []
    saved_heatmaps: list[np.ndarray] = []
    saved_indices: list[int] = []
    bar = tqdm(loader, desc=f"[Val {epoch_label}]", unit="batch", dynamic_ncols=True)
    with torch.no_grad():
        for batch in bar:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target_heatmap"].to(device, non_blocking=True)
            originals = batch["original_xy"].to(device, non_blocking=True)
            if hasattr(criterion, "current_dicom_path"):
                criterion.current_dicom_path = str(batch["dicom_path"][0])
            outputs = model(images)
            loss = criterion(outputs, targets)
            if not torch.isfinite(loss) or not torch.isfinite(outputs).all():
                raise FloatingPointError(f"Non-finite validation output/loss: {batch['dicom_path'][0]}")
            peak, index1, second, index2, x1, y1 = _independent_peaks(outputs, _SUPPRESSION_RADIUS)
            x2 = index2 % heatmap_hw[1]
            y2 = torch.div(index2, heatmap_hw[1], rounding_mode="floor")
            pred_original_x = x1.float() * original_hw[1] / heatmap_hw[1]
            pred_original_y = y1.float() * original_hw[0] / heatmap_hw[0]
            error = torch.sqrt(
                (pred_original_x - originals[:, 0]) ** 2
                + (pred_original_y - originals[:, 1]) ** 2
            )
            margin = peak - second
            ratio = peak / (second.abs() + 1e-8)
            distance = torch.sqrt((x1.float() - x2.float()) ** 2 + (y1.float() - y2.float()) ** 2)
            probabilities = torch.softmax(outputs[:, 0].reshape(outputs.shape[0], -1), dim=1)
            entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=1)
            entropy = entropy / math.log(probabilities.shape[1])
            batch_size = int(images.shape[0])
            losses.append(float(loss))
            weights.append(batch_size)
            errors.extend(error.cpu().numpy().astype(float).tolist())
            peaks.extend(peak.cpu().numpy().astype(float).tolist())
            seconds.extend(second.cpu().numpy().astype(float).tolist())
            margins.extend(margin.cpu().numpy().astype(float).tolist())
            ratios.extend(ratio.cpu().numpy().astype(float).tolist())
            entropies.extend(entropy.cpu().numpy().astype(float).tolist())
            distances.extend(distance.cpu().numpy().astype(float).tolist())
            output_means.extend(outputs.mean(dim=(1, 2, 3)).cpu().numpy().astype(float).tolist())
            output_stds.extend(outputs.std(dim=(1, 2, 3)).cpu().numpy().astype(float).tolist())
            if collect_predictions:
                sample_indices = batch["sample_index"].cpu().numpy().astype(int)
                originals_cpu = batch["original_xy"].cpu().numpy()
                x1_cpu, y1_cpu = x1.cpu().numpy(), y1.cpu().numpy()
                x2_cpu, y2_cpu = x2.cpu().numpy(), y2.cpu().numpy()
                peak_cpu, second_cpu = peak.cpu().numpy(), second.cpu().numpy()
                margin_cpu, ratio_cpu = margin.cpu().numpy(), ratio.cpu().numpy()
                entropy_cpu, distance_cpu = entropy.cpu().numpy(), distance.cpu().numpy()
                error_cpu = error.cpu().numpy()
                saved_heatmaps.append(outputs[:, 0].cpu().numpy().astype(np.float16))
                saved_indices.extend(sample_indices.tolist())
                for position, sample_index in enumerate(sample_indices):
                    source = manifest.iloc[sample_index]
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
                            "pred_x_heatmap": int(x1_cpu[position]),
                            "pred_y_heatmap": int(y1_cpu[position]),
                            "pred_x_original": float(x1_cpu[position] * original_hw[1] / heatmap_hw[1]),
                            "pred_y_original": float(y1_cpu[position] * original_hw[0] / heatmap_hw[0]),
                            "error_px": float(error_cpu[position]),
                            "heatmap_peak_value": float(peak_cpu[position]),
                            "second_peak_value": float(second_cpu[position]),
                            "independent_second_peak_value": float(second_cpu[position]),
                            "peak_margin": float(margin_cpu[position]),
                            "independent_peak_margin": float(margin_cpu[position]),
                            "independent_peak_ratio": float(ratio_cpu[position]),
                            "second_peak_x_heatmap": int(x2_cpu[position]),
                            "second_peak_y_heatmap": int(y2_cpu[position]),
                            "top2_distance_heatmap_px": float(distance_cpu[position]),
                            "heatmap_entropy": float(entropy_cpu[position]),
                            "sample_index": int(sample_index),
                        }
                    )
            bar.set_postfix(loss=f"{losses[-1]:.3g}", mean_px=f"{np.mean(errors):.1f}")
    summary = phase4.summarize_errors(errors)
    summary.update(
        {
            "loss": float(np.average(losses, weights=weights)),
            "pred_heatmap_max_mean": float(np.mean(peaks)),
            "pred_heatmap_max_median": float(np.median(peaks)),
            "pred_heatmap_mean": float(np.mean(output_means)),
            "pred_heatmap_std": float(np.mean(output_stds)),
            "second_peak_mean": float(np.mean(seconds)),
            "independent_second_peak_mean": float(np.mean(seconds)),
            "independent_second_peak_median": float(np.median(seconds)),
            "peak_margin_mean": float(np.mean(margins)),
            "independent_peak_margin_mean": float(np.mean(margins)),
            "independent_peak_margin_median": float(np.median(margins)),
            "independent_peak_ratio_median": float(np.median(ratios)),
            "top2_distance_median": float(np.median(distances)),
            "heatmap_entropy_mean": float(np.mean(entropies)),
            "heatmap_entropy_median": float(np.median(entropies)),
        }
    )
    if not collect_predictions and _ACTIVE_OUTPUT_DIR is not None and epoch_label.split("/")[0].isdigit():
        epoch = int(epoch_label.split("/")[0])
        confidence_path = _ACTIVE_OUTPUT_DIR / "confidence_epoch_log.csv"
        record = {"epoch": epoch, **summary}
        prior = pd.read_csv(confidence_path) if confidence_path.exists() else pd.DataFrame()
        combined = pd.concat([prior.loc[prior.get("epoch", pd.Series(dtype=int)) != epoch], pd.DataFrame([record])], ignore_index=True)
        combined.sort_values("epoch").to_csv(confidence_path, index=False, encoding="utf-8-sig")
        print(
            f"[Confidence] independent radius={_SUPPRESSION_RADIUS} | "
            f"peak median={summary['pred_heatmap_max_median']:.4f} | "
            f"margin median={summary['independent_peak_margin_median']:.4f} | "
            f"entropy={summary['heatmap_entropy_median']:.5f}"
        )
    if collect_predictions and _ACTIVE_OUTPUT_DIR is not None:
        np.savez_compressed(
            _ACTIVE_OUTPUT_DIR / "val_heatmaps_best_float16.npz",
            heatmaps=np.concatenate(saved_heatmaps, axis=0),
            sample_index=np.asarray(saved_indices, dtype=np.int32),
        )
    return summary, pd.DataFrame.from_records(rows) if collect_predictions else None


def save_checkpoint_phase5a(
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
            "phase4_checkpoint_loaded": False,
            "kind": kind,
            "epoch": int(epoch),
            "model_variant": config["model"]["variant"],
            "pretrained": True,
            "pretrained_source": config["model"]["pretrained_hf_hub_id"],
            "initialization": config["model"]["initialization"],
            "task_heatmap_head_fresh": True,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": record,
            "training_state": dict(state),
            "train_generator_state": generator.get_state(),
            "config": phase4.config_for_save(config),
        },
        path,
    )


def environment_text_phase5a(device: torch.device) -> str:
    base = _BASE_ENVIRONMENT_TEXT(device)
    assert _ACTIVE_CONFIG is not None
    return base + "\n" + "\n".join(
        [
            "Phase 5A initialization: pretrained backbone + fresh heatmap head",
            f"Pretrained source: {_ACTIVE_CONFIG['model']['pretrained_hf_hub_id']}",
            "ImageNet mean/std input normalization: False",
            "Phase 3/4 checkpoint initialization: False",
        ]
    )


def save_model_summary_phase5a(
    path: Path, model: nn.Module, config: dict[str, Any], device: torch.device
) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    text = "\n".join(
        [
            "Phase 5A HRNet pretrained single-variable experiment",
            "=" * 72,
            CHECKPOINT_WARNING,
            INITIALIZATION_NOTE,
            f"Variant: {config['model']['variant']}",
            "Pretrained: True",
            f"Pretrained source: {config['model']['pretrained_hf_hub_id']}",
            "Task head: freshly initialized Conv2d(..., 1, kernel_size=1)",
            "ImageNet mean/std normalization: False",
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


def print_header_phase5a(
    logger: logging.Logger,
    config: dict[str, Any],
    device: torch.device,
    train: pd.DataFrame,
    val: pd.DataFrame,
    train_iterations: int,
    val_iterations: int,
) -> None:
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    logger.info("=" * 62)
    logger.info("Phase 5A HRNet Pretrained Initialization Experiment")
    logger.info("=" * 62)
    logger.info(f"Device: {name}")
    logger.info(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        logger.info(f"GPU memory: available {free / 1024**2:.0f} / total {total / 1024**2:.0f} MiB")
    logger.info(f"Model: {config['model']['variant']}")
    logger.info("Pretrained backbone: True")
    logger.info(f"Pretrained source: {config['model']['pretrained_hf_hub_id']}")
    logger.info("Task head: fresh 1-channel heatmap head")
    logger.info("Phase 3 checkpoint: NOT loaded")
    logger.info("Phase 4 checkpoint: NOT loaded")
    logger.info("Input normalization: per-frame min-max only (unchanged from Phase 4)")
    logger.info(
        f"Train: {len(train)} frames / {train['case_identifier'].nunique()} cases / "
        f"{train['series_uid'].nunique()} series | iterations/epoch={train_iterations}"
    )
    logger.info(
        f"Val: {len(val)} frames / {val['case_identifier'].nunique()} cases / "
        f"{val['series_uid'].nunique()} series | iterations/epoch={val_iterations}"
    )
    logger.info(
        f"Adam LR={config['training']['learning_rate']} | WD={config['training']['weight_decay']} | "
        f"Batch={config['data']['batch_size']} | MSE | max_epochs={config['training']['max_epochs']}"
    )
    logger.info(
        f"Independent peak suppression radius: {_SUPPRESSION_RADIUS} heatmap pixels"
    )
    logger.info("=" * 62)


def _merge_confidence_into_training_log(output_dir: Path) -> None:
    train_path = output_dir / "training_log.csv"
    confidence_path = output_dir / "confidence_epoch_log.csv"
    if not train_path.exists() or not confidence_path.exists():
        return
    training = pd.read_csv(train_path)
    confidence = pd.read_csv(confidence_path)
    add_columns = [
        "epoch",
        "independent_second_peak_mean",
        "independent_second_peak_median",
        "independent_peak_margin_mean",
        "independent_peak_margin_median",
        "independent_peak_ratio_median",
        "top2_distance_median",
        "heatmap_entropy_mean",
        "heatmap_entropy_median",
    ]
    existing = [column for column in add_columns if column in confidence.columns]
    training = training.drop(columns=[c for c in existing if c != "epoch" and c in training], errors="ignore")
    training.merge(confidence[existing], on="epoch", how="left").to_csv(
        train_path, index=False, encoding="utf-8-sig"
    )


def _correct_run_summary(output_dir: Path) -> None:
    path = output_dir / "run_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["phase3_checkpoint_loaded"] = False
    summary["phase4_checkpoint_loaded"] = False
    summary["pretrained"] = True
    summary["pretrained_source"] = _ACTIVE_CONFIG["model"]["pretrained_hf_hub_id"]
    summary["formal_training_initialization"] = "pretrained_backbone_fresh_heatmap_head"
    summary["task_heatmap_head_fresh"] = True
    summary["only_intended_change_from_phase4"] = "pretrained: false -> true"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args, _ = parser.parse_known_args()
    config = phase4.load_config(args.config)
    output_dir = phase4.resolve_path(config, config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    global _ACTIVE_CONFIG, _ACTIVE_OUTPUT_DIR, _SUPPRESSION_RADIUS
    _ACTIVE_CONFIG = config
    _ACTIVE_OUTPUT_DIR = output_dir
    _SUPPRESSION_RADIUS = int(config["heatmap_confidence"]["independent_peak_suppression_radius"])

    if not args.resume:
        print("[Pretrained] Verifying source, retained weights, and fresh task head...")
        report = verify_pretrained_loading(config, output_dir)
        print(
            f"[PASS] Pretrained loaded: {report['parameters_different_from_same_seed_random']:,} "
            "retained parameters differ from same-seed random initialization."
        )
    else:
        checkpoint_path = output_dir / "checkpoints" / "last.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot resume; missing {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not bool(checkpoint.get("pretrained", False)) or bool(
            checkpoint.get("phase4_checkpoint_loaded", True)
        ):
            raise RuntimeError("Resume checkpoint does not have valid Phase 5A provenance")

    phase4.CHECKPOINT_WARNING = CHECKPOINT_WARNING
    phase4.PHASE3_WARNING = INITIALIZATION_NOTE
    phase4.build_logger = build_logger
    phase4.build_model = build_pretrained_model
    phase4.validate = validate_phase5a
    phase4.save_checkpoint = save_checkpoint_phase5a
    phase4.environment_text = environment_text_phase5a
    phase4.save_model_summary = save_model_summary_phase5a
    phase4.print_header = print_header_phase5a
    result = phase4.main()
    _merge_confidence_into_training_log(output_dir)
    _correct_run_summary(output_dir)
    print("[Phase 5A] Formal training and best-checkpoint inference complete.")
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[Stopped] User interrupted Phase 5A. Resume with --resume.", file=sys.stderr)
        raise
