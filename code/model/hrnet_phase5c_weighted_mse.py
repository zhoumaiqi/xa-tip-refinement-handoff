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

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import nn

import hrnet_phase4_train as phase4
import hrnet_phase5a_train as phase5a


WARNING = "PHASE 5C: pretrained HRNet, no augmentation, continuous heatmap-weighted MSE; best by Val mean"
NOTE = "ImageNet-1K pretrained backbone + fresh task head; no Phase 3/4/5A/5B checkpoint loaded"
BASE_LOGGER = phase4.build_logger
BASE_TRAIN = phase4.train_one_epoch
BASE_PLUMBING = phase4.plumbing_test
BASE_ENVIRONMENT = phase4.environment_text
ACTIVE: dict[str, Any] | None = None
LAST_CRITERION: WeightedMSE | None = None


class WeightedMSE(nn.Module):
    def __init__(self, alpha: float, fg_threshold: float, absolute_limit: float) -> None:
        super().__init__()
        if alpha != 9.0 or fg_threshold != 0.1 or absolute_limit != 10.0:
            raise ValueError("Phase 5C fixed alpha=9, diagnostic GT>0.1, safety limit=10")
        self.alpha = alpha
        self.fg_threshold = fg_threshold
        self.absolute_limit = absolute_limit
        self.current_dicom_path = "unknown"
        self.reset()

    def reset(self) -> None:
        self.stats = {key: 0.0 for key in ["plain_sse", "weighted_sse", "fg_plain_sse", "bg_plain_sse",
                                              "fg_weighted_sse", "bg_weighted_sse", "pixels", "fg_pixels"]}
        self.pred_min = float("inf")
        self.pred_max = float("-inf")

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if predicted.shape != target.shape or predicted.ndim != 4 or predicted.shape[1:] != (1, 128, 128):
            raise ValueError(f"Invalid Phase 5C heatmap shape {tuple(predicted.shape)} vs {tuple(target.shape)}: {self.current_dicom_path}")
        if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
            raise FloatingPointError(f"Non-finite heatmap: {self.current_dicom_path}")
        minimum, maximum = float(predicted.detach().min()), float(predicted.detach().max())
        if max(abs(minimum), abs(maximum)) > self.absolute_limit:
            raise FloatingPointError(f"Predicted heatmap outside diagnostic ±{self.absolute_limit}: [{minimum},{maximum}], DICOM={self.current_dicom_path}")
        error = (predicted - target).square()
        weighted_error = (1.0 + self.alpha * target) * error
        with torch.no_grad():
            foreground = target > self.fg_threshold
            self.stats["plain_sse"] += float(error.detach().sum())
            self.stats["weighted_sse"] += float(weighted_error.detach().sum())
            self.stats["fg_plain_sse"] += float(error.detach().masked_select(foreground).sum())
            self.stats["bg_plain_sse"] += float(error.detach().masked_select(~foreground).sum())
            self.stats["fg_weighted_sse"] += float(weighted_error.detach().masked_select(foreground).sum())
            self.stats["bg_weighted_sse"] += float(weighted_error.detach().masked_select(~foreground).sum())
            self.stats["pixels"] += float(target.numel())
            self.stats["fg_pixels"] += float(foreground.sum())
            self.pred_min = min(self.pred_min, minimum)
            self.pred_max = max(self.pred_max, maximum)
        return weighted_error.mean()

    def diagnostic(self, prefix: str) -> dict[str, float]:
        s = self.stats
        n = max(s["pixels"], 1.0)
        return {
            f"{prefix}_weighted_loss": s["weighted_sse"] / n,
            f"{prefix}_plain_mse": s["plain_sse"] / n,
            f"{prefix}_fg_plain_sse": s["fg_plain_sse"],
            f"{prefix}_bg_plain_sse": s["bg_plain_sse"],
            f"{prefix}_fg_weighted_sse": s["fg_weighted_sse"],
            f"{prefix}_bg_weighted_sse": s["bg_weighted_sse"],
            f"{prefix}_fg_pixel_fraction": s["fg_pixels"] / n,
            f"{prefix}_fg_plain_contribution_percent": 100 * s["fg_plain_sse"] / max(s["plain_sse"], 1e-12),
            f"{prefix}_fg_weighted_contribution_percent": 100 * s["fg_weighted_sse"] / max(s["weighted_sse"], 1e-12),
            f"{prefix}_pred_min": self.pred_min,
            f"{prefix}_pred_max": self.pred_max,
        }


def build_criterion(config: dict[str, Any]) -> WeightedMSE:
    global LAST_CRITERION
    if config["training"]["loss"] != "heatmap_weighted_mse":
        raise ValueError("Phase 5C requires heatmap_weighted_mse")
    loss = config["weighted_loss"]
    LAST_CRITERION = WeightedMSE(float(loss["alpha"]), float(loss["foreground_diagnostic_threshold"]),
                                 float(loss["prediction_abs_safety_stop"]))
    return LAST_CRITERION


def train_epoch(model: Any, loader: Any, optimizer: Any, criterion: WeightedMSE, *args: Any,
                **kwargs: Any) -> dict[str, Any]:
    criterion.reset()
    metrics = BASE_TRAIN(model, loader, optimizer, criterion, *args, **kwargs)
    extra = criterion.diagnostic("train")
    metrics["_epoch_extra"] = extra
    print(f"[Weighted Train] loss={extra['train_weighted_loss']:.8f} | plain={extra['train_plain_mse']:.8f} | "
          f"FG contribution plain={extra['train_fg_plain_contribution_percent']:.2f}% "
          f"weighted={extra['train_fg_weighted_contribution_percent']:.2f}%", flush=True)
    return metrics


def validate(model: Any, loader: Any, criterion: WeightedMSE, *args: Any,
             **kwargs: Any) -> tuple[dict[str, Any], pd.DataFrame | None]:
    criterion.reset()
    result, predictions = phase5a.validate_phase5a(model, loader, criterion, *args, **kwargs)
    extra = criterion.diagnostic("val")
    result["_epoch_extra"] = extra
    print(f"[Weighted Val] loss={extra['val_weighted_loss']:.8f} | plain={extra['val_plain_mse']:.8f} | "
          f"FG contribution plain={extra['val_fg_plain_contribution_percent']:.2f}% "
          f"weighted={extra['val_fg_weighted_contribution_percent']:.2f}% | "
          f"pred range [{extra['val_pred_min']:.3f},{extra['val_pred_max']:.3f}]", flush=True)
    return result, predictions


def plumbing_test(model: Any, loader: Any, device: Any, config: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    report = BASE_PLUMBING(model, loader, device, config, logger)
    assert LAST_CRITERION is not None
    diagnostics = LAST_CRITERION.diagnostic("sanity")
    report["weighted_loss"] = diagnostics["sanity_weighted_loss"]
    report["plain_unweighted_mse"] = diagnostics["sanity_plain_mse"]
    report["fresh_model_and_optimizer_after_probe"] = True
    logger.info(f"[Gradient Sanity] weighted={report['weighted_loss']:.8f}, "
                f"plain={report['plain_unweighted_mse']:.8f}, grad_norm={report['gradient_l2_norm']:.5f}; "
                "PASS. Probe discarded; fresh pretrained model/optimizer will be built.")
    return report


def build_logger(path: Path, append: bool) -> logging.Logger:
    return BASE_LOGGER(path.parent / "phase5c_run.log", append)


def save_checkpoint(path: Path, epoch: int, model: Any, optimizer: Any, scheduler: Any,
                    record: dict[str, Any], config: dict[str, Any], state: dict[str, Any],
                    generator: torch.Generator, kind: str) -> None:
    torch.save({"warning": WARNING, "phase3_checkpoint_loaded": False,
                "phase4_checkpoint_loaded": False, "phase5a_checkpoint_loaded": False,
                "phase5b_checkpoint_loaded": False, "kind": kind, "epoch": epoch,
                "model_variant": config["model"]["variant"], "pretrained": True,
                "pretrained_source": config["model"]["pretrained_hf_hub_id"],
                "initialization": config["model"]["initialization"], "task_heatmap_head_fresh": True,
                "loss_name": config["training"]["loss"], "loss_config": config["weighted_loss"],
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "metrics": record,
                "training_state": dict(state), "train_generator_state": generator.get_state(),
                "config": phase4.config_for_save(config)}, path)


def environment_text(device: torch.device) -> str:
    return BASE_ENVIRONMENT(device) + "\n" + (
        "Phase 5C: ImageNet pretrained backbone + fresh heatmap head; no earlier checkpoint\n"
        "No augmentation; per-frame min-max; ImageNet mean/std not applied\n"
        "Only intended change from Phase 5A: weighted MSE with alpha=9, no weight normalization\n"
        "Predicted heatmap diagnostic safety stop: absolute output > 10\n"
    )


def save_model_summary(path: Path, model: Any, config: dict[str, Any], device: torch.device) -> None:
    phase5a.save_model_summary_phase5a(path, model, config, device)
    content = path.read_text(encoding="utf-8").replace("Phase 5A", "Phase 5C").replace("PHASE 5A", "PHASE 5C")
    path.write_text(content + "\nLoss: mean((1+9*GT)*(Pred-GT)^2), weights unnormalized.\n", encoding="utf-8")


def print_header(logger: logging.Logger, config: dict[str, Any], device: torch.device,
                 train: pd.DataFrame, val: pd.DataFrame, train_iterations: int,
                 val_iterations: int) -> None:
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    logger.info("=" * 64)
    logger.info("Phase 5C — WEIGHTED MSE SINGLE-VARIABLE EXPERIMENT")
    logger.info("=" * 64)
    logger.info("Changed variable: Plain MSE -> Heatmap-weighted MSE; w = 1 + 9 * GT_heatmap")
    logger.info("Pretrained: True; augmentation: None; everything else frozen from Phase 5A")
    logger.info(f"Device: {name}; torch={torch.__version__}; CUDA={torch.version.cuda}")
    logger.info(f"Train: {len(train)} frames / {train.case_identifier.nunique()} cases / {train.series_uid.nunique()} Series; "
                f"{train_iterations} batches/epoch")
    logger.info(f"Val: {len(val)} frames / {val.case_identifier.nunique()} cases / {val.series_uid.nunique()} Series; "
                f"{val_iterations} batches/epoch")
    logger.info(f"Adam lr={config['training']['learning_rate']} WD={config['training']['weight_decay']} | "
                f"batch={config['data']['batch_size']} | max_epochs={config['training']['max_epochs']}")
    logger.info("Checkpoint selection and early stopping use Val mean original-space localization error")
    logger.info("=" * 64)


def loss_sanity(config: dict[str, Any], output: Path) -> None:
    (output / "figures").mkdir(parents=True, exist_ok=True)
    print("[1/5] Loading frozen Phase 5A configuration and manifest...", flush=True)
    manifest_path = phase4.resolve_path(config, config["project"]["manifest"])
    manifest = pd.read_csv(manifest_path, dtype={"case_code": str, "case_identifier": str})
    train, _, integrity = phase4.split_integrity(manifest, config, manifest_path)
    reference = phase4.resolve_path(config, config["project"]["phase5a_baseline_dir"])
    original = json.loads((reference / "split_integrity.json").read_text(encoding="utf-8"))
    if not original["passed"] or integrity["manifest_sha256"] != original["manifest_sha256"]:
        raise RuntimeError("Phase 5C frozen split differs from Phase 5A")
    print("[PASS] Frozen split integrity: Train 2784/8/12, Val 928/3/4; zero case/Series/DICOM overlaps", flush=True)
    (output / "split_integrity.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "loss_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config["weighted_loss"], handle, sort_keys=False, allow_unicode=True)
    dataset = phase4.XAHeatmapDataset(train, "Train", (512, 512), (128, 128), 2.0, 3, False,
                                      logging.getLogger("phase5c_loss_sanity"))
    chosen: list[int] = []
    for _, series in train.groupby("series_uid", sort=True):
        chosen.extend([int(series.index[0]), int(series.index[-1])])
    if len(chosen) != 24 or train.iloc[chosen].case_code.nunique() != 8:
        raise RuntimeError("Loss sanity sampling did not cover 8 Train cases / 12 Series")
    print("[2/5] Weighted MSE sanity: 24 real Train Gaussian heatmaps, alpha=9...", flush=True)
    rows: list[dict[str, Any]] = []
    for index in chosen:
        sample = dataset._load_sample(index)
        gt = sample["target_heatmap"]
        weight = 1 + 9 * gt
        base = {"sample_index": index, "dicom_path": sample["dicom_path"],
                "case_id": str(train.iloc[index].case_code), "series_id": str(train.iloc[index].series_uid),
                "foreground_fraction_gt_over_0_1": float((gt > .1).float().mean())}

        def record(name: str, prediction: torch.Tensor, altered: int = 0) -> None:
            squared = (prediction - gt).square()
            plain, weighted = float(squared.mean()), float((weight * squared).mean())
            rows.append({**base, "test": name, "altered_pixels": altered,
                         "plain_mse": plain, "weighted_mse": weighted,
                         "weighted_to_plain_ratio": weighted / plain if plain > 0 else math.nan})

        record("perfect", gt)
        record("all_zero", torch.zeros_like(gt))
        if index == chosen[0]:
            foreground = torch.nonzero(gt.reshape(-1) > .5).reshape(-1)
            background = torch.nonzero(gt.reshape(-1) < 1e-8).reshape(-1)[:len(foreground)]
            if not len(foreground) or len(background) != len(foreground):
                raise RuntimeError("Cannot construct equal-size foreground/background test")
            fg_prediction = gt.clone().reshape(-1)
            bg_prediction = gt.clone().reshape(-1)
            fg_prediction[foreground] += .5
            bg_prediction[background] += .5
            record("equal_amplitude_foreground", fg_prediction.reshape_as(gt), len(foreground))
            record("equal_amplitude_background", bg_prediction.reshape_as(gt), len(background))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "loss_sanity_check.csv", index=False, encoding="utf-8-sig")
    zero = frame.loc[frame.test == "all_zero"]
    fg = frame.loc[frame.test == "equal_amplitude_foreground"].iloc[0]
    bg = frame.loc[frame.test == "equal_amplitude_background"].iloc[0]
    if not (frame.loc[frame.test == "perfect", ["plain_mse", "weighted_mse"]].to_numpy() == 0).all():
        raise RuntimeError("Perfect-prediction weighted MSE was nonzero")
    if not (zero.weighted_mse > zero.plain_mse).all():
        raise RuntimeError("All-zero weighted MSE not greater than plain MSE")
    if not np.isclose(fg.plain_mse, bg.plain_mse, rtol=1e-6) or fg.weighted_mse <= 2 * bg.weighted_mse:
        raise RuntimeError("Foreground weighting failed equal-amplitude diagnostic")
    print(f"[Loss Sanity] All-zero plain={zero.plain_mse.mean():.8f}, weighted={zero.weighted_mse.mean():.8f}, "
          f"ratio={zero.weighted_mse.mean()/zero.plain_mse.mean():.2f}x", flush=True)
    print(f"[Loss Sanity] Equal-error FG plain={fg.plain_mse:.8f}, weighted={fg.weighted_mse:.8f}; "
          f"BG plain={bg.plain_mse:.8f}, weighted={bg.weighted_mse:.8f}; PASS", flush=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("hrnet_phase5c_sanity_plot.py")),
                    "--output", str(output)], check=True)


def correct_summary(output: Path) -> None:
    path = output / "run_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary.update({"phase3_checkpoint_loaded": False, "phase4_checkpoint_loaded": False,
                    "phase5a_checkpoint_loaded": False, "phase5b_checkpoint_loaded": False,
                    "pretrained": True, "pretrained_source": ACTIVE["model"]["pretrained_hf_hub_id"],
                    "formal_training_initialization": "pretrained_backbone_fresh_heatmap_head",
                    "task_heatmap_head_fresh": True,
                    "only_intended_change_from_phase5a": "plain MSE -> mean((1+9*GT)*(Pred-GT)^2); no augmentation"})
    best = pd.read_csv(output / "training_log.csv").loc[lambda df: df.epoch == summary["best_epoch"]].iloc[0]
    summary["train_at_best"]["plain_mse"] = float(best.train_plain_mse)
    summary["train_at_best"]["weighted_loss"] = float(best.train_weighted_loss)
    summary["val_at_best"]["plain_mse"] = float(best.val_plain_mse)
    summary["val_at_best"]["weighted_loss"] = float(best.val_weighted_loss)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5C weighted MSE single-variable experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sanity-only", action="store_true")
    args = parser.parse_args()
    config = phase4.load_config(args.config)
    output = phase4.resolve_path(config, config["project"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    reference = phase4.resolve_path(config, config["project"]["phase5a_baseline_dir"])
    baseline = yaml.safe_load((reference / "config_used.yaml").read_text(encoding="utf-8"))
    allowed = {("training", "loss"), ("model", "forbidden_initializations")}
    for section in ["experiment", "data", "model", "training", "heatmap_confidence", "checkpointing", "visualization"]:
        for key, value in baseline[section].items():
            if (section, key) not in allowed and config[section].get(key) != value:
                raise RuntimeError(f"Phase 5A frozen setting changed: {section}.{key}: {value!r} -> {config[section].get(key)!r}")
    if config["training"]["loss"] != "heatmap_weighted_mse" or config["data"]["augmentation"] != "none":
        raise RuntimeError("Only weighted MSE may differ from Phase 5A; augmentation must remain NONE")
    try:
        loss_sanity(config, output)
        if args.sanity_only:
            print("[Sanity-only] Loss algebra PASS. No model training started.", flush=True)
            return 0
        print("[3/5] Pretrained model / weighted gradient sanity; probe will be discarded...", flush=True)
        global ACTIVE
        ACTIVE = config
        phase5a._ACTIVE_CONFIG = config
        phase5a._ACTIVE_OUTPUT_DIR = output
        phase5a._SUPPRESSION_RADIUS = int(config["heatmap_confidence"]["independent_peak_suppression_radius"])
        if args.resume:
            ckpt = torch.load(output / "checkpoints" / "last.pth", map_location="cpu", weights_only=False)
            if not ckpt.get("pretrained") or ckpt.get("phase5a_checkpoint_loaded", True) or ckpt.get("phase5b_checkpoint_loaded", True) or ckpt.get("loss_config") != config["weighted_loss"]:
                raise RuntimeError("Resume checkpoint does not belong to frozen Phase 5C")
        else:
            loading = phase5a.verify_pretrained_loading(config, output)
            print(f"[Pretrained] PASS: {loading['parameters_different_from_same_seed_random']:,} retained backbone params differ from random; fresh head.", flush=True)
        phase4.CHECKPOINT_WARNING = WARNING
        phase4.PHASE3_WARNING = NOTE
        phase4.build_logger = build_logger
        phase4.build_model = phase5a.build_pretrained_model
        phase4.build_criterion = build_criterion
        phase4.train_one_epoch = train_epoch
        phase4.validate = validate
        phase4.plumbing_test = plumbing_test
        phase4.save_checkpoint = save_checkpoint
        phase4.environment_text = environment_text
        phase4.save_model_summary = save_model_summary
        phase4.print_header = print_header
        print("[4/5] Formal training: Train/Val weighted objective + parallel plain-MSE diagnostics...", flush=True)
        sys.argv = [sys.argv[0], "--config", args.config] + (["--resume"] if args.resume else [])
        result = phase4.main()
        phase5a._merge_confidence_into_training_log(output)
        correct_summary(output)
        print("[Training] Phase 5C complete; finalizer will compare strictly to Phase 5A.", flush=True)
        return result
    except BaseException as exc:
        (output / "phase5c_failure.json").write_text(json.dumps({"type": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc()}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
