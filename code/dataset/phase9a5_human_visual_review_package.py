from __future__ import annotations

import argparse
import html
import json
import math
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value))


def normalize_minmax(array: np.ndarray) -> np.ndarray:
    a = array.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return (a - lo) / (hi - lo)


def add_reason(store: OrderedDict[int, list[str]], idx: int, reason: str) -> None:
    idx = int(idx)
    if idx not in store:
        store[idx] = []
    if reason not in store[idx]:
        store[idx].append(reason)


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.nanargmin(np.abs(values - target)))


def select_review_rows(d: pd.DataFrame, count: int, seed: int) -> tuple[pd.DataFrame, dict[int, str]]:
    n = len(d)
    reasons: OrderedDict[int, list[str]] = OrderedDict()
    positions = [(0.00, "temporal_first"), (0.10, "temporal_10%"), (0.25, "temporal_25%"),
                 (0.50, "temporal_50%"), (0.75, "temporal_75%"), (0.90, "temporal_90%"), (1.00, "temporal_last")]
    for frac, label in positions:
        add_reason(reasons, round(frac * (n - 1)), label)
    residual = d.raw_gt_distance.to_numpy(float)
    for q, label in [(0, "residual_min"), (.25, "residual_Q25"), (.5, "residual_median"),
                     (.75, "residual_Q75"), (.95, "residual_P95"), (1, "residual_max")]:
        add_reason(reasons, nearest_index(residual, float(np.quantile(residual, q))), label)
    motion = d.gt_motion.to_numpy(float)
    valid = np.where(np.isfinite(motion))[0]
    if len(valid):
        for i in valid[np.argsort(motion[valid])[-3:]][::-1]:
            add_reason(reasons, int(i), "motion_outlier")
    rng = np.random.default_rng(seed)
    remaining = [i for i in range(n) if i not in reasons]
    rng.shuffle(remaining)
    for i in remaining:
        if len(reasons) >= min(count, n):
            break
        add_reason(reasons, i, "random")
    chosen = list(reasons.keys())[:min(count, n)]
    selected = d.iloc[chosen].copy()
    selected["sampling_reason"] = [";".join(reasons[i]) for i in chosen]
    return selected, {int(d.iloc[i].frame_index): ";".join(reasons[i]) for i in chosen}


def select_suspicious(d: pd.DataFrame, count: int) -> pd.DataFrame:
    reasons: OrderedDict[int, list[str]] = OrderedDict()
    for col, reason in [("raw_gt_distance", "residual_top"), ("gt_motion", "motion_top"),
                        ("residual_change", "residual_change_top")]:
        values = d[col].to_numpy(float)
        valid = np.where(np.isfinite(values))[0]
        for i in valid[np.argsort(values[valid])[-4:]][::-1]:
            add_reason(reasons, int(i), reason)
    margin = d.edge_margin.to_numpy(float)
    for i in np.argsort(margin)[:2]:
        if margin[i] < 32:
            add_reason(reasons, int(i), "near_boundary")
    coords = list(zip(d.true_tip_row, d.true_tip_col))
    run_start = 0
    for i in range(1, len(coords) + 1):
        if i == len(coords) or coords[i] != coords[run_start]:
            if i - run_start >= 8 and i < len(coords):
                add_reason(reasons, i - 1, "long_repeat_before_change")
                add_reason(reasons, i, "change_after_long_repeat")
            run_start = i
    chosen = list(reasons.keys())[:min(count, len(reasons))]
    out = d.iloc[chosen].copy()
    out["sampling_reason"] = [";".join(reasons[i]) for i in chosen]
    return out


def load_image(path: str) -> np.ndarray:
    return normalize_minmax(pydicom.dcmread(path, force=True).pixel_array)


def overlay(ax, row, x_offset=0.0, y_offset=0.0, legend=False) -> None:
    rx, ry = float(row.raw_col) - x_offset, float(row.raw_row) - y_offset
    gx, gy = float(row.true_tip_col) - x_offset, float(row.true_tip_row) - y_offset
    ax.plot([rx, gx], [ry, gy], color="#FFD400", lw=1.0, alpha=.9)
    ax.plot(rx, ry, marker="x", color="red", ms=8, mew=1.6, linestyle="None", label="RAW")
    ax.plot(gx, gy, marker="o", markerfacecolor="none", markeredgecolor="#00D53B", ms=10, mew=1.7, linestyle="None", label="GT")
    ax.plot(gx, gy, marker="+", color="#00D53B", ms=5, mew=1.1, linestyle="None")
    if legend:
        ax.legend(loc="lower right", fontsize=7, framealpha=.75)


def crop_gt_centered(im: np.ndarray, row, size: int) -> tuple[np.ndarray, int, int, bool]:
    h, w = im.shape
    cx, cy = float(row.true_tip_col), float(row.true_tip_row)
    x0, y0 = int(round(cx - size / 2)), int(round(cy - size / 2))
    x1, y1 = x0 + size, y0 + size
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    out = np.zeros((size, size), dtype=np.float32)
    out[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = im[sy0:sy1, sx0:sx1]
    return out, x0, y0, bool(x0 < 0 or y0 < 0 or x1 > w or y1 > h)


def draw_sheet(rows: pd.DataFrame, path: Path, mode: str, zoom_size: int, title: str, cols=4) -> None:
    if len(rows) == 0:
        return
    nrows = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 5.1, nrows * 5.1), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for j, row in enumerate(rows.itertuples()):
        ax = axes.flat[j]
        im = load_image(row.dicom_path)
        padded = False
        if mode == "zoom":
            im, x0, y0, padded = crop_gt_centered(im, row, zoom_size)
            ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            overlay(ax, row, x0, y0, legend=(j == 0))
        else:
            ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            overlay(ax, row, legend=(j == 0))
        extra = " | PAD" if padded else ""
        reason = getattr(row, "sampling_reason", "")
        ax.set_title(f"{row.case_id} / {row.sequence_identifier} / frame {int(row.frame_index)}\nresidual {row.raw_gt_distance:.1f}px{extra}\n{reason}", fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .975))
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def build_series_dataframe(series_row, mapping_row) -> pd.DataFrame:
    csv_path = str(mapping_row.csv_paths).split(";")[0]
    d = pd.read_csv(csv_path)
    for c in ["frame_index", "raw_row", "raw_col", "true_tip_row", "true_tip_col"]:
        d[c] = pd.to_numeric(d[c], errors="raise")
    files = {p.name: str(p) for p in Path(series_row.series_path).rglob("*.dcm")}
    d["dicom_basename"] = d.dicom_file.map(lambda x: Path(str(x).replace("\\", "/")).name)
    d["dicom_path"] = d.dicom_basename.map(files)
    if d.dicom_path.isna().any():
        raise RuntimeError(f"Unmapped DICOM in {series_row.case_id}/{series_row.sequence_identifier}")
    d = d.sort_values("frame_index").reset_index(drop=True)
    d["case_id"] = str(series_row.case_id)
    d["series_id"] = str(series_row.series_id)
    d["sequence_identifier"] = str(series_row.sequence_identifier)
    d["raw_gt_distance"] = np.hypot(d.true_tip_col-d.raw_col, d.true_tip_row-d.raw_row)
    d["gt_motion"] = np.r_[np.nan, np.hypot(np.diff(d.true_tip_col), np.diff(d.true_tip_row))]
    d["residual_change"] = np.r_[np.nan, np.abs(np.diff(d.raw_gt_distance))]
    d["edge_margin"] = np.minimum.reduce([d.true_tip_col, d.true_tip_row, 1023-d.true_tip_col, 1023-d.true_tip_row])
    in_bounds = ((d.raw_row >= 0) & (d.raw_row < 1024) & (d.true_tip_row >= 0) & (d.true_tip_row < 1024) &
                 (d.raw_col >= 0) & (d.raw_col < 1024) & (d.true_tip_col >= 0) & (d.true_tip_col < 1024))
    if not bool(in_bounds.all()):
        bad = d.loc[~in_bounds, ["frame_index", "raw_row", "raw_col", "true_tip_row", "true_tip_col"]]
        raise RuntimeError(f"Out-of-bounds coordinates in {series_row.case_id}/{series_row.sequence_identifier}:\n{bad}")
    return d


def build_dashboard(summary: pd.DataFrame, out: Path) -> None:
    cards = []
    for r in summary.itertuples():
        rel = Path(r.series_dir).relative_to(out).as_posix()
        cards.append(f"""
        <article class="card">
          <h2>{html.escape(str(r.case_id))} / {html.escape(str(r.sequence_identifier))}</h2>
          <p>{int(r.total_frames)} frames · JSON {html.escape(str(r.json_status))} · residual median {r.median_residual:.1f}px · P95 {r.P95_residual:.1f}px</p>
          <p class="status">PENDING HUMAN REVIEW</p>
          <img src="{rel}/overview_full.png" alt="overview">
          <nav>
            <a href="{rel}/overview_full.png">Full overview</a>
            <a href="{rel}/overview_zoom.png">Zoom overview</a>
            <a href="{rel}/temporal_strip_A.png">Temporal A</a>
            <a href="{rel}/temporal_strip_B.png">Temporal B</a>
            <a href="{rel}/temporal_strip_C.png">Temporal C</a>
            <a href="{rel}/suspicious.png">Suspicious</a>
          </nav>
        </article>""")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Phase 9A.5 human visual review</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;background:#f4f6f8;color:#18212b}}h1{{margin-bottom:4px}}.legend{{background:white;padding:12px 16px;border-left:5px solid #d0a900}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px;margin-top:20px}}.card{{background:white;border:1px solid #d9dee3;border-radius:8px;padding:14px;box-shadow:0 2px 6px #0001}}.card img{{width:100%;height:280px;object-fit:cover;object-position:top;border:1px solid #ddd}}.status{{font-weight:bold;color:#9a6700}}nav{{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}}a{{color:#075985}}code{{background:#eef1f4;padding:2px 5px}}</style></head><body>
<h1>Phase 9A.5 human visual review</h1><p>6 cases · 10 Series · existing CSV labels only</p>
<div class="legend"><b>RED X</b> = system raw coarse point · <b>GREEN CIRCLE</b> = existing manually corrected true-tip GT · <b>YELLOW LINE</b> = raw → GT correction vector.<br>These are not model predictions or new annotations. Review images here, then record decisions in <code>human_review_form.csv</code>.</div>
<main class="grid">{''.join(cards)}</main></body></html>"""
    (out / "review_dashboard.html").write_text(page, encoding="utf-8")


def main(config_path: Path) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.parent.parent
    p9 = root / cfg["phase9a_output"]
    out = root / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases").mkdir(exist_ok=True)
    elig = pd.read_csv(p9 / "series_eligibility.csv", dtype={"case_id": str})
    mapping = pd.read_csv(p9 / "csv_dicom_mapping.csv", dtype={"case_id": str})
    wanted = elig[(elig.case_id.isin(cfg["candidate_cases"])) & (elig.tier.eq("B")) & (elig.current_split.eq("unused"))].copy()
    if len(wanted) != 10 or wanted.case_id.nunique() != 6 or int(wanted.n_matched_labels.sum()) != 2997:
        raise RuntimeError(f"Frozen candidate mismatch: cases={wanted.case_id.nunique()}, series={len(wanted)}, frames={wanted.n_matched_labels.sum()}")
    print(f"[Validation] Frozen candidates: {wanted.case_id.nunique()} cases / {len(wanted)} Series / {int(wanted.n_matched_labels.sum())} frames")
    review_rows, frame_rows, summary_rows = [], [], []
    total = len(wanted)
    for i, sr in enumerate(wanted.sort_values(["case_id", "sequence_identifier"]).itertuples(), 1):
        print(f"[{i}/{total}] case {sr.case_id} series {sr.sequence_identifier}")
        mr = mapping[mapping.series_id.eq(sr.series_id)].iloc[0]
        d = build_series_dataframe(sr, mr)
        selected, _ = select_review_rows(d, cfg["review_frames_per_series"], cfg["sampling_seed"] + i)
        suspicious = select_suspicious(d, cfg["suspicious_frames_per_series"])
        sdir = out / "cases" / safe_name(sr.case_id) / safe_name(sr.sequence_identifier)
        sdir.mkdir(parents=True, exist_ok=True)
        print("  Generating full-frame overview")
        draw_sheet(selected, sdir / "overview_full.png", "full", cfg["zoom_size"], f"{sr.case_id} / {sr.sequence_identifier} — full-frame review")
        print("  Generating GT zoom overview")
        draw_sheet(selected, sdir / "overview_zoom.png", "zoom", cfg["zoom_size"], f"{sr.case_id} / {sr.sequence_identifier} — GT-centered 256×256 review")
        centers = [round(.25*(len(d)-1)), round(.50*(len(d)-1)), round(.75*(len(d)-1))]
        temporal_paths = []
        half = cfg["temporal_strip_length"] // 2
        for label, center in zip("ABC", centers):
            start = max(0, min(len(d)-cfg["temporal_strip_length"], center-half))
            strip = d.iloc[start:start+cfg["temporal_strip_length"]].copy()
            strip["sampling_reason"] = f"continuous strip {label}"
            p = sdir / f"temporal_strip_{label}.png"
            draw_sheet(strip, p, "full", cfg["zoom_size"], f"{sr.case_id} / {sr.sequence_identifier} — temporal strip {label}", cols=4)
            temporal_paths.append(str(p))
        print("  Generating suspicious-frame sheet")
        draw_sheet(suspicious, sdir / "suspicious.png", "full", cfg["zoom_size"], f"{sr.case_id} / {sr.sequence_identifier} — numeric-priority review", cols=4)
        for r in selected.itertuples():
            frame_rows.append({"case_id": r.case_id, "series_id": r.series_id, "series": r.sequence_identifier,
                               "frame_index": int(r.frame_index), "raw_row": r.raw_row, "raw_col": r.raw_col,
                               "true_tip_row": r.true_tip_row, "true_tip_col": r.true_tip_col,
                               "raw_gt_distance": r.raw_gt_distance, "sampling_reason": r.sampling_reason,
                               "review_status": "PENDING", "review_note": ""})
        review_rows.append({"case_id": sr.case_id, "series_id": sr.series_id, "series": sr.sequence_identifier,
                            "n_frames": len(d), "n_review_frames": len(selected), "csv_dicom_mapping_status": mr.mapping_class,
                            "json_status": sr.json_status, "reviewer": "", "raw_plausible": "PENDING",
                            "true_tip_correct": "PENDING", "frame_alignment_ok": "PENDING", "row_col_ok": "PENDING",
                            "sustained_drift": "PENDING", "wrong_structure_seen": "PENDING",
                            "overall_decision": "PENDING", "notes": ""})
        summary_rows.append({"case": sr.case_id, "Series": sr.sequence_identifier, "total_frames": len(d),
                             "review_frames": len(selected), "median_residual": float(d.raw_gt_distance.median()),
                             "P95_residual": float(d.raw_gt_distance.quantile(.95)), "max_residual": float(d.raw_gt_distance.max()),
                             "max_gt_motion": float(d.gt_motion.max()), "overview_path": str(sdir / "overview_full.png"),
                             "zoom_path": str(sdir / "overview_zoom.png"), "temporal_path": ";".join(temporal_paths),
                             "suspicious_path": str(sdir / "suspicious.png"), "review_status": "PENDING",
                             "case_id": sr.case_id, "sequence_identifier": sr.sequence_identifier, "series_dir": str(sdir),
                             "json_status": sr.json_status})
    review_df = pd.DataFrame(review_rows)
    frames_df = pd.DataFrame(frame_rows)
    summary_df = pd.DataFrame(summary_rows)
    review_df.to_csv(out / "human_review_form.csv", index=False, encoding="utf-8-sig")
    frames_df.to_csv(out / "human_review_frames.csv", index=False, encoding="utf-8-sig")
    summary_df.drop(columns=["case_id", "sequence_identifier", "series_dir", "json_status"]).to_csv(out / "review_package_summary.csv", index=False, encoding="utf-8-sig")
    build_dashboard(summary_df, out)
    readme = """# Phase 9A.5 — Human visual review package

**RED X** = system raw coarse point  
**GREEN CIRCLE** = existing manually corrected true-tip GT  
**YELLOW LINE** = raw → GT correction vector

All coordinates come from the existing primary CSV. They are not model predictions and are not new Codex annotations. DICOM display uses per-frame min-max normalization only for visualization; original DICOM and CSV files remain unchanged.

## Review procedure

1. Open `review_dashboard.html`.
2. Inspect full overview, GT-centered zoom, all three temporal strips, and suspicious-frame sheet for each Series.
3. Enter the decision in `human_review_form.csv`. Do not edit source annotation CSV files.

PASS: raw is broadly near the intended tip, GT lies at the intended tip, temporal tracking is reasonable, and no clear frame/structure mismatch is visible.

REVIEW: some frames are unclear or suspicious, additional frames are needed, or tip identity cannot be determined confidently. When uncertain, choose REVIEW rather than forcing PASS.

FAIL: many GT points are visibly off-target, labels follow another structure, frame-label alignment is clearly wrong, row/column semantics are wrong, or there is systematic drift.

Use `generate_extra_review_frames.ps1 -Case <case> -Series <series> -FrameRange <start-end> -NumberOfFrames <N>` for an explicitly requested second look. It does not modify annotations. After review, run `summarize_human_review.ps1` to create a descriptive summary; mixed case status is never automatically passed.

All 10 Series remain PENDING HUMAN REVIEW. This package does not change Train eligibility or the frozen split.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    status = {"status": "complete", "cases": int(wanted.case_id.nunique()), "series": int(len(wanted)),
              "candidate_frames": int(wanted.n_matched_labels.sum()), "unique_review_frames": int(len(frames_df)),
              "all_expected_images_present": True, "source_files_modified": False,
              "training": False, "model_inference": False, "test_used": False, "all_review_status": "PENDING"}
    (out / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"[Done] Review package: {len(wanted)} Series / {len(frames_df)} unique sampled frames")
    print(f"Open: {out / 'review_dashboard.html'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    main(args.config.resolve())
