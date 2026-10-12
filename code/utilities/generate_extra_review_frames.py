from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase9a5_human_visual_review_package import build_series_dataframe, draw_sheet, safe_name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--series", required=True)
    p.add_argument("--frame-range", default="")
    p.add_argument("--number-of-frames", type=int, default=20)
    a = p.parse_args()
    cfg_path = a.config.resolve(); root = cfg_path.parent.parent
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    p9 = root / cfg["phase9a_output"]; out = root / cfg["output_dir"]
    elig = pd.read_csv(p9 / "series_eligibility.csv", dtype={"case_id": str})
    mapping = pd.read_csv(p9 / "csv_dicom_mapping.csv", dtype={"case_id": str})
    hit = elig[(elig.case_id.eq(str(a.case))) & (elig.sequence_identifier.eq(a.series)) & (elig.tier.eq("B")) & (elig.current_split.eq("unused"))]
    if len(hit) != 1:
        raise RuntimeError("Case/Series must identify exactly one frozen Phase 9A.5 Tier B candidate")
    sr = next(hit.itertuples()); mr = mapping[mapping.series_id.eq(sr.series_id)].iloc[0]
    d = build_series_dataframe(sr, mr)
    if a.frame_range:
        parts = a.frame_range.replace(":", "-").split("-")
        if len(parts) != 2:
            raise ValueError("FrameRange must be start-end")
        lo, hi = map(int, parts)
        pool = d[(d.frame_index >= lo) & (d.frame_index <= hi)].copy()
    else:
        pool = d.copy()
    if len(pool) == 0:
        raise RuntimeError("Requested frame range contains no labeled frames")
    n = min(max(1, a.number_of_frames), len(pool))
    positions = np.linspace(0, len(pool)-1, n).round().astype(int)
    chosen = pool.iloc[np.unique(positions)].copy()
    chosen["sampling_reason"] = "extra_review_request"
    sdir = out / "cases" / safe_name(sr.case_id) / safe_name(sr.sequence_identifier) / "extra_review"
    sdir.mkdir(parents=True, exist_ok=True)
    suffix = safe_name(a.frame_range or "full") + f"_{len(chosen)}frames"
    draw_sheet(chosen, sdir / f"extra_full_{suffix}.png", "full", cfg["zoom_size"], f"{sr.case_id} / {sr.sequence_identifier} — extra full review")
    draw_sheet(chosen, sdir / f"extra_zoom_{suffix}.png", "zoom", cfg["zoom_size"], f"{sr.case_id} / {sr.sequence_identifier} — extra zoom review")
    chosen[["case_id", "series_id", "sequence_identifier", "frame_index", "raw_row", "raw_col", "true_tip_row", "true_tip_col", "raw_gt_distance", "sampling_reason"]].to_csv(sdir / f"extra_frames_{suffix}.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {len(chosen)} extra review frames to {sdir}")


if __name__ == "__main__":
    main()
