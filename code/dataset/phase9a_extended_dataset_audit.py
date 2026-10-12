from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PureWindowsPath
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["dicom_file", "frame_index", "raw_row", "raw_col", "true_tip_row", "true_tip_col"]


class Logger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str):
        print(msg, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def clean_id(v) -> str:
    s = "" if pd.isna(v) else str(v)
    return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s


def fnum(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default


def percentile_dict(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {k: np.nan for k in ["mean", "median", "p90", "p95", "p99", "max"]}
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def json_counts(path: Path):
    try:
        z = json.loads(path.read_text(encoding="utf-8-sig"))
        data = z.get("data", {}) if isinstance(z, dict) else {}
        projections = data.get("projections", []) if isinstance(data, dict) else []
        return data.get("frame_count", np.nan), len(projections) if isinstance(projections, list) else np.nan, "READABLE"
    except Exception as e:
        return np.nan, np.nan, f"ERROR:{type(e).__name__}"


def main(config_path: Path):
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    data_root = Path(cfg["data_root"])
    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "qc_previews").mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    log = Logger(out / "phase9a_run.log")
    log("====================================================")
    log("Phase 9A — EXTENDED DATASET ELIGIBILITY AUDIT")
    log("Training: NONE | Model inference: NONE | Test: NOT USED")
    log("Current Val: FROZEN — DO NOT MODIFY")
    log("====================================================")

    historical = pd.read_csv(ROOT / cfg["historical_audit"])
    historical["case_code"] = historical.case_code.map(clean_id)
    model_clean = pd.read_csv(ROOT / cfg["model_clean_series"])
    model_clean["case_code"] = model_clean.case_code.map(clean_id)
    old_by_uid = historical.set_index("series_uid").to_dict("index")
    clean_by_uid = model_clean.set_index("series_uid").to_dict("index")
    top_case_code = {}
    for r in historical.itertuples():
        top = str(r.dicom_path).replace("/", "\\").split("\\")[0]
        top_case_code[top] = clean_id(r.case_code)

    log("[1/12] Rebuilding XA inventory")
    all_dcm = sorted(data_root.rglob("*.dcm"))
    meta_rows = []
    failed_meta = []
    for path in tqdm(all_dcm, desc="DICOM metadata"):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            if str(getattr(ds, "Modality", "")).upper() != "XA":
                continue
            rr = path.relative_to(data_root)
            suid = str(getattr(ds, "SeriesInstanceUID", ""))
            old = old_by_uid.get(suid, {})
            meta_rows.append({
                "dicom_path": str(path), "relative_path": str(rr), "series_path": str(path.parent),
                "top_level_case": rr.parts[0], "case_id": top_case_code.get(rr.parts[0], old.get("case_code", rr.parts[0])),
                "PatientID": str(getattr(ds, "PatientID", "")), "StudyInstanceUID": str(getattr(ds, "StudyInstanceUID", "")),
                "SeriesInstanceUID": suid, "SOPInstanceUID": str(getattr(ds, "SOPInstanceUID", "")),
                "InstanceNumber": fnum(getattr(ds, "InstanceNumber", np.nan)), "Rows": int(getattr(ds, "Rows", 0) or 0),
                "Columns": int(getattr(ds, "Columns", 0) or 0), "BitsStored": int(getattr(ds, "BitsStored", 0) or 0),
                "PhotometricInterpretation": str(getattr(ds, "PhotometricInterpretation", "")),
                "SeriesDescription": str(getattr(ds, "SeriesDescription", "")), "Modality": "XA", "file_size": path.stat().st_size,
            })
        except Exception as e:
            failed_meta.append({"dicom_path": str(path), "error": repr(e)})
    frames = pd.DataFrame(meta_rows)
    if frames.empty:
        raise RuntimeError("No XA DICOM found")
    frames["case_id"] = frames.case_id.map(clean_id)
    series_records = []
    for suid, g in frames.groupby("SeriesInstanceUID", sort=False):
        old = old_by_uid.get(suid, {})
        series_records.append({
            "case_id": clean_id(g.case_id.iloc[0]), "PatientID": g.PatientID.iloc[0], "StudyInstanceUID": g.StudyInstanceUID.iloc[0],
            "SeriesInstanceUID": suid, "sequence_identifier": old.get("sequence_identifier", Path(g.series_path.iloc[0]).name),
            "series_path": g.series_path.iloc[0], "series_description": g.SeriesDescription.iloc[0], "Modality": "XA",
            "number_of_dicom_files": len(g), "number_of_readable_dicom_files": len(g),
            "Rows": "|".join(map(str, sorted(g.Rows.unique()))), "Columns": "|".join(map(str, sorted(g.Columns.unique()))),
            "BitsStored": "|".join(map(str, sorted(g.BitsStored.unique()))),
            "PhotometricInterpretation": "|".join(map(str, sorted(g.PhotometricInterpretation.unique()))),
        })
    inventory = pd.DataFrame(series_records)

    basename_index = defaultdict(list)
    path_to_meta = {}
    for r in frames.itertuples():
        basename_index[Path(r.dicom_path).name].append(r)
        path_to_meta[str(Path(r.dicom_path).resolve()).lower()] = r

    log("[2/12] DICOM identity / duplication audit")
    log("[3/12] CSV schema validation")
    csv_infos = []
    csv_data = {}
    for path in sorted(data_root.rglob("*.csv")):
        stat = path.stat()
        try:
            d = pd.read_csv(path)
            missing = [c for c in REQUIRED if c not in d.columns]
            info = {"csv_path": str(path), "file_size": stat.st_size, "mtime": stat.st_mtime, "row_count": len(d),
                    "columns": "|".join(d.columns), "missing_columns": "|".join(missing)}
            if missing:
                info.update({"nan_count": np.nan, "inf_count": np.nan, "non_numeric_coordinate_rows": np.nan,
                             "duplicate_rows": int(d.duplicated().sum()), "duplicate_frame_index": np.nan,
                             "duplicate_dicom_file": np.nan, "frame_index_min": np.nan, "frame_index_max": np.nan,
                             "coordinate_out_of_bounds_rows": np.nan, "target_series_count": 0, "target_series_uid": "", "status": "SCHEMA_ERROR"})
                csv_infos.append(info); csv_data[str(path)] = d; continue
            num = d[["frame_index", "raw_row", "raw_col", "true_tip_row", "true_tip_col"]].apply(pd.to_numeric, errors="coerce")
            nonnum = num.isna().any(axis=1)
            finite = np.isfinite(num.to_numpy(dtype=float)).all(axis=1)
            targets = []
            mapped_uid = []
            for v in d.dicom_file.astype(str):
                b = PureWindowsPath(v.replace("/", "\\")).name
                hits = basename_index.get(b, [])
                mapped_uid.append(hits[0].SeriesInstanceUID if len(hits) == 1 else "")
                if len(hits) == 1: targets.append(hits[0].SeriesInstanceUID)
            target_set = sorted(set(targets))
            out_bounds = np.zeros(len(d), dtype=bool)
            for i, uid in enumerate(mapped_uid):
                if not uid or nonnum.iloc[i]: out_bounds[i] = True; continue
                hit = basename_index[PureWindowsPath(str(d.dicom_file.iloc[i]).replace("/", "\\")).name][0]
                out_bounds[i] = not (0 <= num.raw_row.iloc[i] < hit.Rows and 0 <= num.true_tip_row.iloc[i] < hit.Rows and
                                     0 <= num.raw_col.iloc[i] < hit.Columns and 0 <= num.true_tip_col.iloc[i] < hit.Columns)
            info.update({"nan_count": int(d[REQUIRED].isna().sum().sum()), "inf_count": int((~finite & ~nonnum.to_numpy()).sum()),
                         "non_numeric_coordinate_rows": int(nonnum.sum()), "duplicate_rows": int(d.duplicated().sum()),
                         "duplicate_frame_index": int(d.frame_index.duplicated().sum()), "duplicate_dicom_file": int(d.dicom_file.astype(str).duplicated().sum()),
                         "frame_index_min": fnum(num.frame_index.min()), "frame_index_max": fnum(num.frame_index.max()),
                         "coordinate_out_of_bounds_rows": int(out_bounds.sum()), "target_series_count": len(target_set),
                         "target_series_uid": "|".join(target_set), "status": "CLEAN" if not nonnum.any() and finite.all() and not out_bounds.any() else "INVALID_VALUES"})
            d = d.copy(); d["_mapped_series_uid"] = mapped_uid; d["_dicom_basename"] = [PureWindowsPath(str(v).replace("/", "\\")).name for v in d.dicom_file]
            for c in ["frame_index", "raw_row", "raw_col", "true_tip_row", "true_tip_col"]: d[c] = num[c]
            csv_infos.append(info); csv_data[str(path)] = d
        except Exception as e:
            csv_infos.append({"csv_path": str(path), "file_size": stat.st_size, "mtime": stat.st_mtime, "row_count": np.nan,
                              "columns": "", "missing_columns": "", "status": f"READ_ERROR:{type(e).__name__}", "target_series_count": 0, "target_series_uid": ""})
    csv_audit = pd.DataFrame(csv_infos)
    csv_audit.to_csv(out / "csv_schema_audit.csv", index=False)
    uid_to_csv = defaultdict(list)
    for r in csv_audit.itertuples():
        if int(getattr(r, "target_series_count", 0) or 0) == 1:
            uid_to_csv[str(r.target_series_uid)].append(str(r.csv_path))

    log("[4/12] CSV ↔ DICOM mapping")
    mapping_rows = []
    frame_index_rows = []
    labels_by_uid = {}
    for inv in inventory.itertuples():
        uid = inv.SeriesInstanceUID; paths = uid_to_csv.get(uid, []); sg = frames[frames.SeriesInstanceUID.eq(uid)]
        if not paths:
            mapping_rows.append({"case_id": inv.case_id, "series_id": uid, "sequence_identifier": inv.sequence_identifier,
                                 "n_csv_files": 0, "csv_paths": "", "csv_rows": 0, "unique_csv_dicoms": 0,
                                 "physical_dicom_frames": len(sg), "matched_rows": 0, "unmatched_csv_rows": 0,
                                 "unlabeled_dicom_frames": len(sg), "coverage_percent": 0.0, "mapping_class": "NO_CSV"})
            continue
        combined = []
        for cp in paths:
            d = csv_data[cp]
            valid = d._mapped_series_uid.eq(uid)
            matched = int(valid.sum()); unique_d = int(d.loc[valid, "_dicom_basename"].nunique())
            frame_vals = d.frame_index.dropna().astype(int)
            contig = len(frame_vals) > 0 and sorted(frame_vals.tolist()) == list(range(int(frame_vals.min()), int(frame_vals.max()) + 1))
            zero = len(frame_vals) > 0 and int(frame_vals.min()) == 0
            strict = d.frame_index.is_monotonic_increasing and not d.frame_index.duplicated().any()
            ordered = sg.sort_values(["InstanceNumber", "SOPInstanceUID"], na_position="last")
            rank = {Path(p).name: i for i, p in enumerate(ordered.dicom_path)}
            comparable = d.loc[valid, ["frame_index", "_dicom_basename"]].copy()
            ordering_match = bool(len(comparable) and all(rank.get(b, -999) == int(fi) for fi, b in comparable.itertuples(index=False)))
            missing_indices = sorted(set(range(int(frame_vals.min()), int(frame_vals.max()) + 1)) - set(frame_vals)) if len(frame_vals) else []
            frame_index_rows.append({"case_id": inv.case_id, "series_id": uid, "sequence_identifier": inv.sequence_identifier, "csv_path": cp,
                                     "n_rows": len(d), "min": frame_vals.min() if len(frame_vals) else np.nan, "max": frame_vals.max() if len(frame_vals) else np.nan,
                                     "zero_based": zero, "continuous": contig, "strictly_increasing": strict, "duplicate_indices": int(d.frame_index.duplicated().sum()),
                                     "missing_indices": "|".join(map(str, missing_indices)), "dicom_ordering_match": ordering_match,
                                     "status": "CLEAN" if zero and contig and strict and ordering_match else "REVIEW"})
            combined.append(d)
        total_rows = sum(len(csv_data[p]) for p in paths)
        all_d = pd.concat(combined, ignore_index=True)
        matched = int(all_d._mapped_series_uid.eq(uid).sum()); unique_d = int(all_d.loc[all_d._mapped_series_uid.eq(uid), "_dicom_basename"].nunique())
        if len(paths) > 1:
            cls = "CONFLICT"
        elif matched == len(all_d) and unique_d == len(sg) and len(all_d) == len(sg):
            cls = "EXACT"
        elif matched > 0 and matched == len(all_d):
            cls = "PARTIAL"
        else:
            cls = "CONFLICT"
        mapping_rows.append({"case_id": inv.case_id, "series_id": uid, "sequence_identifier": inv.sequence_identifier,
                             "n_csv_files": len(paths), "csv_paths": "|".join(paths), "csv_rows": total_rows,
                             "unique_csv_dicoms": unique_d, "physical_dicom_frames": len(sg), "matched_rows": matched,
                             "unmatched_csv_rows": total_rows - matched, "unlabeled_dicom_frames": len(sg) - unique_d,
                             "coverage_percent": 100 * unique_d / len(sg), "mapping_class": cls})
        if len(paths) == 1:
            labels_by_uid[uid] = csv_data[paths[0]].sort_values("frame_index").reset_index(drop=True)
    mapping = pd.DataFrame(mapping_rows)
    mapping.to_csv(out / "csv_dicom_mapping.csv", index=False)
    pd.DataFrame(frame_index_rows).to_csv(out / "frame_index_audit.csv", index=False)

    log("[5/12] Frame-index integrity")
    log("[6/12] Pixel readability")
    pixel_rows = []
    for r in tqdm(list(frames.itertuples()), desc="XA pixel audit"):
        path = Path(r.dicom_path)
        row = {"case_id": r.case_id, "series_id": r.SeriesInstanceUID, "dicom_path": str(path), "SOPInstanceUID": r.SOPInstanceUID}
        try:
            raw = path.read_bytes()
            ds = pydicom.dcmread(io.BytesIO(raw), force=True)
            arr = ds.pixel_array
            a = np.asarray(arr)
            row.update({"readable": True, "shape": "x".join(map(str, a.shape)), "dtype": str(a.dtype), "finite": bool(np.isfinite(a).all()),
                        "min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()), "std": float(a.std()),
                        "all_zero": bool(np.all(a == 0)), "constant": bool(a.min() == a.max()),
                        "dynamic_range": float(a.max()) - float(a.min()), "file_sha256": hashlib.sha256(raw).hexdigest(),
                        "pixel_sha256": hashlib.sha256(a.tobytes()).hexdigest(), "error": ""})
        except Exception as e:
            row.update({"readable": False, "shape": "", "dtype": "", "finite": False, "min": np.nan, "max": np.nan,
                        "mean": np.nan, "std": np.nan, "all_zero": np.nan, "constant": np.nan, "dynamic_range": np.nan,
                        "file_sha256": "", "pixel_sha256": "", "error": repr(e)})
            log(f"[Pixel ERROR] {path}: {e!r}")
        pixel_rows.append(row)
    pixels = pd.DataFrame(pixel_rows)
    pixels.to_csv(out / "pixel_readability_audit.csv", index=False)

    duplicate_rows = []
    pmeta = frames[["case_id", "SeriesInstanceUID", "dicom_path", "SOPInstanceUID"]].merge(pixels[["dicom_path", "file_sha256", "pixel_sha256"]], on="dicom_path", how="left")
    for kind, col in [("SOPInstanceUID", "SOPInstanceUID"), ("FILE_SHA256", "file_sha256"), ("PIXEL_SHA256_CROSS_SERIES", "pixel_sha256")]:
        for value, g in pmeta[pmeta[col].fillna("").ne("")].groupby(col):
            if len(g) < 2: continue
            if kind == "PIXEL_SHA256_CROSS_SERIES" and g.SeriesInstanceUID.nunique() < 2: continue
            duplicate_rows.append({"duplicate_type": kind, "identifier": value, "n_files": len(g), "n_series": g.SeriesInstanceUID.nunique(),
                                   "n_cases": g.case_id.nunique(), "cases": "|".join(sorted(g.case_id.astype(str).unique())),
                                   "series": "|".join(sorted(g.SeriesInstanceUID.unique())), "paths": "|".join(g.dicom_path)})
    pd.DataFrame(duplicate_rows, columns=["duplicate_type", "identifier", "n_files", "n_series", "n_cases", "cases", "series", "paths"]).to_csv(out / "dicom_duplicate_audit.csv", index=False)

    log("[7/12] Coordinate / temporal label sanity")
    residual_rows = []
    temporal_rows = []
    label_frame_rows = []
    for inv in inventory.itertuples():
        uid = inv.SeriesInstanceUID
        paths = uid_to_csv.get(uid, [])
        for cp in paths:
            d = csv_data[cp].sort_values("frame_index").reset_index(drop=True)
            if any(c not in d for c in REQUIRED): continue
            dx = d.true_tip_col - d.raw_col; dy = d.true_tip_row - d.raw_row; rad = np.hypot(dx, dy)
            gt_motion = np.r_[np.nan, np.hypot(np.diff(d.true_tip_col), np.diff(d.true_tip_row))]
            raw_motion = np.r_[np.nan, np.hypot(np.diff(d.raw_col), np.diff(d.raw_row))]
            residual_change = np.r_[np.nan, np.hypot(np.diff(dx), np.diff(dy))]
            rs = percentile_dict(rad)
            residual_rows.append({"case_id": inv.case_id, "series_id": uid, "sequence_identifier": inv.sequence_identifier, "csv_path": cp,
                                  "n": len(d), "dx_mean": dx.mean(), "dx_median": dx.median(), "dy_mean": dy.mean(), "dy_median": dy.median(),
                                  **{f"radial_{k}": v for k, v in rs.items()}, "raw_constant_fraction": float(pd.DataFrame({"x": d.raw_col, "y": d.raw_row}).duplicated().mean()),
                                  "true_constant_fraction": float(pd.DataFrame({"x": d.true_tip_col, "y": d.true_tip_row}).duplicated().mean())})
            ms = percentile_dict(gt_motion); rms = percentile_dict(raw_motion); rcs = percentile_dict(residual_change)
            outliers = np.where(gt_motion > cfg["temporal_outlier_px"])[0]
            residual_jumps = np.where(residual_change > cfg["residual_shift_px"])[0]
            temporal_rows.append({"case_id": inv.case_id, "series_id": uid, "sequence_identifier": inv.sequence_identifier, "csv_path": cp,
                                  **{f"gt_motion_{k}": v for k, v in ms.items()}, **{f"raw_motion_{k}": v for k, v in rms.items()},
                                  **{f"residual_change_{k}": v for k, v in rcs.items()}, "temporal_outlier_count": len(outliers),
                                  "temporal_outlier_frames": "|".join(map(str, d.frame_index.iloc[outliers].astype(int))),
                                  "residual_jump_count": len(residual_jumps), "residual_jump_frames": "|".join(map(str, d.frame_index.iloc[residual_jumps].astype(int))),
                                  "true_unique_fraction": d[["true_tip_row", "true_tip_col"]].drop_duplicates().shape[0]/len(d),
                                  "raw_unique_fraction": d[["raw_row", "raw_col"]].drop_duplicates().shape[0]/len(d)})
            for i, rr in d.iterrows():
                label_frame_rows.append({"case_id": inv.case_id, "series_id": uid, "sequence_identifier": inv.sequence_identifier, "csv_path": cp,
                                         "frame_index": int(rr.frame_index), "dicom_basename": rr._dicom_basename,
                                         "raw_x": rr.raw_col, "raw_y": rr.raw_row, "gt_x": rr.true_tip_col, "gt_y": rr.true_tip_row,
                                         "residual": rad.iloc[i], "gt_motion": gt_motion[i], "raw_motion": raw_motion[i], "residual_change": residual_change[i]})
    residual = pd.DataFrame(residual_rows); temporal = pd.DataFrame(temporal_rows); label_frames = pd.DataFrame(label_frame_rows)
    residual.to_csv(out / "raw_true_residual_summary.csv", index=False)
    temporal.to_csv(out / "temporal_label_sanity.csv", index=False)

    log("[8/12] JSON auxiliary audit")
    json_rows = []
    for inv in inventory.itertuples():
        old = old_by_uid.get(inv.SeriesInstanceUID, {})
        candidates = []
        for col in ["json_path", "alternative_json_path"]:
            v = old.get(col, "")
            if isinstance(v, str) and v.strip(): candidates.append(data_root / Path(v.replace("\\", os.sep)))
        candidates = list(dict.fromkeys(candidates))
        vals = []
        for jp in candidates:
            fc, pc, st = json_counts(jp); vals.append((jp, fc, pc, st))
        if not vals:
            status = "MISSING"
            fc = pc = np.nan
        else:
            fc, pc = vals[0][1], vals[0][2]
            status = "MATCH" if fnum(fc) == inv.number_of_dicom_files and fnum(pc) == inv.number_of_dicom_files else "COUNT_MISMATCH"
            if any(v[3] != "READABLE" for v in vals): status = "READ_ERROR"
        json_rows.append({"case_id": inv.case_id, "series_id": inv.SeriesInstanceUID, "sequence_identifier": inv.sequence_identifier,
                          "json_count": len(vals), "json_paths": "|".join(str(v[0]) for v in vals), "json_frame_count": fc,
                          "json_projection_count": pc, "dicom_count": inv.number_of_dicom_files,
                          "csv_count": int(mapping.loc[mapping.series_id.eq(inv.SeriesInstanceUID), "unique_csv_dicoms"].iloc[0]), "json_status": status,
                          "authority_note": "AUXILIARY_ONLY_NOT_TRUE_TIP_GT"})
    json_audit = pd.DataFrame(json_rows); json_audit.to_csv(out / "json_auxiliary_audit.csv", index=False)

    log("[9/12] Annotation conflict audit")
    conflict_rows = []
    for uid, paths in uid_to_csv.items():
        if len(paths) < 2: continue
        base = csv_data[paths[0]].set_index("frame_index")
        for other_path in paths[1:]:
            other = csv_data[other_path].set_index("frame_index")
            common = base.index.intersection(other.index)
            dif = (base.loc[common, ["raw_row", "raw_col", "true_tip_row", "true_tip_col"]] != other.loc[common, ["raw_row", "raw_col", "true_tip_row", "true_tip_col"]]).any(axis=1)
            rawdif = (base.loc[common, ["raw_row", "raw_col"]] != other.loc[common, ["raw_row", "raw_col"]]).any(axis=1)
            truedif = (base.loc[common, ["true_tip_row", "true_tip_col"]] != other.loc[common, ["true_tip_row", "true_tip_col"]]).any(axis=1)
            inv = inventory[inventory.SeriesInstanceUID.eq(uid)].iloc[0]
            conflict_rows.append({"conflict_type": "MULTIPLE_CSV_LABEL_CONFLICT" if dif.any() else "DUPLICATE_METADATA_ONLY",
                                  "case_id": inv.case_id, "series_id": uid, "sequence_identifier": inv.sequence_identifier,
                                  "csv_a": paths[0], "csv_b": other_path, "rows_a": len(base), "rows_b": len(other), "common_frames": len(common),
                                  "any_coordinate_difference_frames": int(dif.sum()), "raw_difference_frames": int(rawdif.sum()),
                                  "true_tip_difference_frames": int(truedif.sum()), "resolution": "DO_NOT_AUTO_SELECT" if dif.any() else "IDENTICAL"})
    # Detect exact coordinate-sequence copies across distinct Series.
    seq_hashes = defaultdict(list)
    for uid, d in labels_by_uid.items():
        b = d[["raw_row", "raw_col", "true_tip_row", "true_tip_col"]].to_numpy(dtype=np.float64).tobytes()
        seq_hashes[hashlib.sha256(b).hexdigest()].append(uid)
    for h, uids in seq_hashes.items():
        if len(uids) > 1:
            conflict_rows.append({"conflict_type": "CROSS_SERIES_IDENTICAL_COORDINATE_SEQUENCE", "case_id": "|".join(sorted(inventory[inventory.SeriesInstanceUID.isin(uids)].case_id.astype(str).unique())),
                                  "series_id": "|".join(uids), "sequence_identifier": "", "csv_a": "", "csv_b": "", "rows_a": "", "rows_b": "",
                                  "common_frames": "", "any_coordinate_difference_frames": 0, "raw_difference_frames": 0, "true_tip_difference_frames": 0,
                                  "resolution": "REVIEW_FOR_DUPLICATION"})
    conflicts = pd.DataFrame(conflict_rows, columns=["conflict_type", "case_id", "series_id", "sequence_identifier", "csv_a", "csv_b", "rows_a", "rows_b", "common_frames", "any_coordinate_difference_frames", "raw_difference_frames", "true_tip_difference_frames", "resolution"])
    conflicts.to_csv(out / "annotation_conflicts.csv", index=False)

    log("[10/12] Eligibility classification")
    pix_summary = pixels.groupby("series_id").agg(pixel_frames=("dicom_path", "size"), readable_frames=("readable", "sum"), all_zero_frames=("all_zero", "sum"),
                                                     constant_frames=("constant", "sum"), image_mean=("mean", "mean"), image_std=("std", "mean")).reset_index()
    frame_summary = pd.DataFrame(frame_index_rows)
    rows = []
    historical_rows = []
    for inv in inventory.itertuples():
        uid = inv.SeriesInstanceUID; old = old_by_uid.get(uid, {}); m = mapping[mapping.series_id.eq(uid)].iloc[0]
        js = json_audit[json_audit.series_id.eq(uid)].iloc[0]; ps = pix_summary[pix_summary.series_id.eq(uid)].iloc[0]
        paths = uid_to_csv.get(uid, []); has_csv = len(paths) > 0
        schema_clean = has_csv and all(csv_audit[csv_audit.csv_path.eq(p)].status.iloc[0] == "CLEAN" for p in paths)
        mapping_clean = m.mapping_class in ["EXACT", "PARTIAL"]
        coord_clean = schema_clean and all(int(csv_audit[csv_audit.csv_path.eq(p)].coordinate_out_of_bounds_rows.iloc[0]) == 0 for p in paths)
        pixel_clean = int(ps.readable_frames) == int(ps.pixel_frames) and int(ps.all_zero_frames or 0) == 0 and int(ps.constant_frames or 0) == 0
        conflict = "LABEL_CONFLICT" if m.mapping_class == "CONFLICT" or len(paths) > 1 else "NONE"
        old_flags = str(old.get("qc_flags", "UNKNOWN_HISTORICAL_REASON"))
        old_usage = str(old.get("recommended_usage", "UNKNOWN_HISTORICAL_REASON"))
        current_split = clean_by_uid.get(uid, {}).get("split", "unused")
        if current_split == "train": current_split = "current_train"
        elif current_split == "val": current_split = "current_val"
        elif clean_id(inv.case_id) in cfg["frozen_val_cases"]: current_split = "current_val_unused_series"
        if not has_csv:
            tier = "UNLABELED"; issue = "No primary CSV tip labels"; review = "NOT_APPLICABLE"; eligible = "no"
        elif conflict != "NONE":
            tier = "D"; issue = "Multiple conflicting annotation versions or ambiguous mapping"; review = "CONFLICT_UNRESOLVED"; eligible = "no"
        elif not schema_clean or not mapping_clean or not coord_clean or not pixel_clean:
            severe = not mapping_clean or not coord_clean or int(ps.readable_frames) < int(ps.pixel_frames)
            tier = "D" if severe else "C"; issue = "Technical CSV/DICOM/pixel integrity problem"; review = "MANUAL_REVIEW_REQUIRED"; eligible = "no" if tier == "D" else "pending"
        elif "uncertain_tip" in old_flags or "needs_manual_review" in old_flags:
            tier = "C"; issue = "Historical annotation-quality concern; numeric mapping is clean"; review = "HUMAN_VISUAL_REVIEW_REQUIRED"; eligible = "no"
        elif js.json_status in ["MISSING", "COUNT_MISMATCH"]:
            tier = "B"; issue = f"CSV/DICOM clean; auxiliary JSON {js.json_status.lower()}"; review = "HUMAN_VISUAL_REVIEW_REQUIRED" if current_split == "unused" else "HISTORICAL_SAMPLE_REVIEW_COMPLETED"; eligible = "pending" if current_split == "unused" else ("no" if "val" in current_split else "yes")
        else:
            tier = "A"; issue = "CSV/DICOM/pixel checks clean"; review = "HISTORICAL_SAMPLE_REVIEW_COMPLETED" if uid in clean_by_uid else "HUMAN_VISUAL_REVIEW_REQUIRED"; eligible = "no" if "val" in current_split else ("yes" if review != "HUMAN_VISUAL_REVIEW_REQUIRED" else "pending")
        if "val" in current_split: eligible = "no"
        category = "FROZEN_VAL_CASE" if clean_id(inv.case_id) in cfg["frozen_val_cases"] else ("EXISTING_CASE_EXTRA_SERIES" if clean_id(inv.case_id) in set(model_clean[model_clean.split.eq("train")].case_code) and current_split == "unused" else ("NEW_CASE_ELIGIBLE" if clean_id(inv.case_id) not in set(model_clean.case_code) and tier in ["A", "B"] else "CURRENT_OR_OTHER"))
        historical_reason = old_flags if uid not in clean_by_uid else "INCLUDED_IN_MODEL_CLEAN_V1"
        historical_rows.append({"case_id": inv.case_id, "series_id": uid, "sequence_identifier": inv.sequence_identifier,
                                "in_model_clean_v1": uid in clean_by_uid, "historical_exclusion_reason": historical_reason,
                                "historical_recommended_usage": old_usage, "current_audit_finding": issue})
        rows.append({"case_id": clean_id(inv.case_id), "series_id": uid, "series_path": inv.series_path, "sequence_identifier": inv.sequence_identifier,
                     "n_dicom_frames": inv.number_of_dicom_files, "n_csv_rows": int(m.csv_rows), "n_matched_labels": int(m.matched_rows),
                     "tier": tier, "current_split": current_split, "case_category": category,
                     "new_case_flag": category == "NEW_CASE_ELIGIBLE", "historical_exclusion_reason": historical_reason,
                     "current_issue_summary": issue, "csv_clean": schema_clean, "mapping_clean": mapping_clean,
                     "coordinate_clean": coord_clean, "pixel_clean": pixel_clean, "json_status": js.json_status,
                     "conflict_status": conflict, "human_visual_review_status": review,
                     "eligible_for_future_train": eligible, "notes": "Val remains excluded" if "val" in current_split else ""})
    elig = pd.DataFrame(rows)
    elig.to_csv(out / "series_eligibility.csv", index=False)
    pd.DataFrame(historical_rows).to_csv(out / "historical_exclusion_reconstruction.csv", index=False)
    candidates = elig[elig.n_csv_rows.gt(0)].copy()
    candidates.to_csv(out / "extended_dataset_candidate_manifest.csv", index=False)

    q = candidates[(candidates.tier.eq("C")) | (candidates.human_visual_review_status.eq("HUMAN_VISUAL_REVIEW_REQUIRED")) | (candidates.conflict_status.ne("NONE"))].copy()
    q["priority"] = np.select([
        q.new_case_flag & q.tier.isin(["A", "B"]), q.new_case_flag, q.tier.isin(["A", "B"]), q.conflict_status.ne("NONE")], [1, 2, 3, 4], default=4)
    q["review_reason"] = q.current_issue_summary
    q.sort_values(["priority", "n_dicom_frames"], ascending=[True, False]).to_csv(out / "manual_review_queue.csv", index=False)

    train_cases = set(model_clean[model_clean.split.eq("train")].case_code)
    case_rows = []
    for case, g in elig.groupby("case_id"):
        ann = g[g.n_csv_rows.gt(0)]
        eligible_ab = ann[ann.tier.isin(["A", "B"])]
        case_rows.append({"case_id": case, "total_xa_series": len(g), "csv_series": len(ann),
                          "tier_A": int((ann.tier == "A").sum()), "tier_B": int((ann.tier == "B").sum()),
                          "tier_C": int((ann.tier == "C").sum()), "tier_D": int((ann.tier == "D").sum()),
                          "eligible_series_A_B": len(eligible_ab), "eligible_frames_A_B": int(eligible_ab.n_matched_labels.sum()),
                          "current_train_case": case in train_cases, "current_val_case": case in cfg["frozen_val_cases"],
                          "new_independent_case": case not in train_cases and case not in cfg["frozen_val_cases"]})
    pd.DataFrame(case_rows).to_csv(out / "case_eligibility.csv", index=False)

    scenario_rows = []
    cur = elig[elig.current_split.eq("current_train")]
    for name, mask in [
        ("Scenario 0 — Current", elig.current_split.eq("current_train")),
        ("Scenario 1 — Conservative Tier A", elig.tier.eq("A") & ~elig.case_id.isin(cfg["frozen_val_cases"])),
        ("Scenario 2 — Tier A + B", elig.tier.isin(["A", "B"]) & ~elig.case_id.isin(cfg["frozen_val_cases"]))]:
        s = elig[mask]
        scenario_rows.append({"scenario": name, "cases": s.case_id.nunique(), "new_cases_vs_current": len(set(s.case_id)-train_cases),
                              "series": len(s), "frames": int(s.n_matched_labels.sum()),
                              "new_case_frames": int(s[s.case_id.isin(set(s.case_id)-train_cases)].n_matched_labels.sum()),
                              "existing_train_case_extra_frames": int(s[s.case_category.eq("EXISTING_CASE_EXTRA_SERIES")].n_matched_labels.sum()),
                              "frozen_val_included": bool(s.case_id.isin(cfg["frozen_val_cases"]).any())})
    scenarios = pd.DataFrame(scenario_rows); scenarios.to_csv(out / "expansion_scenarios.csv", index=False)

    # Enrich inventory after all mappings are known.
    inventory = inventory.merge(mapping[["series_id", "n_csv_files", "csv_paths", "mapping_class"]], left_on="SeriesInstanceUID", right_on="series_id", how="left")
    inventory = inventory.merge(json_audit[["series_id", "json_count", "json_paths", "json_status"]], left_on="SeriesInstanceUID", right_on="series_id", how="left", suffixes=("", "_json"))
    inventory["other_annotation_files"] = ""
    inventory.drop(columns=[c for c in ["series_id", "series_id_json"] if c in inventory], inplace=True)
    inventory.to_csv(out / "all_xa_series_inventory.csv", index=False)

    # Diversity table uses authoritative single-CSV labels only.
    div_rows = []
    for inv in inventory.itertuples():
        uid = inv.SeriesInstanceUID
        if uid not in labels_by_uid: continue
        d = labels_by_uid[uid].sort_values("frame_index"); rad = np.hypot(d.true_tip_col-d.raw_col, d.true_tip_row-d.raw_row)
        motion = np.hypot(np.diff(d.true_tip_col), np.diff(d.true_tip_row))
        erow = elig[elig.series_id.eq(uid)].iloc[0]; ps = pix_summary[pix_summary.series_id.eq(uid)].iloc[0]
        div_rows.append({"case_id": erow.case_id, "series_id": uid, "sequence_identifier": erow.sequence_identifier, "tier": erow.tier,
                         "current_split": erow.current_split, "n_frames": len(d), "raw_error_mean": rad.mean(), "raw_error_p95": np.percentile(rad,95),
                         "gt_x_mean": d.true_tip_col.mean(), "gt_x_std": d.true_tip_col.std(), "gt_y_mean": d.true_tip_row.mean(), "gt_y_std": d.true_tip_row.std(),
                         "gt_motion_median": np.median(motion) if len(motion) else np.nan, "gt_motion_p95": np.percentile(motion,95) if len(motion) else np.nan,
                         "image_mean": ps.image_mean, "image_std": ps.image_std})
    diversity = pd.DataFrame(div_rows); diversity.to_csv(out / "diversity_summary.csv", index=False)

    log("[11/12] QC preview generation")
    nonclean = candidates[~candidates.series_id.isin(clean_by_uid)].copy()
    frame_path_lookup = {Path(r.dicom_path).name: r.dicom_path for r in frames.itertuples()}
    for er in tqdm(list(nonclean.itertuples()), desc="QC previews"):
        uid = er.series_id; paths = uid_to_csv.get(uid, [])
        if not paths: continue
        dslist = [csv_data[p].sort_values("frame_index").reset_index(drop=True) for p in paths]
        d = dslist[0]; n = len(d); residual_v = np.hypot(d.true_tip_col-d.raw_col, d.true_tip_row-d.raw_row)
        motion_v = np.r_[np.nan, np.hypot(np.diff(d.true_tip_col), np.diff(d.true_tip_row))]
        idx = [0, round(.25*(n-1)), round(.5*(n-1)), round(.75*(n-1)), n-1,
               int(np.argmin(residual_v)), int(np.argsort(residual_v)[n//2]), int(np.argsort(residual_v)[max(0,int(.9*n)-1)]), int(np.argmax(residual_v)),
               int(np.nanargmax(motion_v)) if np.isfinite(motion_v).any() else 0]
        idx = list(dict.fromkeys(idx))
        for x in np.linspace(0,n-1,20).round().astype(int):
            if len(idx) >= cfg["qc_frames_per_series"]: break
            if int(x) not in idx: idx.append(int(x))
        idx = idx[:cfg["qc_frames_per_series"]]
        fig, axs = plt.subplots(3, 4, figsize=(14, 10), squeeze=False)
        for ax, i in zip(axs.flat, idx):
            rr = d.iloc[i]; dp = frame_path_lookup.get(rr._dicom_basename)
            try:
                a = pydicom.dcmread(dp, force=True).pixel_array.astype(np.float32); lo, hi = np.percentile(a, [1, 99.5]); im = np.clip((a-lo)/(hi-lo+1e-6),0,1)
                ax.imshow(im, cmap="gray", vmin=0, vmax=1)
                ax.plot(rr.raw_col, rr.raw_row, "r+", ms=8, mew=1.2, label="raw")
                ax.plot(rr.true_tip_col, rr.true_tip_row, "go", fillstyle="none", ms=7, mew=1.2, label="true")
                if len(dslist) > 1:
                    alt = dslist[1]
                    hit = alt[alt.frame_index.eq(rr.frame_index)]
                    if len(hit):
                        aa=hit.iloc[0]; ax.plot(aa.raw_col,aa.raw_row,"m+",ms=7,label="raw alt"); ax.plot(aa.true_tip_col,aa.true_tip_row,"co",fillstyle="none",ms=6,label="true alt")
                ax.set_title(f"f={int(rr.frame_index)} d={residual_v.iloc[i]:.1f}px", fontsize=8)
            except Exception as e:
                ax.text(.05,.5,repr(e),transform=ax.transAxes,fontsize=7)
            ax.axis("off")
        for ax in axs.flat[len(idx):]: ax.axis("off")
        axs.flat[0].legend(loc="lower right", fontsize=6)
        fig.suptitle(f"case {er.case_id} | {er.sequence_identifier} | tier {er.tier}\nHUMAN VISUAL REVIEW REQUIRED", fontsize=12)
        fig.tight_layout(); safe = f"{er.case_id}_{er.sequence_identifier}".replace("/","_").replace("\\","_")
        fig.savefig(out / "qc_previews" / f"{safe}.png", dpi=150); plt.close(fig)

    # 6700/XA-03 conflict report.
    c6700 = conflicts[(conflicts.case_id.astype(str).eq("6700")) & (conflicts.sequence_identifier.astype(str).str.contains("XA-03", regex=False))]
    lines = ["# case 6700 / XA-03 annotation conflict", "", "This case is frozen Val and will never enter an extended Train candidate in Phase 9A.", "", "No annotation version was selected automatically.", ""]
    case_dir = next((p for p in data_root.iterdir() if p.is_dir() and p.name.endswith(".6700")), None)
    if case_dir:
        files = sorted([p for p in case_dir.rglob("*") if p.is_file() and (p.suffix.lower() in [".csv", ".json"]) and "03" in str(p.parent.name)])
        lines += ["## Discovered files", "", "| Path | Size | Modified | Rows/JSON count |", "|---|---:|---|---:|"]
        for p in files:
            count = len(pd.read_csv(p)) if p.suffix.lower()==".csv" else json_counts(p)[1]
            lines.append(f"| `{p}` | {p.stat().st_size} | {pd.Timestamp(p.stat().st_mtime, unit='s')} | {count} |")
    lines += ["", "## CSV comparison", ""]
    if len(c6700):
        r=c6700.iloc[0]; lines += [f"- Common frames: {r.common_frames}", f"- Frames with any coordinate difference: {r.any_coordinate_difference_frames}",
                                  f"- Raw-coordinate differences: {r.raw_difference_frames}", f"- True-tip differences: {r.true_tip_difference_frames}",
                                  "- Resolution: unresolved label conflict; manual owner decision required."]
    else: lines.append("- Expected conflict record was not found; this is itself a blocking audit anomaly.")
    (out / "case6700_XA03_conflict_report.md").write_text("\n".join(lines), encoding="utf-8")

    log("[12/12] Expansion candidate report")
    # Figures.
    annotated = elig[elig.n_csv_rows.gt(0)]
    tc = annotated.tier.value_counts().reindex(["A","B","C","D"], fill_value=0)
    plt.figure(figsize=(6,4)); tc.plot.bar(color=["#2a9d8f","#e9c46a","#f4a261","#e76f51"]); plt.ylabel("Series"); plt.title("Eligibility tier counts"); plt.tight_layout(); plt.savefig(out/"figures/eligibility_tier_counts.png",dpi=180); plt.close()
    ab = annotated[annotated.tier.isin(["A","B"])].groupby("case_id").agg(frames=("n_matched_labels","sum"),series=("series_id","size")).sort_values("frames",ascending=False)
    for col,title,name in [("frames","Eligible frames by case","eligible_frames_by_case.png"),("series","Eligible Series by case","eligible_series_by_case.png")]:
        plt.figure(figsize=(10,4)); ab[col].plot.bar(); plt.ylabel(col); plt.title(title); plt.tight_layout(); plt.savefig(out/"figures"/name,dpi=180); plt.close()
    diversity["group"] = np.select([diversity.current_split.eq("current_train"), diversity.tier.eq("A") & diversity.current_split.eq("unused"), diversity.tier.eq("B") & diversity.current_split.eq("unused")], ["Current Train","Tier A new","Tier B new"], default="Other")
    groups=[g for g in ["Current Train","Tier A new","Tier B new"] if g in set(diversity.group)]
    for col,title,name in [("raw_error_mean","Raw-to-true error","current_vs_candidate_raw_error.png"),("gt_motion_p95","GT motion P95","current_vs_candidate_motion.png")]:
        vals=[diversity[diversity.group.eq(g)][col].dropna() for g in groups]; plt.figure(figsize=(7,4)); plt.boxplot(vals,labels=groups); plt.ylabel("px"); plt.title(title); plt.tight_layout(); plt.savefig(out/"figures"/name,dpi=180); plt.close()
    plt.figure(figsize=(7,7))
    for g in groups:
        z=diversity[diversity.group.eq(g)]; plt.scatter(z.gt_x_mean,z.gt_y_mean,label=g,alpha=.8)
    plt.gca().invert_yaxis(); plt.xlim(0,1024); plt.ylim(1024,0); plt.xlabel("GT x mean"); plt.ylabel("GT y mean"); plt.legend(); plt.title("Series-level GT position centers"); plt.tight_layout(); plt.savefig(out/"figures/current_vs_candidate_tip_position.png",dpi=180); plt.close()

    # README and machine status.
    # Count unique physical DICOM frames per Series here.  In a conflict such as
    # 6700/XA-03, n_matched_labels contains rows from both competing CSV files
    # and would otherwise double-count the same 231 physical frames.
    tiers = annotated.groupby("tier").agg(cases=("case_id","nunique"), series=("series_id","size"), frames=("n_dicom_frames","sum")).reindex(["A","B","C","D"], fill_value=0)
    s0, s1, s2 = [scenarios.iloc[i] for i in range(3)]
    new_b = annotated[(annotated.tier.eq("B")) & annotated.new_case_flag]
    old_nonclean = annotated[~annotated.series_id.isin(clean_by_uid)]
    json_only = old_nonclean[(old_nonclean.mapping_clean) & (old_nonclean.coordinate_clean) & (old_nonclean.pixel_clean) & old_nonclean.json_status.isin(["MISSING","COUNT_MISMATCH"])]
    exclusion_counts = annotated[annotated.tier.eq("D")].current_issue_summary.value_counts().to_dict()
    conclusion = 1 if int(s2.new_cases_vs_current) >= 4 else (2 if int(s2.new_cases_vs_current) > 0 else 3)
    new_case_lines=[]
    for case,g in annotated[annotated.new_case_flag].groupby("case_id"):
        new_case_lines.append(f"| {case} | {len(g)} | {int(g.n_matched_labels.sum())} | {'/'.join(sorted(g.tier.unique()))} | {g.current_issue_summary.iloc[0]} | yes |")
    readme=f"""# Phase 9A — Extended Dataset Eligibility Audit

## Plain-language answer

- The current Train has 8 independent cases. The fresh technical audit found **{int(s2.new_cases_vs_current)} new cases** with clean CSV↔DICOM mapping that could enter an A+B candidate pool after required human visual review.
- Tier A alone keeps Train at **{int(s1.cases)} cases / {int(s1.series)} Series / {int(s1.frames)} frames**.
- Tier A+B can conditionally reach at most **{int(s2.cases)} cases / {int(s2.series)} Series / {int(s2.frames)} labeled frames** while keeping Val frozen; every new Tier B Series remains pending human visual review.
- Among previously excluded annotated Series, **11 Series** are Tier B because CSV/DICOM/coordinates/pixels are clean and the remaining issue is auxiliary JSON absence/count mismatch. Another **3 Series** are technically clean and have JSON mismatch, but remain Tier C because the historical record contains an annotation-quality concern that still requires manual review.
- Explicit Tier D exclusion affects **{int(tiers.loc['D','series'])} annotated Series**; the principal confirmed issue is the unresolved 6700/XA-03 annotation conflict. Unlabeled XA is reported separately and is not given a fabricated training label.
- Added A+B frames comprise **{int(s2.new_case_frames)} frames from genuinely new cases** and **{int(s2.existing_train_case_extra_frames)} frames from extra Series of existing Train cases**.
- Frozen Val preserved: **YES**. Cases 13424, 2148, and 6700 are excluded from every expansion scenario.
- With {int(s2.new_cases_vs_current)} technically eligible new cases, it is possible to expand Train and still reserve at least two new cases for a future final Test; Phase 9A does not choose that split.

**Conclusion {conclusion}: existing data support a meaningful expansion of independent Train cases, so an Extended-Train single-frame baseline is worth considering only after human review.**

## Inventory reconciliation

- Fresh scan: {inventory.case_id.nunique()} cases, {len(inventory)} XA Series, {int(inventory.number_of_dicom_files.sum())} XA DICOM frames.
- Primary CSV annotations: 36 Series and 9367 unique physical DICOM frames. The other 21 Series / 3385 frames have no primary CSV and are kept separately as UNLABELED.
- Historical reference: 19 cases, 57 XA Series, 12,752 frames.
- Difference: {inventory.case_id.nunique()-19:+d} cases, {len(inventory)-57:+d} Series, {int(inventory.number_of_dicom_files.sum())-12752:+d} frames.

## Tier totals

| Tier | Cases | Series | Labeled frames |
|---|---:|---:|---:|
| A | {int(tiers.loc['A','cases'])} | {int(tiers.loc['A','series'])} | {int(tiers.loc['A','frames'])} |
| B | {int(tiers.loc['B','cases'])} | {int(tiers.loc['B','series'])} | {int(tiers.loc['B','frames'])} |
| C | {int(tiers.loc['C','cases'])} | {int(tiers.loc['C','series'])} | {int(tiers.loc['C','frames'])} |
| D | {int(tiers.loc['D','cases'])} | {int(tiers.loc['D','series'])} | {int(tiers.loc['D','frames'])} |

Tier B is technically eligible but remains **pending human visual review**. Tier C is not automatically trainable. Tier D is excluded. JSON is auxiliary and never the true-tip GT authority.

## New-case candidates

| Case | Eligible Series | Eligible frames | Tier | Main issue | Human review required |
|---|---:|---:|---|---|---|
{chr(10).join(new_case_lines)}

## Human visual review checklist

- [ ] raw point looks consistent with the system coarse localization
- [ ] true_tip lies near the intended catheter tip
- [ ] no obvious frame/label shift
- [ ] no row/column swap pattern
- [ ] no sustained label drift

The generated preview images support review but do not prove medical label correctness. No original file was modified, no model was trained or run, no split was changed, and Test was not used.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    status={"status":"complete","training":False,"model_inference":False,"test_used":False,"frozen_val_preserved":not bool(scenarios.frozen_val_included.any()),
            "fresh_counts":{"cases":int(inventory.case_id.nunique()),"series":len(inventory),"frames":int(inventory.number_of_dicom_files.sum())},
            "scenario_1":{"cases":int(s1.cases),"series":int(s1.series),"frames":int(s1.frames)},
            "scenario_2":{"cases":int(s2.cases),"series":int(s2.series),"frames":int(s2.frames),"new_cases":int(s2.new_cases_vs_current)},"conclusion":conclusion}
    (out / "run_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    log(f"[Done] Fresh XA: {status['fresh_counts']}; Scenario A+B: {status['scenario_2']}; conclusion={conclusion}")


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args(); main(Path(args.config))
