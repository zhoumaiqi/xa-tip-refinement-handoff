from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTERNAL_ROOT = Path(r"C:\Users\Walter\Desktop\TIP")


def read_csv_compat(path: Path, **kwargs) -> pd.DataFrame:
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, encoding_errors="replace", **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def read_committed_csv(relative_path: str, **kwargs) -> pd.DataFrame:
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    return pd.read_csv(io.BytesIO(result.stdout), **kwargs)


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rewrite_external_path(path: str, external_root: Path) -> Path:
    normalized = str(path).replace("/", "\\")
    marker = "\\TIP\\"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        return external_root / suffix
    return Path(normalized)


def load_labeled_series() -> pd.DataFrame:
    mapping = read_committed_csv(
        "results/phase9a/csv_dicom_mapping.csv",
        dtype={"case_id": str, "series_id": str},
    )
    split = read_committed_csv(
        "results/phase9b/phase9b_frozen_split.csv",
        dtype={"case_id": str, "series_id": str},
    )
    split = split[split.split_group.isin(["ORIGINAL_TRAIN", "FROZEN_VAL", "EXTENDED_NEW_TRAIN"])]
    selected = split[["split_group", "case_id", "series_id", "sequence_identifier", "frame_count"]].merge(
        mapping[["case_id", "series_id", "csv_paths", "mapping_class"]],
        on=["case_id", "series_id"],
        how="left",
        validate="one_to_one",
    )
    if selected.csv_paths.isna().any() or (selected.mapping_class != "EXACT").any():
        raise RuntimeError("Selected development Series do not all have EXACT CSV/DICOM mappings")
    selected["n_dicom_frames"] = selected["frame_count"].astype(int)
    selected["series_path"] = selected["csv_paths"].map(
        lambda value: str(Path(str(value).split(";")[0]).parent)
    )
    return selected


def build_manifest(external_root: Path) -> pd.DataFrame:
    selected = load_labeled_series()
    rows: list[dict] = []
    for s in selected.itertuples(index=False):
        csv_path = Path(str(s.csv_paths).split(";")[0])
        if not csv_path.exists():
            raise FileNotFoundError(f"Label CSV not found: {csv_path}")
        labels = read_csv_compat(csv_path)
        required = ["frame_index", "raw_row", "raw_col", "true_tip_row", "true_tip_col", "dicom_file"]
        missing = [c for c in required if c not in labels.columns]
        if missing:
            raise RuntimeError(f"Missing columns {missing} in {csv_path}")
        labels = labels.copy()
        for c in required[:-1]:
            labels[c] = pd.to_numeric(labels[c], errors="raise")

        series_path = rewrite_external_path(str(s.series_path), external_root)
        dcm_files = sorted(series_path.glob("*.dcm"))
        by_name = {p.name: p for p in dcm_files}
        labels["basename"] = labels.dicom_file.astype(str).map(
            lambda x: Path(x.replace("\\", "/")).name
        )
        labels["dicom_absolute_path"] = labels.basename.map(by_name)
        if labels.dicom_absolute_path.isna().any():
            missing_names = labels.loc[labels.dicom_absolute_path.isna(), "basename"].head(3).tolist()
            raise RuntimeError(f"DICOM mapping failed for {s.case_id}/{s.series_id}: {missing_names}")
        if len(labels) != int(s.n_dicom_frames):
            raise RuntimeError(
                f"Frame count mismatch for {s.case_id}/{s.series_id}: "
                f"labels={len(labels)} inventory={s.n_dicom_frames}"
            )
        for r in labels.itertuples(index=False):
            rows.append(
                {
                    "split": "train" if s.split_group != "FROZEN_VAL" else "val",
                    "split_group": s.split_group,
                    "case_id": str(s.case_id),
                    "series_id": str(s.series_id),
                    "sequence_identifier": str(s.sequence_identifier),
                    "frame_index": int(r.frame_index),
                    "dicom_absolute_path": str(r.dicom_absolute_path),
                    "raw_col": float(r.raw_col),
                    "raw_row": float(r.raw_row),
                    "true_tip_col": float(r.true_tip_col),
                    "true_tip_row": float(r.true_tip_row),
                }
            )
    manifest = pd.DataFrame(rows).sort_values(
        ["split", "case_id", "series_id", "frame_index"]
    ).reset_index(drop=True)
    expected = {
        "train": (4113, 12, 17),
        "val": (928, 3, 4),
    }
    for split_name, (frames, cases, series) in expected.items():
        part = manifest[manifest.split.eq(split_name)]
        got = (len(part), part.case_id.nunique(), part.series_id.nunique())
        if got != (frames, cases, series):
            raise RuntimeError(f"{split_name} manifest mismatch: {got} != {(frames, cases, series)}")
    if set(manifest[manifest.split.eq("train")].case_id) & {"908", "771"}:
        raise RuntimeError("Reserved Final Test case entered training manifest")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/phase10_augmentation_final/development_manifest.csv",
    )
    args = parser.parse_args()
    manifest = build_manifest(args.external_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        old = pd.read_csv(args.output, dtype={"case_id": str, "series_id": str})
        if canonical_hash(old.to_dict(orient="records")) != canonical_hash(manifest.to_dict(orient="records")):
            raise RuntimeError(f"Refusing to overwrite changed manifest: {args.output}")
        print(f"[Manifest] Existing manifest verified: {args.output}")
    else:
        manifest.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"[Manifest] Written once: {args.output}")
    summary = (
        manifest.groupby(["split", "split_group"])
        .agg(frames=("frame_index", "size"), cases=("case_id", "nunique"), series=("series_id", "nunique"))
        .reset_index()
    )
    summary.to_csv(args.output.with_name("manifest_summary.csv"), index=False)
    (args.output.with_name("manifest_hash.txt")).write_text(
        canonical_hash(manifest.to_dict(orient="records")) + "\n", encoding="ascii"
    )
    print(summary.to_string(index=False))
    print("[Isolation] Final Test cases 908 and 771: absent from development manifest")


if __name__ == "__main__":
    main()
