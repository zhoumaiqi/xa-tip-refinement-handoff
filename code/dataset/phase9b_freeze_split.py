from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def main(config_path: Path) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    p9 = ROOT / cfg["phase9a_dir"]
    manifest = pd.read_csv(ROOT / cfg["model_clean_manifest"], dtype={"case_code": str, "series_uid": str})
    elig = pd.read_csv(p9 / "series_eligibility.csv", dtype={"case_id": str, "series_id": str})
    pixel = pd.read_csv(p9 / "pixel_readability_audit.csv", dtype={"case_id": str, "series_id": str, "SOPInstanceUID": str})

    hashes = sorted(((c, hashlib.sha256((cfg["hash_prefix"] + c).encode("utf-8")).hexdigest()) for c in cfg["candidate_cases"]), key=lambda x: x[1])
    test_cases = [c for c, _ in hashes[:2]]
    new_train_cases = [c for c, _ in hashes[2:]]
    if test_cases != ["908", "771"] or new_train_cases != ["4274", "14-cbct", "2768", "658"]:
        raise RuntimeError(f"Unexpected deterministic split: test={test_cases}, train={new_train_cases}")

    rows = []
    for split_name, split_value in [("ORIGINAL_TRAIN", "train"), ("FROZEN_VAL", "val")]:
        z = manifest[manifest.split.str.lower().eq(split_value)]
        for (case, uid, seq), g in z.groupby(["case_code", "series_uid", "sequence_identifier"], sort=True):
            rows.append({"split_group": split_name, "case_id": str(case), "series_id": str(uid), "sequence_identifier": str(seq),
                         "frame_count": len(g), "source_tier": "A", "human_review": "HISTORICAL_MODEL_CLEAN"})
    candidate = elig[(elig.tier.eq("B")) & (elig.current_split.eq("unused")) & (elig.case_id.isin(cfg["candidate_cases"]))]
    for r in candidate.itertuples():
        split_name = "FINAL_TEST" if str(r.case_id) in test_cases else "EXTENDED_NEW_TRAIN"
        rows.append({"split_group": split_name, "case_id": str(r.case_id), "series_id": str(r.series_id),
                     "sequence_identifier": str(r.sequence_identifier), "frame_count": int(r.n_dicom_frames),
                     "source_tier": "B", "human_review": "PASSED_BY_USER_2026-09-24"})
    split = pd.DataFrame(rows).sort_values(["split_group", "case_id", "series_id"]).reset_index(drop=True)
    expected = {"ORIGINAL_TRAIN": (8, 12, 2784), "FROZEN_VAL": (3, 4, 928),
                "EXTENDED_NEW_TRAIN": (4, 5, 1329), "FINAL_TEST": (2, 5, 1668)}
    summary = {}
    for name, (nc, ns, nf) in expected.items():
        g = split[split.split_group.eq(name)]
        got = (g.case_id.nunique(), len(g), int(g.frame_count.sum()))
        if got != (nc, ns, nf):
            raise RuntimeError(f"Frozen split count mismatch for {name}: {got} != {(nc,ns,nf)}")
        summary[name] = {"cases": nc, "series": ns, "frames": nf, "case_ids": sorted(g.case_id.unique().tolist())}
    if set(summary["ORIGINAL_TRAIN"]["case_ids"]) & set(summary["FROZEN_VAL"]["case_ids"]):
        raise RuntimeError("Original Train/Val case overlap")

    split_records = split.to_dict(orient="records")
    split_hash = canonical_hash(split_records)
    payload = {"algorithm": "SHA256('phase9b|20260924|' + exact_case_id), ascending; first 2 FINAL_TEST, last 4 EXTENDED_NEW_TRAIN",
               "candidate_hash_order": [{"case_id": c, "sha256": h, "assignment": "FINAL_TEST" if c in test_cases else "EXTENDED_NEW_TRAIN"} for c,h in hashes],
               "summary": summary, "extended_train_total": {"cases": 12, "series": 17, "frames": 4113},
               "split_sha256": split_hash, "frozen": True, "test_used": False}
    json_path = out / "phase9b_frozen_split.json"; csv_path = out / "phase9b_frozen_split.csv"; hash_path = out / "split_hash.txt"
    if json_path.exists() or csv_path.exists() or hash_path.exists():
        if not (json_path.exists() and csv_path.exists() and hash_path.exists()):
            raise RuntimeError("Partial frozen split exists; refusing to overwrite")
        old = json.loads(json_path.read_text(encoding="utf-8"))
        old_csv = pd.read_csv(csv_path, dtype={"case_id": str, "series_id": str}).sort_values(["split_group","case_id","series_id"]).reset_index(drop=True)
        if old.get("split_sha256") != split_hash or canonical_hash(old_csv.to_dict(orient="records")) != split_hash or hash_path.read_text().strip() != split_hash:
            raise RuntimeError("Existing frozen split hash mismatch; STOP")
        print(f"[Freeze] Existing split verified: {split_hash}")
    else:
        split.to_csv(csv_path, index=False, encoding="utf-8-sig")
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        hash_path.write_text(split_hash + "\n", encoding="ascii")
        print(f"[Freeze] Split written once: {split_hash}")

    # Identity-only isolation audit. No Test DICOM pixels or GT are read.
    groups = {}
    for group_name in ["ORIGINAL_TRAIN", "EXTENDED_NEW_TRAIN", "FROZEN_VAL", "FINAL_TEST"]:
        pairs = split[split.split_group.eq(group_name)][["case_id", "series_id"]].drop_duplicates()
        z = pixel.merge(pairs, on=["case_id", "series_id"], how="inner")
        groups[group_name] = z
    audit = []
    for other in ["ORIGINAL_TRAIN", "EXTENDED_NEW_TRAIN", "FROZEN_VAL"]:
        t, o = groups["FINAL_TEST"], groups[other]
        audit.extend([
            {"check": f"FINAL_TEST_vs_{other}_case", "overlap_count": len(set(t.case_id)&set(o.case_id))},
            {"check": f"FINAL_TEST_vs_{other}_series", "overlap_count": len(set(t.series_id)&set(o.series_id))},
            {"check": f"FINAL_TEST_vs_{other}_dicom_path", "overlap_count": len(set(t.dicom_path)&set(o.dicom_path))},
            {"check": f"FINAL_TEST_vs_{other}_SOPInstanceUID", "overlap_count": len(set(t.SOPInstanceUID)&set(o.SOPInstanceUID))},
        ])
    leakage = pd.DataFrame(audit)
    leakage.to_csv(out / "leakage_audit.csv", index=False)
    if leakage.overlap_count.sum() != 0:
        raise RuntimeError("FINAL TEST leakage detected; STOP")
    test_access = {"test_cases_reserved": test_cases, "series_count": 5, "frame_count": 1668,
                   "training_reads": 0, "inference_reads": 0, "metric_reads": 0,
                   "pixel_reads": 0, "gt_reads": 0, "test_used": False,
                   "identity_metadata_used_for_leakage_audit_only": True}
    (out / "test_access_audit.json").write_text(json.dumps(test_access, indent=2), encoding="utf-8")
    attestation = """# Human review attestation\n\nThe user confirmed that all 10 Phase 9A Tier-B candidate Series were visually reviewed and passed: no systematic frame-label shift, row/column swap, sustained drift, or wrong-structure labeling was found. Rare shadow/occlusion frames retain minor label uncertainty within a reasonable manual annotation range. No frame was deleted, relabeled, or down-weighted.\n"""
    (out / "human_review_attestation.md").write_text(attestation, encoding="utf-8")
    print("[Isolation] FINAL TEST overlaps: 0 cases / 0 Series / 0 paths / 0 SOP UIDs")
    print(f"[Split] Extended Train: 12 cases / 17 Series / 4113 frames; Final Test: {test_cases} RESERVED / NOT USED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True, type=Path)
    a = ap.parse_args(); main((ROOT / a.config).resolve() if not a.config.is_absolute() else a.config)
