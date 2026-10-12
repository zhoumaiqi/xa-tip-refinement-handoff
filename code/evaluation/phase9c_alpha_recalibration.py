from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "phase9c_alpha_recalibration.json"
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
OUT = ROOT / CFG["output_dir"]
OOF_DIR = OUT / "oof"
VAL_DIR = OUT / "val"
FIG_DIR = OUT / "figures"
for directory in (OUT, OOF_DIR, VAL_DIR, FIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

ALPHAS = [float(x) for x in CFG["alpha_grid"]]
HIST_ALPHA = float(CFG["historical_alpha"])
TIE_TOL = float(CFG["tie_tolerance_px"])
KEYS = ["case_id", "series_id", "frame_index", "base_seed"]
REQUIRED = KEYS + ["raw_x", "raw_y", "gt_x", "gt_y", "candidate_x", "candidate_y", "final_x", "final_y"]


def banner() -> None:
    print("=" * 52)
    print("Phase 9C — ALPHA RECALIBRATION AUDIT")
    print("=" * 52)
    print("Training: NONE")
    print("Model inference: NONE")
    print("Models: EXISTING PREDICTIONS ONLY")
    print("Alpha grid: 0 / 0.25 / 0.50 / 0.75 / 1.00")
    print("Primary: Case-macro Mean")
    print("Nested case audit: YES")
    print("Frozen Val: ONLY AFTER OOF ALPHA FREEZE")
    print("Final Test: SEALED")
    print("-" * 52)


def as_str_ids(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["case_id"] = d["case_id"].astype(str)
    d["series_id"] = d["series_id"].astype(str)
    d["frame_index"] = pd.to_numeric(d["frame_index"], errors="raise").astype(int)
    d["base_seed"] = pd.to_numeric(d["base_seed"], errors="raise").astype(int)
    return d


def load_predictions(path: Path, method: str, split: str, expected_rows: int) -> pd.DataFrame:
    d = as_str_ids(pd.read_csv(path))
    missing = [c for c in REQUIRED if c not in d.columns]
    if missing:
        raise RuntimeError(f"{method} {split}: missing columns {missing}")
    if len(d) != expected_rows:
        raise RuntimeError(f"{method} {split}: {len(d)} rows, expected {expected_rows}")
    if d[KEYS].duplicated().any():
        raise RuntimeError(f"{method} {split}: duplicate matched keys")
    numeric = ["raw_x", "raw_y", "gt_x", "gt_y", "candidate_x", "candidate_y", "final_x", "final_y"]
    if not np.isfinite(d[numeric].to_numpy(float)).all():
        raise RuntimeError(f"{method} {split}: non-finite coordinates")
    d["method"] = method
    d["split"] = split
    return d


def candidate_audit(d: pd.DataFrame, source: Path) -> dict:
    rebuilt_x = d.raw_x + HIST_ALPHA * (d.candidate_x - d.raw_x)
    rebuilt_y = d.raw_y + HIST_ALPHA * (d.candidate_y - d.raw_y)
    discrepancy = np.hypot(rebuilt_x - d.final_x, rebuilt_y - d.final_y)
    return {
        "split": d.split.iloc[0],
        "method": d.method.iloc[0],
        "source": str(source.relative_to(ROOT)),
        "n": len(d),
        "candidate_x_present": True,
        "candidate_y_present": True,
        "candidate_reconstruction_needed": False,
        "alpha075_roundtrip_max_difference_px": float(discrepancy.max()),
        "alpha075_roundtrip_mean_difference_px": float(discrepancy.mean()),
        "roundtrip_pass_lt_1e6": bool(discrepancy.max() < 1e-6),
        "status": "DIRECT_CANDIDATE_AVAILABLE" if discrepancy.max() < 1e-6 else "FAIL",
    }


def assert_matched(a: pd.DataFrame, b: pd.DataFrame, split: str) -> None:
    ka = a[KEYS].sort_values(KEYS).reset_index(drop=True)
    kb = b[KEYS].sort_values(KEYS).reset_index(drop=True)
    if not ka.equals(kb):
        raise RuntimeError(f"{split}: BASELINE and EXTENDED frame keys are not identical")
    for c in ["raw_x", "raw_y", "gt_x", "gt_y"]:
        aa = a.sort_values(KEYS)[c].to_numpy(float)
        bb = b.sort_values(KEYS)[c].to_numpy(float)
        if not np.allclose(aa, bb, atol=1e-9, rtol=0):
            raise RuntimeError(f"{split}: matched {c} differs")


def alpha_frame(d: pd.DataFrame, alpha: float) -> pd.DataFrame:
    z = d.copy()
    z["alpha"] = float(alpha)
    z["pred_x"] = z.raw_x + alpha * (z.candidate_x - z.raw_x)
    z["pred_y"] = z.raw_y + alpha * (z.candidate_y - z.raw_y)
    z["error_px"] = np.hypot(z.pred_x - z.gt_x, z.pred_y - z.gt_y)
    z["raw_error_px_calc"] = np.hypot(z.raw_x - z.gt_x, z.raw_y - z.gt_y)
    z["gain_px_calc"] = z.raw_error_px_calc - z.error_px
    return z


def metric_row(z: pd.DataFrame, **extra) -> dict:
    e = z.error_px.to_numpy(float)
    gain = z.gain_px_calc.to_numpy(float)
    case_means = z.groupby("case_id").error_px.mean()
    return {
        **extra,
        "n": len(z),
        "n_cases": int(z.case_id.nunique()),
        "case_macro_mean": float(case_means.mean()),
        "frame_mean": float(e.mean()),
        "median": float(np.median(e)),
        "p95": float(np.percentile(e, 95)),
        "p99": float(np.percentile(e, 99)),
        "max": float(e.max()),
        "lt5": float(np.mean(e < 5)),
        "lt10": float(np.mean(e < 10)),
        "lt20": float(np.mean(e < 20)),
        "lt40": float(np.mean(e < 40)),
        "harmful_gt5": float(np.mean(gain < -5)),
        "harmful_gt10": float(np.mean(gain < -10)),
        "harmful_gt20": float(np.mean(gain < -20)),
        "improved_pct": float(np.mean(gain > 0)),
        "worsened_pct": float(np.mean(gain < 0)),
        "mean_gain": float(gain.mean()),
    }


def easy_rows(z: pd.DataFrame, **extra) -> list[dict]:
    out = []
    for name, mask in [("raw_lt5", z.raw_error_px_calc < 5), ("raw_lt10", z.raw_error_px_calc < 10)]:
        g = z.loc[mask]
        gain = g.gain_px_calc.to_numpy(float)
        out.append({**extra, "bin": name, "n": len(g), "mean_error": float(g.error_px.mean()),
                    "median_error": float(g.error_px.median()), "harmful_gt5": float(np.mean(gain < -5)),
                    "harmful_gt10": float(np.mean(gain < -10))})
    return out


def hard_rows(z: pd.DataFrame, **extra) -> list[dict]:
    out = []
    for name, lo, hi in [("raw_20_40", 20, 40), ("raw_40_60", 40, 60)]:
        g = z.loc[(z.raw_error_px_calc >= lo) & (z.raw_error_px_calc < hi)]
        gain = g.gain_px_calc.to_numpy(float)
        out.append({**extra, "bin": name, "n": len(g), "mean_error": float(g.error_px.mean()),
                    "median_error": float(g.error_px.median()), "mean_gain": float(gain.mean()),
                    "median_gain": float(np.median(gain)), "improved_pct": float(np.mean(gain > 0))})
    return out


def persistent_row(z: pd.DataFrame, **extra) -> dict:
    harmful_total = 0
    persistent_frames = 0
    segments = []
    for _, g in z.sort_values(["case_id", "series_id", "frame_index"]).groupby(["case_id", "series_id"], sort=False):
        frames = g.frame_index.to_numpy(int)
        bad = (g.gain_px_calc.to_numpy(float) < -10)
        harmful_total += int(bad.sum())
        run = 0
        prev = None
        for frame, flag in zip(frames, bad):
            if flag and (prev is None or frame == prev + 1):
                run += 1
            elif flag:
                if run:
                    segments.append(run)
                run = 1
            else:
                if run:
                    segments.append(run)
                run = 0
            prev = frame
        if run:
            segments.append(run)
    persistent_frames = int(sum(s for s in segments if s >= 3))
    return {**extra, "total_harmful_frames": harmful_total, "persistent_frames_ge3": persistent_frames,
            "segments_ge3": int(sum(s >= 3 for s in segments)), "segments_ge5": int(sum(s >= 5 for s in segments)),
            "segments_ge10": int(sum(s >= 10 for s in segments)), "max_segment": int(max(segments, default=0))}


def choose_alpha(scores: dict[float, float]) -> float:
    best = min(scores.values())
    eligible = [a for a in ALPHAS if scores[a] <= best + TIE_TOL]
    return float(min(eligible))


def evaluate_oof(method_frames: dict[str, pd.DataFrame]):
    grid_seed, grid_case, easy, hard, persist = [], [], [], [], []
    expanded: dict[tuple[str, int, float], pd.DataFrame] = {}
    for method, d in method_frames.items():
        for seed, sd in d.groupby("base_seed"):
            for alpha in ALPHAS:
                z = alpha_frame(sd, alpha)
                expanded[(method, int(seed), alpha)] = z
                grid_seed.append(metric_row(z, split="OOF", method=method, base_seed=int(seed), alpha=alpha))
                for case_id, cg in z.groupby("case_id"):
                    grid_case.append(metric_row(cg, split="OOF", method=method, base_seed=int(seed), case_id=case_id, alpha=alpha))
                easy.extend(easy_rows(z, split="OOF", method=method, base_seed=int(seed), alpha=alpha))
                hard.extend(hard_rows(z, split="OOF", method=method, base_seed=int(seed), alpha=alpha))
                persist.append(persistent_row(z, split="OOF", method=method, base_seed=int(seed), alpha=alpha))
    return expanded, pd.DataFrame(grid_seed), pd.DataFrame(grid_case), pd.DataFrame(easy), pd.DataFrame(hard), pd.DataFrame(persist)


def nested_audit(method_frames: dict[str, pd.DataFrame]):
    choices, performance, nested_frames = [], [], []
    for method, d in method_frames.items():
        for seed, sd in d.groupby("base_seed"):
            cases = sorted(sd.case_id.unique())
            for outer in cases:
                train = sd.loc[sd.case_id != outer]
                scores = {}
                for alpha in ALPHAS:
                    z = alpha_frame(train, alpha)
                    scores[alpha] = float(z.groupby("case_id").error_px.mean().mean())
                selected = choose_alpha(scores)
                outer_df = sd.loc[sd.case_id == outer]
                fixed_z = alpha_frame(outer_df, HIST_ALPHA)
                selected_z = alpha_frame(outer_df, selected)
                selected_z["selected_alpha"] = selected
                nested_frames.append(selected_z)
                fixed_m = metric_row(fixed_z)
                selected_m = metric_row(selected_z)
                choices.append({"method": method, "base_seed": int(seed), "outer_case": outer,
                                "selected_alpha": selected, "selection_best_score": min(scores.values()),
                                **{f"score_alpha_{a:g}": scores[a] for a in ALPHAS}})
                delta = selected_m["frame_mean"] - fixed_m["frame_mean"]
                performance.append({"method": method, "base_seed": int(seed), "outer_case": outer,
                                    "selected_alpha": selected, "n": len(outer_df),
                                    "fixed075_mean": fixed_m["frame_mean"], "selected_mean": selected_m["frame_mean"],
                                    "delta_selected_minus_fixed075": delta,
                                    "outcome": "improved" if delta < -TIE_TOL else ("worsened" if delta > TIE_TOL else "tie"),
                                    "fixed075_p95": fixed_m["p95"], "selected_p95": selected_m["p95"],
                                    "fixed075_harmful_gt10": fixed_m["harmful_gt10"],
                                    "selected_harmful_gt10": selected_m["harmful_gt10"]})
    return pd.DataFrame(choices), pd.DataFrame(performance), pd.concat(nested_frames, ignore_index=True)


def landscapes(grid_case: pd.DataFrame, grid_seed: pd.DataFrame):
    case_rows = []
    for (method, case_id), g in grid_case.groupby(["method", "case_id"]):
        scores = {a: float(g.loc[g.alpha.eq(a), "frame_mean"].mean()) for a in ALPHAS}
        case_rows.append({"method": method, "case_id": case_id, "best_alpha": choose_alpha(scores),
                          **{f"mean_error_alpha_{a:g}": scores[a] for a in ALPHAS}})
    seed_rows = []
    for (method, seed), g in grid_seed.groupby(["method", "base_seed"]):
        scores = {a: float(g.loc[g.alpha.eq(a), "case_macro_mean"].iloc[0]) for a in ALPHAS}
        seed_rows.append({"method": method, "base_seed": int(seed), "best_alpha": choose_alpha(scores),
                          **{f"case_macro_alpha_{a:g}": scores[a] for a in ALPHAS}})
    return pd.DataFrame(case_rows), pd.DataFrame(seed_rows)


def global_selection(grid_seed: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    selections = {}
    rows = []
    for method, g in grid_seed.groupby("method"):
        scores = {a: float(g.loc[g.alpha.eq(a), "case_macro_mean"].mean()) for a in ALPHAS}
        selected = choose_alpha(scores)
        selections[method] = selected
        for a in ALPHAS:
            ag = g.loc[g.alpha.eq(a)]
            rows.append({"method": method, "alpha": a,
                         "case_macro_mean_5seed": scores[a],
                         "case_macro_sd": float(ag.case_macro_mean.std(ddof=1)),
                         "frame_mean_5seed": float(ag.frame_mean.mean()), "median_5seed": float(ag["median"].mean()),
                         "p95_5seed": float(ag.p95.mean()), "p99_5seed": float(ag.p99.mean()),
                         "harmful_gt10_5seed": float(ag.harmful_gt10.mean()),
                         "mean_gain_5seed": float(ag.mean_gain.mean()), "is_selected": a == selected})
    return selections, pd.DataFrame(rows)


def nested_summary(perf: pd.DataFrame, nested_frames: pd.DataFrame, method: str) -> dict:
    p = perf.loc[perf.method.eq(method)]
    z = nested_frames.loc[nested_frames.method.eq(method)]
    fixed_parts = []
    source = z.drop(columns=[c for c in ["pred_x", "pred_y", "error_px", "raw_error_px_calc", "gain_px_calc", "alpha"] if c in z.columns])
    for _, g in source.groupby(["base_seed", "case_id"]):
        fixed_parts.append(alpha_frame(g, HIST_ALPHA))
    f = pd.concat(fixed_parts, ignore_index=True)
    sm = metric_row(z)
    fm = metric_row(f)
    return {"method": method, "n_outer_evaluations": len(p),
            "selected_case_macro_mean": float(p.selected_mean.mean()),
            "fixed075_case_macro_mean": float(p.fixed075_mean.mean()),
            "mean_delta_selected_minus_fixed075": float(p.delta_selected_minus_fixed075.mean()),
            "outer_improved": int((p.outcome == "improved").sum()), "outer_worsened": int((p.outcome == "worsened").sum()),
            "outer_tie": int((p.outcome == "tie").sum()), "selected_frame_mean": sm["frame_mean"],
            "fixed075_frame_mean": fm["frame_mean"], "selected_p95": sm["p95"], "fixed075_p95": fm["p95"],
            "selected_harmful_gt10": sm["harmful_gt10"], "fixed075_harmful_gt10": fm["harmful_gt10"]}


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def plot_oof(global_table, choices, case_land, easy, hard):
    for method, filename in [("BASELINE_8", "oof_alpha_curve_baseline.png"), ("EXTENDED_12", "oof_alpha_curve_extended.png")]:
        g = global_table.loc[global_table.method.eq(method)]
        plt.figure(figsize=(7, 5)); plt.plot(g.alpha, g.case_macro_mean_5seed, marker="o", label="Case-macro Mean")
        plt.plot(g.alpha, g.p95_5seed, marker="s", label="P95"); plt.xlabel("alpha"); plt.ylabel("px")
        plt.title(f"{method} OOF alpha curve"); plt.legend(); plt.grid(alpha=.25); savefig(filename)
    counts = choices.groupby(["method", "selected_alpha"]).size().unstack(fill_value=0).reindex(columns=ALPHAS, fill_value=0)
    counts.T.plot(kind="bar", figsize=(8, 5)); plt.xlabel("Nested-selected alpha"); plt.ylabel("Count / 40")
    plt.title("Nested alpha choice distribution"); savefig("alpha_choice_distribution.png")
    pivot = case_land.pivot(index="case_id", columns="method", values="best_alpha")
    plt.figure(figsize=(8, 5)); plt.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=20); plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="Best alpha"); plt.title("Case alpha landscape (5-seed mean)"); savefig("case_alpha_landscape.png")
    e = easy.groupby(["method", "alpha", "bin"], as_index=False).harmful_gt10.mean()
    for (m, b), g in e.groupby(["method", "bin"]): plt.plot(g.alpha, g.harmful_gt10, marker="o", label=f"{m} {b}")
    plt.xlabel("alpha"); plt.ylabel("Harmful >10 fraction"); plt.title("OOF easy-case safety vs alpha"); plt.legend(fontsize=8); plt.grid(alpha=.25); savefig("easy_safety_vs_alpha.png")
    h = hard.groupby(["method", "alpha", "bin"], as_index=False).mean_gain.mean()
    for (m, b), g in h.groupby(["method", "bin"]): plt.plot(g.alpha, g.mean_gain, marker="o", label=f"{m} {b}")
    plt.axhline(0, color="black", lw=1); plt.xlabel("alpha"); plt.ylabel("Mean gain vs raw (px)"); plt.title("OOF hard-case rescue vs alpha"); plt.legend(fontsize=8); plt.grid(alpha=.25); savefig("hard_rescue_vs_alpha.png")


def evaluate_val(method_frames: dict[str, pd.DataFrame], selections: dict[str, float]):
    metrics, by_case, easy, hard, persist = [], [], [], [], []
    selected_labels = {}
    for method, d in method_frames.items():
        chosen = selections[method]
        for seed, sd in d.groupby("base_seed"):
            for alpha in ALPHAS:
                z = alpha_frame(sd, alpha)
                metrics.append(metric_row(z, split="Val", method=method, base_seed=int(seed), alpha=alpha,
                                          role="selected" if alpha == chosen else ("historical" if alpha == HIST_ALPHA else "grid")))
            for role, alpha in [("historical075", HIST_ALPHA), ("oof_selected", chosen)]:
                z = alpha_frame(sd, alpha)
                selected_labels[(method, int(seed), role)] = z
                for case_id, cg in z.groupby("case_id"):
                    by_case.append(metric_row(cg, split="Val", method=method, base_seed=int(seed), role=role,
                                              alpha=alpha, case_id=case_id))
                easy.extend(easy_rows(z, split="Val", method=method, base_seed=int(seed), role=role, alpha=alpha))
                hard.extend(hard_rows(z, split="Val", method=method, base_seed=int(seed), role=role, alpha=alpha))
                persist.append(persistent_row(z, split="Val", method=method, base_seed=int(seed), role=role, alpha=alpha))
    return pd.DataFrame(metrics), pd.DataFrame(by_case), pd.DataFrame(easy), pd.DataFrame(hard), pd.DataFrame(persist)


def pct(x):
    return f"{100*x:.2f}%"


def main() -> None:
    banner()
    (OUT / "config.json").write_text(json.dumps(CFG, indent=2), encoding="utf-8")
    test_audit = {"reserved_test_cases": CFG["reserved_test_cases"], "pixel_reads": 0, "gt_reads": 0,
                  "training_reads": 0, "inference_reads": 0, "metric_reads": 0, "test_used": False,
                  "note": "Phase9C used only saved Phase7D/Phase9B prediction CSVs."}
    (OUT / "test_access_audit.json").write_text(json.dumps(test_audit, indent=2), encoding="utf-8")

    print("[1/8] Loading candidate coordinates")
    bo_path = ROOT / CFG["baseline_oof_predictions"]
    eo_path = ROOT / CFG["extended_oof_predictions"]
    bo = load_predictions(bo_path, "BASELINE_8", "OOF", CFG["expected_oof_rows_per_method"])
    eo = load_predictions(eo_path, "EXTENDED_12", "OOF", CFG["expected_oof_rows_per_method"])
    audits = [candidate_audit(bo, bo_path), candidate_audit(eo, eo_path)]
    if any(x["status"] == "FAIL" for x in audits):
        pd.DataFrame(audits).to_csv(OUT / "candidate_availability_audit.csv", index=False)
        raise RuntimeError("Candidate alpha=.75 roundtrip failed; inference is forbidden, stopping")

    print("[2/8] Verifying matched OOF frames")
    assert_matched(bo, eo, "OOF")
    expected_cases = set(CFG["old_cases"])
    if set(bo.case_id.unique()) != expected_cases or bo.case_id.nunique() != 8:
        raise RuntimeError("OOF cases do not match the frozen 8 old cases")
    if any(c in set(bo.case_id) for c in CFG["reserved_test_cases"]):
        raise RuntimeError("Final Test case leaked into OOF")

    print("[3/8] Evaluating alpha grid")
    method_oof = {"BASELINE_8": bo, "EXTENDED_12": eo}
    expanded, grid_seed, grid_case, easy, hard, persist = evaluate_oof(method_oof)
    grid_seed.loc[grid_seed.method.eq("BASELINE_8")].to_csv(OOF_DIR / "alpha_grid_metrics_baseline.csv", index=False)
    grid_seed.loc[grid_seed.method.eq("EXTENDED_12")].to_csv(OOF_DIR / "alpha_grid_metrics_extended.csv", index=False)
    grid_seed.to_csv(OOF_DIR / "alpha_metrics_by_seed.csv", index=False)
    grid_case.to_csv(OOF_DIR / "alpha_metrics_by_case_seed.csv", index=False)
    easy.to_csv(OOF_DIR / "easy_safety_by_alpha.csv", index=False)
    hard.to_csv(OOF_DIR / "hard_rescue_by_alpha.csv", index=False)
    persist.to_csv(OOF_DIR / "persistent_harmful_by_alpha.csv", index=False)

    print("[4/8] Nested case-level alpha audit")
    choices, performance, nested_frames = nested_audit(method_oof)
    choices.to_csv(OOF_DIR / "nested_alpha_choices.csv", index=False)
    performance.to_csv(OOF_DIR / "nested_alpha_performance.csv", index=False)
    case_land, seed_land = landscapes(grid_case, grid_seed)
    case_land.to_csv(OOF_DIR / "case_alpha_landscape.csv", index=False)
    seed_land.to_csv(OOF_DIR / "seed_alpha_landscape.csv", index=False)
    nested_summaries = {m: nested_summary(performance, nested_frames, m) for m in method_oof}

    print("[5/8] Freezing global alpha")
    selections, global_table = global_selection(grid_seed)
    plot_oof(global_table, choices, case_land, easy, hard)
    ext_fixed = global_table[(global_table.method == "EXTENDED_12") & (global_table.alpha == HIST_ALPHA)].iloc[0]
    ext_sel = global_table[(global_table.method == "EXTENDED_12") & (global_table.alpha == selections["EXTENDED_12"])].iloc[0]
    ns = nested_summaries["EXTENDED_12"]
    global_better = ext_sel.case_macro_mean_5seed < ext_fixed.case_macro_mean_5seed - TIE_TOL
    nested_better = ns["selected_case_macro_mean"] < ns["fixed075_case_macro_mean"] - TIE_TOL
    safety_ok = ext_sel.harmful_gt10_5seed <= ext_fixed.harmful_gt10_5seed + 1e-12
    if global_better and nested_better and safety_ok:
        oof_class = "A_RECALIBRATION_USEFUL"
    elif abs(ext_sel.case_macro_mean_5seed - ext_fixed.case_macro_mean_5seed) <= TIE_TOL or safety_ok:
        oof_class = "B_TRADEOFF_ONLY"
    else:
        oof_class = "C_NO_USEFUL_RECALIBRATION"
    decision = {
        "frozen_before_val": True,
        "alpha_grid": ALPHAS,
        "tie_tolerance_px": TIE_TOL,
        "baseline_global_alpha": selections["BASELINE_8"],
        "extended_global_alpha": selections["EXTENDED_12"],
        "nested_choice_distribution": {m: {str(a): int(((choices.method == m) & (choices.selected_alpha == a)).sum()) for a in ALPHAS} for m in method_oof},
        "nested_summary": nested_summaries,
        "global_alpha_table": global_table.to_dict("records"),
        "extended_oof_decision": oof_class,
        "test_used": False,
    }
    (OOF_DIR / "oof_alpha_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"      Frozen OOF alpha: BASELINE={selections['BASELINE_8']}, EXTENDED={selections['EXTENDED_12']} | {oof_class}")

    print("[6/8] Recomputing frozen Val")
    bv_path = ROOT / CFG["baseline_val_predictions"]
    ev_path = ROOT / CFG["extended_val_predictions"]
    bv = load_predictions(bv_path, "BASELINE_8", "Val", CFG["expected_val_rows_per_method"])
    ev = load_predictions(ev_path, "EXTENDED_12", "Val", CFG["expected_val_rows_per_method"])
    audits.extend([candidate_audit(bv, bv_path), candidate_audit(ev, ev_path)])
    pd.DataFrame(audits).to_csv(OUT / "candidate_availability_audit.csv", index=False)
    if any(x["status"] == "FAIL" for x in audits):
        raise RuntimeError("Val candidate roundtrip failed")
    assert_matched(bv, ev, "Val")
    if any(c in set(bv.case_id) for c in CFG["reserved_test_cases"]):
        raise RuntimeError("Final Test case leaked into Val")
    vm, vc, ve, vh, vp = evaluate_val({"BASELINE_8": bv, "EXTENDED_12": ev}, selections)
    vm.to_csv(VAL_DIR / "val_metrics_by_alpha.csv", index=False)
    vc.to_csv(VAL_DIR / "val_by_case.csv", index=False)
    ve.to_csv(VAL_DIR / "val_easy_safety.csv", index=False)
    vh.to_csv(VAL_DIR / "val_hard_rescue.csv", index=False)
    vp.to_csv(VAL_DIR / "val_persistent_harmful.csv", index=False)

    print("[7/8] Safety / rescue analysis")
    plot_rows = []
    for method in ["BASELINE_8", "EXTENDED_12"]:
        for role, alpha in [("historical075", HIST_ALPHA), ("oof_selected", selections[method])]:
            g = vm[(vm.method == method) & (vm.alpha == alpha)]
            plot_rows.append({"method": method, "role": role, "mean": g.frame_mean.mean(), "p95": g.p95.mean(), "harmful_gt10": g.harmful_gt10.mean()})
    pv = pd.DataFrame(plot_rows)
    pv.assign(label=pv.method + "\n" + pv.role).plot(x="label", y=["mean", "p95"], kind="bar", figsize=(9, 5))
    plt.ylabel("px"); plt.title("Frozen Val alpha comparison (5-seed mean)"); plt.xticks(rotation=20); savefig("val_alpha_comparison.png")

    def val_summary(method, alpha):
        g = vm[(vm.method == method) & (vm.alpha == alpha)]
        return {"alpha": alpha, "mean": float(g.frame_mean.mean()), "mean_sd": float(g.frame_mean.std(ddof=1)),
                "median": float(g["median"].mean()), "p95": float(g.p95.mean()), "p99": float(g.p99.mean()),
                "harmful_gt5": float(g.harmful_gt5.mean()), "harmful_gt10": float(g.harmful_gt10.mean()),
                "harmful_gt20": float(g.harmful_gt20.mean())}
    final_decision = {"oof_decision": oof_class, "baseline_selected_alpha": selections["BASELINE_8"],
                      "extended_selected_alpha": selections["EXTENDED_12"],
                      "baseline_val_historical075": val_summary("BASELINE_8", HIST_ALPHA),
                      "baseline_val_selected": val_summary("BASELINE_8", selections["BASELINE_8"]),
                      "extended_val_historical075": val_summary("EXTENDED_12", HIST_ALPHA),
                      "extended_val_selected": val_summary("EXTENDED_12", selections["EXTENDED_12"]),
                      "test_used": False}

    ext_global_improvement = float(ext_fixed.case_macro_mean_5seed - ext_sel.case_macro_mean_5seed)
    heterogeneity = int(case_land[(case_land.method == "EXTENDED_12")].best_alpha.nunique())
    ext_v075 = final_decision["extended_val_historical075"]
    ext_vsel = final_decision["extended_val_selected"]
    val_mean_improves = ext_vsel["mean"] < ext_v075["mean"] - TIE_TOL
    if global_better and nested_better and ext_global_improvement >= 0.25 and val_mean_improves:
        final_choice = 1
    elif selections["EXTENDED_12"] == HIST_ALPHA and abs(ext_global_improvement) <= TIE_TOL:
        final_choice = 4
    elif not nested_better and heterogeneity >= 3:
        final_choice = 3
    else:
        final_choice = 2
    final_decision["final_choice"] = final_choice
    final_decision["interpretation"] = {1: "Recalibration materially improves the extended model.",
                                         2: "Recalibration changes the safety/rescue trade-off but does not solve generalization.",
                                         3: "Case alpha preferences are heterogeneous; one fixed alpha is limiting.",
                                         4: "Alpha 0.75 remains near-optimal; correction strength is not the main issue."}[final_choice]
    (VAL_DIR / "final_decision.json").write_text(json.dumps(final_decision, indent=2), encoding="utf-8")

    print("[8/8] Final report")
    bc = choices[choices.method == "BASELINE_8"].selected_alpha.value_counts().reindex(ALPHAS, fill_value=0)
    ec = choices[choices.method == "EXTENDED_12"].selected_alpha.value_counts().reindex(ALPHAS, fill_value=0)
    ext_v075 = final_decision["extended_val_historical075"]
    ext_vsel = final_decision["extended_val_selected"]
    lines = [
        "# Phase 9C — Fixed Refinement-Strength Recalibration Audit", "",
        "## Plain-language result", "",
        f"- BASELINE-8 OOF-selected global alpha: **{selections['BASELINE_8']:.2f}** (historical anchor 0.75).",
        f"- EXTENDED-12 OOF-selected global alpha: **{selections['EXTENDED_12']:.2f}**.",
        f"- EXTENDED OOF case-macro Mean: alpha .75 = **{ext_fixed.case_macro_mean_5seed:.3f}px**; selected alpha = **{ext_sel.case_macro_mean_5seed:.3f}px**.",
        f"- Strict nested EXTENDED: selected = **{ns['selected_case_macro_mean']:.3f}px** vs fixed .75 = **{ns['fixed075_case_macro_mean']:.3f}px**; outcomes {ns['outer_improved']} improved / {ns['outer_worsened']} worsened / {ns['outer_tie']} ties out of 40.",
        f"- Frozen Val EXTENDED: alpha .75 Mean = **{ext_v075['mean']:.3f}px**; selected alpha Mean = **{ext_vsel['mean']:.3f}px**.",
        f"- Final choice: **{final_choice}** — {final_decision['interpretation']}",
        "- No training, model inference, checkpoint loading, or Final Test access occurred.", "",
        "## Nested alpha choices (40 outer case-seed evaluations per method)", "",
        "| alpha | BASELINE-8 | EXTENDED-12 |", "|---:|---:|---:|",
    ]
    lines += [f"| {a:.2f} | {int(bc[a])} | {int(ec[a])} |" for a in ALPHAS]
    lines += ["", "## OOF global alpha landscape", "",
              "The selection metric is the five-seed mean of the eight-case macro Mean; ties within 0.05px choose the smaller alpha.", "",
              "| Method | alpha | Case-macro Mean | P95 | Harmful >10 |", "|---|---:|---:|---:|---:|"]
    for _, r in global_table.iterrows():
        lines.append(f"| {r.method} | {r.alpha:.2f} | {r.case_macro_mean_5seed:.3f} | {r.p95_5seed:.3f} | {pct(r.harmful_gt10_5seed)} |")
    lines += ["", "## Interpretation limits", "",
              "Alpha recalibration is deterministic post-processing of saved raw/candidate coordinates. It cannot establish that data expansion improved the visual model itself. Nested selection is an audit of cross-case stability, not a case-adaptive deployment rule. Final Test cases 908 and 771 remain sealed.", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "run_status.json").write_text(json.dumps({"status": "complete", "training": False, "inference": False,
                                                      "baseline_selected_alpha": selections["BASELINE_8"],
                                                      "extended_selected_alpha": selections["EXTENDED_12"],
                                                      "final_choice": final_choice, "test_used": False}, indent=2), encoding="utf-8")
    print(f"Done. BASELINE alpha={selections['BASELINE_8']}; EXTENDED alpha={selections['EXTENDED_12']}; final choice={final_choice}; Test NOT USED")


if __name__ == "__main__":
    main()
