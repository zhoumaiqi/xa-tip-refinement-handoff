from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\Walter\Documents\Codex\2026-09-18\files-pasted-by-the-user-xa")
OUT = ROOT / "outputs" / "phase6g_oof_shrinkage_refinement"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

oof = pd.read_csv(OUT / "oof_predictions.csv")
am = pd.read_csv(OUT / "oof_alpha_metrics.csv")
abc = pd.read_csv(OUT / "oof_alpha_by_case.csv")
abr = pd.read_csv(OUT / "oof_alpha_by_raw_bin.csv")
val = pd.read_csv(OUT / "val_predictions.csv")
vm = pd.read_csv(OUT / "val_metrics.csv")
vbc = pd.read_csv(OUT / "val_by_case.csv")
frozen = json.loads((OUT / "frozen_design.json").read_text(encoding="utf-8"))
folds = pd.read_csv(OUT / "oof_fold_summary.csv")

# Correct semantic separation: overall comparison vs case-stratified comparison.
vm.to_csv(OUT / "val_comparison.csv", index=False)
vbc.to_csv(OUT / "val_by_case.csv", index=False)

plt.style.use("seaborn-v0_8-whitegrid")

fig, ax = plt.subplots(2, 2, figsize=(11, 8), dpi=160)
ax = ax.ravel()
ax[0].plot(am.alpha, am.case_macro_mean_error_px, "o-", lw=2)
ax[0].axvline(frozen["alpha_frozen"], color="tab:red", ls="--", label="frozen alpha")
ax[0].set(xlabel="alpha", ylabel="Case-macro mean error (px)", title="Primary OOF selection metric")
ax[0].legend()
ax[1].plot(am.alpha, am.mean_px, "o-", label="Mean")
ax[1].plot(am.alpha, am.median_px, "o-", label="Median")
ax[1].plot(am.alpha, am.p95_px, "o-", label="P95")
ax[1].set(xlabel="alpha", ylabel="Error (px)", title="OOF frame-weighted errors"); ax[1].legend()
ax[2].plot(am.alpha, am.harmful_gt_10px_pct, "o-", color="tab:red")
ax[2].set(xlabel="alpha", ylabel="Harmful >10 (%)", title="OOF safety")
ax[3].plot(am.alpha, am.improved_pct, "o-", label="Improved")
ax[3].plot(am.alpha, am.worsened_pct, "o-", label="Worsened")
ax[3].set(xlabel="alpha", ylabel="Frames (%)", title="Paired outcomes vs raw"); ax[3].legend()
fig.tight_layout(); fig.savefig(FIG / "oof_alpha_curve.png"); plt.close(fig)

p = abc.pivot(index="case_id", columns="alpha", values="mean_px")
fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
p.plot(kind="bar", ax=ax, width=.85)
ax.set(ylabel="OOF mean error (px)", title="OOF alpha stability by held-out case")
ax.legend(title="alpha", ncol=5); fig.tight_layout(); fig.savefig(FIG / "oof_alpha_by_case.png"); plt.close(fig)

methods = ["System raw", "Train-only global mean residual", "Phase6B full alpha=1", "Phase6F residual", "Phase6G frozen shrinkage"]
errcols = ["raw_error_px", "global_correction_error_px", "phase6b_error_px", "phase6f_error_px", "phase6g_error_px"]
fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
for name, col in zip(methods, errcols):
    x = np.sort(val[col].to_numpy()); y = np.arange(1, len(x)+1)/len(x)
    ax.plot(x, y, label=name)
ax.set(xlabel="Localization error (px)", ylabel="Empirical CDF", xlim=(0, 70), title="One-shot Val error CDF")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "val_error_cdf.png"); plt.close(fig)

vp = vbc[vbc.method.isin(["System raw", "Phase6B", "Phase6F", "Phase6G"])].pivot(index="case_id", columns="method", values="mean_px")
fig, ax = plt.subplots(figsize=(10, 6), dpi=160); vp.plot(kind="bar", ax=ax)
ax.set(ylabel="Mean error (px)", title="Val case comparison"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "val_case_comparison.png"); plt.close(fig)

easy = pd.read_csv(OUT / "easy_case_safety.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=160)
for a, b in zip(axes, ["0-5", "5-10"]):
    s = easy[easy.raw_bin == b]
    a.bar(s.method, s.mean_px); a.set(title=f"Val raw error {b}px", ylabel="Mean output error (px)"); a.tick_params(axis="x", rotation=25)
fig.tight_layout(); fig.savefig(FIG / "easy_case_safety.png"); plt.close(fig)

hard = pd.read_csv(OUT / "hard_case_rescue.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=160)
for a, b in zip(axes, ["20-40", "40-60"]):
    s = hard[hard.raw_bin == b]
    a.axhline(float(s.raw_mean_px.iloc[0]), color="black", ls="--", label="Raw")
    a.bar(s.method, s.mean_px); a.set(title=f"Val raw error {b}px", ylabel="Mean output error (px)"); a.tick_params(axis="x", rotation=25); a.legend()
fig.tight_layout(); fig.savefig(FIG / "hard_case_rescue.png"); plt.close(fig)

for (case, series), s in val.groupby(["case_id", "series_id"], sort=False):
    s = s.sort_values("frame_index")
    fig, ax = plt.subplots(figsize=(12, 4), dpi=140)
    for lab, col in [("Raw", "raw_error_px"), ("6B", "phase6b_error_px"), ("6F", "phase6f_error_px"), ("6G", "phase6g_error_px")]:
        ax.plot(s.frame_index, s[col], label=lab, lw=1.2)
    ax.set(xlabel="Frame", ylabel="Error (px)", title=f"Case {case} | {s.sequence_identifier.iloc[0]}")
    ax.legend(ncol=4); fig.tight_layout()
    safe_series = str(s.sequence_identifier.iloc[0]).replace("/", "_")
    fig.savefig(FIG / f"trajectory_{case}_{safe_series}.png"); plt.close(fig)

best_by_case = abc.loc[abc.groupby("case_id").mean_px.idxmin(), ["case_id", "alpha", "mean_px"]].sort_values("case_id")
best_counts = best_by_case.alpha.value_counts().sort_index().to_dict()
sel = am.loc[am.alpha == frozen["alpha_frozen"]].iloc[0]
full = am.loc[am.alpha == 1.0].iloc[0]

def row(method):
    return vm.loc[vm.method == method].iloc[0]
raw, b6, f6, g6 = [row(x) for x in ["System raw", "Phase6B full alpha=1", "Phase6F residual", "Phase6G frozen shrinkage"]]

readme = f"""# Phase 6G — Case-OOF Conservative Heatmap Shrinkage

## Status and frozen design

Completed. The prescribed 8-case leave-one-case-out experiment used Phase 6B's raw-centered 256×256 HRNet heatmap model, fresh ImageNet initialization per fold, weighted MSE, and no augmentation/jitter/noise/temporal input. Every OOF prediction came from a model that did not see that case. No Test data were used.

The alpha grid was fixed in advance: `0, 0.25, 0.50, 0.75, 1.00`. **Alpha was selected only by the 8-case OOF case-macro mean error before final Val inference.** The frozen alpha is **{frozen['alpha_frozen']:.2f}**. Final full-Train duration was frozen at the rounded median fold-best epoch: **{frozen['final_training_epochs']} epochs**. Val was run once and was not used to revise alpha or training duration.

## OOF selection

| alpha | case-macro mean px | frame mean px | median px | P95 px | harmful >10 % |
|---:|---:|---:|---:|---:|---:|
"""
for _, r in am.iterrows():
    readme += f"| {r.alpha:.2f} | {r.case_macro_mean_error_px:.3f} | {r.mean_px:.3f} | {r.median_px:.3f} | {r.p95_px:.3f} | {r.harmful_gt_10px_pct:.2f} |\n"
readme += f"""

Alpha 0.75 minimized the sole primary criterion (case-macro mean error: {sel.case_macro_mean_error_px:.3f}px versus {full.case_macro_mean_error_px:.3f}px for full correction). It also reduced OOF harmful >10px corrections from {full.harmful_gt_10px_pct:.2f}% to {sel.harmful_gt_10px_pct:.2f}%, while retaining most hard-case correction capacity. Case-specific optima were heterogeneous: `{best_counts}` (see `oof_alpha_by_case.csv`), so the global shrinkage is a useful compromise, not a universal case optimum.

## One-shot Val result

| method | Mean px | Median px | P95 px |
|---|---:|---:|---:|
| Raw | {raw.mean_px:.3f} | {raw.median_px:.3f} | {raw.p95_px:.3f} |
| Phase 6B full correction | {b6.mean_px:.3f} | {b6.median_px:.3f} | {b6.p95_px:.3f} |
| Phase 6F residual regression | {f6.mean_px:.3f} | {f6.median_px:.3f} | {f6.p95_px:.3f} |
| Phase 6G alpha={frozen['alpha_frozen']:.2f} | {g6.mean_px:.3f} | {g6.median_px:.3f} | {g6.p95_px:.3f} |

Phase 6G achieved the lowest overall Val mean ({g6.mean_px:.3f}px), essentially tied with 6B on mean but with a lower P95 ({g6.p95_px:.3f} vs {b6.p95_px:.3f}px) and substantially fewer harmful >10px corrections ({g6.harmful_gt_10px_pct:.2f}% vs {b6.harmful_gt_10px_pct:.2f}%). It does not make easy cases fully safe: for raw-error 0–5px, 6G still worsened most frames and remains materially less safe than 6F.

## Interpretation

The evidence supports outcome **2**: fixed shrinkage helps the safety/rescue tradeoff, but held-out cases prefer materially different alpha values. Phase 6G is worth retaining as a stronger fixed-fusion baseline, while any future adaptive fusion must be calibrated with case-level OOF predictions. No adaptive gate, temporal model, structural supervision, 384 ROI, or Test evaluation was run here.

## Run/recovery audit

The original reporting process terminated after the one-shot Val predictions were safely saved because an Intel OpenMP runtime was initialized twice while starting plotting. Reporting was recovered strictly from already saved OOF and one-shot Val CSV files; training, alpha selection, and Val inference were not rerun. An earlier fold-2 inference indexing failure was repaired by reindexing the held-out `Subset` locally; its saved best fold checkpoint was used without retraining, and case isolation was unchanged. Both incidents are preserved in `recovery_incident_initial.json` and `run_status.json`.

## Key artifacts

- `oof_predictions.csv`: all 2,784 case-OOF predictions.
- `oof_alpha_metrics.csv`, `oof_alpha_by_case.csv`, `oof_alpha_by_raw_bin.csv`: selection, stability, safety, and rescue evidence.
- `frozen_design.json`: pre-Val alpha and epoch freeze record.
- `val_predictions.csv`, `val_comparison.csv`, `val_by_case.csv`: one-shot Val results.
- `persistent_failure_summary.csv`: harmful >10px temporal-run description (analysis only; no temporal inference).
- `figures/`: alpha curves, case comparison, CDF, safety/rescue plots, and per-Series trajectories.
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")

status = {
    "status": "complete_after_report_recovery",
    "alpha_frozen_before_val": frozen["alpha_frozen"],
    "final_epochs_frozen_before_val": frozen["final_training_epochs"],
    "oof_frames": int(len(oof)), "val_frames_one_shot": int(len(val)),
    "val_rerun_during_recovery": False,
    "test_used": False,
    "notes": ["Fold-2 saved checkpoint inference recovered with local Subset sample reindexing; no retraining.", "Plot/report process recovered from saved results after duplicate OpenMP runtime termination."],
}
(OUT / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
print("[Recovered] Reports and figures generated from frozen saved predictions.")
print(f"[Done] alpha={frozen['alpha_frozen']:.2f}; Val 6G mean={g6.mean_px:.3f}, median={g6.median_px:.3f}, P95={g6.p95_px:.3f}")
