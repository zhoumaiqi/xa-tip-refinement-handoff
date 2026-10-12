from pathlib import Path
import importlib.machinery, importlib.util, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase8b_explicit_temporal_difference"
S2 = OUT / "stage2"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)


def load_source(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def mean_sd(df, fields):
    return {f: {"mean": float(df[f].mean()), "sd": float(df[f].std(ddof=1))} for f in fields}


metrics = pd.read_csv(S2 / "metrics_by_seed.csv").query("alpha == 0.75")
val_metrics = pd.read_csv(S2 / "val_metrics_by_seed.csv").query("alpha == 0.75")
easy = pd.read_csv(S2 / "easy_safety_by_seed.csv")
val_easy = pd.read_csv(S2 / "val_easy_safety_by_seed.csv")
hard = pd.read_csv(S2 / "hard_rescue_by_seed.csv")
val_hard = pd.read_csv(S2 / "val_hard_rescue_by_seed.csv")
pers = pd.read_csv(S2 / "persistent_by_seed.csv")
val_pers = pd.read_csv(S2 / "val_persistent_by_seed.csv")
deltas = pd.read_csv(S2 / "temporal_delta_by_seed.csv")
val_deltas = pd.read_csv(S2 / "val_delta_by_seed.csv")
decision = json.loads((S2 / "final_decision.json").read_text(encoding="utf-8"))
oof_decision = json.loads((S2 / "oof_decision.json").read_text(encoding="utf-8"))

summary = {
    "diff_scale": json.loads((OUT / "stage0/difference_scale.json").read_text())["diff_scale"],
    "stage1": json.loads((OUT / "stage1/stage1_decision.json").read_text()),
    "oof": {a: mean_sd(metrics[metrics.arm.eq(a)], ["case_macro_mean", "mean", "median", "p95", "harmful_gt10"]) for a in ["F", "G"]},
    "val": {a: mean_sd(val_metrics[val_metrics.arm.eq(a)], ["case_macro_mean", "mean", "median", "p95", "harmful_gt10"]) for a in ["F", "G"]},
    "oof_G_wins": int((deltas.delta_G_minus_F < 0).sum()),
    "val_G_wins": int((val_deltas.delta_G_minus_F < 0).sum()),
    "oof_mean_G_minus_F": float(deltas.delta_G_minus_F.mean()),
    "val_mean_G_minus_F": float(val_deltas.delta_G_minus_F.mean()),
    "oof_win_val_loss": int(((deltas.delta_G_minus_F < 0) & (val_deltas.delta_G_minus_F > 0)).sum()),
    "decision": decision,
    "test_used": False,
}
(OUT / "final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def grouped_plot(df, value, title, ylabel, path):
    g = df.groupby(["bin", "arm"])[value].agg(["mean", "std"]).reset_index()
    bins = list(dict.fromkeys(g.bin))
    x = np.arange(len(bins)); width = .34
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, arm in enumerate(["F", "G"]):
        q = g[g.arm.eq(arm)].set_index("bin").reindex(bins)
        ax.bar(x + (j-.5)*width, q["mean"], width, yerr=q["std"], capsize=3, label=arm)
    ax.set_xticks(x, bins); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


grouped_plot(pd.concat([easy.assign(domain="OOF")]), "harmful_gt10", "OOF easy-case harmful >10", "fraction", FIG / "easy_safety.png")
grouped_plot(pd.concat([hard.assign(domain="OOF")]), "mean_gain", "OOF hard-case rescue", "mean gain (px)", FIG / "hard_rescue.png")

g = pers.groupby("arm")[["total_harmful_frames", "persistent_frames_ge3"]].agg(["mean", "std"])
fig, ax = plt.subplots(figsize=(6, 4.5)); x = np.arange(2); width=.34
for j, arm in enumerate(["F", "G"]):
    vals = [g.loc[arm, (c, "mean")] for c in ["total_harmful_frames", "persistent_frames_ge3"]]
    err = [g.loc[arm, (c, "std")] for c in ["total_harmful_frames", "persistent_frames_ge3"]]
    ax.bar(x+(j-.5)*width, vals, width, yerr=err, capsize=3, label=arm)
ax.set_xticks(x, ["harmful frames", "persistent >=3 frames"]); ax.set_ylabel("count / seed"); ax.set_title("OOF persistent harmful behavior"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "persistent_harmful.png", dpi=180); plt.close(fig)


def make_examples(kind, out_name):
    seed = 20260918
    f = pd.read_csv(S2 / "val_F_all_seeds.csv").query("base_seed == @seed").copy()
    g = pd.read_csv(S2 / "val_G_all_seeds.csv").query("base_seed == @seed").copy()
    keys = ["case_id", "series_id", "frame_index"]
    for d in (f, g):
        d["case_id"] = d.case_id.astype(str); d["series_id"] = d.series_id.astype(str); d["frame_index"] = d.frame_index.astype(int)
    z = f.merge(g, on=keys, suffixes=("_F", "_G")); z["G_gain"] = z.final_error_px_F-z.final_error_px_G
    z = z.nlargest(6, "G_gain") if kind == "rescue" else z.nsmallest(6, "G_gain")
    p7d = load_source("phase7d_for_8b_final", ROOT / "scripts/phase7d_repeat_seed_stability.src")
    cfg = json.loads((OUT / "config.json").read_text())
    d = pd.read_csv(ROOT / cfg["manifest"]); d["case_code"] = d.case_code.astype(str)
    w = p7d.P7B.build_windows(d); wv = w[w.split.eq("val")].reset_index(drop=True)
    base = p7d.P7B.WindowDataset(wv, str(ROOT / cfg["phase7b_dir"] / "cache/val_stacks.npy"), "temporal", cfg["sigma"])
    lookup = {(str(r.case_id), str(r.series_id), int(r.frame_index)): i for i, r in base.w.iterrows()}
    fig, axs = plt.subplots(len(z), 3, figsize=(10, 2.7*len(z)), squeeze=False)
    for row, r in enumerate(z.itertuples()):
        i = lookup[(str(r.case_id), str(r.series_id), int(r.frame_index))]; x = np.asarray(base.x[i], dtype=np.float32)
        current, d1, d2 = x[2], x[2]-x[1], x[1]-x[0]
        for col, (im, title, cmap, lo, hi) in enumerate([(current, "current", "gray", 0, 1), (d1, "D1 signed", "coolwarm", -.5, .5), (d2, "D2 signed", "coolwarm", -.5, .5)]):
            ax=axs[row,col]; ax.imshow(im,cmap=cmap,vmin=lo,vmax=hi); ax.set_title(title); ax.axis("off")
        ax=axs[row,0]; x0=float(r.crop_x0_F); y0=float(r.crop_y0_F)
        ax.plot(float(r.gt_x_F)-x0,float(r.gt_y_F)-y0,"r+",ms=8,mew=1.3,label="GT")
        ax.plot(float(r.final_x_F)-x0,float(r.final_y_F)-y0,"bo",ms=4,label="F")
        ax.plot(float(r.final_x_G)-x0,float(r.final_y_G)-y0,"go",ms=4,label="G")
        ax.legend(loc="lower right",fontsize=6); ax.text(2,12,f"case {r.case_id} frame {r.frame_index}\nF {r.final_error_px_F:.1f} G {r.final_error_px_G:.1f} gain {r.G_gain:.1f}",color="yellow",fontsize=7,bbox=dict(facecolor="black",alpha=.5,pad=1))
    fig.suptitle(f"Difference {kind} examples (seed {seed})", y=.999); fig.tight_layout(); fig.savefig(FIG/out_name,dpi=170); plt.close(fig)


make_examples("rescue", "difference_rescue_examples.png")
make_examples("failure", "difference_failure_examples.png")

stage1 = summary["stage1"]
oof_F, oof_G = summary["oof"]["F"], summary["oof"]["G"]
val_F, val_G = summary["val"]["F"], summary["val"]["G"]
readme = f"""# Phase 8B — Explicit Temporal-Difference Input

## Plain-language result

Giving the model signed change maps helped in some Train-case OOF runs, but it was not reliable: G won **{summary['oof_G_wins']}/5** OOF seeds and was on average **{summary['oof_mean_G_minus_F']:+.3f} px** relative to F (positive means worse). On the independent matched Val cases, G won **{summary['val_G_wins']}/5** seeds and was on average **{summary['val_mean_G_minus_F']:+.3f} px worse**. Therefore the frozen conclusion is **2 — explicit differences have partial benefit but remain cross-case unstable**.

## Frozen design and audit

- Train-only `DIFF_SCALE` (pooled |D1|/|D2| P99.5): **{summary['diff_scale']:.8f}**.
- F: `[current, 0.5, 0.5]`; G: `[current, signed D1, signed D2]` encoded around 0.5.
- Shared current `raw_t` crop, 256×256; weighted heatmap MSE; 3 epochs; fixed alpha=.75.
- Stage 0 clipping stayed below the prespecified warning threshold.
- Paired initialization hashes matched for every F/G run. No NaN, collapse, or coordinate bug was observed.
- OOF was frozen before Val. **Test was not used.**

## Stage 1

- PASS: F/G case-macro Mean **{stage1['F_case_macro_mean']:.3f}/{stage1['G_case_macro_mean']:.3f} px**; G wins/ties/losses **{stage1['wins']}/{stage1['ties']}/{stage1['losses']}**.
- Harmful >10 decreased from **{stage1['F_harmful_gt10']*100:.2f}%** to **{stage1['G_harmful_gt10']*100:.2f}%**.

## Five-seed OOF (mean ± SD)

- Case-macro Mean: F **{oof_F['case_macro_mean']['mean']:.3f} ± {oof_F['case_macro_mean']['sd']:.3f}**, G **{oof_G['case_macro_mean']['mean']:.3f} ± {oof_G['case_macro_mean']['sd']:.3f} px**.
- Frame Mean: F **{oof_F['mean']['mean']:.3f}**, G **{oof_G['mean']['mean']:.3f} px**; P95: F **{oof_F['p95']['mean']:.3f}**, G **{oof_G['p95']['mean']:.3f} px**.
- Harmful >10: F **{oof_F['harmful_gt10']['mean']*100:.2f}%**, G **{oof_G['harmful_gt10']['mean']*100:.2f}%**.
- OOF wins were 3/5, but seed 20260922 had a large G regression; the case-macro average therefore did not improve.

## Matched Val, 920 frames (mean ± SD over 5 seeds)

- Mean: F **{val_F['mean']['mean']:.3f} ± {val_F['mean']['sd']:.3f}**, G **{val_G['mean']['mean']:.3f} ± {val_G['mean']['sd']:.3f} px**.
- Median: F **{val_F['median']['mean']:.3f}**, G **{val_G['median']['mean']:.3f} px**.
- P95: F **{val_F['p95']['mean']:.3f}**, G **{val_G['p95']['mean']:.3f} px**.
- Harmful >10: F **{val_F['harmful_gt10']['mean']*100:.2f}%**, G **{val_G['harmful_gt10']['mean']*100:.2f}%**.
- G won 0/5 seeds; OOF-win/Val-loss reversal occurred in **{summary['oof_win_val_loss']}** seeds.

## Interpretation

OOF easy-case safety improved and OOF P95 decreased, but hard-case rescue was mixed and independent-Val easy safety, hard rescue, and persistent failures worsened. A frame difference is not catheter motion: it can contain patient/C-arm/background/contrast/noise changes. The evidence supports only that this representation sometimes helps; it does not support adopting it as the temporal baseline.

No Phase 8C, longer window, gate, alpha tuning, new normalization, augmentation, or Test was run.
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")
(OUT / "terminal_summary.txt").write_text(readme, encoding="utf-8")
print("Phase8B final reporting complete")
