from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

root = Path.cwd()
out = root / "outputs/phase7b_causal_3frame_hrnet"
fig = out / "figures"
fig.mkdir(exist_ok=True)

def bar_from_csv(filename, value, output, ylabel):
    d = pd.read_csv(out / filename)
    d.pivot(index="raw_bin", columns="method", values=value).plot(kind="bar", figsize=(10,5))
    plt.ylabel(ylabel); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(fig/output,dpi=180); plt.close()

bar_from_csv("oof_easy_case_safety.csv","mean","easy_case_safety.png","Mean error (px)")
bar_from_csv("oof_hard_case_rescue.csv","mean_gain_px","hard_case_rescue.png","Mean gain vs raw (px)")

p = pd.read_csv(out/"oof_persistent_harmful.csv").set_index("method")
p[["total_harmful_frames","persistent_ge3_frames","segments_ge5","segments_ge10"]].plot(kind="bar",figsize=(10,5))
plt.ylabel("Count");plt.xticks(rotation=0);plt.tight_layout();plt.savefig(fig/"persistent_harmful_comparison.png",dpi=180);plt.close()

c = pd.read_csv(out/"confidence_comparison.csv")
q=c[(c["group"]=="overall") & (c["metric"].isin(["peak","margin"]))].pivot_table(index=["split","metric"],columns="model",values="mean",aggfunc="first")
q.plot(kind="bar",figsize=(10,5));plt.ylabel("Mean value");plt.xticks(rotation=20);plt.tight_layout();plt.savefig(fig/"confidence_comparison.png",dpi=180);plt.close()

logs=[]
for path in (out/"folds").glob("fold_*_*_*/training_log.csv"):
    d=pd.read_csv(path); d["mode"]="TEMPORAL" if "temporal" in str(path.parent) else "CONTROL";logs.append(d)
if logs:
    d=pd.concat(logs,ignore_index=True)
    for mode,g in d.groupby("mode"):
        z=g.groupby("epoch")["train_loss"].agg(["mean","std"])
        plt.plot(z.index,z["mean"],marker="o",label=mode)
        plt.fill_between(z.index,z["mean"]-z["std"],z["mean"]+z["std"],alpha=.15)
    plt.xlabel("Epoch");plt.ylabel("Train loss across OOF folds");plt.legend();plt.tight_layout();plt.savefig(fig/"paired_training_curves.png",dpi=180);plt.close()

v=pd.read_csv(out/"val_by_case.csv")
v=v[v.method.isin(["Raw","Historical Phase6G alpha=.75","CONTROL alpha=.75","TEMPORAL alpha=.75"])]
v.pivot(index="case_id",columns="method",values="mean_px").plot(kind="bar",figsize=(11,5))
plt.ylabel("Mean error (px)");plt.xticks(rotation=0);plt.tight_layout();plt.savefig(fig/"val_case_comparison.png",dpi=180);plt.close()

print("[Figures] Phase 7B required quantitative figures finalized.")
