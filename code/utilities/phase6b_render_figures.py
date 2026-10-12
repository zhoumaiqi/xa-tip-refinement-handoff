"""Render Phase 6B figures in a separate process without importing PyTorch."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path,dpi=150,facecolor="white")
    plt.close(fig)


def source_crop(path: str, x0: int, y0: int) -> np.ndarray:
    ds=pydicom.dcmread(path)
    pixel=np.asarray(ds.pixel_array,dtype=np.float32)
    if pixel.shape!=(1024,1024) or str(getattr(ds,"PhotometricInterpretation",""))!="MONOCHROME2":
        raise ValueError(f"Unexpected image geometry: {path}")
    low,high=float(pixel.min()),float(pixel.max())
    if high<=low: raise ValueError(f"Constant image: {path}")
    pixel=(pixel-low)/(high-low)
    image=np.zeros((256,256),dtype=np.float32)
    sx0,sy0=max(0,x0),max(0,y0)
    sx1,sy1=min(1024,x0+256),min(1024,y0+256)
    if sx1>sx0 and sy1>sy0:
        image[sy0-y0:sy1-y0,sx0-x0:sx1-x0]=pixel[sy0:sy1,sx0:sx1]
    return image


def montage(frame: pd.DataFrame, name: str, figdir: Path) -> None:
    fig,axes=plt.subplots(3,4,figsize=(15,12))
    for ax in axes.flat: ax.axis("off")
    for ax,row in zip(axes.flat,frame.head(12).itertuples(index=False)):
        image=source_crop(str(row.dicom_path),int(row.crop_x0),int(row.crop_y0))
        ax.imshow(image,cmap="gray",vmin=0,vmax=1)
        ax.scatter(float(row.raw_x)-float(row.crop_x0),float(row.raw_y)-float(row.crop_y0),
                   s=44,edgecolors="cyan",facecolors="none",linewidths=1.2)
        ax.scatter(float(row.gt_local_x),float(row.gt_local_y),
                   s=44,edgecolors="red",facecolors="none",linewidths=1.2)
        ax.scatter(float(row.pred_x_local),float(row.pred_y_local),
                   s=44,marker="x",c="lime",linewidths=1.3)
        ax.set_title(f"{row.case_id}/{row.sequence_identifier} f{row.frame_index}\n"
                     f"raw {row.raw_error_px:.1f}→model {row.model_error_px:.1f}px",fontsize=9)
        ax.set_xlim(0,255);ax.set_ylim(255,0);ax.axis("off")
    fig.suptitle(name.replace("_"," ").title()+" | raw cyan / GT red / model green",fontsize=14)
    save(fig,figdir/f"{name}.png")


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    args=p.parse_args()
    output=Path(args.output)
    figdir=output/"figures"
    figdir.mkdir(exist_ok=True)
    val=pd.read_csv(output/"val_predictions.csv",dtype={"case_id":str,"series_id":str,"sequence_identifier":str})
    log=pd.read_csv(output/"training_log.csv")
    print("[Figures] Loss, CDF, paired error, gain, and subgroup charts...",flush=True)
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(log.epoch,log.train_loss,label="Train weighted MSE")
    ax.plot(log.epoch,log.val_loss,label="Val weighted MSE")
    ax.set(xlabel="Epoch",ylabel="Weighted MSE",title="Training loss")
    ax.grid(alpha=.2);ax.legend();save(fig,figdir/"training_loss.png")

    fig,ax=plt.subplots(figsize=(8,5))
    for name,values in [("System raw",val.raw_error_px),("Train-only global correction",val.global_correction_error_px),
                        ("Local HRNet",val.model_error_px)]:
        ordered=np.sort(values.to_numpy());ax.plot(ordered,np.arange(1,len(ordered)+1)/len(ordered),label=name)
    ax.set(xlabel="Original-space localization error (px)",ylabel="Cumulative fraction",title="Val error CDF")
    ax.grid(alpha=.2);ax.legend();save(fig,figdir/"raw_vs_model_cdf.png")

    fig,ax=plt.subplots(figsize=(7,7))
    for case,g in val.groupby("case_id"):
        ax.scatter(g.raw_error_px,g.model_error_px,s=8,alpha=.45,label=case)
    limit=max(60,float(np.percentile(np.r_[val.raw_error_px,val.model_error_px],99)))
    ax.plot([0,limit],[0,limit],"k--",lw=1,label="Equal error")
    ax.set(xlim=(0,limit),ylim=(0,limit),xlabel="Raw error (px)",ylabel="Model error (px)",title="Val paired errors")
    ax.grid(alpha=.2);ax.legend();save(fig,figdir/"raw_vs_model_scatter.png")

    fig,ax=plt.subplots(figsize=(8,5))
    ax.hist(val.gain_px,bins=60,color="steelblue",edgecolor="white")
    ax.axvline(0,color="black",ls="--")
    ax.set(xlabel="Gain = raw error − model error (px)",ylabel="Val frames",title="Refinement gain")
    save(fig,figdir/"gain_histogram.png")

    fig,ax=plt.subplots(figsize=(8,5))
    for case,g in val.groupby("case_id"):
        ax.scatter(g.raw_error_px,g.gain_px,s=9,alpha=.45,label=case)
    ax.axhline(0,color="black",ls="--")
    ax.set(xlabel="Raw error (px)",ylabel="Gain (px)",title="Gain versus raw error")
    ax.grid(alpha=.2);ax.legend();save(fig,figdir/"gain_vs_raw_error.png")

    for cols,filename in [(["case_id"],"metrics_by_case.png"),
                          (["case_id","sequence_identifier"],"metrics_by_series.png")]:
        grouped=val.groupby(cols,sort=True).agg(raw_mean=("raw_error_px","mean"),
                                                 model_mean=("model_error_px","mean")).reset_index()
        labels=grouped[cols].astype(str).agg("/".join,axis=1)
        x=np.arange(len(grouped));fig,ax=plt.subplots(figsize=(max(8,len(x)*2),5))
        ax.bar(x-.2,grouped.raw_mean,width=.4,label="Raw mean")
        ax.bar(x+.2,grouped.model_mean,width=.4,label="Model mean")
        ax.set_xticks(x,labels,rotation=25,ha="right")
        ax.set(ylabel="Mean error (px)",title="Val by "+" / ".join(cols))
        ax.grid(axis="y",alpha=.2);ax.legend();save(fig,figdir/filename)

    print("[Figures] Four 256-crop montage categories...",flush=True)
    rescues=val.sort_values("gain_px",ascending=False).head(12)
    regressions=val[val.raw_error_px<10].sort_values("gain_px").head(12)
    if len(regressions)<12:
        regressions=pd.concat([regressions,val.sort_values("gain_px").head(12-len(regressions))]).drop_duplicates("dicom_path")
    good=val[(val.raw_error_px<10)&(val.model_error_px<10)].sort_values("model_error_px").head(12)
    fail=val[(val.raw_error_px>=20)&(val.model_error_px>=20)].sort_values("model_error_px",ascending=False).head(12)
    for name,frame in [("best_rescues",rescues),("worst_regressions",regressions),
                       ("both_good",good),("both_fail",fail)]:
        montage(frame,name,figdir)

    print("[Figures] Unsmoothed Val Series trajectories...",flush=True)
    for (case,series),group in val.groupby(["case_id","sequence_identifier"],sort=True):
        g=group.sort_values("frame_index")
        frames=g.frame_index.to_numpy()
        fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True)
        for ax,axis in zip(axes[:2],["x","y"]):
            ax.plot(frames,g[f"gt_{axis}"],lw=1,label="GT")
            ax.plot(frames,g[f"raw_{axis}"],lw=1,label="Raw")
            ax.plot(frames,g[f"pred_{axis}_original"],lw=1,label="Model")
            ax.set_ylabel(f"{axis} (original px)");ax.grid(alpha=.2);ax.legend(ncol=3)
        axes[2].plot(frames,g.raw_error_px,lw=1,label="Raw error")
        axes[2].plot(frames,g.model_error_px,lw=1,label="Model error")
        axes[2].set(xlabel="Frame index",ylabel="Error (px)");axes[2].grid(alpha=.2);axes[2].legend()
        fig.suptitle(f"Val trajectory {case}/{series} (unsmoothed)")
        safe="".join(c if c.isalnum() or c in "-_" else "_" for c in str(series))
        save(fig,figdir/f"trajectory_{case}_{safe}.png")
    print(f"[Figures] Saved {len(list(figdir.glob('*.png')))} PNG files",flush=True)


if __name__=="__main__":
    main()
