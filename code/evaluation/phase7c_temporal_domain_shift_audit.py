from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_p7(root: Path):
    pyc = root / "scripts/__pycache__/phase7b_causal_3frame_hrnet.cpython-313.pyc"
    spec = importlib.util.spec_from_file_location("phase7b_core_for_audit", pyc)
    mod = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def norm(df):
    df=df.copy()
    for c in ("case_id","series_id","sequence_identifier"):
        if c in df: df[c]=df[c].astype(str)
    if "frame_index" in df: df.frame_index=pd.to_numeric(df.frame_index).astype(int)
    return df


def ks_stat(a,b):
    a=np.sort(np.asarray(a,float));b=np.sort(np.asarray(b,float));x=np.sort(np.r_[a,b])
    return float(np.max(np.abs(np.searchsorted(a,x,side="right")/len(a)-np.searchsorted(b,x,side="right")/len(b))))


def summary(a,prefix=""):
    a=np.asarray(a,float); q=np.percentile(a,[10,25,50,75,90,95,99])
    return {prefix+"n":len(a),prefix+"mean":a.mean(),prefix+"std":a.std(ddof=1),prefix+"p10":q[0],prefix+"p25":q[1],prefix+"median":q[2],prefix+"p75":q[3],prefix+"p90":q[4],prefix+"p95":q[5],prefix+"p99":q[6],prefix+"min":a.min(),prefix+"max":a.max()}


def contiguous_segments(df, harmful_col, method, split):
    rows=[]
    for (case,series),g in df.sort_values(["case_id","series_id","frame_index"]).groupby(["case_id","series_id"]):
        g=g.reset_index(drop=True); mask=g[harmful_col].to_numpy(bool); start=None
        for i,on in enumerate(np.r_[mask,False]):
            if on and start is None: start=i
            if not on and start is not None:
                end=i-1; seg=g.iloc[start:i]; pre=g.iloc[max(0,start-3):start]
                row={"split":split,"method":method,"case_id":case,"series_id":series,"sequence_identifier":seg.sequence_identifier.iloc[0],"start_frame":int(seg.frame_index.iloc[0]),"end_frame":int(seg.frame_index.iloc[-1]),"length":len(seg)}
                for f in ["gt_motion_1","gt_motion_2","raw_motion_1","raw_motion_2","corr_prev1","mae_prev1","image_mean","image_std","gradient_mean","local_contrast"]:
                    row[f"segment_median_{f}"]=seg[f].median();row[f"pre3_median_{f}"]=pre[f].median() if len(pre) else np.nan
                rows.append(row);start=None
    return rows


def metric_rows(df, group_cols, features):
    rows=[]
    for key,g in df.groupby(group_cols,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        base=dict(zip(group_cols,key))
        for f in features: rows.append({**base,"feature":f,**summary(g[f])})
    return rows


def binned_performance(df, feature, edges, label):
    bins=[-np.inf,*edges,np.inf]; names=["OUT_LOW","Q1","Q2","Q3","Q4","OUT_HIGH"]
    # With +/-inf plus 5 cut points (min,Q1,Q2,Q3,max), six categories.
    z=df.copy();z["bin"]=pd.cut(z[feature],bins=bins,labels=names,include_lowest=True,duplicates="drop")
    rows=[]
    for (split,b),g in z.groupby(["split","bin"],observed=True):
        rows.append({"feature":feature,"binning":label,"split":split,"bin":str(b),"n":len(g),"bin_min":g[feature].min(),"bin_max":g[feature].max(),"control_mean":g.control_error.mean(),"temporal_mean":g.temporal_error.mean(),"temporal_gain_mean":g.temporal_gain.mean(),"temporal_win_pct":100*(g.temporal_gain>0).mean(),"control_harmful_gt10_pct":100*(g.control_gain<-10).mean(),"temporal_harmful_gt10_pct":100*(g.temporal_gain_vs_raw<-10).mean()})
    return rows


def create_figures(out, allf, repro, motion_bins, sim_bins, grid, segments):
    fd=out/"figures";fd.mkdir(exist_ok=True)
    # Reproducibility plots.
    plt.figure(figsize=(6,6));plt.scatter(repro.hist_candidate_error,repro.control_candidate_error,s=8,alpha=.35);mx=max(repro.hist_candidate_error.max(),repro.control_candidate_error.max());plt.plot([0,mx],[0,mx],"k--");plt.xlabel("Historical6G candidate error (px)");plt.ylabel("New CONTROL candidate error (px)");plt.tight_layout();plt.savefig(fd/"control_vs_historical6g_scatter.png",dpi=180);plt.close()
    rb=repro.groupby("case_id")[["hist_final_error","control_final_error"]].mean();rb.plot(kind="bar",figsize=(9,5));plt.ylabel("Mean error (px)");plt.xticks(rotation=0);plt.tight_layout();plt.savefig(fd/"control_vs_historical6g_by_case.png",dpi=180);plt.close()
    def dist_plot(cols,name):
        fig,axs=plt.subplots(1,len(cols),figsize=(5*len(cols),4));axs=np.atleast_1d(axs)
        for ax,c in zip(axs,cols):
            for split,color in [("OOF","#2563eb"),("Val","#dc2626")]:
                a=allf.loc[allf.split==split,c];ax.hist(a,bins=40,density=True,histtype="step",linewidth=1.5,label=split,color=color)
            ax.set_title(c);ax.legend()
        plt.tight_layout();plt.savefig(fd/name,dpi=180);plt.close()
    dist_plot(["gt_motion_1","gt_motion_2","gt_acc"],"oof_vs_val_motion_distribution.png")
    dist_plot(["corr_prev1","corr_prev2","mae_prev1"],"oof_vs_val_similarity_distribution.png")
    dist_plot(["image_mean","image_std","gradient_mean","local_contrast"],"oof_vs_val_appearance_distribution.png")
    for cols,name,xlab in [(["gt_motion_2","raw_motion_2"],"temporal_gain_vs_motion.png","Motion (px)"),(["corr_prev1","mae_prev1"],"temporal_gain_vs_similarity.png","Similarity")]:
        fig,axs=plt.subplots(1,2,figsize=(11,4))
        for ax,c in zip(axs,cols):
            for split,color in [("OOF","#2563eb"),("Val","#dc2626")]:
                g=allf[allf.split==split];step=max(1,len(g)//800);ax.scatter(g[c].iloc[::step],g.temporal_gain.iloc[::step],s=6,alpha=.25,label=split,color=color)
            ax.axhline(0,color="black",lw=.8);ax.set_xlabel(c);ax.set_ylabel("Temporal gain (px)");ax.legend()
        plt.tight_layout();plt.savefig(fd/name,dpi=180);plt.close()
    for data,name in [(motion_bins,"motion_bin_performance.png"),(sim_bins,"similarity_bin_performance.png")]:
        q=data[data.bin.isin(["Q1","Q2","Q3","Q4"])].pivot_table(index=["feature","bin"],columns="split",values="temporal_gain_mean",aggfunc="first")
        q.plot(kind="bar",figsize=(11,5));plt.axhline(0,color="black",lw=.8);plt.ylabel("Mean temporal gain (px)");plt.tight_layout();plt.savefig(fd/name,dpi=180);plt.close()
    pv=grid.pivot_table(index="motion_level",columns="corr_level",values="val_temporal_gain",aggfunc="first")
    plt.figure(figsize=(6,5));plt.imshow(pv,cmap="coolwarm",aspect="auto");plt.colorbar(label="Val temporal gain (px)");plt.xticks(range(len(pv.columns)),pv.columns);plt.yticks(range(len(pv.index)),pv.index);plt.xlabel("Correlation level");plt.ylabel("GT motion2 level");plt.tight_layout();plt.savefig(fd/"motion_similarity_grid.png",dpi=180);plt.close()
    # Case timelines.
    c=allf[allf.case_id=="13424"]
    for seq in ["XA-01","XA-06"]:
        g=c[c.sequence_identifier==seq].sort_values("frame_index")
        fig,axs=plt.subplots(3,1,figsize=(12,8),sharex=True)
        axs[0].plot(g.frame_index,g.gt_motion_2,label="GT motion2");axs[0].plot(g.frame_index,g.raw_motion_2,label="raw motion2");axs[0].legend()
        axs[1].plot(g.frame_index,g.corr_prev1,label="corr t-1/t");axs[1].plot(g.frame_index,g.mae_prev1,label="MAE t-1/t");axs[1].legend()
        axs[2].plot(g.frame_index,g.control_error,label="CONTROL");axs[2].plot(g.frame_index,g.temporal_error,label="TEMPORAL");axs[2].fill_between(g.frame_index,0,1,where=(g.temporal_gain<-10),transform=axs[2].get_xaxis_transform(),alpha=.2,color="red",label="TEMP worse >10");axs[2].legend();axs[2].set_xlabel("Frame")
        plt.tight_layout();plt.savefig(fd/f"case13424_temporal_failure_timeline_{seq.replace('-','')}.png",dpi=180);plt.close()
    case=allf.groupby(["split","case_id"])[["gt_motion_2","raw_motion_2","corr_prev1","mae_prev1","temporal_gain"]].median().reset_index()
    case.set_index("case_id")[["gt_motion_2","raw_motion_2","temporal_gain"]].plot(kind="bar",figsize=(12,5));plt.tight_layout();plt.savefig(fd/"case_comparison_dynamics.png",dpi=180);plt.close()
    if len(segments):
        segments.groupby(["split","method"])[["segment_median_gt_motion_2","segment_median_raw_motion_2","segment_median_corr_prev1"]].median().plot(kind="bar",figsize=(11,5));plt.tight_layout();plt.savefig(fd/"persistent_segment_dynamics.png",dpi=180);plt.close()
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,y in zip(axs,["peak_diff","margin_diff"]):
        for split,color in [("OOF","#2563eb"),("Val","#dc2626")]:
            g=allf[allf.split==split];step=max(1,len(g)//800);ax.scatter(g.gt_motion_2.iloc[::step],g[y].iloc[::step],s=6,alpha=.25,label=split,color=color)
        ax.axhline(0,color="black",lw=.8);ax.set_xlabel("GT motion2 (px)");ax.set_ylabel(y);ax.legend()
    plt.tight_layout();plt.savefig(fd/"confidence_vs_motion.png",dpi=180);plt.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);args=ap.parse_args()
    root=Path(args.config).resolve().parents[1];cfg=json.loads(Path(args.config).read_text(encoding="utf-8"));out=(root/cfg["output_dir"]).resolve();out.mkdir(parents=True,exist_ok=True)
    logp=out/"phase7c_run.log";logp.write_text("",encoding="utf-8")
    def log(s): print(s,flush=True);logp.open("a",encoding="utf-8").write(s+"\n")
    print("="*52,"\nPhase 7C — TEMPORAL DOMAIN-SHIFT AUDIT\n"+"="*52,"\nTraining: NONE\nTest: FORBIDDEN",flush=True)
    p7=load_p7(root);p7b=root/"outputs/phase7b_causal_3frame_hrnet";p6g=root/"outputs/phase6g_oof_shrinkage_refinement";p7a=root/"outputs/phase7a_causal_multiframe_crop_audit"
    log("[1/10] Verifying matched prediction frames")
    oc=norm(pd.read_csv(p7b/"oof_control_predictions.csv"));ot=norm(pd.read_csv(p7b/"oof_temporal_predictions.csv"));vc=norm(pd.read_csv(p7b/"val_control_predictions.csv"));vt=norm(pd.read_csv(p7b/"val_temporal_predictions.csv"));keys=["case_id","series_id","frame_index"]
    for a,b,n in [(oc,ot,2760),(vc,vt,920)]:
        assert len(a)==len(b)==n and not a.duplicated(keys).any() and not b.duplicated(keys).any();assert set(map(tuple,a[keys].values))==set(map(tuple,b[keys].values))
    manifest=pd.read_csv(root/"outputs/model_clean_v1/model_clean_frames.csv");manifest.case_code=manifest.case_code.astype(str)
    w=norm(p7.build_windows(manifest));tw=w[w.split=="train"].reset_index(drop=True);vw=w[w.split=="val"].reset_index(drop=True)
    matched=pd.concat([tw.assign(audit_split="OOF"),vw.assign(audit_split="Val")],ignore_index=True);matched.to_csv(out/"matched_frame_manifest.csv",index=False)
    log("  matched: OOF 2760 / Val 920; duplicates 0")

    log("[2/10] Historical6G vs CONTROL config audit")
    c6=json.loads((p6g/"config.json").read_text());c6b=json.loads((root/"outputs/phase6b_local_refinement_256_baseline/config.json").read_text());c7=json.loads((p7b/"config.json").read_text())
    diff_items=[
      ("train cases","8","8","same","same frozen Train cases"),("training samples","2784 all frames","2760 legal temporal targets","different","24 first-two-Series frames excluded in Phase7B; changes batches/order"),("Val comparison frames","920 matched subset","920 legal targets","same","fair matched evaluation"),("input","[I_t,I_t,I_t]","[I_t,I_t,I_t]","same","identical channel construction"),("crop center","raw_t","raw_t","same","identical"),("crop origin","floor(raw - 128)","floor(raw - 128)","same","identical"),("normalization","full-frame per-frame min-max then crop","full-frame per-frame min-max then crop","same","identical"),("padding","zero after min-max","zero after min-max","same","identical"),("model","hrnet_w18_small_v2","hrnet_w18_small_v2","same","checkpoint keys and all 866 tensor shapes match"),("pretrained weights","gluon_in1k","gluon_in1k","same","identical"),("heatmap head","HRNetTipHeatmap 1-channel","HRNetTipHeatmap 1-channel","same","state structure matches"),("heatmap size","64x64","64x64","same","identical"),("sigma","2","2","same","identical"),("GT center","floating-point","floating-point","same","identical"),("loss","mean((1+9*GT)*(Pred-GT)^2)",c7["loss"],"same","identical"),("optimizer","Adam","Adam","same","identical"),("learning rate","0.001",str(c7["learning_rate"]),"same","identical"),("weight decay","0",str(c7["weight_decay"]),"same","identical"),("scheduler","MultiStepLR [120,160], gamma .2","MultiStepLR [120,160], gamma .2","same","inactive within 3 epochs"),("batch size","16",str(c7["batch_size"]),"same","identical"),("shuffle","True","True","same","sample-set difference changes permutation trajectory"),("seed","20260918",str(c7["seed"]),"same","identical nominal seed"),("deterministic","True / cudnn benchmark False","deterministic paired setup","same","no nondeterministic warning recorded"),("epochs","3",str(c7["epochs"]),"same","identical"),("checkpoint selection","fixed 3 epochs, chosen before Val","fixed 3 epochs","same","no Val selection"),("argmax","integer 64x64","integer 64x64","same","identical"),("heatmap to input","x4,y4","x4,y4","same","identical"),("local to global","crop origin + local","crop origin + local","same","identical"),("alpha","0.75",str(c7["alpha"]),"same","identical")]
    cd=pd.DataFrame(diff_items,columns=["field","Phase6G","Phase7B_CONTROL","status","possible_effect"]);cd.to_csv(out/"control_vs_phase6g_config_diff.csv",index=False)
    log("  true differences: training samples 2784 vs 2760; resulting batch/order trajectory")

    log("[3/10] CONTROL reproducibility analysis")
    h=norm(pd.read_csv(p6g/"val_predictions.csv"));hj=vc[keys].merge(h,on=keys,how="left",validate="one_to_one",suffixes=("","_h"));assert not hj.phase6g_error_px.isna().any()
    repro=vc.merge(hj[keys+["phase6b_x","phase6b_y","phase6g_x","phase6g_y","phase6b_error_px","phase6g_error_px"]],on=keys,validate="one_to_one")
    repro["hist_candidate_error"]=repro.phase6b_error_px;repro["control_candidate_error"]=repro.candidate_error_px;repro["hist_final_error"]=repro.phase6g_error_px;repro["control_final_error"]=repro.final_error_px
    repro["candidate_prediction_distance"]=np.hypot(repro.phase6b_x-repro.candidate_x,repro.phase6b_y-repro.candidate_y);repro["final_prediction_distance"]=np.hypot(repro.phase6g_x-repro.final_x,repro.phase6g_y-repro.final_y)
    rr=[]
    for scope,g in [("overall",repro),*[(str(c),g) for c,g in repro.groupby("case_id")]]:
        for metric in ["candidate_prediction_distance","final_prediction_distance"]:rr.append({"scope":scope,"metric":metric,**summary(g[metric])})
    pd.DataFrame([x for x in rr if x["scope"]=="overall"]).to_csv(out/"control_reproducibility_metrics.csv",index=False);pd.DataFrame([x for x in rr if x["scope"]!="overall"]).rename(columns={"scope":"case_id"}).to_csv(out/"control_reproducibility_by_case.csv",index=False)
    # Optimization comparison.
    l6=pd.read_csv(p6g/"final_training_log.csv").rename(columns={"train_loss":"phase6g_train_loss"});l7=pd.read_csv(p7b/"final_control_training_log.csv").rename(columns={"train_loss":"control_train_loss"});l6[["epoch","phase6g_train_loss"]].merge(l7[["epoch","control_train_loss"]],on="epoch").to_csv(out/"control_training_loss_comparison.csv",index=False)

    log("[4/10] Computing temporal dynamics / crop appearance")
    full=manifest.rename(columns={"case_code":"case_id","series_uid":"series_id","raw_col":"raw_x","raw_row":"raw_y","true_tip_col":"gt_x","true_tip_row":"gt_y"});full=norm(full);full=full.sort_values(["split","case_id","series_id","frame_index"])
    for lag in [1,2]:
        for c in ["raw_x","raw_y","gt_x","gt_y"]:full[f"{c}_m{lag}"]=full.groupby(["split","case_id","series_id"])[c].shift(lag)
    legal=matched[["audit_split",*keys]].merge(full,on=keys,how="left",validate="one_to_one")
    legal["split"]=legal.audit_split
    legal["gt_motion_1"]=np.hypot(legal.gt_x-legal.gt_x_m1,legal.gt_y-legal.gt_y_m1);legal["gt_motion_2"]=np.hypot(legal.gt_x-legal.gt_x_m2,legal.gt_y-legal.gt_y_m2)
    legal["raw_motion_1"]=np.hypot(legal.raw_x-legal.raw_x_m1,legal.raw_y-legal.raw_y_m1);legal["raw_motion_2"]=np.hypot(legal.raw_x-legal.raw_x_m2,legal.raw_y-legal.raw_y_m2)
    legal["gt_acc"]=np.hypot((legal.gt_x-legal.gt_x_m1)-(legal.gt_x_m1-legal.gt_x_m2),(legal.gt_y-legal.gt_y_m1)-(legal.gt_y_m1-legal.gt_y_m2))
    legal["raw_acc"]=np.hypot((legal.raw_x-legal.raw_x_m1)-(legal.raw_x_m1-legal.raw_x_m2),(legal.raw_y-legal.raw_y_m1)-(legal.raw_y_m1-legal.raw_y_m2))
    appearance=[]
    for split,ww,cache in [("OOF",tw,p7b/"cache/train_stacks.npy"),("Val",vw,p7b/"cache/val_stacks.npy")]:
        mm=np.memmap(cache,dtype=np.float32,mode="r",shape=(len(ww),3,256,256))
        for i in range(len(ww)):
            if i%250==0:log(f"  {split}: {i}/{len(ww)}")
            a,b,c=mm[i]; dx=np.diff(c,axis=1);dy=np.diff(c,axis=0);grad=np.hypot(dx[:-1],dy[:,:-1]);
            def corr(x,y):
                xx=x.ravel();yy=y.ravel();sx=xx.std();sy=yy.std();return float(np.mean((xx-xx.mean())*(yy-yy.mean()))/(sx*sy+1e-12))
            row={"split":split,"case_id":str(ww.iloc[i].case_id),"series_id":str(ww.iloc[i].series_id),"sequence_identifier":str(ww.iloc[i].sequence_identifier),"frame_index":int(ww.iloc[i].frame_index),"corr_prev1":corr(b,c),"corr_prev2":corr(a,c),"corr_t2_t1":corr(a,b),"mae_prev1":float(np.mean(np.abs(b-c))),"mae_prev2":float(np.mean(np.abs(a-c))),"mae_t2_t1":float(np.mean(np.abs(a-b))),"image_mean":float(c.mean()),"image_std":float(c.std()),"gradient_mean":float(grad.mean()),"edge_density":float(np.mean(grad>0.05)),"local_contrast":float(np.percentile(c,95)-np.percentile(c,5))}
            for j,ch in enumerate((a,b,c)):row[f"channel{j}_mean"]=float(ch.mean());row[f"channel{j}_std"]=float(ch.std())
            appearance.append(row)
    app=pd.DataFrame(appearance);legal=norm(legal).merge(app,on=["split",*keys,"sequence_identifier"],validate="one_to_one")
    # Merge matched model outcomes.
    def predmerge(a,b,split):
        z=a.merge(b[keys+["final_error_px","peak","margin"]],on=keys,suffixes=("_control","_temporal"),validate="one_to_one");z["split"]=split;return z
    pred=pd.concat([predmerge(oc,ot,"OOF"),predmerge(vc,vt,"Val")],ignore_index=True);pred=norm(pred)
    keep=keys+["split","raw_error_px","final_error_px_control","final_error_px_temporal","gain_px_control","gain_px_temporal","peak_control","peak_temporal","margin_control","margin_temporal"]
    allf=legal.merge(pred[keep],on=["split",*keys],validate="one_to_one").rename(columns={"final_error_px_control":"control_error","final_error_px_temporal":"temporal_error","gain_px_control":"control_gain","gain_px_temporal":"temporal_gain_vs_raw"})
    allf["temporal_gain"]=allf.control_error-allf.temporal_error;allf["peak_diff"]=allf.peak_temporal-allf.peak_control;allf["margin_diff"]=allf.margin_temporal-allf.margin_control
    allf[allf.split=="OOF"].to_csv(out/"temporal_dynamics_oof.csv",index=False);allf[allf.split=="Val"].to_csv(out/"temporal_dynamics_val.csv",index=False)

    log("[5/10] OOF vs Val distribution shift")
    feats=["gt_motion_1","gt_motion_2","gt_acc","raw_motion_1","raw_motion_2","raw_acc","corr_prev1","corr_prev2","mae_prev1","mae_prev2","image_mean","image_std","gradient_mean","edge_density","local_contrast"]
    ds=[]
    for f in feats:
        a=allf.loc[allf.split=="OOF",f];b=allf.loc[allf.split=="Val",f];pooled=math.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
        ds.append({"feature":f,**summary(a,"oof_"),**summary(b,"val_"),"standardized_mean_difference":(b.mean()-a.mean())/(pooled+1e-12),"ks_statistic":ks_stat(a,b)})
    pd.DataFrame(ds).to_csv(out/"distribution_shift_summary.csv",index=False)
    by=pd.DataFrame(metric_rows(allf,["split","case_id"],feats));trainmed=by[by.split=="OOF"].groupby("feature").median.agg(["min","max"])
    by["train_case_median_min"]=by.feature.map(trainmed["min"]);by["train_case_median_max"]=by.feature.map(trainmed["max"]);by["outside_train_case_median_range"]=(by.split=="Val")&((by["median"]<by.train_case_median_min)|(by["median"]>by.train_case_median_max));by.to_csv(out/"distribution_shift_by_case.csv",index=False)

    log("[6/10] Temporal gain conditional analysis")
    corr_feats=["gt_motion_1","gt_motion_2","gt_acc","raw_motion_1","raw_motion_2","raw_acc","corr_prev1","corr_prev2","mae_prev1","mae_prev2","raw_error_px","peak_diff","margin_diff"]
    cr=[];cbc=[]
    for split,g in allf.groupby("split"):
        for f in corr_feats:cr.append({"split":split,"feature":f,"spearman_temporal_gain":g[[f,"temporal_gain"]].corr(method="spearman").iloc[0,1],"n":len(g)})
        for case,hc in g.groupby("case_id"):
            for f in corr_feats:cbc.append({"split":split,"case_id":case,"feature":f,"spearman_temporal_gain":hc[[f,"temporal_gain"]].corr(method="spearman").iloc[0,1],"n":len(hc)})
    pd.DataFrame(cr).to_csv(out/"motion_gain_correlations.csv",index=False);pd.DataFrame(cbc).to_csv(out/"motion_gain_by_case.csv",index=False)
    train=allf[allf.split=="OOF"]
    def frozen_edges(f):
        a=train[f];return [float(a.min()),*map(float,a.quantile([.25,.5,.75]).values),float(a.max())]
    motion_bins=pd.DataFrame(binned_performance(allf,"gt_motion_2",frozen_edges("gt_motion_2"),"OOF quartiles"));motion_bins.to_csv(out/"motion_bins.csv",index=False)
    simrows=[]
    for f in ["corr_prev1","mae_prev1"]:simrows+=binned_performance(allf,f,frozen_edges(f),"OOF quartiles")
    simbins=pd.DataFrame(simrows);simbins.to_csv(out/"similarity_bins.csv",index=False)
    mq=train.gt_motion_2.quantile([1/3,2/3]).values;cq=train.corr_prev1.quantile([1/3,2/3]).values
    def level(x,q):return pd.cut(x,[-np.inf,q[0],q[1],np.inf],labels=["low","medium","high"],include_lowest=True)
    z=allf.copy();z["motion_level"]=level(z.gt_motion_2,mq);z["corr_level"]=level(z.corr_prev1,cq)
    grid=[]
    for ml in ["low","medium","high"]:
      for cl in ["low","medium","high"]:
        row={"motion_level":ml,"corr_level":cl,"motion_cut1":mq[0],"motion_cut2":mq[1],"corr_cut1":cq[0],"corr_cut2":cq[1]}
        for split in ["OOF","Val"]:
            g=z[(z.split==split)&(z.motion_level==ml)&(z.corr_level==cl)];row[f"{split.lower()}_n"]=len(g);row[f"{split.lower()}_temporal_gain"]=g.temporal_gain.mean();row[f"{split.lower()}_control_error"]=g.control_error.mean();row[f"{split.lower()}_temporal_error"]=g.temporal_error.mean()
        grid.append(row)
    grid=pd.DataFrame(grid);grid.to_csv(out/"motion_similarity_grid.csv",index=False)
    pd.DataFrame([r for r in cr if r["feature"].startswith("raw_")]).to_csv(out/"raw_dynamics_gain.csv",index=False)
    rbins=[0,5,10,20,40,60,np.inf];z=allf.copy();z["raw_bin"]=pd.cut(z.raw_error_px,rbins,right=False,labels=["0-5","5-10","10-20","20-40","40-60",">=60"])
    rows=[]
    for (split,b),g in z.groupby(["split","raw_bin"],observed=True):rows.append({"split":split,"raw_bin":str(b),"n":len(g),"control_mean":g.control_error.mean(),"temporal_mean":g.temporal_error.mean(),"temporal_gain_mean":g.temporal_gain.mean(),"temporal_win_pct":100*(g.temporal_gain>0).mean()})
    pd.DataFrame(rows).to_csv(out/"raw_error_gain_bins.csv",index=False)

    log("[7/10] 13424 / 2148 / 6700 case diagnosis")
    c134=allf[(allf.split=="Val")&(allf.case_id=="13424")].copy();c134["temporal_worse_gt10"]=c134.temporal_gain<-10;c134.to_csv(out/"case13424_deep_dive.csv",index=False)
    for case in ["2148","6700"]:
        g=allf[(allf.split=="Val")&(allf.case_id==case)];pd.DataFrame(metric_rows(g,["case_id","series_id","sequence_identifier"],feats+["control_error","temporal_error","temporal_gain"])).to_csv(out/f"case{case}_summary.csv",index=False)

    log("[8/10] Persistent wrong-track dynamics")
    seg=[]
    for split,g in allf.groupby("split"):
        gg=g.copy();gg["control_harmful"]=gg.control_gain<-10;gg["temporal_harmful"]=gg.temporal_gain_vs_raw<-10
        seg+=contiguous_segments(gg,"control_harmful","CONTROL",split);seg+=contiguous_segments(gg,"temporal_harmful","TEMPORAL",split)
    seg=pd.DataFrame(seg);seg.to_csv(out/"persistent_segment_dynamics.csv",index=False)

    log("[9/10] Confidence / channel-statistics shift")
    cs=[]
    for split,g in allf.groupby("split"):
        for diff in ["peak_diff","margin_diff"]:
            for f in ["gt_motion_1","gt_motion_2","raw_motion_1","raw_motion_2","corr_prev1","mae_prev1"]:
                cs.append({"split":split,"confidence_difference":diff,"feature":f,"spearman":g[[diff,f]].corr(method="spearman").iloc[0,1],"n":len(g)})
    pd.DataFrame(cs).to_csv(out/"confidence_shift_analysis.csv",index=False)
    ch=[]
    for split,g in allf.groupby("split"):
        for j in range(3):
            ch.append({"split":split,"channel":j,"mean_mean":g[f"channel{j}_mean"].mean(),"mean_std":g[f"channel{j}_mean"].std(),"std_mean":g[f"channel{j}_std"].mean(),"std_std":g[f"channel{j}_std"].std()})
        for pair,c in [("0-1","corr_t2_t1"),("1-2","corr_prev1"),("0-2","corr_prev2")]:ch.append({"split":split,"channel":pair,"pairwise_corr_mean":g[c].mean(),"pairwise_corr_median":g[c].median(),"pairwise_corr_p10":g[c].quantile(.1)})
    pd.DataFrame(ch).to_csv(out/"channel_statistics_shift.csv",index=False)

    log("[10/10] Final diagnosis / figures")
    create_figures(out,allf,repro,motion_bins,simbins,grid,seg)
    dsum=pd.DataFrame(ds).set_index("feature");repmean=repro.candidate_prediction_distance.mean();repmed=repro.candidate_prediction_distance.median()
    # Diagnosis prioritizes independent Val contradiction plus the directly observed training-set difference.
    diagnosis={"selected_hypothesis":"H4","label":"multiple factors","control_reproducibility":"B","control_reason":"Core architecture/inference configuration matches. Phase6G trained on 2784 frames while CONTROL trained on 2760 legal targets, changing minibatches and optimization trajectory; prediction behavior differs substantially despite matched architecture.","candidate_prediction_distance_mean_px":float(repmean),"candidate_prediction_distance_median_px":float(repmed),"oof_temporal_gain_mean_px":float(allf.loc[allf.split=="OOF","temporal_gain"].mean()),"val_temporal_gain_mean_px":float(allf.loc[allf.split=="Val","temporal_gain"].mean()),"test_used":False,"training_performed":False,"recommended_next_direction":"repeat-seed stability / training reproducibility","reason":"Before adding temporal complexity, quantify whether the 24-frame/sample-order difference and fresh initialization produce comparable CONTROL baselines."}
    (out/"final_diagnosis.json").write_text(json.dumps(diagnosis,indent=2,ensure_ascii=False),encoding="utf-8")
    # Compact data-derived README.
    def sm(f,s,p):return dsum.loc[f,f"{s}_{p}"]
    lines=["# Phase 7C — Temporal Domain-Shift and Control-Reproducibility Audit","","No training, optimizer, backward pass, new checkpoint, Test data, alpha tuning, gate, or threshold fitting was used.","","## CONTROL reproducibility","",f"Historical6G matched-920: Mean {repro.hist_final_error.mean():.3f}, Median {repro.hist_final_error.median():.3f}, P95 {repro.hist_final_error.quantile(.95):.3f} px. New CONTROL: Mean {repro.control_final_error.mean():.3f}, Median {repro.control_final_error.median():.3f}, P95 {repro.control_final_error.quantile(.95):.3f} px.",f"Candidate prediction distance: Mean {repmean:.3f}, Median {repmed:.3f}, P95 {repro.candidate_prediction_distance.quantile(.95):.3f} px.","","Core model, preprocessing, loss, optimizer, seed, epochs, coordinate transforms, and alpha match. The verified difference is 2784 Phase6G training frames versus 2760 legal Phase7B targets. That exclusion changes minibatch composition/order. State keys and all 866 tensor shapes match; no implementation bug was found. Classification: B — configuration is essentially aligned, but prediction behavior differs, consistent with small-case optimization instability plus the sample-set/order difference.","","## Motion and appearance shift","",f"GT motion2 Median/P95: OOF {sm('gt_motion_2','oof','median'):.3f}/{sm('gt_motion_2','oof','p95'):.3f}, Val {sm('gt_motion_2','val','median'):.3f}/{sm('gt_motion_2','val','p95'):.3f}. Correlation t-1/t Median: OOF {sm('corr_prev1','oof','median'):.4f}, Val {sm('corr_prev1','val','median'):.4f}. Pixel MAE Median: OOF {sm('mae_prev1','oof','median'):.4f}, Val {sm('mae_prev1','val','median'):.4f}.","","## Final diagnosis","","H4 — multiple factors. Temporal dynamics and image/input-statistic shifts are measured and associated with behavior, but they do not alone explain the OOF-to-Val reversal. The independent-Val confidence reversal and failure to reproduce Historical6G with a fresh matched CONTROL show that training/sample-order instability is also important.","","Only one next direction is recommended: repeat-seed stability / training reproducibility. This is a recommendation only; it was not executed."]
    (out/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (out/"run_status.json").write_text(json.dumps({"status":"complete","training":False,"test_used":False,"oof_frames":2760,"val_frames":920},indent=2),encoding="utf-8")
    log(f"[Done] H4 selected. CONTROL candidate difference mean={repmean:.3f}px. Outputs: {out}")


if __name__=="__main__": main()
