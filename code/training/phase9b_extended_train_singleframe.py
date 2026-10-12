from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


def load_source(name: str, path: Path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


P7D = load_source("phase7d_runtime_for_9b", ROOT / "scripts/phase7d_repeat_seed_stability.src")
P7B = P7D.P7B


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_new_train_rows(split: pd.DataFrame, p9: Path) -> pd.DataFrame:
    train_series = split[split.split_group.eq("EXTENDED_NEW_TRAIN")][["case_id", "series_id", "sequence_identifier"]]
    mapping = pd.read_csv(p9 / "csv_dicom_mapping.csv", dtype={"case_id": str, "series_id": str})
    pixels = pd.read_csv(p9 / "pixel_readability_audit.csv", dtype={"case_id": str, "series_id": str})
    rows = []
    for s in train_series.itertuples():
        m = mapping[(mapping.case_id.eq(s.case_id)) & (mapping.series_id.eq(s.series_id))]
        if len(m) != 1 or m.iloc[0].mapping_class != "EXACT":
            raise RuntimeError(f"New Train mapping is not EXACT: {s.case_id}/{s.sequence_identifier}")
        csv_path = str(m.iloc[0].csv_paths).split(";")[0]
        d = pd.read_csv(csv_path)
        for c in ["frame_index", "raw_row", "raw_col", "true_tip_row", "true_tip_col"]:
            d[c] = pd.to_numeric(d[c], errors="raise")
        pf = pixels[(pixels.case_id.eq(s.case_id)) & (pixels.series_id.eq(s.series_id))]
        file_map = {Path(p).name: p for p in pf.dicom_path.astype(str)}
        d["basename"] = d.dicom_file.map(lambda x: Path(str(x).replace("\\", "/")).name)
        d["path_t"] = d.basename.map(file_map)
        if d.path_t.isna().any() or len(d) != len(pf):
            raise RuntimeError(f"New Train frame mapping mismatch: {s.case_id}/{s.sequence_identifier}")
        for r in d.itertuples():
            raw_x, raw_y = float(r.raw_col), float(r.raw_row)
            gt_x, gt_y = float(r.true_tip_col), float(r.true_tip_row)
            x0, y0 = P7B.origin(raw_x, raw_y, 256)
            rows.append({"split": "train", "case_id": str(s.case_id), "series_id": str(s.series_id),
                         "sequence_identifier": str(s.sequence_identifier), "frame_index": int(r.frame_index),
                         "path_t": str(r.path_t), "raw_x": raw_x, "raw_y": raw_y, "gt_x": gt_x, "gt_y": gt_y,
                         "crop_x0": x0, "crop_y0": y0, "gt_local_x": gt_x-x0, "gt_local_y": gt_y-y0})
    out = pd.DataFrame(rows).sort_values(["case_id", "series_id", "frame_index"]).reset_index(drop=True)
    if len(out) != 1329 or out.case_id.nunique() != 4 or out.series_id.nunique() != 5:
        raise RuntimeError(f"New Train rows mismatch: {len(out)} frames, {out.case_id.nunique()} cases, {out.series_id.nunique()} Series")
    return out


def build_extended_cache(old_rows: pd.DataFrame, new_rows: pd.DataFrame, old_cache: Path, out_path: Path, log) -> None:
    n = len(old_rows) + len(new_rows)
    if out_path.exists():
        a = np.load(out_path, mmap_mode="r")
        if a.shape == (n, 1, 256, 256) and a.dtype == np.float32:
            log(f"[Cache] Reusing {out_path}: {a.shape}")
            return
        raise RuntimeError("Existing extended cache shape mismatch")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".building.npy")
    if tmp.exists():
        tmp.unlink()
    dst = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, 1, 256, 256))
    src = np.load(old_cache, mmap_mode="r")
    if src.shape != (len(old_rows), 1, 256, 256):
        raise RuntimeError(f"Historical cache mismatch: {src.shape}")
    dst[:len(old_rows)] = src
    for j, r in tqdm(enumerate(new_rows.itertuples(), start=len(old_rows)), total=len(new_rows), desc="Cache new Train frames"):
        image = P7B.mmread(str(r.path_t))
        dst[j, 0] = P7B.crop(image, int(r.crop_x0), int(r.crop_y0), 256)
    dst.flush(); del dst
    tmp.replace(out_path)
    log(f"[Cache] Saved {out_path}: {(n,1,256,256)}")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    z = P7D.normalize_prediction(df)
    z["case_id"] = z.case_id.astype(str); z["series_id"] = z.series_id.astype(str); z["frame_index"] = z.frame_index.astype(int)
    return z


def run_extended(run_dir: Path, cfg_hash: str, split_hash: str, seed: int, fold: int, held: str,
                 dataset, train_idx, held_idx, device, p5, log, scope="OOF", infer_dataset=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    done = run_dir / "complete.json"; pred_path = run_dir / "predictions.csv"; ckpt = run_dir / "checkpoint.pth"; train_log = run_dir / "training_log.csv"
    expected = {"config_hash": cfg_hash, "split_hash": split_hash, "seed": seed, "fold": fold, "held_case": held, "scope": scope}
    if done.exists() and pred_path.exists() and ckpt.exists() and train_log.exists():
        meta = json.loads(done.read_text(encoding="utf-8"))
        if all(meta.get(k) == v for k, v in expected.items()):
            log(f"  [Resume] {scope} seed={seed} fold={fold} held={held}")
            return normalize(pd.read_csv(pred_path)), pd.read_csv(train_log)
        raise RuntimeError(f"Resume metadata mismatch: {run_dir}")
    effective_seed = seed + fold if scope == "OOF" else seed
    model, logs = P7B.train_model(p5, dataset, train_idx, effective_seed, device,
                                  f"Phase9B {scope} seed={seed} held={held}", 3, 16, log)
    prediction_dataset = dataset if infer_dataset is None else infer_dataset
    pred = normalize(P7B.infer(model, prediction_dataset, held_idx, device, 0.75, 16))
    pred.to_csv(pred_path, index=False); logs.to_csv(train_log, index=False)
    torch.save({"model_state_dict": model.state_dict(), "seed": seed, "effective_seed": effective_seed,
                "fold": fold, "held_case": held, "scope": scope, "epochs": 3,
                "split_hash": split_hash, "config_hash": cfg_hash}, ckpt)
    meta = {**expected, "effective_seed": effective_seed, "train_cases": sorted(dataset.w.iloc[train_idx].case_id.astype(str).unique().tolist()),
            "train_frames": int(len(train_idx)), "prediction_frames": int(len(held_idx)), "epochs": 3,
            "complete": True, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    done.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return pred, logs


def metric_row(g: pd.DataFrame, method: str, seed: int, split_name: str, alpha: float = .75) -> dict:
    col = "final_error_px" if alpha == .75 else "candidate_error_px"
    e = g[col].to_numpy(float); raw = g.raw_error_px.to_numpy(float); gain = raw-e
    return {"split": split_name, "method": method, "base_seed": seed, "alpha": alpha, "n": len(g),
            "mean": e.mean(), "median": np.median(e), "p95": np.percentile(e,95), "p99": np.percentile(e,99), "max": e.max(),
            "lt5": np.mean(e<5), "lt10": np.mean(e<10), "lt20": np.mean(e<20), "lt40": np.mean(e<40),
            "mean_gain": gain.mean(), "median_gain": np.median(gain), "improved_pct": np.mean(gain>0), "worsened_pct": np.mean(gain<0),
            "harmful_gt5": np.mean(gain<-5), "harmful_gt10": np.mean(gain<-10), "harmful_gt20": np.mean(gain<-20),
            "case_macro_mean": g.assign(_e=e).groupby("case_id")["_e"].mean().mean()}


def safety_rows(df: pd.DataFrame, method: str, seed: int, split_name: str) -> list[dict]:
    out=[]
    for hi,label in [(5,"raw_lt5"),(10,"raw_lt10")]:
        g=df[df.raw_error_px < hi]; gain=g.gain_px.to_numpy(float)
        out.append({"split":split_name,"method":method,"base_seed":seed,"bin":label,"n":len(g),
                    "mean_error":g.final_error_px.mean(),"median_error":g.final_error_px.median(),
                    "harmful_gt5":np.mean(gain<-5),"harmful_gt10":np.mean(gain<-10)})
    return out


def rescue_rows(df: pd.DataFrame, method: str, seed: int, split_name: str) -> list[dict]:
    out=[]
    for lo,hi in [(20,40),(40,60)]:
        g=df[(df.raw_error_px>=lo)&(df.raw_error_px<hi)]
        out.append({"split":split_name,"method":method,"base_seed":seed,"bin":f"raw_{lo}_{hi}","n":len(g),
                    "mean_error":g.final_error_px.mean(),"mean_gain":g.gain_px.mean(),"median_gain":g.gain_px.median(),
                    "improved_pct":np.mean(g.gain_px>0)})
    return out


def persistent(df: pd.DataFrame, method: str, seed: int, split_name: str) -> dict:
    row=P7D.persistent_rows(df,method,seed,split_name); row["method"]=row.pop("arm"); return row


def correction_rows(df: pd.DataFrame, method: str, seed: int, split_name: str) -> list[dict]:
    mag=np.hypot(df.final_x-df.raw_x,df.final_y-df.raw_y)
    bins=[(np.ones(len(df),bool),"overall"),(df.raw_error_px<10,"easy_raw_lt10"),((df.raw_error_px>=20)&(df.raw_error_px<60),"hard_raw_20_60")]
    out=[]
    for mask,label in bins:
        v=mag[np.asarray(mask)]; out.append({"split":split_name,"method":method,"base_seed":seed,"bin":label,"n":len(v),
                                            "mean":v.mean(),"median":np.median(v),"p95":np.percentile(v,95)})
    return out


def savefig(path: Path):
    plt.tight_layout(); plt.savefig(path,dpi=180,bbox_inches="tight"); plt.close()


def paired_analysis(baseline: pd.DataFrame, extended: pd.DataFrame, out_dir: Path, split_name: str):
    metric=[]; cases=[]; safety=[]; rescue=[]; persist=[]; correction=[]; confidence=[]
    seeds=sorted(extended.base_seed.unique())
    for seed in seeds:
        for method,source in [("BASELINE_8",baseline),("EXTENDED_12",extended)]:
            g=source[source.base_seed.eq(seed)].copy()
            for alpha in [1.0,.75]: metric.append(metric_row(g,method,seed,split_name,alpha))
            for case,cg in g.groupby("case_id"):
                cases.append({"split":split_name,"method":method,"base_seed":seed,"case_id":case,"n":len(cg),
                              "mean":cg.final_error_px.mean(),"median":cg.final_error_px.median(),"p95":np.percentile(cg.final_error_px,95)})
            safety += safety_rows(g,method,seed,split_name); rescue += rescue_rows(g,method,seed,split_name)
            persist.append(persistent(g,method,seed,split_name)); correction += correction_rows(g,method,seed,split_name)
            confidence.append({"split":split_name,"method":method,"base_seed":seed,"peak_mean":g.peak.mean(),"peak_median":g.peak.median(),"margin_mean":g.margin.mean(),"margin_median":g.margin.median()})
    return tuple(map(pd.DataFrame,[metric,cases,safety,rescue,persist,correction,confidence]))


def main(config_path: Path) -> None:
    cfg=json.loads(config_path.read_text(encoding="utf-8")); out=ROOT/cfg["output_dir"]; out.mkdir(parents=True,exist_ok=True)
    (out/"oof").mkdir(exist_ok=True); (out/"final").mkdir(exist_ok=True); (out/"figures").mkdir(exist_ok=True); (out/"cache").mkdir(exist_ok=True)
    log=P7D.Logger(out/"phase9b_run.log"); cfg_hash=canonical_hash(cfg); split_hash=(out/"split_hash.txt").read_text().strip()
    (out/"config.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    log("="*52);log("Phase 9B — EXTENDED-TRAIN SINGLE-FRAME BASELINE");log("="*52)
    log("Scientific variable: TRAINING DATA EXPANSION | Temporal: NONE | Test: RESERVED / NOT USED")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda": raise RuntimeError("CUDA required")
    log(f"GPU: {torch.cuda.get_device_name(0)} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
    p5=P7B.load_p5(ROOT); P7D.CFG=cfg
    split=pd.read_csv(out/"phase9b_frozen_split.csv",dtype={"case_id":str,"series_id":str})
    if canonical_hash(split.sort_values(["split_group","case_id","series_id"]).reset_index(drop=True).to_dict(orient="records"))!=split_hash:
        raise RuntimeError("Frozen split hash mismatch")
    manifest=pd.read_csv(ROOT/cfg["model_clean_manifest"],dtype={"case_code":str,"series_uid":str})
    old=P7D.full_rows(manifest,"train"); new=build_new_train_rows(split,ROOT/cfg["phase9a_dir"]); ext=pd.concat([old,new],ignore_index=True)
    if (len(ext),ext.case_id.nunique(),ext.series_id.nunique())!=(4113,12,17): raise RuntimeError("Extended manifest mismatch")
    ext.to_csv(out/"final/extended_final_manifest.csv",index=False)
    dist=ext.groupby("case_id").agg(frames=("frame_index","size"),series=("series_id","nunique")).reset_index();dist["frame_fraction"]=dist.frames/len(ext);dist["imbalance_warning"]=dist.frame_fraction>.30;dist.to_csv(out/"train_case_frame_distribution.csv",index=False)
    build_extended_cache(old,new,ROOT/cfg["phase7d_dir"]/"cache/train_full_single.npy",out/"cache/extended_full_single.npy",log)
    ds=P7D.FullDataset(ext,out/"cache/extended_full_single.npy",2.0)
    windows=P7B.build_windows(manifest); old_legal=windows[windows.split.eq("train")].reset_index(drop=True)
    cases=sorted(old_legal.case_id.astype(str).unique().tolist())
    if len(old_legal)!=2760 or len(cases)!=8: raise RuntimeError("Matched old-case OOF manifest mismatch")
    baseline_parts=[]; extended_parts=[]; training_logs=[]
    total=40; counter=0
    for seed in cfg["seeds"]:
        for fold,held in enumerate(cases,1):
            counter+=1; log(f"[4/8] OOF model {counter}/{total} | seed {seed} | held old case {held}")
            train_idx=np.where(ext.case_id.astype(str).ne(held))[0]
            held_idx=np.where((ext.case_id.astype(str).eq(held)) & (ext.frame_index.astype(int).ge(2)))[0]
            rdir=out/f"oof/seed_{seed}/fold_{fold}_{held}"
            pred,tl=run_extended(rdir,cfg_hash,split_hash,seed,fold,held,ds,train_idx,held_idx,device,p5,log)
            pred["base_seed"]=seed;pred["fold"]=fold;pred["held_case"]=held;pred["method"]="EXTENDED_12";extended_parts.append(pred)
            tl=tl.copy();tl["base_seed"]=seed;tl["fold"]=fold;tl["held_case"]=held;tl["scope"]="OOF";training_logs.append(tl)
            bp=ROOT/cfg["phase7d_dir"]/f"oof/seed_{seed}/fold_{fold}_{held}/arm_A/predictions.csv"
            b=normalize(pd.read_csv(bp));b["base_seed"]=seed;b["fold"]=fold;b["held_case"]=held;b["method"]="BASELINE_8";baseline_parts.append(b)
    baseline=pd.concat(baseline_parts,ignore_index=True); extended=pd.concat(extended_parts,ignore_index=True)
    if len(baseline)!=13800 or len(extended)!=13800: raise RuntimeError(f"OOF prediction mismatch {len(baseline)}/{len(extended)}")
    extended.to_csv(out/"oof/extended_oof_predictions.csv",index=False); baseline.to_csv(out/"oof/baseline_oof_predictions_reused.csv",index=False)
    om,oc,os,oh,op,ocr,oconf=paired_analysis(baseline,extended,out/"oof","OOF")
    om.to_csv(out/"oof/oof_metrics_by_seed.csv",index=False);oc.to_csv(out/"oof/oof_metrics_by_case_seed.csv",index=False)
    os.to_csv(out/"oof/easy_safety.csv",index=False);oh.to_csv(out/"oof/hard_rescue.csv",index=False);op.to_csv(out/"oof/persistent_harmful.csv",index=False)
    ocr.to_csv(out/"oof/correction_magnitude.csv",index=False);oconf.to_csv(out/"oof/confidence.csv",index=False)
    primary=om[om.alpha.eq(.75)].pivot(index="base_seed",columns="method",values="case_macro_mean").reset_index();primary["delta_extended_minus_baseline"]=primary.EXTENDED_12-primary.BASELINE_8
    primary.to_csv(out/"oof/baseline_vs_extended_by_seed.csv",index=False)
    cm=oc.pivot_table(index=["base_seed","case_id"],columns="method",values="mean").reset_index();cm["delta_extended_minus_baseline"]=cm.EXTENDED_12-cm.BASELINE_8;cm.to_csv(out/"oof/case_seed_matrix.csv",index=False)
    wins=int((primary.delta_extended_minus_baseline<0).sum()); avg_delta=float(primary.delta_extended_minus_baseline.mean())
    sb=os[(os.method.eq("BASELINE_8"))&(os.bin.eq("raw_lt10"))].harmful_gt10.mean();se=os[(os.method.eq("EXTENDED_12"))&(os.bin.eq("raw_lt10"))].harmful_gt10.mean()
    if wins>=4 and avg_delta<0 and se<=sb+.01: oof_code="OOF-A"
    elif wins in (2,3) or avg_delta<0: oof_code="OOF-B"
    else: oof_code="OOF-C"
    decision={"decision":oof_code,"baseline_case_macro_mean_5seed":float(primary.BASELINE_8.mean()),"extended_case_macro_mean_5seed":float(primary.EXTENDED_12.mean()),
              "mean_delta_extended_minus_baseline_px":avg_delta,"extended_seed_wins":wins,"extended_seed_losses":5-wins,
              "baseline_macro_sd":float(primary.BASELINE_8.std(ddof=1)),"extended_macro_sd":float(primary.EXTENDED_12.std(ddof=1)),
              "baseline_p95_mean":float(om[(om.method.eq("BASELINE_8"))&(om.alpha.eq(.75))].p95.mean()),
              "extended_p95_mean":float(om[(om.method.eq("EXTENDED_12"))&(om.alpha.eq(.75))].p95.mean()),
              "baseline_harmful_gt10_mean":float(om[(om.method.eq("BASELINE_8"))&(om.alpha.eq(.75))].harmful_gt10.mean()),
              "extended_harmful_gt10_mean":float(om[(om.method.eq("EXTENDED_12"))&(om.alpha.eq(.75))].harmful_gt10.mean()),
              "frozen_before_val":True,"split_hash":split_hash,"test_used":False}
    (out/"oof/oof_decision.json").write_text(json.dumps(decision,indent=2),encoding="utf-8")
    log(f"[5/8] Frozen OOF conclusion: {oof_code}; Extended wins {wins}/5; mean delta {avg_delta:+.3f}px")
    # OOF figures before any final Val inference
    fig=out/"figures"
    primary.set_index("base_seed")[["BASELINE_8","EXTENDED_12"]].plot(marker="o");plt.ylabel("Case-macro mean error (px)");plt.title("Matched old-case OOF");savefig(fig/"oof_baseline_vs_extended.png")
    plt.axhline(0,color="black",lw=.8);plt.plot(primary.base_seed.astype(str),primary.delta_extended_minus_baseline,marker="o");plt.ylabel("Extended - Baseline (px)");plt.title("OOF delta by seed");savefig(fig/"oof_delta_by_seed.png")
    hm=cm.pivot(index="case_id",columns="base_seed",values="delta_extended_minus_baseline");plt.figure(figsize=(8,5));v=max(abs(hm.to_numpy().min()),abs(hm.to_numpy().max()));im=plt.imshow(hm,aspect="auto",cmap="coolwarm",vmin=-v,vmax=v);plt.colorbar(im,label="Extended - Baseline (px)");plt.xticks(range(len(hm.columns)),hm.columns,rotation=30);plt.yticks(range(len(hm.index)),hm.index);plt.title("Case × seed delta");savefig(fig/"case_seed_heatmap.png")
    for data,name,col,title in [(os,"easy_safety_oof.png","harmful_gt10","OOF easy-case harmful >10"),(oh,"hard_rescue_oof.png","mean_gain","OOF hard-case mean gain")]:
        data.pivot_table(index="base_seed",columns=["method","bin"],values=col).plot(marker="o");plt.ylabel(col);plt.title(title);savefig(fig/name)
    # Full Train models and matched Val only after frozen OOF decision.
    wv=windows[windows.split.eq("val")].reset_index(drop=True)
    dv=P7B.WindowDataset(wv,str(ROOT/cfg["phase7b_dir"]/"cache/val_stacks.npy"),"control",2.0)
    val_base=[];val_ext=[]
    for seed in cfg["seeds"]:
        log(f"[6/8] Full Extended Train model seed {seed}")
        rdir=out/f"final/seed_{seed}"
        pred,tl=run_extended(rdir,cfg_hash,split_hash,seed,0,"VAL",ds,np.arange(len(ds)),np.arange(len(dv)),device,p5,log,scope="FINAL",infer_dataset=dv)
        pred["base_seed"]=seed;pred["method"]="EXTENDED_12";val_ext.append(pred)
        tl=tl.copy();tl["base_seed"]=seed;tl["fold"]=0;tl["held_case"]="VAL";tl["scope"]="FINAL";training_logs.append(tl)
        bp=ROOT/cfg["phase7d_dir"]/f"final/seed_{seed}/arm_A/predictions.csv"
        b=normalize(pd.read_csv(bp));b["base_seed"]=seed;b["method"]="BASELINE_8";val_base.append(b)
    vb=pd.concat(val_base,ignore_index=True);ve=pd.concat(val_ext,ignore_index=True)
    if len(vb)!=4600 or len(ve)!=4600: raise RuntimeError(f"Val prediction mismatch {len(vb)}/{len(ve)}")
    ve.to_csv(out/"final/val_extended_predictions.csv",index=False);vb.to_csv(out/"final/val_baseline_predictions_reused.csv",index=False)
    pd.concat(training_logs,ignore_index=True).to_csv(out/"final/extended_training_logs.csv",index=False)
    vm,vc,vs,vh,vp,vcr,vconf=paired_analysis(vb,ve,out/"final","Val")
    vm.to_csv(out/"final/val_metrics_by_seed.csv",index=False);vc.to_csv(out/"final/val_by_case_seed.csv",index=False)
    vs.to_csv(out/"final/val_easy_safety.csv",index=False);vh.to_csv(out/"final/val_hard_rescue.csv",index=False);vp.to_csv(out/"final/val_persistent_harmful.csv",index=False)
    vcr.to_csv(out/"final/val_correction_magnitude.csv",index=False);vconf.to_csv(out/"final/val_confidence.csv",index=False)
    vprim=vm[vm.alpha.eq(.75)].pivot(index="base_seed",columns="method",values="mean").reset_index();vprim["delta_extended_minus_baseline"]=vprim.EXTENDED_12-vprim.BASELINE_8;vprim.to_csv(out/"final/val_delta_by_seed.csv",index=False)
    val_wins=int((vprim.delta_extended_minus_baseline<0).sum())
    if oof_code=="OOF-A" and val_wins>=4: final_choice=1
    elif oof_code=="OOF-C" and val_wins<=2: final_choice=3
    else: final_choice=2
    final_decision={"conclusion":final_choice,"oof_decision":oof_code,"oof_extended_wins":wins,"val_extended_wins":val_wins,
                    "oof_mean_delta_px":avg_delta,"val_mean_delta_px":float(vprim.delta_extended_minus_baseline.mean()),
                    "test_used":False,"test_status":"RESERVED_NOT_USED","split_hash":split_hash}
    (out/"final/final_decision.json").write_text(json.dumps(final_decision,indent=2),encoding="utf-8")
    vprim.set_index("base_seed")[["BASELINE_8","EXTENDED_12"]].plot(marker="o");plt.ylabel("Val mean error (px)");plt.title("Matched frozen Val");savefig(fig/"val_baseline_vs_extended.png")
    plt.axhline(0,color="black",lw=.8);plt.plot(vprim.base_seed.astype(str),vprim.delta_extended_minus_baseline,marker="o");plt.ylabel("Extended - Baseline (px)");plt.title("Val delta by seed");savefig(fig/"val_delta_by_seed.png")
    vc.groupby(["case_id","method"]).mean(numeric_only=True)["mean"].unstack().plot(kind="bar");plt.ylabel("Mean error (px)");plt.title("Val by case, 5-seed average");savefig(fig/"val_by_case.png")
    for data,name,col,title in [(vs,"val_easy_safety.png","harmful_gt10","Val easy-case harmful >10"),(vh,"val_hard_rescue.png","mean_gain","Val hard-case mean gain")]:
        data.pivot_table(index="base_seed",columns=["method","bin"],values=col).plot(marker="o");plt.ylabel(col);plt.title(title);savefig(fig/name)
    both=pd.concat([op.assign(scope="OOF"),vp.assign(scope="Val")]);both.pivot_table(index="base_seed",columns=["scope","method"],values="persistent_frames_ge3").plot(marker="o");plt.ylabel("Persistent harmful frames >=3");plt.title("Persistent harmful behavior");savefig(fig/"persistent_harmful.png")
    # Re-run identity-only leakage assertion and preserve zero-access Test audit.
    leakage=pd.read_csv(out/"leakage_audit.csv")
    if leakage.overlap_count.sum()!=0: raise RuntimeError("Post-training Test leakage audit failed")
    access=json.loads((out/"test_access_audit.json").read_text());
    if any(access[k]!=0 for k in ["training_reads","inference_reads","metric_reads","pixel_reads","gt_reads"]): raise RuntimeError("Test access counter nonzero")
    # README / status
    casewins=cm.assign(win=cm.delta_extended_minus_baseline<0).groupby("case_id").win.sum().sort_values(ascending=False)
    vcase=vc.groupby(["case_id","method"]).mean(numeric_only=True)["mean"].unstack()
    readme=f"""# Phase 9B — Extended-Train Single-Frame Baseline\n\n## Plain-language result\n\n- Deterministic hash assigned **4274, 14-cbct, 2768, 658** to Extended Train and permanently reserved **908, 771** as Final Test. No case characteristic or result influenced this assignment.\n- Train expanded from 8 cases / 12 Series / 2784 frames to **12 cases / 17 Series / 4113 frames**.\n- On matched old-case OOF, Extended won **{wins}/5 seeds**. Mean case-macro change was **{avg_delta:+.3f} px** (negative is better). Baseline seed SD was {primary.BASELINE_8.std(ddof=1):.3f}px; Extended seed SD was {primary.EXTENDED_12.std(ddof=1):.3f}px. Frozen OOF decision: **{oof_code}**.\n- On the frozen matched 920-frame Val, Extended won **{val_wins}/5 seeds**; mean change was {vprim.delta_extended_minus_baseline.mean():+.3f}px.\n- Final conclusion: **{final_choice}**. This is not a Final-Test claim; cases 908 and 771 remain unopened for training/inference/metrics.\n\n## Case consistency\n\nExtended OOF wins across five seeds:\n\n{casewins.to_string()}\n\nVal five-seed mean error by case:\n\n{vcase.to_string()}\n\n## Frozen method\n\nSingle current XA frame copied to three channels; raw-centered 256×256 crop; full-frame per-frame min-max; ImageNet-pretrained `hrnet_w18_small_v2.gluon_in1k`; 64×64 heatmap; sigma 2; weighted MSE `mean((1+9*GT)*(Pred-GT)^2)`; Adam 0.001, weight decay 0; batch 16; three fixed epochs; integer argmax; alpha 0.75. No augmentation, temporal input, gate, Test access, or parameter tuning. Phase7D ARM-A baseline predictions/checkpoints were reused without seed selection.\n\nExtended full Train has 258 optimizer steps per epoch versus 174 for the historical 8-case Train. Therefore any improvement supports the complete training-data expansion intervention (more independent cases plus more samples/updates), not a claim that case diversity alone is proven causal.\n\nRare visually obscured frames retain minor label uncertainty, but human review found no systematic annotation problem. No frame was deleted, relabeled, or down-weighted.\n"""
    (out/"README.md").write_text(readme,encoding="utf-8")
    status={"status":"complete","oof_models_trained":40,"final_models_trained":5,"oof_rows_extended":len(extended),"val_rows_extended":len(ve),
            "extended_train":{"cases":12,"series":17,"frames":4113},"final_test":{"cases":["908","771"],"status":"RESERVED_NOT_USED"},
            "test_used":False,"training_reads_test":0,"inference_reads_test":0,"metric_reads_test":0,"split_hash":split_hash}
    (out/"run_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    log(f"[8/8] Done. OOF Extended wins {wins}/5; Val wins {val_wins}/5; conclusion={final_choice}; Test NOT USED")


if __name__ == "__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True,type=Path);a=ap.parse_args()
    main((ROOT/a.config).resolve() if not a.config.is_absolute() else a.config)
