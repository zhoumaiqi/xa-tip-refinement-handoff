from __future__ import annotations

import argparse, hashlib, importlib.util, json, math, os, random, sys, time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


def load_p5(root: Path):
    p=root/"scripts/__pycache__/hrnet_phase5a_train.cpython-313.pyc"
    s=importlib.util.spec_from_file_location("phase5a_runtime",p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)


def mmread(path:str)->np.ndarray:
    try: a=pydicom.dcmread(path).pixel_array.astype(np.float32)
    except Exception as e: raise RuntimeError(f"DICOM decode failed: {path}\n{e}") from e
    lo,hi=float(a.min()),float(a.max())
    return (a-lo)/(hi-lo) if hi>lo else np.zeros_like(a,dtype=np.float32)


def origin(x,y,s=256): return int(math.floor(float(x)-s/2)),int(math.floor(float(y)-s/2))
def crop(img,x0,y0,s=256):
    out=np.zeros((s,s),np.float32); x1,y1=x0+s,y0+s
    sx0,sy0,sx1,sy1=max(0,x0),max(0,y0),min(img.shape[1],x1),min(img.shape[0],y1)
    if sx1>sx0 and sy1>sy0: out[sy0-y0:sy1-y0,sx0-x0:sx1-x0]=img[sy0:sy1,sx0:sx1]
    return out


def heatmap(cx,cy,size=64,sigma=2.):
    yy,xx=np.mgrid[0:size,0:size].astype(np.float32)
    return np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sigma*sigma)).astype(np.float32)


def build_windows(d):
    rows=[]
    for (split,case,ser),g in d.groupby(["split","case_code","series_uid"],sort=True):
        g=g.sort_values("frame_index").reset_index(drop=True); ix=g.frame_index.to_numpy(int)
        for i in range(2,len(g)):
            if ix[i-2]+1!=ix[i-1] or ix[i-1]+1!=ix[i]: continue
            a,b,c=g.iloc[i-2],g.iloc[i-1],g.iloc[i]; x0,y0=origin(c.raw_col,c.raw_row)
            rows.append({"split":split,"case_id":str(case),"series_id":str(ser),"sequence_identifier":str(c.sequence_identifier),
                "frame_index":int(c.frame_index),"path_t2":str(a.dicom_absolute_path),"path_t1":str(b.dicom_absolute_path),"path_t":str(c.dicom_absolute_path),
                "raw_x":float(c.raw_col),"raw_y":float(c.raw_row),"gt_x":float(c.true_tip_col),"gt_y":float(c.true_tip_row),
                "crop_x0":x0,"crop_y0":y0,"gt_local_x":float(c.true_tip_col)-x0,"gt_local_y":float(c.true_tip_row)-y0})
    return pd.DataFrame(rows)


def build_cache(w, path, logger):
    shape=(len(w),3,256,256)
    if path.exists():
        a=np.load(path,mmap_mode="r")
        if a.shape==shape and a.dtype==np.float32: logger(f"  reuse cache {path.name}: {shape}"); return
        raise RuntimeError(f"Cache shape mismatch: {path} {a.shape} != {shape}")
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".building.npy"); a=np.lib.format.open_memmap(tmp,mode="w+",dtype=np.float32,shape=shape)
    for (case,ser),g in w.groupby(["case_id","series_id"],sort=True):
        unique={}; paths=pd.unique(g[["path_t2","path_t1","path_t"]].values.ravel())
        for p in tqdm(paths,desc=f"Cache {case}/{g.sequence_identifier.iloc[0]}",leave=False): unique[p]=mmread(p)
        for idx,r in g.iterrows():
            a[idx,0]=crop(unique[r.path_t2],r.crop_x0,r.crop_y0); a[idx,1]=crop(unique[r.path_t1],r.crop_x0,r.crop_y0); a[idx,2]=crop(unique[r.path_t],r.crop_x0,r.crop_y0)
        a.flush(); del unique
    del a; tmp.replace(path); logger(f"  saved cache {path.name}: {shape}")


class WindowDataset(Dataset):
    def __init__(self,w,cache_path,mode,sigma=2.):
        self.w=w.reset_index(drop=True); self.x=np.load(cache_path,mmap_mode="r"); self.mode=mode
        self.y=np.stack([heatmap(r.gt_local_x/4.,r.gt_local_y/4.,64,sigma) for _,r in self.w.iterrows()])[:,None]
    def __len__(self): return len(self.w)
    def __getitem__(self,i):
        x=np.asarray(self.x[i],dtype=np.float32)
        if self.mode=="control": x=np.repeat(x[2:3],3,axis=0)
        return torch.from_numpy(x.copy()),torch.from_numpy(self.y[i]),int(i)


class WMSE(nn.Module):
    def forward(self,p,y): return torch.mean((1.+9.*y)*(p-y)**2)


def second_peak(h,radius=6):
    flat=int(np.argmax(h)); y,x=np.unravel_index(flat,h.shape); peak=float(h[y,x]); z=h.copy()
    yy,xx=np.ogrid[:h.shape[0],:h.shape[1]]; z[(xx-x)**2+(yy-y)**2<=radius*radius]=-np.inf
    sec=float(np.max(z)); return x,y,peak,sec,peak-sec


def create_model(p5,seed,device):
    seed_all(seed); m=p5.HRNetTipHeatmap("hrnet_w18_small_v2",True,1,(64,64)).to(device)
    with torch.no_grad():
        o=m(torch.zeros(1,3,256,256,device=device))
    if tuple(o.shape)!=(1,1,64,64): raise RuntimeError(f"Output shape mismatch: {tuple(o.shape)}")
    return m


def train_model(p5,dataset,indices,seed,device,label,epochs,batch,logger):
    model=create_model(p5,seed,device); opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
    sch=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[120,160],gamma=.2); crit=WMSE()
    gen=torch.Generator().manual_seed(seed)
    loader=DataLoader(Subset(dataset,indices),batch_size=batch,shuffle=True,num_workers=0,pin_memory=True,generator=gen)
    logs=[]
    for ep in range(1,epochs+1):
        model.train(); losses=[]; bar=tqdm(loader,desc=f"{label} epoch {ep}/{epochs}",leave=True)
        for x,y,_ in bar:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
            pred=model(x); loss=crit(pred,y)
            if not torch.isfinite(loss): raise RuntimeError(f"Non-finite loss in {label} epoch {ep}")
            loss.backward(); opt.step(); losses.append(float(loss.detach().cpu())); bar.set_postfix(loss=f"{np.mean(losses):.6f}")
        logs.append({"model":label,"epoch":ep,"train_loss":float(np.mean(losses)),"learning_rate":opt.param_groups[0]["lr"]}); sch.step()
        logger(f"  {label} epoch {ep}/{epochs}: loss={logs[-1]['train_loss']:.6f}")
    return model,pd.DataFrame(logs)


def infer(model,dataset,indices,device,alpha=.75,batch=16):
    model.eval(); rows=[]; loader=DataLoader(Subset(dataset,indices),batch_size=batch,shuffle=False,num_workers=0,pin_memory=True)
    with torch.no_grad():
        for x,_,ids in tqdm(loader,desc="Inference",leave=False):
            hm=model(x.to(device,non_blocking=True)).detach().cpu().numpy()[:,0]
            for h,ii in zip(hm,ids.numpy()):
                r=dataset.w.iloc[int(ii)]; hx,hy,pk,sec,mar=second_peak(h)
                cx=r.crop_x0+4.*hx; cy=r.crop_y0+4.*hy
                fx=r.raw_x+alpha*(cx-r.raw_x); fy=r.raw_y+alpha*(cy-r.raw_y)
                rawerr=math.hypot(r.raw_x-r.gt_x,r.raw_y-r.gt_y); ce=math.hypot(cx-r.gt_x,cy-r.gt_y); fe=math.hypot(fx-r.gt_x,fy-r.gt_y)
                rows.append({"case_id":r.case_id,"series_id":r.series_id,"sequence_identifier":r.sequence_identifier,"frame_index":int(r.frame_index),"dicom_path":r.path_t,
                    "raw_x":r.raw_x,"raw_y":r.raw_y,"gt_x":r.gt_x,"gt_y":r.gt_y,"crop_x0":r.crop_x0,"crop_y0":r.crop_y0,
                    "candidate_x":cx,"candidate_y":cy,"final_x":fx,"final_y":fy,"raw_error_px":rawerr,"candidate_error_px":ce,"final_error_px":fe,
                    "gain_px":rawerr-fe,"peak":pk,"second_peak":sec,"margin":mar,"pred_x_heatmap":hx,"pred_y_heatmap":hy})
    return pd.DataFrame(rows)


def estats(a):
    a=np.asarray(a,float); return {"n":len(a),"mean":a.mean(),"median":np.median(a),"p95":np.percentile(a,95),"p99":np.percentile(a,99),"max":a.max(),
        "lt5_pct":100*np.mean(a<5),"lt10_pct":100*np.mean(a<10),"lt20_pct":100*np.mean(a<20),"lt40_pct":100*np.mean(a<40)}


def paired_stats(df,error_col):
    e=df[error_col].to_numpy(float); raw=df.raw_error_px.to_numpy(float); gain=raw-e
    return estats(e)|{"mean_gain_px":gain.mean(),"median_gain_px":np.median(gain),"improved_pct":100*np.mean(gain>0),"worsened_pct":100*np.mean(gain<0),
        "harmful_gt5_pct":100*np.mean(gain<-5),"harmful_gt10_pct":100*np.mean(gain<-10),"harmful_gt20_pct":100*np.mean(gain<-20)}


def add_historical(w,p):
    h=pd.read_csv(p); h["case_id"]=h.case_id.astype(str); h["series_id"]=h.series_id.astype(str)
    if "phase6g_error_px" in h: h=h.rename(columns={"phase6g_error_px":"hist_error_px","phase6g_x":"hist_x","phase6g_y":"hist_y"})
    else: h["hist_error_px"]=h["error_alpha_0.75"]; h["hist_x"]=h.raw_x+.75*(h.model_x-h.raw_x); h["hist_y"]=h.raw_y+.75*(h.model_y-h.raw_y)
    return w.merge(h[["case_id","series_id","frame_index","hist_x","hist_y","hist_error_px"]],on=["case_id","series_id","frame_index"],how="left",validate="one_to_one")


def comparison(control,temporal,hist=None):
    rows=[]
    methods=[("Raw","raw_error_px",control),("CONTROL candidate alpha=1","candidate_error_px",control),("CONTROL alpha=.75","final_error_px",control),
             ("TEMPORAL candidate alpha=1","candidate_error_px",temporal),("TEMPORAL alpha=.75","final_error_px",temporal)]
    if hist is not None: methods.insert(1,("Historical Phase6G alpha=.75","hist_error_px",hist))
    for name,col,z in methods:
        s=paired_stats(z,col) if name!="Raw" else estats(z[col])|{"mean_gain_px":0.,"median_gain_px":0.,"improved_pct":0.,"worsened_pct":0.,"harmful_gt5_pct":0.,"harmful_gt10_pct":0.,"harmful_gt20_pct":0.}
        cases=z.groupby("case_id")[col].mean(); rows.append({"method":name,**s,"case_macro_mean_px":cases.mean()})
    return pd.DataFrame(rows)


def by_case(control,temporal,hist=None):
    rows=[]
    for case in sorted(control.case_id.unique()):
        c=control[control.case_id==case]; t=temporal[temporal.case_id==case]
        methods=[("Raw",c.raw_error_px),("CONTROL candidate alpha=1",c.candidate_error_px),("CONTROL alpha=.75",c.final_error_px),
                 ("TEMPORAL candidate alpha=1",t.candidate_error_px),("TEMPORAL alpha=.75",t.final_error_px)]
        if hist is not None:
            h=hist[hist.case_id==case]; methods.insert(1,("Historical Phase6G alpha=.75",h.hist_error_px))
        for name,a in methods: rows.append({"case_id":case,"method":name,**estats(a)})
    return pd.DataFrame(rows)


def safety(control,temporal):
    rows=[]
    for label,lo,hi in [("raw_0_5",0,5),("raw_5_10",5,10),("raw_lt5",0,5),("raw_lt10",0,10)]:
        for name,z,col in [("Raw",control,"raw_error_px"),("CONTROL alpha=.75",control,"final_error_px"),("TEMPORAL alpha=.75",temporal,"final_error_px")]:
            m=(z.raw_error_px>=lo)&(z.raw_error_px<hi); a=z.loc[m,col].to_numpy(float); raw=z.loc[m,"raw_error_px"].to_numpy(float); gain=raw-a
            rows.append({"raw_bin":label,"method":name,"n":len(a),"mean":a.mean(),"median":np.median(a),"worsened_pct":100*np.mean(gain<0),"harmful_gt5_pct":100*np.mean(gain<-5),"harmful_gt10_pct":100*np.mean(gain<-10)})
    return pd.DataFrame(rows)


def rescue(control,temporal):
    rows=[]
    for label,lo,hi in [("raw_20_40",20,40),("raw_40_60",40,60)]:
        for name,z,col in [("Raw",control,"raw_error_px"),("CONTROL alpha=.75",control,"final_error_px"),("TEMPORAL alpha=.75",temporal,"final_error_px")]:
            m=(z.raw_error_px>=lo)&(z.raw_error_px<hi); a=z.loc[m,col].to_numpy(float); raw=z.loc[m,"raw_error_px"].to_numpy(float); gain=raw-a
            rows.append({"raw_bin":label,"method":name,"n":len(a),"mean":a.mean(),"median":np.median(a),"mean_gain_px":gain.mean(),"improved_pct":100*np.mean(gain>0)})
    return pd.DataFrame(rows)


def persistent(control,temporal):
    sums=[]; segs=[]
    for name,z in [("CONTROL alpha=.75",control),("TEMPORAL alpha=.75",temporal)]:
        total=shortf=persf=ge3=ge5=ge10=0; mx=0
        for (case,ser),g in z.sort_values("frame_index").groupby(["case_id","series_id"]):
            bad=(g.raw_error_px-g.final_error_px<-10).to_numpy(); frames=g.frame_index.to_numpy(); start=None
            for i,v in enumerate(bad.tolist()+[False]):
                if v and start is None:start=i
                if not v and start is not None:
                    L=i-start; total+=L; mx=max(mx,L); shortf+=L if L<3 else 0; persf+=L if L>=3 else 0; ge3+=L>=3; ge5+=L>=5; ge10+=L>=10
                    segs.append({"method":name,"case_id":case,"series_id":ser,"start_frame":int(frames[start]),"end_frame":int(frames[i-1]),"length":L}); start=None
        sums.append({"method":name,"total_harmful_frames":total,"harmful_frame_pct":100*total/len(z),"short_lt3_frames":shortf,"persistent_ge3_frames":persf,
                     "persistent_ge3_segments":ge3,"segments_ge5":ge5,"segments_ge10":ge10,"max_segment_length":mx})
    return pd.DataFrame(sums),pd.DataFrame(segs)


def confidence(control,temporal,split):
    rows=[]
    for name,z in [("CONTROL",control),("TEMPORAL",temporal)]:
        cats={"overall":np.ones(len(z),bool),"beneficial":z.gain_px.to_numpy()>0,"harmful_gt10":z.gain_px.to_numpy()<-10}
        for cat,m in cats.items():
            for case,g in [("ALL",z[m])]+[(c,q) for c,q in z[m].groupby("case_id")]:
                for metric in ("peak","margin"):
                    a=g[metric].to_numpy(float); rows.append({"split":split,"model":name,"group":cat,"case_id":case,"metric":metric,"n":len(a),"mean":a.mean() if len(a) else np.nan,"median":np.median(a) if len(a) else np.nan,"p95":np.percentile(a,95) if len(a) else np.nan})
    return pd.DataFrame(rows)


def trajectory(control,temporal,split):
    rows=[]
    base=control[["case_id","series_id","frame_index","gt_x","gt_y","raw_x","raw_y"]].copy()
    base=base.merge(control[["case_id","series_id","frame_index","final_x","final_y"]].rename(columns={"final_x":"control_x","final_y":"control_y"}),on=["case_id","series_id","frame_index"])
    base=base.merge(temporal[["case_id","series_id","frame_index","final_x","final_y"]].rename(columns={"final_x":"temporal_x","final_y":"temporal_y"}),on=["case_id","series_id","frame_index"])
    for name,x,y in [("GT","gt_x","gt_y"),("Raw","raw_x","raw_y"),("CONTROL","control_x","control_y"),("TEMPORAL","temporal_x","temporal_y")]:
        for (case,ser),g in base.sort_values("frame_index").groupby(["case_id","series_id"]):
            xy=g[[x,y]].to_numpy(float); v=np.diff(xy,axis=0); d1=np.linalg.norm(v,axis=1); d2=np.linalg.norm(xy[2:]-xy[:-2],axis=1); acc=np.linalg.norm(np.diff(v,axis=0),axis=1)
            for metric,a in [("1frame_displacement",d1),("2frame_displacement",d2),("acceleration",acc)]: rows.append({"split":split,"method":name,"case_id":case,"series_id":ser,"metric":metric,**estats(a)})
    return pd.DataFrame(rows)


def plot_basic(out,oofcomp,oofcase,safe,hard,pers,conf,logs):
    f=out/"figures"; f.mkdir(exist_ok=True)
    # Case comparison
    p=oofcase[oofcase.method.isin(["CONTROL alpha=.75","TEMPORAL alpha=.75"])].pivot(index="case_id",columns="method",values="mean")
    ax=p.plot(kind="bar",figsize=(10,5)); ax.set_ylabel("Mean error (px)"); plt.tight_layout(); plt.savefig(f/"oof_case_comparison.png",dpi=180); plt.close()
    # CDF
    fig,ax=plt.subplots(figsize=(7,5))
    for name,z,col in [("Raw",GLOBAL["oc"],"raw_error_px"),("Historical6G",GLOBAL["oh"],"hist_error_px"),("CONTROL .75",GLOBAL["oc"],"final_error_px"),("TEMPORAL .75",GLOBAL["ot"],"final_error_px")]:
        a=np.sort(z[col]); ax.plot(a,np.arange(1,len(a)+1)/len(a),label=name)
    ax.set_xlabel("Error (px)"); ax.set_ylabel("CDF"); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(f/"oof_error_cdf.png",dpi=180); plt.close(fig)
    for df,name,file in [(safe,"Easy-case safety","easy_case_safety.png"),(hard,"Hard-case rescue","hard_case_rescue.png")]:
        pv=df.pivot(index="raw_bin",columns="method",values="mean"); ax=pv.plot(kind="bar",figsize=(9,4.5)); ax.set_ylabel("Mean error (px)"); ax.set_title(name); plt.tight_layout(); plt.savefig(f/file,dpi=180); plt.close()
    ax=pers.set_index("method")[["harmful_frame_pct","persistent_ge3_frames"]].plot(kind="bar",secondary_y="persistent_ge3_frames",figsize=(8,4.5)); plt.tight_layout(); plt.savefig(f/"persistent_harmful_comparison.png",dpi=180); plt.close()
    co=conf[(conf.group=="overall")&(conf.case_id=="ALL")].pivot(index="metric",columns="model",values="median"); ax=co.plot(kind="bar",figsize=(7,4)); plt.tight_layout(); plt.savefig(f/"confidence_comparison.png",dpi=180); plt.close()
    if len(logs):
        fig,ax=plt.subplots(figsize=(9,5))
        for (fold,model),g in logs.groupby(["fold","model"]): ax.plot(g.epoch,g.train_loss,alpha=.45,label=f"{fold}-{model}")
        ax.set_xlabel("Epoch");ax.set_ylabel("Train weighted MSE");ax.set_xticks([1,2,3]); fig.tight_layout();fig.savefig(f/"paired_training_curves.png",dpi=180);plt.close(fig)


GLOBAL={}


def montage(out,w,cache,control,temporal,segs):
    x=np.load(cache,mmap_mode="r"); key={(r.case_id,r.series_id,int(r.frame_index)):i for i,r in w.iterrows()}
    c=control.set_index(["case_id","series_id","frame_index"]); t=temporal.set_index(["case_id","series_id","frame_index"])
    # categories based strictly on offline outcomes
    cg=control.raw_error_px-control.final_error_px; tg=temporal.raw_error_px-temporal.final_error_px
    masks={"temporal_rescues_control.png":(cg<-10)&(tg>cg+10),"both_persistent_fail.png":(cg<-10)&(tg<-10),
           "temporal_new_regressions.png":(tg<-10)&(cg>=-10),"both_success.png":(control.final_error_px<10)&(temporal.final_error_px<10)}
    for file,mask in masks.items():
        sel=control[mask].head(12); fig,axs=plt.subplots(4,9,figsize=(18,8),squeeze=False)
        for ax in axs.ravel():ax.axis("off")
        for j,(_,r) in enumerate(sel.iterrows()):
            k=(r.case_id,r.series_id,int(r.frame_index)); i=key[k]; tt=t.loc[k]; rr=j//3; base=(j%3)*3
            for z,lab in enumerate(["t-2","t-1","t"]):
                ax=axs[rr,base+z];ax.imshow(x[i,z],cmap="gray",vmin=0,vmax=1);ax.axis("off");ax.set_title(f"{r.case_id}/{r.sequence_identifier} f{int(r.frame_index)} {lab}",fontsize=6)
                if z==2:
                    x0,y0=r.crop_x0,r.crop_y0
                    ax.scatter(r.raw_x-x0,r.raw_y-y0,c="cyan",marker="+",s=30);ax.scatter(r.gt_x-x0,r.gt_y-y0,facecolors="none",edgecolors="red",s=30)
                    ax.scatter(r.final_x-x0,r.final_y-y0,facecolors="none",edgecolors="yellow",s=24);ax.scatter(tt.final_x-x0,tt.final_y-y0,facecolors="none",edgecolors="lime",s=24)
                    ax.text(2,252,f"raw {r.raw_error_px:.1f} C {r.final_error_px:.1f} T {tt.final_error_px:.1f}",color="white",fontsize=5,va="bottom")
        fig.suptitle(file.replace(".png","")+" | raw cyan, GT red, CONTROL yellow, TEMPORAL green");fig.tight_layout();fig.savefig(out/"figures"/file,dpi=170);plt.close(fig)


def val_plots(out,vc,vt,vh):
    f=out/"figures"; fig,ax=plt.subplots(figsize=(7,5))
    for name,z,col in [("Raw",vc,"raw_error_px"),("Historical6G",vh,"hist_error_px"),("CONTROL .75",vc,"final_error_px"),("TEMPORAL .75",vt,"final_error_px")]:
        a=np.sort(z[col]);ax.plot(a,np.arange(1,len(a)+1)/len(a),label=name)
    ax.set_xlabel("Error (px)");ax.set_ylabel("CDF");ax.legend();fig.tight_layout();fig.savefig(f/"val_error_cdf.png",dpi=180);plt.close(fig)
    for (case,ser),g in vc.sort_values("frame_index").groupby(["case_id","series_id"]):
        t=vt[(vt.case_id==case)&(vt.series_id==ser)].sort_values("frame_index"); h=vh[(vh.case_id==case)&(vh.series_id==ser)].sort_values("frame_index")
        fig,ax=plt.subplots(figsize=(11,4));ax.plot(g.frame_index,g.raw_error_px,label="Raw",alpha=.8);ax.plot(g.frame_index,h.hist_error_px,label="Historical6G",alpha=.8);ax.plot(g.frame_index,g.final_error_px,label="CONTROL",alpha=.8);ax.plot(t.frame_index,t.final_error_px,label="TEMPORAL",alpha=.8);ax.set_xlabel("Frame");ax.set_ylabel("Error px");ax.legend();fig.tight_layout();fig.savefig(f/f"trajectory_{case}_{g.sequence_identifier.iloc[0]}.png",dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args();cfg=json.load(open(a.config,encoding="utf-8"))
    root=Path(a.config).resolve().parents[1];out=(root/cfg["output_dir"]).resolve();out.mkdir(parents=True,exist_ok=True);(out/"checkpoints").mkdir(exist_ok=True);(out/"folds").mkdir(exist_ok=True);(out/"cache").mkdir(exist_ok=True)
    logp=out/"phase7b_run.log"
    def log(s=""):
        print(s,flush=True)
        with logp.open("a",encoding="utf-8") as f:f.write(str(s)+"\n")
    logp.touch(); log("="*52);log("Phase 7B — CAUSAL 3-FRAME HRNET");log("="*52);log("Scientific variable: INPUT ONLY\nCONTROL: [I_t,I_t,I_t]\nTEMPORAL: [I_t-2,I_t-1,I_t]\nShared crop center: raw_t\nROI: 256x256\nTarget: true_tip_t\nAlpha: FIXED 0.75\nEpochs: FIXED 3\nCase-level OOF: 8 folds\nFuture frames: FORBIDDEN\nNew labels: NONE\nTest: FORBIDDEN\n")
    device=torch.device("cuda");log(f"Device: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | CUDA {torch.version.cuda}")
    p5=load_p5(root);d=pd.read_csv(root/cfg["manifest"]);d.case_code=d.case_code.astype(str);d.frame_index=d.frame_index.astype(int)
    tr=d[d.split=="train"].copy();va=d[d.split=="val"].copy()
    if (len(tr),tr.case_code.nunique(),tr.series_uid.nunique())!=(2784,8,12) or (len(va),va.case_code.nunique(),va.series_uid.nunique())!=(928,3,4):raise RuntimeError("Frozen split mismatch")
    if set(tr.case_code)&set(va.case_code) or set(tr.series_uid)&set(va.series_uid) or set(tr.dicom_absolute_path)&set(va.dicom_absolute_path):raise RuntimeError("Split leakage")
    w=build_windows(d);tw=w[w.split=="train"].reset_index(drop=True);vw=w[w.split=="val"].reset_index(drop=True)
    if len(tw)!=2760 or len(vw)!=920:raise RuntimeError(f"Window mismatch train={len(tw)} val={len(vw)}")
    log("[1/14] Verifying frozen windows / matched frames: PASS (2760 Train, 920 Val; intersections 0)")
    cfg["resolved_manifest"]=str((root/cfg["manifest"]).resolve());json.dump(cfg,open(out/"config.json","w",encoding="utf-8"),indent=2)
    build_cache(tw,out/"cache/train_stacks.npy",log);build_cache(vw,out/"cache/val_stacks.npy",log)
    ds_c=WindowDataset(tw,out/"cache/train_stacks.npy","control");ds_t=WindowDataset(tw,out/"cache/train_stacks.npy","temporal")
    cases=sorted(tw.case_id.unique()); all_logs=[]; foldsum=[]
    for fi,case in enumerate(cases,1):
        seed=cfg["seed"]+fi;train_idx=np.where(tw.case_id.to_numpy()!=case)[0].tolist();hold_idx=np.where(tw.case_id.to_numpy()==case)[0].tolist()
        log(f"[{fi+1}/14] OOF fold {fi}/8 held case={case} train={len(train_idx)} held={len(hold_idx)} seed={seed}")
        row={"fold":fi,"held_case":case,"seed":seed,"train_cases":";".join(sorted(set(cases)-{case})),"train_windows":len(train_idx),"held_windows":len(hold_idx)}
        for mode,ds in [("control",ds_c),("temporal",ds_t)]:
            fd=out/"folds"/f"fold_{fi}_{case}_{mode}";fd.mkdir(exist_ok=True);predp=fd/"predictions.csv";ckp=out/"checkpoints"/f"{mode}_fold_{fi}_{case}.pth";done=fd/"complete.json"
            if predp.exists() and ckp.exists() and done.exists(): log(f"  {mode.upper()}: reuse completed artifact");pred=pd.read_csv(predp);lg=pd.read_csv(fd/"training_log.csv")
            else:
                model,lg=train_model(p5,ds,train_idx,seed,device,mode.upper(),cfg["epochs"],cfg["batch_size"],log)
                pred=infer(model,ds,hold_idx,device,cfg["alpha"],cfg["batch_size"]);torch.save({"model_state_dict":model.state_dict(),"fold":fi,"held_case":case,"mode":mode,"seed":seed,"epochs":3},ckp)
                pred.to_csv(predp,index=False);lg.to_csv(fd/"training_log.csv",index=False);json.dump({"complete":True,"held_case":case,"mode":mode,"seed":seed,"n":len(pred)},open(done,"w"),indent=2);del model;torch.cuda.empty_cache()
            lg["fold"]=fi;lg["held_case"]=case;all_logs.append(lg);row[f"{mode}_mean"]=pred.final_error_px.mean();row[f"{mode}_candidate_mean"]=pred.candidate_error_px.mean()
        foldsum.append(row);pd.DataFrame(foldsum).to_csv(out/"oof_fold_summary.csv",index=False)
    oc=pd.concat([pd.read_csv(out/"folds"/f"fold_{i}_{c}_control"/"predictions.csv") for i,c in enumerate(cases,1)],ignore_index=True)
    ot=pd.concat([pd.read_csv(out/"folds"/f"fold_{i}_{c}_temporal"/"predictions.csv") for i,c in enumerate(cases,1)],ignore_index=True)
    oc.to_csv(out/"oof_control_predictions.csv",index=False);ot.to_csv(out/"oof_temporal_predictions.csv",index=False)
    oh=add_historical(tw,root/cfg["phase6g_dir"]/"oof_predictions.csv");
    if oh.hist_error_px.isna().any():raise RuntimeError("Historical Phase6G OOF match incomplete")
    comp=comparison(oc,ot,oh);comp.to_csv(out/"oof_comparison.csv",index=False);bc=by_case(oc,ot,oh);bc.to_csv(out/"oof_metrics_by_case.csv",index=False)
    safe=safety(oc,ot);safe.to_csv(out/"oof_easy_case_safety.csv",index=False);hard=rescue(oc,ot);hard.to_csv(out/"oof_hard_case_rescue.csv",index=False);pers,segs=persistent(oc,ot);pers.to_csv(out/"oof_persistent_harmful.csv",index=False);segs.to_csv(out/"oof_persistent_segments.csv",index=False)
    conf=confidence(oc,ot,"OOF");traj=trajectory(oc,ot,"OOF")
    cm=comp.set_index("method");cb=bc.pivot(index="case_id",columns="method",values="mean");delta=cb["TEMPORAL alpha=.75"]-cb["CONTROL alpha=.75"]
    wins=int((delta<0).sum());losses=int((delta>0).sum());ties=int((np.isclose(delta,0)).sum())
    cp=pers.set_index("method"); better=cm.loc["TEMPORAL alpha=.75","case_macro_mean_px"]<cm.loc["CONTROL alpha=.75","case_macro_mean_px"]
    noninfer=wins+ties>=5; noharm=cp.loc["TEMPORAL alpha=.75","harmful_frame_pct"]<=cp.loc["CONTROL alpha=.75","harmful_frame_pct"]+0.5 and cp.loc["TEMPORAL alpha=.75","persistent_ge3_frames"]<=cp.loc["CONTROL alpha=.75","persistent_ge3_frames"]
    if better and noninfer and noharm: decision="A"
    elif better: decision="B"
    else: decision="C"
    dec={"frozen_before_val":True,"decision":decision,"control_case_macro_mean_px":cm.loc["CONTROL alpha=.75","case_macro_mean_px"],"temporal_case_macro_mean_px":cm.loc["TEMPORAL alpha=.75","case_macro_mean_px"],
         "control_p95_px":cm.loc["CONTROL alpha=.75","p95"],"temporal_p95_px":cm.loc["TEMPORAL alpha=.75","p95"],"control_harmful_gt10_pct":cm.loc["CONTROL alpha=.75","harmful_gt10_pct"],"temporal_harmful_gt10_pct":cm.loc["TEMPORAL alpha=.75","harmful_gt10_pct"],
         "control_persistent_ge3_frames":int(cp.loc["CONTROL alpha=.75","persistent_ge3_frames"]),"temporal_persistent_ge3_frames":int(cp.loc["TEMPORAL alpha=.75","persistent_ge3_frames"]),"wins":wins,"losses":losses,"ties":ties,"rule":"Prespecified Phase7B A/B/C rule"}
    json.dump(dec,open(out/"oof_decision.json","w"),indent=2);log(f"[10/14] OOF paired analysis complete | wins/losses/ties={wins}/{losses}/{ties}");log(f"[11/14] OOF conclusion FROZEN: {decision}")
    GLOBAL.update({"oc":oc,"ot":ot,"oh":oh});logs=pd.concat(all_logs,ignore_index=True);plot_basic(out,comp,bc,safe,hard,pers,conf,logs);montage(out,tw,out/"cache/train_stacks.npy",oc,ot,segs)
    # Full train CONTROL then TEMPORAL; Val is not read until both are frozen.
    final_logs={};final_models={};final_seed=cfg["seed"]
    for step,(mode,ds) in enumerate([("control",ds_c),("temporal",ds_t)],12):
        log(f"[{step}/14] Full-Train {mode.upper()} (2760 windows, fixed 3 epochs)");ckp=out/"checkpoints"/f"final_{mode}.pth";lp=out/f"final_{mode}_training_log.csv"
        if ckp.exists() and lp.exists():
            model=create_model(p5,final_seed,device);model.load_state_dict(torch.load(ckp,map_location=device,weights_only=False)["model_state_dict"]);lg=pd.read_csv(lp);log("  reuse frozen full-Train checkpoint")
        else:
            model,lg=train_model(p5,ds,list(range(len(tw))),final_seed,device,mode.upper(),3,16,log);torch.save({"model_state_dict":model.state_dict(),"mode":mode,"seed":final_seed,"epochs":3,"val_unseen":True},ckp);lg.to_csv(lp,index=False)
        final_models[mode]=model;final_logs[mode]=lg
    log("[14/14] One-shot matched Val comparison (920 legal windows)")
    vdc=WindowDataset(vw,out/"cache/val_stacks.npy","control");vdt=WindowDataset(vw,out/"cache/val_stacks.npy","temporal")
    vc=infer(final_models["control"],vdc,list(range(len(vw))),device,.75,16);vt=infer(final_models["temporal"],vdt,list(range(len(vw))),device,.75,16)
    vc.to_csv(out/"val_control_predictions.csv",index=False);vt.to_csv(out/"val_temporal_predictions.csv",index=False)
    vh=add_historical(vw,root/cfg["phase6g_dir"]/"val_predictions.csv");
    if vh.hist_error_px.isna().any():raise RuntimeError("Historical Phase6G Val match incomplete")
    vcomp=comparison(vc,vt,vh);vcomp.to_csv(out/"val_comparison.csv",index=False);vbc=by_case(vc,vt,vh);vbc.to_csv(out/"val_by_case.csv",index=False)
    vs=safety(vc,vt);vs.to_csv(out/"val_easy_case_safety.csv",index=False);vr=rescue(vc,vt);vr.to_csv(out/"val_hard_case_rescue.csv",index=False);vp,vsegs=persistent(vc,vt);vp.to_csv(out/"val_persistent_harmful.csv",index=False);vsegs.to_csv(out/"val_persistent_segments.csv",index=False)
    pd.concat([conf,confidence(vc,vt,"Val")],ignore_index=True).to_csv(out/"confidence_comparison.csv",index=False);pd.concat([traj,trajectory(vc,vt,"Val")],ignore_index=True).to_csv(out/"trajectory_summary.csv",index=False);val_plots(out,vc,vt,vh)
    # README after frozen Val evaluation.
    vcm=vcomp.set_index("method");vp2=vp.set_index("method")
    lines=["# Phase 7B — Strict Case-OOF Causal 3-Frame HRNet","","Completed without Test data or new labels. The only scientific variable was channel content: CONTROL `[I_t,I_t,I_t]` versus TEMPORAL `[I_t-2,I_t-1,I_t]`. Both used the same raw_t-centered 256 crop, ImageNet initialization, seed/order, weighted MSE, optimizer, three epochs, and frozen alpha=0.75.","",
      "The OOF decision was saved to `oof_decision.json` before either final Val prediction was generated. No Val result changed epochs, alpha, preprocessing, ROI, input ordering, or decision rule.","","## Strict 8-case OOF","",
      f"- CONTROL alpha=.75 case-macro mean: {cm.loc['CONTROL alpha=.75','case_macro_mean_px']:.3f}px.",f"- TEMPORAL alpha=.75 case-macro mean: {cm.loc['TEMPORAL alpha=.75','case_macro_mean_px']:.3f}px.",f"- Case wins/losses/ties: {wins}/{losses}/{ties}.",
      f"- Frozen OOF decision: **{decision}**.","","## Matched Val (920 legal targets)","","| Method | Mean | Median | P95 | harmful >10 |","|---|---:|---:|---:|---:|"]
    for method in ["Raw","Historical Phase6G alpha=.75","CONTROL alpha=.75","TEMPORAL alpha=.75"]:
        r=vcm.loc[method];lines.append(f"| {method} | {r['mean']:.3f} | {r['median']:.3f} | {r['p95']:.3f} | {r['harmful_gt10_pct']:.3f}% |")
    lines += ["","## Important interpretation constraints","","Frames inside a Series are correlated; no frame-level independent-patient significance claim was made. Primary evidence is the 8-case OOF case-macro result and case win/loss pattern.","",
      "The first two frames of every Series are not predicted: no temporal padding or frame duplication was used. Complete-sequence handling is a later deployment decision.","",
      "TEMPORAL channels represent time rather than RGB, so ImageNet first-layer RGB semantics are imperfect. A negative result would reject this simple channel-stacking implementation as a reliable mainline, not prove that all temporal visual information is useless.","",
      "No gate, smoothing, temporal ablation, alternate initialization, jitter/noise, 384 ROI, structural annotation, or Test evaluation was run."]
    (out/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    json.dump({"status":"complete","oof_decision":decision,"oof_windows":len(oc),"val_windows":len(vc),"test_used":False},open(out/"run_status.json","w"),indent=2)
    log("[Done] Phase 7B complete. STOP.");log(f"Outputs: {out}")


if __name__=="__main__": main()
