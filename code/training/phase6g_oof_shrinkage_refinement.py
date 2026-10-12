from __future__ import annotations

import argparse, importlib.util, json, logging, math, random, traceback
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]


def load_phase6b():
    p = ROOT / "scripts" / "__pycache__" / "phase6b_local_refinement_256_baseline.cpython-313.pyc"
    spec = importlib.util.spec_from_file_location("phase6b_runtime", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def q(a, p): return float(np.percentile(np.asarray(a,float),p))


def metrics(err, raw=None):
    e=np.asarray(err,float); d={"n":len(e),"mean_px":float(e.mean()),"median_px":float(np.median(e)),"p95_px":q(e,95),"p99_px":q(e,99),"max_px":float(e.max())}
    for t in (5,10,20,40): d[f"lt_{t}px_pct"]=float((e<t).mean()*100)
    if raw is not None:
        r=np.asarray(raw,float); g=r-e
        d.update(mean_gain_px=float(g.mean()),median_gain_px=float(np.median(g)),improved_pct=float((g>0).mean()*100),worsened_pct=float((g<0).mean()*100))
        for t in (5,10,20): d[f"harmful_gt_{t}px_pct"]=float((g < -t).mean()*100)
    return d


def alpha_eval(df, alpha):
    px=df.raw_x + alpha*(df.model_x-df.raw_x); py=df.raw_y + alpha*(df.model_y-df.raw_y)
    return np.hypot(px-df.gt_x,py-df.gt_y)


def persistent(df, error_col, method):
    rows=[]; harmful=(df[error_col]-df.raw_error_px)>10
    tmp=df.assign(_harmful=harmful).sort_values(["case_id","series_id","frame_index"])
    for (cc,sid,seq),g in tmp.groupby(["case_id","series_id","sequence_identifier"],sort=False):
        run=[]
        for _,r in g.iterrows():
            if r._harmful: run.append(int(r.frame_index))
            elif run:
                rows.append((method,str(cc),str(sid),str(seq),run[0],run[-1],len(run))); run=[]
        if run: rows.append((method,str(cc),str(sid),str(seq),run[0],run[-1],len(run)))
    n=int(harmful.sum()); lens=[x[-1] for x in rows]
    return rows,{"method":method,"harmful_gt10_frames":n,"isolated_or_short_frames":sum(x for x in lens if x<3),"persistent_ge3_frames":sum(x for x in lens if x>=3),"segments_ge3":sum(x>=3 for x in lens),"segments_ge5":sum(x>=5 for x in lens),"segments_ge10":sum(x>=10 for x in lens),"max_segment_length":max(lens,default=0)}


def build_model(p6b, basecfg, device, seed):
    seed_all(seed)
    model=p6b.phase5a.build_pretrained_model(basecfg,device)
    return model


def train_fold(p6b, basecfg, cfg, all_ds, train_df, case, fold_id, device, out, logger):
    fit_idx=train_df.index[train_df.case_code.astype(str)!=case].tolist(); hold_idx=train_df.index[train_df.case_code.astype(str)==case].tolist()
    fit_df=train_df.loc[fit_idx].reset_index(drop=True); hold_df=train_df.loc[hold_idx].reset_index(drop=True)
    fd=out/"folds"/f"fold_{fold_id:02d}_{case}"; fd.mkdir(parents=True,exist_ok=True)
    done=fd/"complete.json"
    if done.exists() and (fd/"oof_predictions.csv").exists():
        print(f"[Fold {fold_id}/8] Resume completed case {case}"); return json.loads(done.read_text()),pd.read_csv(fd/"oof_predictions.csv")
    model=build_model(p6b,basecfg,device,int(cfg["seed"])+fold_id)
    crit=p6b.build_criterion(basecfg); opt=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.0); sched=torch.optim.lr_scheduler.MultiStepLR(opt,[120,160],gamma=.2)
    fit_loader=DataLoader(Subset(all_ds,fit_idx),batch_size=16,shuffle=True,num_workers=0,pin_memory=True)
    hold_loader=DataLoader(Subset(all_ds,hold_idx),batch_size=16,shuffle=False,num_workers=0,pin_memory=True)
    best=math.inf; best_epoch=0; patience=0; logs=[]; ck=fd/"best.pth"
    for epoch in range(1,int(cfg["max_epochs"])+1):
        tr=p6b.train_epoch(model,fit_loader,opt,crit,device=device,original_hw=(256,256),heatmap_hw=(64,64),epoch=epoch,max_epochs=int(cfg["max_epochs"]),gradient_explosion_norm=10000.0)
        va,_=p6b.validate_epoch(model,hold_loader,crit,device=device,manifest=hold_df,original_hw=(256,256),heatmap_hw=(64,64),epoch_label=f"Fold {fold_id} held",collect_predictions=False)
        valmean=float(va["mean_error_px"]); improved=valmean < best-float(cfg["early_stopping_min_delta_px"])
        if improved:
            best=valmean;best_epoch=epoch;patience=0;torch.save({"model_state_dict":model.state_dict(),"epoch":epoch,"held_case":case},ck)
        else: patience+=1
        row={"fold":fold_id,"held_case":case,"epoch":epoch,"lr":opt.param_groups[0]["lr"],"train_loss":tr["loss"],"train_mean_error_px":tr["mean_error_px"],"heldout_loss":va["loss"],"heldout_mean_error_px":valmean,"heldout_median_error_px":va["median_error_px"],"heldout_p95_error_px":va["p95_error_px"],"is_best":improved,"early_stop_counter":patience};logs.append(row);pd.DataFrame(logs).to_csv(fd/"training_log.csv",index=False)
        print(f"[Fold {fold_id}/8] E{epoch:03d} fit={tr['mean_error_px']:.2f}px held={valmean:.2f}px best={best:.2f}@{best_epoch} patience={patience}/15")
        sched.step()
        if patience>=int(cfg["early_stopping_patience"]): break
    state=torch.load(ck,map_location=device,weights_only=False);model.load_state_dict(state["model_state_dict"])
    fit_dx=(fit_df.true_tip_col-fit_df.raw_col).mean();fit_dy=(fit_df.true_tip_row-fit_df.raw_row).mean();base=p6b.base_predictions(hold_df,(fit_dx,fit_dy))
    pred=p6b.inference(model,Subset(all_ds,hold_idx),device,"OOF",base,int(cfg["peak_suppression_radius"]))
    pred=pred.rename(columns={"pred_x_original":"model_x","pred_y_original":"model_y","heatmap_peak_value":"peak","peak_margin":"margin"})
    pred["fold_id"]=fold_id;pred["held_case"]=case;pred["correction_dx"]=pred.model_x-pred.raw_x;pred["correction_dy"]=pred.model_y-pred.raw_y;pred["correction_magnitude"]=np.hypot(pred.correction_dx,pred.correction_dy)
    pred.to_csv(fd/"oof_predictions.csv",index=False)
    info={"fold":fold_id,"held_case":case,"fit_cases":sorted(fit_df.case_code.astype(str).unique().tolist()),"fit_frames":len(fit_df),"heldout_frames":len(hold_df),"best_epoch":best_epoch,"stop_epoch":epoch,"best_heldout_mean_px":best,"fresh_imagenet_initialization":True,"old_tip_checkpoint_loaded":False};done.write_text(json.dumps(info,indent=2),encoding="utf-8")
    del model,opt,fit_loader,hold_loader;torch.cuda.empty_cache();return info,pred


def make_alpha_tables(oof, alphas, out):
    frame=[];bycase=[]
    for a in alphas:
        e=alpha_eval(oof,a); oof[f"error_alpha_{a:g}"]=e
        case_means=[]
        for cc,g in oof.assign(_e=e).groupby("case_id",sort=False):
            z=metrics(g._e,g.raw_error_px);z.update(alpha=a,case_id=str(cc));bycase.append(z);case_means.append(z["mean_px"])
        z=metrics(e,oof.raw_error_px);z.update(alpha=a,case_macro_mean_error_px=float(np.mean(case_means)));frame.append(z)
    fm=pd.DataFrame(frame);bc=pd.DataFrame(bycase);fm.to_csv(out/"oof_alpha_metrics.csv",index=False);bc.to_csv(out/"oof_alpha_by_case.csv",index=False)
    rows=[]
    for lo,hi in [(0,5),(5,10),(20,40),(40,60)]:
        g=oof[(oof.raw_error_px>=lo)&(oof.raw_error_px<hi)]
        for a in alphas:
            z=metrics(alpha_eval(g,a),g.raw_error_px);z.update(raw_bin=f"{lo}-{hi}",alpha=a,raw_mean_px=float(g.raw_error_px.mean()));rows.append(z)
    pd.DataFrame(rows).to_csv(out/"oof_alpha_by_raw_bin.csv",index=False)
    best=fm.sort_values(["case_macro_mean_error_px","alpha"],ascending=[True,True]).iloc[0];sel=fm.copy();sel["is_selected"]=sel.alpha.eq(best.alpha);sel["selection_metric"]="case_macro_mean_error_px";sel["selected_before_val"]=True;sel.to_csv(out/"alpha_selection.csv",index=False)
    return oof,fm,bc,float(best.alpha)


def final_train(p6b,basecfg,cfg,all_ds,train_df,epochs,device,out):
    model=build_model(p6b,basecfg,device,int(cfg["seed"])+1000);crit=p6b.build_criterion(basecfg);opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.0);sched=torch.optim.lr_scheduler.MultiStepLR(opt,[120,160],gamma=.2)
    loader=DataLoader(all_ds,batch_size=16,shuffle=True,num_workers=0,pin_memory=True);rows=[]
    for ep in range(1,epochs+1):
        tr=p6b.train_epoch(model,loader,opt,crit,device=device,original_hw=(256,256),heatmap_hw=(64,64),epoch=ep,max_epochs=epochs,gradient_explosion_norm=10000.0);rows.append({"epoch":ep,"lr":opt.param_groups[0]["lr"],"train_loss":tr["loss"],"train_mean_error_px":tr["mean_error_px"],"train_median_error_px":tr["median_error_px"]});pd.DataFrame(rows).to_csv(out/"final_training_log.csv",index=False);print(f"[Final] E{ep}/{epochs} loss={tr['loss']:.6f} mean={tr['mean_error_px']:.2f}px");sched.step()
    (out/"checkpoints").mkdir(exist_ok=True);torch.save({"model_state_dict":model.state_dict(),"epochs":epochs,"selected_without_val":True},out/"checkpoints"/"final_full_train.pth");return model


def figures(alpha_metrics,bycase,val,alpha,out):
    f=out/"figures";f.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,5));ax.plot(alpha_metrics.alpha,alpha_metrics.case_macro_mean_error_px,'o-',label='case-macro mean');ax.plot(alpha_metrics.alpha,alpha_metrics.mean_px,'o-',label='frame mean');ax.plot(alpha_metrics.alpha,alpha_metrics.median_px,'o-',label='median');ax.plot(alpha_metrics.alpha,alpha_metrics.p95_px,'o-',label='P95');ax2=ax.twinx();ax2.plot(alpha_metrics.alpha,alpha_metrics.harmful_gt_10px_pct,'s--',color='crimson',label='harmful >10%');ax.axvline(alpha,color='k',ls=':');ax.set_xlabel('alpha');ax.set_ylabel('Error px');ax2.set_ylabel('Harmful >10 (%)');ax.legend(loc='upper left');ax2.legend(loc='upper right');fig.tight_layout();fig.savefig(f/"oof_alpha_curve.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6));
    for cc,g in bycase.groupby('case_id'):ax.plot(g.alpha,g.mean_px,'o-',label=str(cc))
    ax.set_xlabel('alpha');ax.set_ylabel('OOF case mean error px');ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(f/"oof_alpha_by_case.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));
    for col,label in [('raw_error_px','raw'),('global_mean_error_px','global mean'),('phase6b_error_px','6B'),('phase6f_error_px','6F'),('phase6g_error_px','6G')]:
        x=np.sort(val[col]);ax.plot(x,np.arange(1,len(x)+1)/len(x),label=label)
    ax.set_xlim(0,100);ax.set_xlabel('Val error px');ax.set_ylabel('CDF');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(f/"val_error_cdf.png",dpi=180);plt.close(fig)
    vc=pd.read_csv(out/"val_by_case.csv");fig,ax=plt.subplots(figsize=(10,5));p=vc.pivot(index='case_id',columns='method',values='mean_px');p.plot.bar(ax=ax);ax.set_ylabel('Mean error px');fig.tight_layout();fig.savefig(f/"val_case_comparison.png",dpi=180);plt.close(fig)
    easy=pd.read_csv(out/"easy_case_safety.csv");fig,ax=plt.subplots(figsize=(9,5));easy.pivot(index='method',columns='raw_bin',values='harmful_gt_10px_pct').plot.bar(ax=ax);ax.set_ylabel('Harmful >10 (%)');fig.tight_layout();fig.savefig(f/"easy_case_safety.png",dpi=180);plt.close(fig)
    hard=pd.read_csv(out/"hard_case_rescue.csv");fig,ax=plt.subplots(figsize=(9,5));hard.pivot(index='method',columns='raw_bin',values='mean_gain_px').plot.bar(ax=ax);ax.set_ylabel('Mean gain px');fig.tight_layout();fig.savefig(f/"hard_case_rescue.png",dpi=180);plt.close(fig)
    for (cc,sid,seq),g in val.groupby(['case_id','series_id','sequence_identifier'],sort=False):
        g=g.sort_values('frame_index');fig,ax=plt.subplots(figsize=(12,4))
        for col,label in [('raw_error_px','raw'),('phase6b_error_px','6B'),('phase6f_error_px','6F'),('phase6g_error_px','6G')]:ax.plot(g.frame_index,g[col],label=label,lw=1)
        ax.set_xlabel('Frame');ax.set_ylabel('Error px');ax.set_title(f'case {cc} | {seq}');ax.legend(ncol=4);ax.grid(alpha=.2);fig.tight_layout();safe=str(seq).replace('/','-');fig.savefig(f/f"trajectory_{cc}_{safe}.png",dpi=180);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args();cfg=json.loads((ROOT/args.config).read_text());out=ROOT/cfg['output_dir'];out.mkdir(parents=True,exist_ok=True);(out/'folds').mkdir(exist_ok=True);(out/'logs').mkdir(exist_ok=True);(out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    try:
        p6b=load_phase6b();basecfg=json.loads((ROOT/cfg['phase6b_config']).read_text());device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');seed_all(int(cfg['seed']))
        print('[1/8] Verifying frozen data');manifest=pd.read_csv(ROOT/cfg['manifest']);train=manifest[manifest.split=='train'].copy().reset_index(drop=True);valmeta=manifest[manifest.split=='val'].copy().reset_index(drop=True);train.case_code=train.case_code.astype(str);valmeta.case_code=valmeta.case_code.astype(str)
        integrity={'train_frames':len(train),'train_cases':sorted(train.case_code.unique().tolist()),'train_series':int(train.series_uid.nunique()),'val_frames_reserved_not_loaded':len(valmeta),'val_cases_reserved_not_loaded':sorted(valmeta.case_code.unique().tolist()),'case_overlap':len(set(train.case_code)&set(valmeta.case_code)),'series_overlap':len(set(train.series_uid)&set(valmeta.series_uid)),'path_overlap':len(set(train.dicom_absolute_path)&set(valmeta.dicom_absolute_path)),'val_used_for_alpha_selection':False};(out/'split_integrity.json').write_text(json.dumps(integrity,indent=2),encoding='utf-8')
        if (len(train),train.case_code.nunique(),train.series_uid.nunique())!=(2784,8,12) or any(integrity[k] for k in ['case_overlap','series_overlap','path_overlap']):raise RuntimeError(f'Integrity check failed: {integrity}')
        logger=logging.getLogger('phase6g');logger.setLevel(logging.INFO);logger.addHandler(logging.StreamHandler());print(f'Device: {device} | {torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU"}')
        all_ds=p6b.LocalTipDataset(train,'Train frozen',(256,256),(64,64),2.0,3,True,logger)
        print('[2/8] OOF heatmap folds');summ=[];preds=[]
        for i,case in enumerate(sorted(train.case_code.unique()),1):info,pred=train_fold(p6b,basecfg,cfg,all_ds,train,case,i,device,out,logger);summ.append(info);preds.append(pred)
        pd.DataFrame(summ).to_csv(out/'oof_fold_summary.csv',index=False);oof=pd.concat(preds,ignore_index=True);oof.to_csv(out/'oof_predictions.csv',index=False)
        print('[3/8] OOF alpha evaluation');oof,am,bc,alpha=make_alpha_tables(oof,[0.,.25,.5,.75,1.],out);oof.to_csv(out/'oof_predictions.csv',index=False)
        print('[4/8] Case-macro alpha selection');print(am[['alpha','case_macro_mean_error_px','mean_px','median_px','p95_px','harmful_gt_10px_pct']].to_string(index=False))
        print(f'[5/8] Freeze alpha = {alpha:.2f} (OOF only)');best_epochs=pd.DataFrame(summ).best_epoch.astype(int).tolist();final_epochs=int(math.floor(float(np.median(best_epochs))+.5));(out/'frozen_design.json').write_text(json.dumps({'alpha_frozen':alpha,'alpha_candidates':[0,.25,.5,.75,1],'criterion':'lowest 8-case macro mean error','fold_best_epochs':best_epochs,'median_best_epoch':float(np.median(best_epochs)),'final_training_epochs':final_epochs,'selected_before_val':True},indent=2),encoding='utf-8')
        print(f'[6/8] Final full-Train heatmap | epochs={final_epochs}');model=final_train(p6b,basecfg,cfg,all_ds,train,final_epochs,device,out)
        if (out/'val_predictions.csv').exists():raise RuntimeError('One-shot Val guard: val_predictions.csv already exists; refusing repeated inference')
        print('[7/8] One-shot Val inference');val_ds=p6b.LocalTipDataset(valmeta,'Val one-shot',(256,256),(64,64),2.0,3,True,logger);dx=float((train.true_tip_col-train.raw_col).mean());dy=float((train.true_tip_row-train.raw_row).mean());base=p6b.base_predictions(valmeta,(dx,dy));v=p6b.inference(model,val_ds,device,'Val',base,6).rename(columns={'pred_x_original':'phase6b_x','pred_y_original':'phase6b_y','model_error_px':'phase6b_error_px','heatmap_peak_value':'peak','peak_margin':'margin'})
        v['phase6f_error_px']=pd.read_csv(ROOT/cfg['phase6f_val_predictions']).set_index('dicom_absolute_path').loc[v.dicom_path,'model_error_px'].to_numpy();v['phase6g_x']=v.raw_x+alpha*(v.phase6b_x-v.raw_x);v['phase6g_y']=v.raw_y+alpha*(v.phase6b_y-v.raw_y);v['phase6g_error_px']=np.hypot(v.phase6g_x-v.gt_x,v.phase6g_y-v.gt_y);v['correction_dx']=v.phase6b_x-v.raw_x;v['correction_dy']=v.phase6b_y-v.raw_y;v['correction_magnitude']=np.hypot(v.correction_dx,v.correction_dy);v['effective_correction_magnitude']=alpha*v.correction_magnitude;v.to_csv(out/'val_predictions.csv',index=False)
        print('[8/8] Safety / rescue comparison');methods={'System raw':'raw_error_px','Train-only global mean residual':'global_correction_error_px','Phase6B full alpha=1':'phase6b_error_px','Phase6F residual':'phase6f_error_px','Phase6G frozen shrinkage':'phase6g_error_px'};rows=[];cases=[]
        for name,col in methods.items():z=metrics(v[col],v.raw_error_px);z['method']=name;rows.append(z)
        pd.DataFrame(rows).to_csv(out/'val_metrics.csv',index=False)
        for cc,g in v.groupby('case_id',sort=False):
            for name,col in {'System raw':'raw_error_px','Phase6B':'phase6b_error_px','Phase6F':'phase6f_error_px','Phase6G':'phase6g_error_px'}.items():z=metrics(g[col],g.raw_error_px);z.update(case_id=str(cc),method=name);cases.append(z)
        pd.DataFrame(cases).to_csv(out/'val_by_case.csv',index=False);pd.DataFrame(cases).to_csv(out/'val_comparison.csv',index=False)
        er=[];hr=[]
        for lo,hi in [(0,5),(5,10)]:
            g=v[(v.raw_error_px>=lo)&(v.raw_error_px<hi)]
            for name,col in {'Raw':'raw_error_px','6B':'phase6b_error_px','6F':'phase6f_error_px','6G':'phase6g_error_px'}.items():z=metrics(g[col],g.raw_error_px);z.update(raw_bin=f'{lo}-{hi}',method=name);er.append(z)
        for lo,hi in [(20,40),(40,60)]:
            g=v[(v.raw_error_px>=lo)&(v.raw_error_px<hi)]
            for name,col in {'6B':'phase6b_error_px','6F':'phase6f_error_px','6G':'phase6g_error_px'}.items():z=metrics(g[col],g.raw_error_px);z.update(raw_bin=f'{lo}-{hi}',method=name,raw_mean_px=float(g.raw_error_px.mean()));hr.append(z)
        pd.DataFrame(er).to_csv(out/'easy_case_safety.csv',index=False);pd.DataFrame(hr).to_csv(out/'hard_case_rescue.csv',index=False)
        ps=[];segs=[]
        for name,col in [('Phase6B','phase6b_error_px'),('Phase6F','phase6f_error_px'),('Phase6G','phase6g_error_px')]:sg,sm=persistent(v,col,name);segs+=sg;ps.append(sm)
        pd.DataFrame(ps).to_csv(out/'persistent_failure_summary.csv',index=False);pd.DataFrame(segs,columns=['method','case_id','series_id','sequence_identifier','start_frame','end_frame','length']).to_csv(out/'persistent_failure_segments.csv',index=False)
        corr=[]
        for a in [0,.25,.5,.75,1]:x=a*oof.correction_magnitude.to_numpy();corr.append({'alpha':a,'median_px':np.median(x),'p90_px':q(x,90),'p95_px':q(x,95),'max_px':x.max()})
        pd.DataFrame(corr).to_csv(out/'correction_distribution.csv',index=False);figures(am,bc,v,alpha,out)
        rawm=rows[0];b6=rows[2];f6=rows[3];g6=rows[4];case_df=pd.DataFrame(cases);lines=['# Phase 6G — Case-OOF Conservative Heatmap Shrinkage','',f'Frozen alpha: **{alpha:.2f}**, selected before any final Val inference by the lowest 8-case OOF case-macro mean error.','',f'Fold best epochs: `{best_epochs}`; final full-Train epochs: **{final_epochs}**.','', '## OOF alpha selection','',am.to_markdown(index=False),'','## One-shot Val','',pd.DataFrame(rows)[['method','mean_px','median_px','p95_px','p99_px','max_px','lt_5px_pct','lt_10px_pct','lt_20px_pct','lt_40px_pct']].to_markdown(index=False),'','## Val by case','',case_df[['case_id','method','mean_px','median_px','p95_px']].to_markdown(index=False),'','## Interpretation','',f'Phase6G mean/median/P95 = {g6["mean_px"]:.3f}/{g6["median_px"]:.3f}/{g6["p95_px"]:.3f} px; raw = {rawm["mean_px"]:.3f}/{rawm["median_px"]:.3f}/{rawm["p95_px"]:.3f}; Phase6B = {b6["mean_px"]:.3f}/{b6["median_px"]:.3f}/{b6["p95_px"]:.3f}; Phase6F = {f6["mean_px"]:.3f}/{f6["median_px"]:.3f}/{f6["p95_px"]:.3f}.','', 'No adaptive alpha, confidence gate, temporal model, structural supervision, 384 ROI, or Test data was used. No next-stage experiment was started.'];(out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
        (out/'run_summary.json').write_text(json.dumps({'status':'completed','alpha_frozen':alpha,'fold_best_epochs':best_epochs,'final_epochs':final_epochs,'val_used_for_alpha_selection':False,'val_inference_count':1},indent=2),encoding='utf-8');print('[Done] Phase 6G complete; STOP.')
    except BaseException as e:
        (out/'phase6g_failure.json').write_text(json.dumps({'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()},indent=2),encoding='utf-8');raise


if __name__=='__main__':main()
