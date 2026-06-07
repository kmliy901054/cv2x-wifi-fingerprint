"""Measure the D_stratified split for real (was only an estimate in the deck).

Same Cascade-aggressive recipe as Split A's headline: GP-synth fit on D's TRAIN
only, 5 seeds, geometric-median ensemble. Saves predictions + prints stats so we
can show A vs D honestly (both in-distribution; A=0.752 is the committed number).
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import data, models, synthetic

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PRED = Path(__file__).parent / 'outputs' / 'predictions'
CFG = dict(embed_dim=48, model_dim=192, num_heads=4, num_sab=3, dropout=0.3)
WD=1e-3; JIT=4.0; BS=64; LR=1e-3; EP=300; PAT=50; SEEDS=[42,43,44,45,46]
MAXAP=50; NSYN=5000
FS=0.25; CS=1.0; CFW=0.3; CCW=0.15; MW=0.55


def gm(p,eps=1e-5,it=100):
    m=p.mean(0)
    for _ in range(it):
        d=np.clip(np.linalg.norm(p-m[None],axis=-1),eps,None); w=1/d; w/=w.sum(0,keepdims=True)
        n=(w[:,:,None]*p).sum(0)
        if np.max(np.linalg.norm(n-m,axis=-1))<eps: break
        m=n
    return m


def train_one(model,tl,vl,dev,tag):
    model=model.to(dev); opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WD)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EP); best=1e9; bs=None; bad=0
    for ep in range(1,EP+1):
        model.train()
        for idx,val,mask,y in tl:
            idx,val,mask,y=idx.to(dev),val.to(dev),mask.to(dev),y.to(dev)
            opt.zero_grad(); loss=model.loss(model(idx,val,mask),y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        sch.step(); model.eval(); v=0;n=0
        with torch.no_grad():
            for idx,val,mask,y in vl:
                idx,val,mask,y=idx.to(dev),val.to(dev),mask.to(dev),y.to(dev)
                v+=float(model.loss(model(idx,val,mask),y).item())*y.size(0); n+=y.size(0)
        v/=max(1,n)
        if v<best-1e-4: best=v; bad=0; bs={k:vv.detach().cpu().clone() for k,vv in model.state_dict().items()}
        else:
            bad+=1
            if bad>=PAT: break
        if ep==1 or ep%40==0: print(f'  [{tag}] ep{ep} val {v:.4f} best {best:.4f}',flush=True)
    if bs: model.load_state_dict(bs)
    return model,best


def main():
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device',dev,flush=True)
    records=data.load_records(); bssids=data.build_bssid_vocab(records,min_count=10)
    X,y,sess=data.build_arrays(records,bssids); splits=data.make_splits(sess)
    tr,te=splits['D_stratified']
    print(f'D split: train {len(tr)} / test {len(te)}',flush=True)
    si,sv,sm=data.build_set_input(records,bssids,max_aps=MAXAP)
    fxy,_,_=data.build_heatmap_grid(cell_size=0.4); fmask=data.build_free_mask(fxy)
    cxy,_,_=data.build_heatmap_grid(cell_size=1.6); cmask=data.build_free_mask(cxy)
    f2c=data.build_fine_to_coarse(fine_cell=0.4,coarse_cell=1.6)

    print('[gen] GP synth on D-train ...',flush=True); t0=time.time()
    trecs=[{'x':float(y[i,0]),'y':float(y[i,1]),'session':sess[i],
            'aps':{b:float(X[i,j]) for j,b in enumerate(bssids) if X[i,j]>-99.5}} for i in tr]
    gps=synthetic.fit_per_ap_gps(trecs,bssids,min_samples=15,verbose=False)
    rxy=np.array([(r['x'],r['y']) for r in trecs],dtype=np.float32)
    fp=synthetic.free_cell_positions(near_real_only=rxy); rng=np.random.RandomState(42)
    ch=rng.choice(len(fp),min(NSYN,len(fp)),replace=False)
    Xs,ys=synthetic.synthesize(gps,fp[ch],bssids,records=trecs,detect_knn_k=10,
                                fallback_threshold=-85.0,unc_threshold=10.0,seed=42)
    print(f'[gen] {time.time()-t0:.0f}s',flush=True)
    gi,gv,gm_=synthetic.build_set_form(Xs,bssids,max_aps=MAXAP)
    itr=np.concatenate([si[tr],gi]); vtr=np.concatenate([sv[tr],gv])
    mtr=np.concatenate([sm[tr],gm_]); ytr=np.concatenate([y[tr],ys.astype(np.float32)])

    tl_v=DataLoader(data.SetDataset(si[te],sv[te],sm[te],y[te],jitter=0.0),batch_size=256)
    preds=[]
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s)
        print(f'== seed {s} ==',flush=True)
        tl=DataLoader(data.SetDataset(itr,vtr,mtr,ytr,jitter=JIT),batch_size=BS,shuffle=True)
        m=models.SetTransformerHeatmapCascade(num_bssids=len(bssids),
            fine_cell_xy=fxy,fine_free_mask=fmask.astype(np.float32),
            coarse_cell_xy=cxy,coarse_free_mask=cmask.astype(np.float32),
            fine_to_coarse=f2c,fine_sigma=FS,coarse_sigma=CS,
            ce_fine_w=CFW,ce_coarse_w=CCW,mse_w=MW,**CFG)
        m,_=train_one(m,tl,tl_v,dev,f's{s}'); m.eval()
        with torch.no_grad():
            pr=m.predict_xy(torch.from_numpy(si[te]).to(dev),torch.from_numpy(sv[te]).to(dev),
                            torch.from_numpy(sm[te]).to(dev))
        e=np.linalg.norm(pr-y[te],axis=1)
        print(f'  seed {s}: median {np.median(e):.3f} mean {e.mean():.3f} p90 {np.percentile(e,90):.3f}',flush=True)
        preds.append(pr)
    pts=np.stack(preds,0); pred=gm(pts); e=np.linalg.norm(pred-y[te],axis=1)
    print(f'\n=== D_stratified Cascade-aggressive 5-seed geom-median ===')
    print(f'  median {np.median(e):.3f}  mean {e.mean():.3f}  p90 {np.percentile(e,90):.3f}  n={len(te)}',flush=True)
    np.savez(PRED/'D_stratified__CascadeAggressiveEnsemble.npz',
             pred=pred,y_true=y[te],err=e,test_idx=te)
    print('[save] D_stratified__CascadeAggressiveEnsemble.npz',flush=True)


if __name__=='__main__':
    main()
