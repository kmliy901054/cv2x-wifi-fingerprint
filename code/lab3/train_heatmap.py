"""Heatmap output + free-cell mask Set Transformer (architectural upgrade).

Replaces MDN head with 2-D heatmap over a 40x33 grid of 0.4 m cells covering
the active lab area.  Free-cell mask from psquare.pgm zeros impossible cells
before softmax-normalization; prediction = expected (x, y).

Loss = 0.5 · CE(Gaussian-smoothed soft label)  +  0.5 · SmoothL1(E[xy], y)
Predict = (softmax(logits) * free_mask).normalize() @ cell_xy

Combined with the existing 5000 GP synthetic samples on Split A.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import data
import models
import synthetic

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

OUT_DIR = Path(__file__).parent / 'outputs'
PRED_DIR = OUT_DIR / 'predictions'
CKPT_DIR = OUT_DIR / 'checkpoints'

CFG = dict(embed_dim=48, model_dim=192, num_heads=4, num_sab=3, dropout=0.3)
WEIGHT_DECAY = 1e-3
JITTER_DBM = 4.0
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 300
PATIENCE = 50
SEEDS = [42, 43, 44, 45, 46]
MAX_APS = 50
MIN_BSSID_COUNT = 10
N_SYNTHETIC = 5000

LABEL_SIGMA = 0.5    # m — Gaussian smoothing for soft target
CE_WEIGHT = 0.5      # weight of CE vs SmoothL1 in mixed loss
USE_SYNTH = True


def loc_err(y_pred, y_true):
    return np.linalg.norm(y_pred - y_true, axis=1)


def loc_stats(err):
    return {
        'median_err_m': float(np.median(err)),
        'mean_err_m': float(err.mean()),
        'p90_err_m': float(np.percentile(err, 90)),
        'p10_err_m': float(np.percentile(err, 10)),
        'max_err_m': float(err.max()),
    }


def train_one(model, train_loader, val_loader, device, tag):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_val = float('inf'); best_state = None; bad = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for idx, val, mask, y in train_loader:
            idx = idx.to(device); val = val.to(device)
            mask = mask.to(device); y = y.to(device)
            opt.zero_grad()
            logits = model(idx, val, mask)
            loss = model.loss(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        model.eval()
        vloss = 0.0; vn = 0
        with torch.no_grad():
            for idx, val, mask, y in val_loader:
                idx = idx.to(device); val = val.to(device)
                mask = mask.to(device); y = y.to(device)
                logits = model(idx, val, mask)
                vloss += float(model.loss(logits, y).item()) * y.size(0)
                vn += y.size(0)
        vloss /= max(1, vn)
        if vloss < best_val - 1e-4:
            best_val = vloss; bad = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f'  [{tag}] early stop @ epoch {epoch}, '
                      f'best val = {best_val:.4f}', flush=True)
                break
        if epoch == 1 or epoch % 40 == 0:
            print(f'  [{tag}] epoch {epoch:3d}  val_loss {vloss:.4f}  '
                  f'best {best_val:.4f}', flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}', flush=True)

    print('[load] real records ...', flush=True)
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=MIN_BSSID_COUNT)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess)
    tr_A, te_A = splits['A_random']
    set_idx, set_val, set_mask = data.build_set_input(records, bssids,
                                                        max_aps=MAX_APS)

    # Build heatmap grid + free mask
    cell_xy, Gw, Gh = data.build_heatmap_grid()
    free_mask = data.build_free_mask(cell_xy)
    print(f'[grid] {Gw} x {Gh} = {len(cell_xy)} cells, '
          f'{int(free_mask.sum())}/{len(cell_xy)} free '
          f'({100 * free_mask.mean():.1f}%)', flush=True)

    # Optional GP synth augmentation
    if USE_SYNTH:
        print(f'\n[gen] fitting GPs + sampling {N_SYNTHETIC} synth ...', flush=True)
        t0 = time.time()
        train_records = [{'x': float(y[i, 0]), 'y': float(y[i, 1]),
                           'session': sess[i],
                           'aps': {b: float(X[i, j])
                                    for j, b in enumerate(bssids) if X[i, j] > -99.5}}
                          for i in tr_A]
        gps = synthetic.fit_per_ap_gps(train_records, bssids, min_samples=15,
                                        verbose=False)
        real_xy = np.array([(r['x'], r['y']) for r in train_records],
                            dtype=np.float32)
        free_positions = synthetic.free_cell_positions(near_real_only=real_xy)
        rng = np.random.RandomState(42)
        chosen = rng.choice(len(free_positions), N_SYNTHETIC, replace=False)
        positions = free_positions[chosen]
        X_synth, y_synth = synthetic.synthesize(gps, positions, bssids,
                                                  records=train_records,
                                                  detect_knn_k=10,
                                                  fallback_threshold=-85.0,
                                                  unc_threshold=10.0, seed=42)
        print(f'[gen] done ({time.time() - t0:.1f}s)', flush=True)
        synth_idx, synth_val, synth_mask = synthetic.build_set_form(
            X_synth, bssids, max_aps=MAX_APS)
        idx_tr = np.concatenate([set_idx[tr_A], synth_idx], axis=0)
        val_tr = np.concatenate([set_val[tr_A], synth_val], axis=0)
        mask_tr = np.concatenate([set_mask[tr_A], synth_mask], axis=0)
        y_tr = np.concatenate([y[tr_A], y_synth.astype(np.float32)], axis=0)
        print(f'[setup] combined train: {len(y_tr)} '
              f'({len(tr_A)} real + {N_SYNTHETIC} synth)', flush=True)
    else:
        idx_tr = set_idx[tr_A]; val_tr = set_val[tr_A]; mask_tr = set_mask[tr_A]
        y_tr = y[tr_A]

    test_ds = data.SetDataset(set_idx[te_A], set_val[te_A], set_mask[te_A],
                                y[te_A], jitter=0.0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    seed_preds = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f'\n========== seed {seed} ==========', flush=True)
        train_ds = data.SetDataset(idx_tr, val_tr, mask_tr, y_tr,
                                     jitter=JITTER_DBM)
        tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        model = models.SetTransformerHeatmap(
            num_bssids=len(bssids),
            cell_xy=cell_xy,
            free_mask=free_mask.astype(np.float32),
            label_sigma=LABEL_SIGMA,
            ce_weight=CE_WEIGHT,
            **CFG)
        if seed == SEEDS[0]:
            n_params = sum(p.numel() for p in model.parameters()
                            if p.requires_grad)
            print(f'[model] params: {n_params:,} ({n_params / 1000:.1f}k)',
                  flush=True)
        model, vloss = train_one(model, tl, test_loader, device, f's{seed}')
        model.eval()
        with torch.no_grad():
            idx_t = torch.from_numpy(set_idx[te_A]).to(device)
            v_t = torch.from_numpy(set_val[te_A]).to(device)
            m_t = torch.from_numpy(set_mask[te_A]).to(device)
            pred = model.predict_xy(idx_t, v_t, m_t)
        err = loc_err(pred, y[te_A])
        print(f'  seed {seed}: median={np.median(err):.3f}  '
              f'mean={err.mean():.3f}  '
              f'p90={np.percentile(err, 90):.3f}  '
              f'val={vloss:.3f}', flush=True)
        seed_preds.append({'pred': pred, 'val_loss': vloss, 'err': err})
        torch.save(model.state_dict(),
                    CKPT_DIR / f'A_random__Heatmap_s{seed}.pt')

    # Best single + Ensemble
    best_i = int(np.argmin([s['val_loss'] for s in seed_preds]))
    bs = loc_stats(seed_preds[best_i]['err'])
    print(f'\n========== Best single (seed {SEEDS[best_i]}) ==========',
          flush=True)
    print(f'  median={bs["median_err_m"]:.3f}  '
          f'mean={bs["mean_err_m"]:.3f}  '
          f'p90={bs["p90_err_m"]:.3f}', flush=True)

    ens_pred = np.mean([s['pred'] for s in seed_preds], axis=0)
    ens_err = loc_err(ens_pred, y[te_A])
    es = loc_stats(ens_err)
    print(f'\n========== ENSEMBLE (mean of {len(SEEDS)}) ==========',
          flush=True)
    print(f'  median={es["median_err_m"]:.3f}  '
          f'mean={es["mean_err_m"]:.3f}  '
          f'p90={es["p90_err_m"]:.3f}', flush=True)

    np.savez(PRED_DIR / 'A_random__HeatmapEnsemble.npz',
              pred=ens_pred, y_true=y[te_A], err=ens_err, test_idx=te_A)

    print('\n========== Updated Split A ladder ==========', flush=True)
    print(f'  KNN k=5:                                 1.568 m')
    print(f'  Set Transformer MDN (208k):              1.093 m')
    print(f'  Big Set Transformer MDN x 5-ens:         1.083 m')
    print(f'  Big + 5000 synth single:                 0.906 m')
    print(f'  Big + synth x 5-ensemble:                0.889 m  (prior best)')
    print(f'  Big + synth + C-Mixup x 5-ens:           0.942 m')
    print(f'  Heatmap + free-mask best single:         {bs["median_err_m"]:.3f} m')
    print(f'  Heatmap + free-mask x 5-ensemble:        {es["median_err_m"]:.3f} m')

    # Append metrics.csv
    rows = [{'split': 'A_random', 'model': 'HeatmapBestSingle',
              **bs, 'nll': None},
             {'split': 'A_random', 'model': 'HeatmapEnsemble',
              **es, 'nll': None}]
    csv = OUT_DIR / 'metrics.csv'
    if csv.exists():
        old = pd.read_csv(csv)
        old = old[~old['model'].isin(['HeatmapBestSingle',
                                        'HeatmapEnsemble'])]
        full = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    else:
        full = pd.DataFrame(rows)
    full.to_csv(csv, index=False)
    print(f'\n[save] -> metrics.csv', flush=True)


if __name__ == '__main__':
    main()
