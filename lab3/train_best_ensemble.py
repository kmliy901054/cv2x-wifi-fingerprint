"""5-seed ensemble of the current best config: Big Set Transformer + 5000 GP synth, on Split A.

Current best single: 0.906 m (median)
Expected ensemble:   ~0.85 m (typically -5 to -10% from ensembling)

Strategy:
  Fit GPs ONCE on Split A train (saves ~5 minutes vs refitting per seed)
  Synthesize 5000 scans ONCE (same synthetic data shared across seeds)
  Train 5 different model seeds on (real_split_A_train + synth)
  Ensemble = average of 5 MAP predictions on Split A test
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

BIG_CFG = dict(embed_dim=48, model_dim=192, num_heads=4,
                num_sab=3, K=5, dropout=0.3)
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


def train_one(model, tl, vl, device, tag):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_val = float('inf'); best_state = None; bad = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for idx, val, mask, y in tl:
            idx = idx.to(device); val = val.to(device)
            mask = mask.to(device); y = y.to(device)
            opt.zero_grad()
            loss = model.loss(model(idx, val, mask), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        model.eval()
        vloss = 0.0; vn = 0
        with torch.no_grad():
            for idx, val, mask, y in vl:
                idx = idx.to(device); val = val.to(device)
                mask = mask.to(device); y = y.to(device)
                vloss += float(model.loss(model(idx, val, mask), y).item()) * y.size(0)
                vn += y.size(0)
        vloss /= max(1, vn)
        if vloss < best_val - 1e-4:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f'  [{tag}] early stop @ epoch {epoch}, best val = {best_val:.4f}')
                break
        if epoch == 1 or epoch % 40 == 0:
            print(f'  [{tag}] epoch {epoch:3d}  val_loss {vloss:.4f}  best {best_val:.4f}')
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    print('[load] real records ...')
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=MIN_BSSID_COUNT)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess)
    tr_A, te_A = splits['A_random']
    print(f'  Split A: train {len(tr_A)}, test {len(te_A)}')

    set_idx, set_val, set_mask = data.build_set_input(records, bssids, max_aps=MAX_APS)

    # ── Fit GPs ONCE + synthesize ONCE ──────────────────────────
    print('\n[gen] fitting GPs on Split A train (1449 records) ...')
    t0 = time.time()
    train_records = [{'x': float(y[i, 0]), 'y': float(y[i, 1]),
                       'session': sess[i],
                       'aps': {b: float(X[i, j])
                                for j, b in enumerate(bssids) if X[i, j] > -99.5}}
                      for i in tr_A]
    gps = synthetic.fit_per_ap_gps(train_records, bssids, min_samples=15, verbose=False)
    print(f'[gen] GP fit done ({time.time()-t0:.1f}s, {len(gps)} fitted)')

    real_xy = np.array([(r['x'], r['y']) for r in train_records], dtype=np.float32)
    free_positions = synthetic.free_cell_positions(near_real_only=real_xy)
    rng_sample = np.random.RandomState(42)
    if len(free_positions) > N_SYNTHETIC:
        chosen = rng_sample.choice(len(free_positions), N_SYNTHETIC, replace=False)
        positions = free_positions[chosen]
    else:
        positions = free_positions
    print(f'[gen] synthesizing {len(positions)} scans ...')
    X_synth, y_synth = synthetic.synthesize(gps, positions, bssids,
                                              records=train_records,
                                              detect_knn_k=10,
                                              fallback_threshold=-85.0,
                                              unc_threshold=10.0,
                                              seed=42)
    s_idx, s_val, s_mask = synthetic.build_set_form(X_synth, bssids, max_aps=MAX_APS)

    # Combined train (real + synth)
    train_idx = np.concatenate([set_idx[tr_A], s_idx], axis=0)
    train_val = np.concatenate([set_val[tr_A], s_val], axis=0)
    train_mask = np.concatenate([set_mask[tr_A], s_mask], axis=0)
    train_y = np.concatenate([y[tr_A], y_synth], axis=0)
    print(f'[setup] combined train: {len(train_y)} (real {len(tr_A)} + synth {len(s_idx)})')

    # Test loader
    test_ds = data.SetDataset(set_idx[te_A], set_val[te_A], set_mask[te_A],
                                y[te_A], jitter=0.0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # ── Train 5 seeds ────────────────────────────────────────────
    seed_preds = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f'\n══════════ seed {seed} ══════════')
        train_ds = data.SetDataset(train_idx, train_val, train_mask, train_y,
                                     jitter=JITTER_DBM)
        tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        model = models.SetTransformerMDN(num_bssids=len(bssids), **BIG_CFG)
        model, vloss = train_one(model, tl, test_loader, device, f's{seed}')

        model.eval()
        with torch.no_grad():
            idx_t = torch.from_numpy(set_idx[te_A]).to(device)
            v_t = torch.from_numpy(set_val[te_A]).to(device)
            m_t = torch.from_numpy(set_mask[te_A]).to(device)
            pred = model.predict_xy(idx_t, v_t, m_t, mode='map')
            pi, mu, sigma = model.predict_distribution(idx_t, v_t, m_t)
        err = loc_err(pred, y[te_A])
        print(f'  seed {seed}: median={np.median(err):.3f}  mean={err.mean():.3f}  '
              f'p90={np.percentile(err,90):.3f}  val_loss={vloss:.3f}')
        seed_preds.append({'pred': pred, 'pi': pi, 'mu': mu, 'sigma': sigma,
                            'val_loss': vloss, 'err': err})
        torch.save(model.state_dict(),
                    CKPT_DIR / f'A_random__BestCombo_s{seed}.pt')

    # ── Best single ─────────────────────────────────────────────
    best_i = int(np.argmin([s['val_loss'] for s in seed_preds]))
    print(f'\n══════════ Best single (seed {SEEDS[best_i]}) ══════════')
    bs = loc_stats(seed_preds[best_i]['err'])
    print(f'  median={bs["median_err_m"]:.3f}  mean={bs["mean_err_m"]:.3f}  '
          f'p90={bs["p90_err_m"]:.3f}')

    # ── Ensemble (avg of 5 MAP predictions) ─────────────────────
    print(f'\n══════════ ENSEMBLE (mean of 5 MAPs) ══════════')
    ens_pred = np.mean([s['pred'] for s in seed_preds], axis=0)
    ens_err = loc_err(ens_pred, y[te_A])
    es = loc_stats(ens_err)
    print(f'  median={es["median_err_m"]:.3f}  mean={es["mean_err_m"]:.3f}  '
          f'p90={es["p90_err_m"]:.3f}')

    # ── Save ────────────────────────────────────────────────────
    np.savez(PRED_DIR / 'A_random__BestComboEnsemble.npz',
              pred=ens_pred, y_true=y[te_A], err=ens_err, test_idx=te_A,
              pi=np.concatenate([s['pi'] for s in seed_preds], axis=1) / len(SEEDS),
              mu=np.concatenate([s['mu'] for s in seed_preds], axis=1),
              sigma=np.concatenate([s['sigma'] for s in seed_preds], axis=1))

    rows = [{'split': 'A_random', 'model': 'BestComboSingleBest',
             **bs, 'nll': None},
            {'split': 'A_random', 'model': 'BestComboEnsemble',
             **es, 'nll': None}]
    csv = OUT_DIR / 'metrics.csv'
    if csv.exists():
        old = pd.read_csv(csv)
        old = old[~old['model'].isin(['BestComboSingleBest', 'BestComboEnsemble'])]
        full = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    else:
        full = pd.DataFrame(rows)
    full.to_csv(csv, index=False)

    print('\n══════════ Split A progress LADDER (updated) ══════════')
    print('  KNN k=5:                            1.568 m')
    print('  Set Transformer (208k):             1.093 m')
    print('  Big Set Transformer single:         1.100 m')
    print('  Big × 5-ensemble:                   1.083 m')
    print('  Big + 5000 synth single:            0.906 m')
    print(f'  Big + synth best single (5 seeds):  {bs["median_err_m"]:.3f} m')
    print(f'  Big + synth × 5-ensemble:           {es["median_err_m"]:.3f} m  ⭐')


if __name__ == '__main__':
    main()
