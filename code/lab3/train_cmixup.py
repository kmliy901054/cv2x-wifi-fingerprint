"""C-Mixup (regression-aware mixup) for Set Transformer.

Implements C-Mixup from arXiv 2405.17938:
  For each sample i, sample partner j ~ Categorical with prob ∝ exp(-||y_i - y_j||²/2b²)
  Sample λ ~ Beta(α, α)
  Mix dense RSSI:  x_mix = λ x_i + (1-λ) x_j
  Mix label:       y_mix = λ y_i + (1-λ) y_j
  Re-form scan as set, feed Set Transformer

Quick sanity experiments:
  1. Real-only + C-Mixup        (compare to Big baseline 1.10 m)
  2. Real + 5000 synth + C-Mixup (compare to Big+synth 0.91 m)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

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
SEED = 42
MAX_APS = 50
MIN_BSSID_COUNT = 10
N_SYNTHETIC = 5000

# C-Mixup hyperparameters
CMIXUP_BANDWIDTH = 2.0      # metres — Gaussian kernel σ on label distance
CMIXUP_ALPHA = 2.0          # Beta(α, α), peak at λ=0.5
CMIXUP_PROB = 1.0           # probability to apply mixup per sample

RSSI_MISSING = -100.0


def build_cmixup_sampler(y_train, bandwidth=CMIXUP_BANDWIDTH):
    """Return (N, N) row-stochastic matrix.  sampler[i, j] = P(j mixes with i)."""
    from scipy.spatial.distance import cdist
    d = cdist(y_train, y_train)
    w = np.exp(-d ** 2 / (2 * bandwidth ** 2)).astype(np.float64)
    np.fill_diagonal(w, 0.0)
    w = w / w.sum(axis=1, keepdims=True)
    return w


def dense_to_set_form(x_dense, num_bssids, max_aps=MAX_APS,
                       detect_threshold=-99.5):
    """Convert one dense RSSI vector (D,) to set form (idx, val, mask)."""
    PAD = num_bssids
    idx = np.full(max_aps, PAD, dtype=np.int64)
    val = np.zeros(max_aps, dtype=np.float32)
    mask = np.zeros(max_aps, dtype=np.float32)
    seen = x_dense > detect_threshold
    if not seen.any():
        return idx, val, mask
    order = np.argsort(-x_dense)
    order = order[seen[order]]
    if len(order) > max_aps:
        order = order[:max_aps]
    for k, j in enumerate(order):
        idx[k] = j
        val[k] = (x_dense[j] - RSSI_MISSING) / 20.0
        mask[k] = 1.0
    return idx, val, mask


class CMixupSetDataset(Dataset):
    """Dataset that does C-Mixup on dense RSSI then yields set-form inputs."""

    def __init__(self, X_dense, y, num_bssids, max_aps=MAX_APS,
                 jitter_dBm=JITTER_DBM, alpha=CMIXUP_ALPHA, prob=CMIXUP_PROB,
                 sampler=None):
        self.X = X_dense.astype(np.float32).copy()       # (N, D), -100 for missing
        self.y = y.astype(np.float32).copy()              # (N, 2)
        self.num_bssids = num_bssids
        self.max_aps = max_aps
        self.jitter = jitter_dBm
        self.alpha = alpha
        self.prob = prob
        self.sampler = sampler  # (N, N) row-stoch, may be None → uniform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        do_mixup = (self.sampler is not None) and (np.random.rand() < self.prob)
        if do_mixup:
            # Sample partner per C-Mixup distribution
            j = np.random.choice(len(self.y), p=self.sampler[i])
            lam = np.random.beta(self.alpha, self.alpha)
            # Mix in dense RSSI space (treats missing as -100)
            x = lam * self.X[i] + (1 - lam) * self.X[j]
            y_m = lam * self.y[i] + (1 - lam) * self.y[j]
        else:
            x = self.X[i].copy()
            y_m = self.y[i].copy()
        if self.jitter > 0:
            seen_mask = x > RSSI_MISSING + 0.5
            noise = (np.random.rand(*x.shape) * 2 - 1) * self.jitter
            x = np.where(seen_mask, x + noise.astype(np.float32), x)
        idx, val, mask = dense_to_set_form(x, self.num_bssids, self.max_aps)
        return (torch.from_numpy(idx), torch.from_numpy(val),
                torch.from_numpy(mask), torch.from_numpy(y_m))


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
    hist = {'epoch': [], 'train_loss': [], 'val_loss': []}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tloss = 0.0; nb = 0
        for idx, val, mask, y in train_loader:
            idx = idx.to(device); val = val.to(device)
            mask = mask.to(device); y = y.to(device)
            opt.zero_grad()
            loss = model.loss(model(idx, val, mask), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tloss += float(loss.item()); nb += 1
        sched.step()
        model.eval()
        vloss = 0.0; vn = 0
        with torch.no_grad():
            for idx, val, mask, y in val_loader:
                idx = idx.to(device); val = val.to(device)
                mask = mask.to(device); y = y.to(device)
                vloss += float(model.loss(model(idx, val, mask), y).item()) * y.size(0)
                vn += y.size(0)
        vloss /= max(1, vn)
        hist['epoch'].append(epoch)
        hist['train_loss'].append(tloss / max(1, nb))
        hist['val_loss'].append(vloss)
        if vloss < best_val - 1e-4:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f'  [{tag}] early stop @ epoch {epoch}, best val = {best_val:.4f}')
                break
        if epoch == 1 or epoch % 30 == 0:
            print(f'  [{tag}] epoch {epoch:3d}  train {hist["train_loss"][-1]:.4f}  '
                  f'val {vloss:.4f}  best {best_val:.4f}')
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


def evaluate(model, set_idx, set_val, set_mask, y_te, device):
    model.eval()
    with torch.no_grad():
        idx_t = torch.from_numpy(set_idx).to(device)
        v_t = torch.from_numpy(set_val).to(device)
        m_t = torch.from_numpy(set_mask).to(device)
        pred = model.predict_xy(idx_t, v_t, m_t, mode='map')
    return pred, loc_err(pred, y_te)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    torch.manual_seed(SEED); np.random.seed(SEED)

    print('[load] real records ...')
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=MIN_BSSID_COUNT)
    X, y, sess = data.build_arrays(records, bssids)
    splits = data.make_splits(sess)
    set_idx, set_val, set_mask = data.build_set_input(records, bssids, max_aps=MAX_APS)
    tr_A, te_A = splits['A_random']
    print(f'  Split A: train {len(tr_A)}, test {len(te_A)}')

    # ── Build C-Mixup sampler for real Split A train ─────────────
    print('\n[cmixup] building label-distance sampler '
          f'(bandwidth={CMIXUP_BANDWIDTH}m, alpha={CMIXUP_ALPHA})')
    sampler_real = build_cmixup_sampler(y[tr_A], bandwidth=CMIXUP_BANDWIDTH)

    # Test loader (always real Split A test)
    test_ds = data.SetDataset(set_idx[te_A], set_val[te_A], set_mask[te_A],
                                y[te_A], jitter=0.0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    results = []

    # ── Experiment 1: Real + C-Mixup (no synth) ──────────────────
    print('\n══════════ EXP 1: Real Split A train + C-Mixup (no synth) ══════════')
    train_ds_1 = CMixupSetDataset(X[tr_A], y[tr_A], len(bssids),
                                    jitter_dBm=JITTER_DBM, sampler=sampler_real)
    tl_1 = DataLoader(train_ds_1, batch_size=BATCH_SIZE, shuffle=True)
    model_1 = models.SetTransformerMDN(num_bssids=len(bssids), **BIG_CFG)
    model_1, hist_1 = train_one(model_1, tl_1, test_loader, device, 'E1_real_cmixup')
    pred_1, err_1 = evaluate(model_1, set_idx[te_A], set_val[te_A], set_mask[te_A],
                              y[te_A], device)
    s_1 = loc_stats(err_1)
    print(f'  EXP 1 result: median={s_1["median_err_m"]:.3f}  '
          f'mean={s_1["mean_err_m"]:.3f}  p90={s_1["p90_err_m"]:.3f}')
    results.append({'split': 'A_random', 'model': 'BigCMixupRealOnly',
                    **s_1, 'nll': None})
    torch.save(model_1.state_dict(), CKPT_DIR / 'A_random__BigCMixupRealOnly.pt')
    np.savez(PRED_DIR / 'A_random__BigCMixupRealOnly.npz',
              pred=pred_1, y_true=y[te_A], err=err_1, test_idx=te_A)

    # ── Build combined train (real + synth) + new sampler ───────
    print('\n[gen] generating 5000 synth from Split A train ...')
    train_records = [{'x': float(y[i, 0]), 'y': float(y[i, 1]),
                       'session': sess[i],
                       'aps': {b: float(X[i, j])
                                for j, b in enumerate(bssids) if X[i, j] > -99.5}}
                      for i in tr_A]
    gps = synthetic.fit_per_ap_gps(train_records, bssids, min_samples=15, verbose=False)
    real_xy = np.array([(r['x'], r['y']) for r in train_records], dtype=np.float32)
    free_positions = synthetic.free_cell_positions(near_real_only=real_xy)
    rng = np.random.RandomState(SEED)
    if len(free_positions) > N_SYNTHETIC:
        chosen = rng.choice(len(free_positions), N_SYNTHETIC, replace=False)
        positions = free_positions[chosen]
    else:
        positions = free_positions
    X_synth, y_synth = synthetic.synthesize(gps, positions, bssids,
                                              records=train_records,
                                              detect_knn_k=10,
                                              fallback_threshold=-85.0,
                                              unc_threshold=10.0, seed=SEED)
    print(f'[gen] synth: {X_synth.shape}, mean APs/scan {(X_synth > -99.5).sum(1).mean():.1f}')

    # Combined dense + label
    X_combined = np.concatenate([X[tr_A], X_synth.astype(np.float32)], axis=0)
    y_combined = np.concatenate([y[tr_A], y_synth.astype(np.float32)], axis=0)
    print(f'[setup] combined train: {len(y_combined)} (real {len(tr_A)} + synth {len(X_synth)})')

    print('[cmixup] building sampler for combined train (this may take ~10s) ...')
    sampler_combined = build_cmixup_sampler(y_combined, bandwidth=CMIXUP_BANDWIDTH)

    # ── Experiment 2: Real + synth + C-Mixup ─────────────────────
    print('\n══════════ EXP 2: Real + 5000 synth + C-Mixup ══════════')
    train_ds_2 = CMixupSetDataset(X_combined, y_combined, len(bssids),
                                    jitter_dBm=JITTER_DBM, sampler=sampler_combined)
    tl_2 = DataLoader(train_ds_2, batch_size=BATCH_SIZE, shuffle=True)
    model_2 = models.SetTransformerMDN(num_bssids=len(bssids), **BIG_CFG)
    model_2, hist_2 = train_one(model_2, tl_2, test_loader, device, 'E2_synth_cmixup')
    pred_2, err_2 = evaluate(model_2, set_idx[te_A], set_val[te_A], set_mask[te_A],
                              y[te_A], device)
    s_2 = loc_stats(err_2)
    print(f'  EXP 2 result: median={s_2["median_err_m"]:.3f}  '
          f'mean={s_2["mean_err_m"]:.3f}  p90={s_2["p90_err_m"]:.3f}')
    results.append({'split': 'A_random', 'model': 'BigCMixupSynth',
                    **s_2, 'nll': None})
    torch.save(model_2.state_dict(), CKPT_DIR / 'A_random__BigCMixupSynth.pt')
    np.savez(PRED_DIR / 'A_random__BigCMixupSynth.npz',
              pred=pred_2, y_true=y[te_A], err=err_2, test_idx=te_A)

    # ── Summary ──────────────────────────────────────────────────
    print('\n══════════ Comparison on Split A ══════════')
    print(f'  Big baseline (real only):           1.100 m')
    print(f'  Big + C-Mixup (real only):          {s_1["median_err_m"]:.3f} m   '
          f'Δ vs 1.100: {s_1["median_err_m"]-1.100:+.3f}')
    print(f'  Big + 5000 synth (single):          0.906 m')
    print(f'  Big + 5000 synth × 5-ensemble:      0.889 m  ⭐')
    print(f'  Big + synth + C-Mixup (this):       {s_2["median_err_m"]:.3f} m   '
          f'Δ vs 0.906: {s_2["median_err_m"]-0.906:+.3f}')

    # Save to metrics.csv
    csv = OUT_DIR / 'metrics.csv'
    new_df = pd.DataFrame(results)
    if csv.exists():
        old = pd.read_csv(csv)
        old = old[~old['model'].isin(['BigCMixupRealOnly', 'BigCMixupSynth'])]
        full = pd.concat([old, new_df], ignore_index=True)
    else:
        full = new_df
    full.to_csv(csv, index=False)
    print(f'\n[save] -> metrics.csv')


if __name__ == '__main__':
    main()
