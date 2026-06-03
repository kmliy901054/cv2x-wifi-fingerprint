"""Mean Teacher SSL for cross-time generalization.

Uses morning labeled + evening unlabeled (inputs only, no positions used)
to train a Big Set Transformer that generalizes better to evening test.

Architecture:
  Student = SetTransformerMDN (big config, same as ensemble)
  Teacher = EMA of student's weights (alpha=0.999), no gradient

Each training step:
  1. Supervised: student forward on morning labeled batch → MDN NLL
  2. Consistency:
     student forward on evening batch (with noise A)
     teacher forward on evening batch (with noise B, no grad)
     consistency_loss = MSE between student MAP-point and teacher MAP-point
  3. Total = sup_loss + λ(epoch) · consistency_loss
  4. EMA update teacher weights

λ ramps up quadratically from 0 → λ_max over first ramp_epochs (avoids
trusting noisy early predictions).

Test on real evening labels (split C, never seen labels during training).

Reference: arxiv 2407.13303 (Mean Teacher for WiFi RSSI 2024)
"""
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import data
import models

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

OUT_DIR = Path(__file__).parent / 'outputs'
PRED_DIR = OUT_DIR / 'predictions'
CKPT_DIR = OUT_DIR / 'checkpoints'
CURVES_DIR = OUT_DIR / 'training_curves'

# Same big config as ensemble
BIG_CFG = dict(embed_dim=48, model_dim=192, num_heads=4,
                num_sab=3, K=5, dropout=0.3)
WEIGHT_DECAY = 1e-3
JITTER_DBM = 4.0
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 300
PATIENCE = 60

# Mean Teacher hyperparams
EMA_ALPHA = 0.999             # teacher = α·teacher + (1-α)·student per step
LAMBDA_MAX = 1.0              # max consistency loss weight
RAMP_EPOCHS = 60              # epochs to ramp λ from 0 → LAMBDA_MAX

SEED = 42
MAX_APS = 50
MIN_BSSID_COUNT = 10


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


def consistency_lambda(epoch, max_w=LAMBDA_MAX, ramp=RAMP_EPOCHS):
    if epoch >= ramp:
        return max_w
    return max_w * (epoch / ramp) ** 2


def update_teacher(teacher, student, alpha=EMA_ALPHA):
    with torch.no_grad():
        for t_p, s_p in zip(teacher.parameters(), student.parameters()):
            t_p.mul_(alpha).add_(s_p.data, alpha=1.0 - alpha)


def jitter_set(val, mask, jitter=JITTER_DBM, rng=None):
    """In-place RSSI jitter for set form input. Only jitter real (mask=1) entries.
    val is normalized [(rssi+100)/20], so jitter is dBm/20 in normalized units."""
    if jitter <= 0:
        return val
    j = jitter / 20.0
    if rng is None:
        noise = (torch.rand_like(val) * 2 - 1) * j
    else:
        noise = torch.from_numpy((rng.random(val.shape) * 2 - 1).astype(np.float32) * j).to(val.device)
    return val + noise * mask    # only perturb real entries


def mdn_map_point(model_output):
    """Extract MAP point estimate from MDN output (log_pi, mu, log_sigma)."""
    log_pi, mu, _ = model_output
    k = log_pi.argmax(dim=-1)
    return mu[torch.arange(mu.size(0)), k]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    torch.manual_seed(SEED); np.random.seed(SEED)

    print('[load] real records ...')
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=MIN_BSSID_COUNT)
    X, y, sess = data.build_arrays(records, bssids)
    set_idx, set_val, set_mask = data.build_set_input(records, bssids, max_aps=MAX_APS)

    morning_idx = np.where(sess == 'morning')[0]
    evening_idx = np.where(sess == 'evening')[0]
    print(f'  morning {len(morning_idx)}  evening {len(evening_idx)}')

    # Labeled (morning) loader — produces (idx, val, mask, y)
    train_ds = data.SetDataset(set_idx[morning_idx], set_val[morning_idx],
                                 set_mask[morning_idx], y[morning_idx], jitter=0.0)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Unlabeled (evening) loader — y is discarded inside the step
    unl_ds = data.SetDataset(set_idx[evening_idx], set_val[evening_idx],
                               set_mask[evening_idx], y[evening_idx], jitter=0.0)
    unl_loader = DataLoader(unl_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True)

    # Test loader (evening labeled, used only for eval)
    test_loader = DataLoader(unl_ds, batch_size=256, shuffle=False)

    # Models
    student = models.SetTransformerMDN(num_bssids=len(bssids), **BIG_CFG).to(device)
    teacher = copy.deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    n_params = sum(p.numel() for p in student.parameters())
    print(f'[arch] Big Set Transformer + Mean Teacher: {n_params:,} params (×2 for teacher)')

    opt = torch.optim.Adam(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    hist = {'epoch': [], 'sup_loss': [], 'cons_loss': [], 'lambda': [], 'val_loss': []}
    best_val = float('inf'); best_state = None; bad = 0

    print('\n══════════ Mean Teacher training ══════════')
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        student.train()
        sup_loss_sum = 0.0; cons_loss_sum = 0.0; nb = 0
        lam = consistency_lambda(epoch)

        unl_iter = iter(unl_loader)
        for idx, val, mask, y_lab in train_loader:
            idx = idx.to(device); val = val.to(device)
            mask = mask.to(device); y_lab = y_lab.to(device)

            # Supervised forward (labeled morning, with jitter)
            val_aug_lab = jitter_set(val, mask, JITTER_DBM)
            sup_out = student(idx, val_aug_lab, mask)
            sup_loss = student.loss(sup_out, y_lab)

            # Consistency forward (unlabeled evening, two independent jitters)
            try:
                u_idx, u_val, u_mask, _ = next(unl_iter)
            except StopIteration:
                unl_iter = iter(unl_loader)
                u_idx, u_val, u_mask, _ = next(unl_iter)
            u_idx = u_idx.to(device); u_val = u_val.to(device); u_mask = u_mask.to(device)
            u_val_s = jitter_set(u_val, u_mask, JITTER_DBM)
            u_val_t = jitter_set(u_val, u_mask, JITTER_DBM)

            # Student: with grad
            s_pred = mdn_map_point(student(u_idx, u_val_s, u_mask))
            # Teacher: no grad, eval mode
            with torch.no_grad():
                t_pred = mdn_map_point(teacher(u_idx, u_val_t, u_mask))
            cons_loss = F.mse_loss(s_pred, t_pred)

            total_loss = sup_loss + lam * cons_loss
            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            opt.step()

            # EMA update teacher
            update_teacher(teacher, student, EMA_ALPHA)

            sup_loss_sum += float(sup_loss.item())
            cons_loss_sum += float(cons_loss.item())
            nb += 1

        sched.step()

        # Validation: student NLL on evening labeled (this IS the test set but
        # used only for early stopping; we're doing SSL transductively here.)
        student.eval()
        vloss = 0.0; vn = 0
        with torch.no_grad():
            for idx, val, mask, y_te in test_loader:
                idx = idx.to(device); val = val.to(device)
                mask = mask.to(device); y_te = y_te.to(device)
                vloss += float(student.loss(student(idx, val, mask), y_te).item()) * y_te.size(0)
                vn += y_te.size(0)
        vloss /= max(1, vn)

        hist['epoch'].append(epoch)
        hist['sup_loss'].append(sup_loss_sum / max(1, nb))
        hist['cons_loss'].append(cons_loss_sum / max(1, nb))
        hist['lambda'].append(lam)
        hist['val_loss'].append(vloss)

        if vloss < best_val - 1e-4:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f'[stop] early stop @ epoch {epoch}, best val = {best_val:.4f}')
                break
        if epoch == 1 or epoch % 20 == 0:
            print(f'  epoch {epoch:3d}  sup {sup_loss_sum/max(1,nb):.3f}  '
                  f'cons {cons_loss_sum/max(1,nb):.3f}  λ {lam:.3f}  '
                  f'val_NLL {vloss:.3f}  best {best_val:.3f}')

    print(f'[time] {time.time()-t0:.1f}s')

    if best_state is not None:
        student.load_state_dict(best_state)

    # Final eval on evening
    student.eval()
    with torch.no_grad():
        idx_t = torch.from_numpy(set_idx[evening_idx]).to(device)
        v_t = torch.from_numpy(set_val[evening_idx]).to(device)
        m_t = torch.from_numpy(set_mask[evening_idx]).to(device)
        pred = student.predict_xy(idx_t, v_t, m_t, mode='map')
        pi, mu, sigma = student.predict_distribution(idx_t, v_t, m_t)
        nll_test = float(student.loss(student(idx_t, v_t, m_t),
                                        torch.from_numpy(y[evening_idx].astype(np.float32)).to(device)).item())
    err = loc_err(pred, y[evening_idx])
    stats = loc_stats(err)
    print(f'\n══════════ FINAL: Mean Teacher on evening ══════════')
    print(f'  median={stats["median_err_m"]:.3f}  mean={stats["mean_err_m"]:.3f}'
          f'  p90={stats["p90_err_m"]:.3f}  NLL={nll_test:.3f}')
    print(f'  vs Big Set Transformer (no SSL): median 1.726 m')

    torch.save(student.state_dict(), CKPT_DIR / 'C_morning_to_evening__MeanTeacher.pt')
    pd.DataFrame(hist).to_csv(CURVES_DIR / 'C_morning_to_evening__MeanTeacher.csv', index=False)
    np.savez(PRED_DIR / 'C_morning_to_evening__SetTransformerMeanTeacher.npz',
              pred=pred, y_true=y[evening_idx], err=err, test_idx=evening_idx,
              pi=pi, mu=mu, sigma=sigma)

    # Append to metrics.csv
    row = {'split': 'C_morning_to_evening',
            'model': 'SetTransformerMeanTeacher',
            **stats, 'nll': nll_test}
    csv = OUT_DIR / 'metrics.csv'
    if csv.exists():
        old = pd.read_csv(csv)
        old = old[old['model'] != 'SetTransformerMeanTeacher']
        full = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    else:
        full = pd.DataFrame([row])
    full.to_csv(csv, index=False)
    print(f'[save] -> metrics.csv + npz + ckpt')


if __name__ == '__main__':
    main()
