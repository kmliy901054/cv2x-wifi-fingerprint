"""Shared data loading + splitting + augmentation for Lab 3.

Builds:
  X[N, D]    RSSI matrix (dBm, -100 for missing AP)
  y[N, 2]    pose (x, y) in meters in map frame
  sess[N]    'morning' or 'evening'
  bssids     list of BSSIDs corresponding to columns of X
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
WIFI_DIR = ROOT / 'wifi'

RSSI_MISSING = -100.0


def load_records():
    out = []
    for jf in sorted(WIFI_DIR.glob('wifi_*.jsonl')):
        session = 'morning' if '20260517' in jf.name else 'evening'
        with open(jf, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pose = d.get('pose') or {}
                if pose.get('frame_id') != 'map':
                    continue
                x, y = pose.get('x'), pose.get('y')
                if x is None or y is None:
                    continue
                aps = {}
                for ap in d.get('aps', []):
                    bssid = (ap.get('bssid') or '').upper()
                    rssi = ap.get('rssi')
                    if not bssid or rssi is None:
                        continue
                    aps[bssid] = max(aps.get(bssid, -200), float(rssi))
                out.append({
                    'x': float(x), 'y': float(y),
                    'session': session,
                    'aps': aps,
                })
    return out


def build_bssid_vocab(records, min_count=10):
    """Return BSSIDs that appear ≥min_count times, sorted by frequency desc."""
    cnt = Counter()
    for r in records:
        for b in r['aps']:
            cnt[b] += 1
    return [b for b, c in cnt.most_common() if c >= min_count]


def build_arrays(records, bssids):
    bidx = {b: i for i, b in enumerate(bssids)}
    N, D = len(records), len(bssids)
    X = np.full((N, D), RSSI_MISSING, dtype=np.float32)
    y = np.zeros((N, 2), dtype=np.float32)
    sess = np.empty(N, dtype=object)
    for i, r in enumerate(records):
        for b, rssi in r['aps'].items():
            j = bidx.get(b)
            if j is not None:
                X[i, j] = rssi
        y[i] = (r['x'], r['y'])
        sess[i] = r['session']
    return X, y, sess


def make_splits(sess, seed=42):
    """Return dict of named splits.  Each split is (train_idx, test_idx).

    A: random 80/20 (all data) — in-distribution baseline
    B: morning random 80/20 — same-time-of-day in-distribution
    C: morning all → evening all — cross-time generalization (worst case)
    D: stratified 80/20 across session — guarantees train AND test see both
       morning and evening; tests "fingerprint database covers all times" setup
    """
    rng = np.random.RandomState(seed)
    n = len(sess)
    all_idx = rng.permutation(n)
    cut = int(0.8 * n)
    tr_A, te_A = all_idx[:cut], all_idx[cut:]

    morning_idx = np.where(sess == 'morning')[0]
    morning_perm = rng.permutation(morning_idx)
    cut_m = int(0.8 * len(morning_perm))
    tr_B, te_B = morning_perm[:cut_m], morning_perm[cut_m:]

    tr_C = np.where(sess == 'morning')[0]
    te_C = np.where(sess == 'evening')[0]

    # Split D: stratified — early 80% of each session goes to train,
    # remaining 20% of each session goes to test.  Guarantees both
    # train and test cover both sessions.
    evening_idx = np.where(sess == 'evening')[0]
    rng_d = np.random.RandomState(seed + 1)   # independent permutation
    m_perm = rng_d.permutation(morning_idx)
    e_perm = rng_d.permutation(evening_idx)
    cut_m_d = int(0.8 * len(m_perm))
    cut_e_d = int(0.8 * len(e_perm))
    tr_D = np.concatenate([m_perm[:cut_m_d], e_perm[:cut_e_d]])
    te_D = np.concatenate([m_perm[cut_m_d:], e_perm[cut_e_d:]])

    return {
        'A_random':              (tr_A, te_A),
        'B_morning_holdout':     (tr_B, te_B),
        'C_morning_to_evening':  (tr_C, te_C),
        'D_stratified':          (tr_D, te_D),
    }


def normalize_X(X, ref_X=None):
    """Per-feature: shift so RSSI_MISSING (-100) → 0, scale by 20 dBm.
    Result roughly in [0, 3] for strong signals.  ref_X allows fitting on
    train only and applying to test; pass ref_X=None to use built-in const.
    """
    return ((X - RSSI_MISSING) / 20.0).astype(np.float32)


def build_mask(X):
    """Presence mask: 1.0 where AP was actually seen, 0.0 where filled with -100."""
    return (X > RSSI_MISSING + 0.5).astype(np.float32)


def build_set_input(records, bssids, max_aps=50):
    """For Set Transformer: each scan becomes (bssid_idx[max_aps], rssi[max_aps], mask[max_aps]).

    bssid_idx in [0, len(bssids)],  index len(bssids) reserved for PAD.
    rssi 是已正規化的 ((rssi - -100)/20).  Mask 1 = real, 0 = pad.
    """
    bidx = {b: i for i, b in enumerate(bssids)}
    PAD = len(bssids)
    N = len(records)
    idx = np.full((N, max_aps), PAD, dtype=np.int64)
    val = np.zeros((N, max_aps), dtype=np.float32)
    mask = np.zeros((N, max_aps), dtype=np.float32)
    truncated = 0
    for i, r in enumerate(records):
        # sort APs by RSSI desc so we keep strongest if truncated
        ranked = sorted(((b, v) for b, v in r['aps'].items() if b in bidx),
                         key=lambda kv: kv[1], reverse=True)
        if len(ranked) > max_aps:
            ranked = ranked[:max_aps]
            truncated += 1
        for j, (b, v) in enumerate(ranked):
            idx[i, j] = bidx[b]
            val[i, j] = (v - RSSI_MISSING) / 20.0   # normalize
            mask[i, j] = 1.0
    if truncated:
        print(f'[set_input] truncated {truncated}/{N} scans to {max_aps} APs (keep strongest)')
    return idx, val, mask


def build_set_input_v2(records, bssids, max_aps=50):
    """Enhanced set input with D1+D2 features.

    Returns (idx, features, mask) where features has shape (N, max_aps, 3):
      [:, :, 0] = rssi_norm        = (rssi + 100) / 20         in ~[0, 3]
      [:, :, 1] = rank_norm        = 1 - rank/max_aps           in [0, 1]
                                       (rank 1 strongest = 1.0)
      [:, :, 2] = rel_rssi_norm    = (rssi - max_rssi_in_scan) / 20  in [-2.0, 0]
                                       (strongest AP in scan = 0,
                                        weak APs negative)
    """
    bidx = {b: i for i, b in enumerate(bssids)}
    PAD = len(bssids)
    N = len(records)
    idx = np.full((N, max_aps), PAD, dtype=np.int64)
    features = np.zeros((N, max_aps, 3), dtype=np.float32)
    mask = np.zeros((N, max_aps), dtype=np.float32)
    for i, r in enumerate(records):
        ranked = sorted(((b, v) for b, v in r['aps'].items() if b in bidx),
                         key=lambda kv: kv[1], reverse=True)
        if not ranked:
            continue
        if len(ranked) > max_aps:
            ranked = ranked[:max_aps]
        max_rssi = ranked[0][1]
        for j, (b, v) in enumerate(ranked):
            idx[i, j] = bidx[b]
            features[i, j, 0] = (v - RSSI_MISSING) / 20.0
            features[i, j, 1] = 1.0 - (j / max_aps)         # rank: top = 1, weak = 0
            features[i, j, 2] = (v - max_rssi) / 20.0       # 0 for strongest, neg for weaker
            mask[i, j] = 1.0
    return idx, features, mask


class SetDatasetV2(Dataset):
    """SetDataset for enhanced features (3 channels) + optional AP dropout.

    ap_dropout: at training time, randomly mask out this fraction of REAL APs
                per scan (mask[j]=1 → 0).  Augmentation simulating ESP32 missed
                detections.  Set 0.0 for eval.
    """
    def __init__(self, idx, features, mask, y, jitter=0.0, ap_dropout=0.0):
        self.idx = torch.from_numpy(idx)
        self.feat_orig = features.copy()
        self.mask = torch.from_numpy(mask)
        self.y = torch.from_numpy(y.astype(np.float32))
        self.jitter = jitter / 20.0   # dBm jitter → normalized units
        self.ap_dropout = float(ap_dropout)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        feat = self.feat_orig[i].copy()
        m = self.mask[i].clone()
        if self.jitter > 0:
            m_np = m.numpy()
            noise = (np.random.rand(feat.shape[0]) * 2 - 1) * self.jitter
            # only jitter RSSI channel (channel 0) of real entries
            feat[:, 0] = feat[:, 0] + (noise * m_np).astype(np.float32)
            # also recompute rel_rssi channel (rel to new max)
            real_rssi = feat[m_np > 0.5, 0]
            if len(real_rssi) > 0:
                feat[:, 2] = (feat[:, 0] - real_rssi.max())
                # zero out channel for pad slots
                feat[m_np < 0.5, 2] = 0.0
        if self.ap_dropout > 0:
            n_real = int(m.sum().item())
            if n_real > 1:
                n_drop = int(n_real * self.ap_dropout + np.random.rand())  # round randomly
                if n_drop > 0:
                    real_positions = np.where(m.numpy() > 0.5)[0]
                    drop_idx = np.random.choice(real_positions, size=min(n_drop, n_real - 1),
                                                  replace=False)
                    m[drop_idx] = 0.0
                    feat[drop_idx] = 0.0
        return self.idx[i], torch.from_numpy(feat), m, self.y[i]


class RSSIDataset(Dataset):
    """Optional RSSI augmentation: add U[-jitter, +jitter] dBm per element.

    mode:
      'vec'     plain (X_norm, y)            — for MLP / MDN with -100 padding
      'masked'  (X_norm, presence_mask, y)   — for MaskedMLP / MaskedMDN
    """
    def __init__(self, X, y, jitter_dBm=0.0, mode='vec'):
        self.X_orig = X.copy()    # unnormalized (-100 missing) so we can augment then normalize
        self.y = torch.from_numpy(y.astype(np.float32))
        self.jitter = float(jitter_dBm)
        self.mode = mode

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X_orig[i].copy()
        if self.jitter > 0:
            noise = (np.random.rand(*x.shape) * 2 - 1) * self.jitter
            mask_seen = x > RSSI_MISSING + 0.5
            x = np.where(mask_seen, x + noise.astype(np.float32), x)
        mask = (x > RSSI_MISSING + 0.5).astype(np.float32)
        x_n = normalize_X(x)
        if self.mode == 'masked':
            return (torch.from_numpy(x_n), torch.from_numpy(mask), self.y[i])
        return torch.from_numpy(x_n), self.y[i]


class SetDataset(Dataset):
    """For Set Transformer: returns (bssid_idx[M], rssi[M], mask[M], y)."""
    def __init__(self, idx, val, mask, y, jitter=0.0):
        self.idx = torch.from_numpy(idx)
        self.val_orig = val.copy()
        self.mask = torch.from_numpy(mask)
        self.y = torch.from_numpy(y.astype(np.float32))
        self.jitter = jitter / 20.0   # convert dBm jitter to normalized units

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        v = self.val_orig[i].copy()
        if self.jitter > 0:
            # only jitter "real" entries (mask=1)
            m_np = self.mask[i].numpy()
            noise = (np.random.rand(*v.shape) * 2 - 1) * self.jitter
            v = v + (noise * m_np).astype(np.float32)
        return self.idx[i], torch.from_numpy(v), self.mask[i], self.y[i]


def make_loaders(X_tr, y_tr, X_te, y_te, batch_size=64, jitter_dBm=2.0, mode='vec'):
    train_ds = RSSIDataset(X_tr, y_tr, jitter_dBm=jitter_dBm, mode=mode)
    test_ds = RSSIDataset(X_te, y_te, jitter_dBm=0.0, mode=mode)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=256, shuffle=False))


# ─────────────────────────────────────────────────────────────────────
# Heatmap-output helpers (used by SetTransformerHeatmap)
# Discretizes the lab area into a 2-D grid; each cell becomes a class for
# cross-entropy training, and predictions are recovered as the expected
# value of softmax(logits) × free_cell_mask.
# ─────────────────────────────────────────────────────────────────────

HEATMAP_BOUNDS = (-6.0, 10.0, -2.0, 11.0)   # (x_min, x_max, y_min, y_max) covers all samples
HEATMAP_CELL_SIZE = 0.4                      # meters per cell


def build_heatmap_grid(bounds=HEATMAP_BOUNDS, cell_size=HEATMAP_CELL_SIZE):
    """Return (cell_xy, Gw, Gh).
    cell_xy: (G, 2) float32 centre of each grid cell, ordered row-major
             (y outer index, x inner index). G = Gw * Gh.
    """
    x_min, x_max, y_min, y_max = bounds
    Gw = int(np.ceil((x_max - x_min) / cell_size))
    Gh = int(np.ceil((y_max - y_min) / cell_size))
    xs = x_min + (np.arange(Gw) + 0.5) * cell_size
    ys = y_min + (np.arange(Gh) + 0.5) * cell_size
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    cell_xy = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    return cell_xy, Gw, Gh


def build_free_mask(cell_xy, free_pixel_min=200):
    """For each grid cell return True iff its centre falls in a free pixel of
    psquare.pgm (occupancy grid: 254 = free, 0 = occupied).  Returns (G,) bool.
    """
    import yaml
    from PIL import Image
    map_yaml = ROOT / 'map' / 'psquare.yaml'
    with open(map_yaml) as f:
        m = yaml.safe_load(f)
    pgm = np.array(Image.open(map_yaml.parent / m['image']))
    ox, oy = m['origin'][0], m['origin'][1]
    res = m['resolution']
    H, W = pgm.shape
    px = ((cell_xy[:, 0] - ox) / res).astype(np.int64)
    py_img = H - 1 - ((cell_xy[:, 1] - oy) / res).astype(np.int64)   # image y inverted
    valid = (px >= 0) & (px < W) & (py_img >= 0) & (py_img < H)
    free = np.zeros(len(cell_xy), dtype=bool)
    free[valid] = pgm[py_img[valid], px[valid]] >= free_pixel_min
    return free


def xy_to_cell_idx(xy, bounds=HEATMAP_BOUNDS, cell_size=HEATMAP_CELL_SIZE):
    """Map (N, 2) coordinates → (N,) int64 cell indices for cross-entropy targets."""
    x_min, x_max, y_min, y_max = bounds
    Gw = int(np.ceil((x_max - x_min) / cell_size))
    Gh = int(np.ceil((y_max - y_min) / cell_size))
    cx = np.clip(((xy[:, 0] - x_min) / cell_size).astype(np.int64), 0, Gw - 1)
    cy = np.clip(((xy[:, 1] - y_min) / cell_size).astype(np.int64), 0, Gh - 1)
    return (cy * Gw + cx).astype(np.int64)


# ─────────────────────────────────────────────────────────────────────
# Cascade-helper: map each fine cell to its containing coarse cell.
# Used by SetTransformerHeatmapCascade so the coarse prediction can
# regularize the fine prediction (P_fine(c) is gated by P_coarse(parent(c))).
# ─────────────────────────────────────────────────────────────────────

def build_fine_to_coarse(fine_bounds=HEATMAP_BOUNDS, fine_cell=HEATMAP_CELL_SIZE,
                          coarse_bounds=HEATMAP_BOUNDS, coarse_cell=1.6):
    """For every fine cell index, return the coarse cell index that contains it.

    Same bounds by default; coarse_cell is typically 4× the fine cell so each
    coarse cell holds roughly 4×4=16 fine cells.  Returns (G_fine,) int64.
    """
    fine_xy, fGw, fGh = build_heatmap_grid(fine_bounds, fine_cell)
    return xy_to_cell_idx(fine_xy, bounds=coarse_bounds, cell_size=coarse_cell)


def build_3level_grids(fine_cell=0.25, medium_cell=0.6, coarse_cell=1.6,
                        bounds=HEATMAP_BOUNDS):
    """Build a 3-resolution hierarchical grid and child→parent mappings.

    Returns a dict with:
        fine_xy, medium_xy, coarse_xy           (G_*, 2) float32 cell centres
        fine_mask, medium_mask, coarse_mask     (G_*,) bool free-cell masks
        fine_to_medium                          (G_fine,)  int64 parent indices
        medium_to_coarse                        (G_medium,) int64 parent indices
        fine_to_coarse                          (G_fine,)  int64 grandparent indices
        Gw_*, Gh_*                              grid dimensions at each level
    """
    f_xy, fGw, fGh = build_heatmap_grid(bounds, fine_cell)
    m_xy, mGw, mGh = build_heatmap_grid(bounds, medium_cell)
    c_xy, cGw, cGh = build_heatmap_grid(bounds, coarse_cell)
    f_mask = build_free_mask(f_xy)
    m_mask = build_free_mask(m_xy)
    c_mask = build_free_mask(c_xy)
    f2m = xy_to_cell_idx(f_xy, bounds=bounds, cell_size=medium_cell)
    m2c = xy_to_cell_idx(m_xy, bounds=bounds, cell_size=coarse_cell)
    f2c = xy_to_cell_idx(f_xy, bounds=bounds, cell_size=coarse_cell)
    return {
        'fine_xy': f_xy, 'medium_xy': m_xy, 'coarse_xy': c_xy,
        'fine_mask': f_mask, 'medium_mask': m_mask, 'coarse_mask': c_mask,
        'fine_to_medium': f2m, 'medium_to_coarse': m2c, 'fine_to_coarse': f2c,
        'Gw_f': fGw, 'Gh_f': fGh,
        'Gw_m': mGw, 'Gh_m': mGh,
        'Gw_c': cGw, 'Gh_c': cGh,
    }
