"""Synthetic RSSI data via per-BSSID Gaussian Process kriging.

Approach (matches agent SOTA review, baseline B):
  1. For each BSSID, fit a 2D GP:  (x, y) → RSSI  (using only real detections)
  2. Sample N positions from psquare.pgm free-space cells
  3. At each position, predict per-BSSID RSSI:
        - if GP's predicted mean < detect_threshold (e.g. -85 dBm) OR
          GP's predicted std > unc_threshold: → mark "not detected" (-100)
        - else: sample from N(mean, std)  (so synth has realistic noise)
  4. Return (X_synth[N, 80], y_synth[N, 2])

Key idea: fills SPATIAL coverage gaps — 1812 real records cover ~200-400 unique
positions, but psquare.pgm has ~75000 free cells.  Standard noise augmentation
just adds variation at the SAME (x, y), doesn't fill the gap.

This is the cheap "Baseline 2" from the SOTA review.  Multi-Wall physics sim
(baseline 1) is more principled but takes a day to implement.
"""
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import data

ROOT = Path(__file__).resolve().parents[2]
MAP_YAML = ROOT / 'map' / 'psquare.yaml'


def fit_per_ap_gps(records, bssids, min_samples=15, verbose=True):
    """Returns dict {bssid: trained GaussianProcessRegressor}.

    BSSIDs with < min_samples real detections are skipped — GP would be unreliable.
    """
    gps = {}
    skipped = 0
    if verbose:
        print(f'[fit] training {len(bssids)} GPs (only those with >={min_samples} detections)...')
    t0 = time.time()
    for i, bssid in enumerate(bssids):
        pts_xy = []
        pts_rs = []
        for r in records:
            if bssid in r['aps']:
                pts_xy.append((r['x'], r['y']))
                pts_rs.append(r['aps'][bssid])
        if len(pts_rs) < min_samples:
            skipped += 1
            continue
        X = np.array(pts_xy, dtype=np.float64)
        y = np.array(pts_rs, dtype=np.float64)
        # Kernel: constant * RBF (length scale ~2m typical indoor) + WhiteKernel noise
        # Bounded length scale to avoid runaway during optimization
        kernel = (ConstantKernel(constant_value=100.0,
                                   constant_value_bounds=(1.0, 1e4))
                  * RBF(length_scale=2.0,
                         length_scale_bounds=(0.5, 10.0))
                  + WhiteKernel(noise_level=10.0,
                                  noise_level_bounds=(0.1, 50.0)))
        try:
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                            n_restarts_optimizer=2, alpha=1e-6,
                                            random_state=42)
            gp.fit(X, y)
            gps[bssid] = gp
        except Exception as e:
            print(f'  [warn] skip {bssid}: {e}')
            skipped += 1
        if verbose and (i + 1) % 20 == 0:
            print(f'  fit {i+1}/{len(bssids)}  ({time.time()-t0:.1f}s)')
    if verbose:
        print(f'[fit] done: {len(gps)} GPs trained, {skipped} skipped  ({time.time()-t0:.1f}s)')
    return gps


def load_map():
    with open(MAP_YAML) as f:
        m = yaml.safe_load(f)
    pgm = np.array(Image.open(MAP_YAML.parent / m['image']))
    return pgm, m['origin'][0], m['origin'][1], m['resolution'], pgm.shape[1], pgm.shape[0]


def free_cell_positions(unknown_pixel=128, free_pixel_min=200, near_real_only=None):
    """Return (M, 2) array of (x, y) at the centre of each free-space cell.

    Args:
      free_pixel_min: pixel >= this is treated as free (typical pgm: free=254, occ=0)
      near_real_only: if a real (x, y) array is given, keep only free cells within
                      some buffer of any real sample — avoids synthesizing in areas
                      the robot never went (e.g. behind walls)
    """
    pgm, ox, oy, res, W, H = load_map()
    free = pgm >= free_pixel_min
    ys, xs = np.where(free)
    world_x = ox + (xs + 0.5) * res
    world_y = oy + (H - ys - 0.5) * res     # image y inverted
    positions = np.stack([world_x, world_y], axis=1)
    if near_real_only is not None:
        from scipy.spatial import cKDTree
        tree = cKDTree(near_real_only)
        d, _ = tree.query(positions, k=1)
        # keep cells within 2 m of any real sample
        keep = d <= 2.0
        positions = positions[keep]
    return positions.astype(np.float32)


def fit_detection_knn(records, bssids, k=10):
    """KNN-based per-BSSID detection probability classifier (fix C, simplified).

    Returns (nn_fitted, det_matrix[N_scans, D_bssids]) which together let you
    estimate P(detected | x, y) for any query position by:
        idxs = nn.kneighbors([(x, y)])  → indices of K nearest real scans
        P = det_matrix[idxs].mean(axis=0)
    """
    from sklearn.neighbors import NearestNeighbors
    all_xy = np.array([(r['x'], r['y']) for r in records], dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=k).fit(all_xy)
    bidx = {b: i for i, b in enumerate(bssids)}
    D = len(bssids)
    det = np.zeros((len(records), D), dtype=bool)
    for i, r in enumerate(records):
        for b in r['aps']:
            j = bidx.get(b)
            if j is not None:
                det[i, j] = True
    return nn, det


def synthesize(gps, positions, bssids, records=None,
                detect_knn_k=10, fallback_threshold=-90.0,
                unc_threshold=10.0, seed=42):
    """Generate synthetic (X_synth, y_synth) — Fix C (probability-based detection).

    Two-stage decision per (position, BSSID):
      1. Detection:  Bernoulli(p) with p = KNN-estimated P(detected | x, y)
         → handles "is this BSSID even reachable from this location?"
      2. If detected:  GP predicts (mean, std) → sample RSSI ~ N(mean, std)
         → handles "given it's detected, how strong?"

    Falls back to old threshold logic if records is None (legacy mode).
    """
    rng = np.random.RandomState(seed)
    N, D = len(positions), len(bssids)
    X = np.full((N, D), -100.0, dtype=np.float32)
    bidx = {b: i for i, b in enumerate(bssids)}

    # Stage 1: KNN detection probability for all (position, BSSID) pairs
    det_prob = None
    if records is not None:
        print(f'[synth] fitting KNN detection classifier (k={detect_knn_k})...')
        nn, det_matrix = fit_detection_knn(records, bssids, k=detect_knn_k)
        # det_prob[i, j] = P(BSSID j detected at position i)
        _, idxs = nn.kneighbors(positions)        # (N, k)
        det_prob = det_matrix[idxs].mean(axis=1)  # (N, D)
        # Bernoulli sample
        will_detect = rng.random((N, D)) < det_prob   # (N, D) bool

    # Stage 2: per-BSSID GP for RSSI value
    print(f'[synth] sampling {N} synthetic scans over {len(gps)} fitted APs...')
    t0 = time.time()
    for k_idx, (bssid, gp) in enumerate(gps.items()):
        j = bidx[bssid]
        mean, std = gp.predict(positions, return_std=True)
        # Combine criteria: KNN detection prob (if available) AND GP sanity
        if det_prob is not None:
            detected = will_detect[:, j] & (std <= unc_threshold)
        else:
            detected = (mean >= fallback_threshold) & (std <= unc_threshold)
        noise = rng.standard_normal(N) * std
        rssi = np.clip(mean + noise, -100.0, -25.0).astype(np.float32)
        X[:, j] = np.where(detected, rssi, -100.0)
        if (k_idx + 1) % 20 == 0:
            kept = detected.sum()
            print(f'  {k_idx+1}/{len(gps)}  {bssid[:17]}  detected {kept}/{N} positions')
    print(f'[synth] done  ({time.time()-t0:.1f}s)')
    return X, positions.astype(np.float32)


def build_set_form(X_synth, bssids, max_aps=50):
    """Convert dense synthetic X[N, 80] → set form (idx, val, mask) for Set Transformer."""
    N, D = X_synth.shape
    PAD = D
    idx = np.full((N, max_aps), PAD, dtype=np.int64)
    val = np.zeros((N, max_aps), dtype=np.float32)
    mask = np.zeros((N, max_aps), dtype=np.float32)
    truncated = 0
    for i in range(N):
        # Get indices of detected APs (RSSI > -99 = not missing) ranked by strength
        seen = X_synth[i] > -99.5
        if not seen.any():
            continue
        ordered_j = np.argsort(-X_synth[i])   # strongest first
        ordered_j = ordered_j[seen[ordered_j]]
        if len(ordered_j) > max_aps:
            ordered_j = ordered_j[:max_aps]
            truncated += 1
        for k, j in enumerate(ordered_j):
            idx[i, k] = j
            val[i, k] = (X_synth[i, j] - (-100.0)) / 20.0
            mask[i, k] = 1.0
    if truncated:
        print(f'[set] truncated {truncated}/{N} synth scans to {max_aps} APs')
    return idx, val, mask


# ─────────────────────────────────────────────────────────────────────
# CLI: fit + synthesize + save
# ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    OUT_DIR = Path(__file__).parent / 'outputs'

    ap = argparse.ArgumentParser()
    ap.add_argument('--n-synthetic', type=int, default=5000,
                     help='How many synthetic scans to generate')
    ap.add_argument('--source', default='morning', choices=['morning', 'all'],
                     help='Which real records to fit the GPs on')
    ap.add_argument('--detect-threshold', type=float, default=-85.0)
    ap.add_argument('--unc-threshold', type=float, default=8.0)
    ap.add_argument('--min-samples', type=int, default=15)
    ap.add_argument('--near-real-buffer', type=float, default=2.0,
                     help='Only synthesize within this many metres of any real sample (m)')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'[load] records...')
    records = data.load_records()
    bssids = data.build_bssid_vocab(records, min_count=10)

    if args.source == 'morning':
        train_records = [r for r in records if r['session'] == 'morning']
        print(f'       fitting on {len(train_records)} morning records')
    else:
        train_records = records
        print(f'       fitting on all {len(train_records)} records')

    gps = fit_per_ap_gps(train_records, bssids, min_samples=args.min_samples)

    # Real positions (for buffer filter)
    real_xy = np.array([(r['x'], r['y']) for r in train_records], dtype=np.float32)

    # Position pool
    print(f'[map]  loading free cells from psquare.pgm ...')
    free_positions = free_cell_positions(near_real_only=real_xy)
    print(f'       {len(free_positions)} free cells within {args.near_real_buffer}m of real samples')

    # Sample N from free positions
    rng = np.random.RandomState(42)
    if len(free_positions) > args.n_synthetic:
        chosen = rng.choice(len(free_positions), args.n_synthetic, replace=False)
        positions = free_positions[chosen]
    else:
        positions = free_positions
    print(f'       sampling {len(positions)} synthetic positions')

    # Synthesize — fix C: pass records to enable KNN detection probability
    X_synth, y_synth = synthesize(gps, positions, bssids,
                                    records=train_records,
                                    detect_knn_k=10,
                                    fallback_threshold=args.detect_threshold,
                                    unc_threshold=args.unc_threshold)

    # Save
    out_npz = OUT_DIR / f'synthetic_{args.source}_{len(positions)}.npz'
    np.savez(out_npz,
              X_synth=X_synth, y_synth=y_synth,
              bssids=np.array(bssids),
              n_aps_fitted=len(gps))
    print(f'[save] -> {out_npz}')

    # Quick stats
    aps_per_scan = (X_synth > -99.5).sum(axis=1)
    print(f'\n=== synthetic data stats ===')
    print(f'  scans: {len(X_synth)}')
    print(f'  APs/scan: mean {aps_per_scan.mean():.1f}, '
          f'median {int(np.median(aps_per_scan))}, '
          f'range [{aps_per_scan.min()}, {aps_per_scan.max()}]')
    print(f'  (real avg was 27.7, EDA fingerprint dataset)')
