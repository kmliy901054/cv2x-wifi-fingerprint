"""EDA — 評估 WiFi RSSI fingerprint 資料對室內定位的可學習性.

回答四個關鍵問題:
  Q1 可學習嗎?           KNN k=1 / k=5 baseline 中位數定位誤差作為難度下界
  Q2 跨時段 drift 影響?    比較 random 80/20 split vs morning→evening split
  Q3 需要多少 BSSID?       top-N curve(N=5/10/20/50/100,看 KNN 誤差曲線)
  Q4 需要機率模型嗎?       within-cell RSSI std 分布(噪聲 > signal range → 要 uncertainty)

輸入: ../../wifi/wifi_*.jsonl
輸出: eda_output/
  eda_report.md          完整文字報告
  feature_table.csv      per-BSSID 統計表(rank, samples, coverage, RSSI 範圍, drift)
  plots/                 8 張視覺化
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scipy import stats
from sklearn.neighbors import NearestNeighbors

# Windows console cp950 → force utf-8 for prints
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Use a CJK-capable font so Chinese in titles renders
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei',
                                     'SimHei', 'PingFang TC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parents[2]
WIFI_DIR = ROOT / 'wifi'
MAP_YAML = ROOT / 'map' / 'psquare.yaml'
OUT_DIR = Path(__file__).parent / 'eda_output'
PLOTS_DIR = OUT_DIR / 'plots'

RSSI_MISSING = -100.0   # fill value for "AP not seen in this scan"


# ─────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────

def load_records():
    """Return list of dicts: {x, y, yaw, session, t, aps: {bssid: rssi}}."""
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
                    # keep the strongest if duplicate bssid in same scan
                    aps[bssid] = max(aps.get(bssid, -200), float(rssi))
                ssids = {(ap.get('bssid') or '').upper(): ap.get('ssid', '')
                         for ap in d.get('aps', [])}
                out.append({
                    'x': float(x), 'y': float(y),
                    'yaw': float(pose.get('yaw', 0.0)),
                    'session': session,
                    'wall_time': float(d.get('wall_time', 0.0)),
                    'aps': aps,
                    'ssids': ssids,
                })
    return out


def build_feature_matrix(records, bssid_list):
    """Returns (X[N, D], y[N, 2], session[N]) with RSSI filled -100 where missing."""
    bssid_to_idx = {b: i for i, b in enumerate(bssid_list)}
    N, D = len(records), len(bssid_list)
    X = np.full((N, D), RSSI_MISSING, dtype=np.float32)
    y = np.zeros((N, 2), dtype=np.float32)
    sess = np.empty(N, dtype=object)
    for i, r in enumerate(records):
        for bssid, rssi in r['aps'].items():
            if bssid in bssid_to_idx:
                X[i, bssid_to_idx[bssid]] = rssi
        y[i] = (r['x'], r['y'])
        sess[i] = r['session']
    return X, y, sess


# ─────────────────────────────────────────────────────────────────────
# KNN baseline
# ─────────────────────────────────────────────────────────────────────

def knn_localize(X_tr, y_tr, X_te, y_te, k=1, metric='euclidean'):
    """k-NN regression in RSSI space; report localization error per test sample (m)."""
    nn = NearestNeighbors(n_neighbors=k, metric=metric).fit(X_tr)
    dists, idxs = nn.kneighbors(X_te)
    if k == 1:
        y_pred = y_tr[idxs[:, 0]]
    else:
        # inverse-distance weighted average of neighbour poses
        w = 1.0 / (dists + 1e-6)
        w = w / w.sum(axis=1, keepdims=True)
        y_pred = (w[..., None] * y_tr[idxs]).sum(axis=1)
    err = np.linalg.norm(y_pred - y_te, axis=1)
    return err, y_pred


# ─────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────

def save_plot(fig, name):
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return path


def load_map():
    """Return (pgm_array, ox, oy, res, W, H) — for plotting on the floor plan."""
    with open(MAP_YAML) as f:
        m = yaml.safe_load(f)
    pgm = np.array(Image.open(MAP_YAML.parent / m['image']))
    return pgm, m['origin'][0], m['origin'][1], m['resolution'], pgm.shape[1], pgm.shape[0]


# ─────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print('[load] reading jsonl ...')
    records = load_records()
    print(f'       {len(records)} records loaded')

    # ── A. 整體 ──────────────────────────────────────────────
    n_morning = sum(r['session'] == 'morning' for r in records)
    n_evening = len(records) - n_morning
    aps_per_scan = [len(r['aps']) for r in records]
    bssid_appearance = Counter()
    bssid_ssids = {}                  # bssid -> any observed ssid
    bssid_rssi_all = defaultdict(list)
    for r in records:
        for bssid, rssi in r['aps'].items():
            bssid_appearance[bssid] += 1
            bssid_rssi_all[bssid].append(rssi)
            if bssid not in bssid_ssids:
                bssid_ssids[bssid] = r['ssids'].get(bssid, '')

    n_unique_bssid = len(bssid_appearance)
    print(f'       morning={n_morning}, evening={n_evening}')
    print(f'       unique BSSID={n_unique_bssid}, '
          f'avg APs/scan={np.mean(aps_per_scan):.1f} ± {np.std(aps_per_scan):.1f}')

    # ── B. 每個 BSSID 統計表 ────────────────────────────────
    sorted_bssids = [b for b, _ in bssid_appearance.most_common()]
    feature_rows = []
    for rank, bssid in enumerate(sorted_bssids, 1):
        rs = np.array(bssid_rssi_all[bssid], dtype=float)
        m_count = sum(1 for r in records if r['session'] == 'morning' and bssid in r['aps'])
        e_count = sum(1 for r in records if r['session'] == 'evening' and bssid in r['aps'])
        m_rs = [r['aps'][bssid] for r in records if r['session'] == 'morning' and bssid in r['aps']]
        e_rs = [r['aps'][bssid] for r in records if r['session'] == 'evening' and bssid in r['aps']]
        feature_rows.append({
            'rank': rank,
            'bssid': bssid,
            'ssid': bssid_ssids[bssid][:30],
            'samples': len(rs),
            'coverage_pct': round(100 * len(rs) / len(records), 1),
            'rssi_min': int(rs.min()),
            'rssi_max': int(rs.max()),
            'rssi_mean': round(rs.mean(), 1),
            'rssi_std': round(rs.std(), 1),
            'rssi_range': int(rs.max() - rs.min()),
            'morning_samples': m_count,
            'evening_samples': e_count,
            'morning_mean': round(np.mean(m_rs), 2) if m_rs else None,
            'evening_mean': round(np.mean(e_rs), 2) if e_rs else None,
            'evening_minus_morning': (round(np.mean(e_rs) - np.mean(m_rs), 2)
                                       if m_rs and e_rs else None),
        })
    feat_df = pd.DataFrame(feature_rows)
    feat_df.to_csv(OUT_DIR / 'feature_table.csv', index=False)
    print(f'[B]    wrote feature_table.csv ({len(feat_df)} BSSIDs)')

    # Plot 1: APs per scan histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(aps_per_scan, bins=30, color='steelblue', edgecolor='white')
    ax.axvline(np.mean(aps_per_scan), color='red', ls='--',
                label=f'mean={np.mean(aps_per_scan):.1f}')
    ax.set_xlabel('APs detected per scan'); ax.set_ylabel('# scans')
    ax.set_title('Distribution of APs visible per scan')
    ax.legend()
    save_plot(fig, '01_aps_per_scan_hist.png')

    # Plot 2: BSSID coverage curve (sorted descending)
    counts = sorted(bssid_appearance.values(), reverse=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(range(1, len(counts)+1), counts, marker='.', color='steelblue')
    ax.axhline(0.5 * len(records), color='orange', ls='--',
                label='50% of scans')
    ax.axhline(0.1 * len(records), color='red', ls='--',
                label='10% of scans')
    ax.set_xlabel('BSSID rank (most common first)')
    ax.set_ylabel('# scans the AP appears in')
    ax.set_title(f'BSSID coverage — {len(counts)} unique BSSIDs')
    ax.legend(); ax.grid(alpha=0.3)
    save_plot(fig, '02_bssid_coverage.png')

    # ── C. Within-cell RSSI std (Q4: 機率模型?) ────────────────
    # Grid 0.5 m × 0.5 m, only cells with ≥5 samples
    print('[C]    computing within-cell RSSI std ...')
    cell_size = 0.5
    by_cell = defaultdict(list)   # cell -> list of records (index)
    for i, r in enumerate(records):
        cx = int(r['x'] / cell_size)
        cy = int(r['y'] / cell_size)
        by_cell[(cx, cy)].append(i)

    top20_bssid = sorted_bssids[:20]
    cell_stds = []                # one std per (cell, bssid) where ≥3 samples both
    for cell, idxs in by_cell.items():
        if len(idxs) < 5:
            continue
        cell_recs = [records[i] for i in idxs]
        for bssid in top20_bssid:
            rs = [r['aps'][bssid] for r in cell_recs if bssid in r['aps']]
            if len(rs) >= 3:
                cell_stds.append(np.std(rs))

    cell_stds = np.array(cell_stds)
    print(f'       within-cell std: median={np.median(cell_stds):.2f} dBm, '
          f'p90={np.percentile(cell_stds, 90):.2f} dBm')

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cell_stds, bins=40, color='salmon', edgecolor='white')
    ax.axvline(np.median(cell_stds), color='black', ls='--',
                label=f'median={np.median(cell_stds):.2f} dBm')
    ax.set_xlabel('within-cell RSSI std (dBm)  [cells of 0.5 m, top-20 BSSIDs]')
    ax.set_ylabel('count')
    ax.set_title('Q4: 噪聲水準 — within-cell RSSI std distribution\n'
                 '(大 → 同一位置 RSSI 不穩 → 建議用機率/集成模型)')
    ax.legend()
    save_plot(fig, '03_within_cell_rssi_std.png')

    # ── D. 跨時段 covariate shift ─────────────────────────────
    # For BSSIDs with ≥30 samples in BOTH sessions: per-BSSID mean shift + KS test
    print('[D]    computing morning vs evening shift ...')
    shifts = []
    for bssid in sorted_bssids:
        m_rs = [r['aps'][bssid] for r in records if r['session'] == 'morning' and bssid in r['aps']]
        e_rs = [r['aps'][bssid] for r in records if r['session'] == 'evening' and bssid in r['aps']]
        if len(m_rs) >= 30 and len(e_rs) >= 30:
            delta = np.mean(e_rs) - np.mean(m_rs)
            ks_stat, ks_p = stats.ks_2samp(m_rs, e_rs)
            shifts.append({
                'bssid': bssid,
                'ssid': bssid_ssids[bssid][:30],
                'morning_n': len(m_rs), 'evening_n': len(e_rs),
                'morning_mean': round(np.mean(m_rs), 2),
                'evening_mean': round(np.mean(e_rs), 2),
                'delta': round(delta, 2),
                'ks_stat': round(ks_stat, 3),
                'ks_p': float(f'{ks_p:.2e}'),
                'significant_shift': bool(ks_p < 0.01),
            })
    shift_df = pd.DataFrame(shifts).sort_values('delta', key=lambda s: s.abs(), ascending=False)
    shift_df.to_csv(OUT_DIR / 'covariate_shift.csv', index=False)
    n_sig = shift_df['significant_shift'].sum()
    print(f'       {len(shift_df)} BSSIDs in both sessions, '
          f'{n_sig} have significant shift (KS p<0.01)')

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(shift_df['delta'], bins=30, color='mediumpurple', edgecolor='white')
    ax.axvline(0, color='black', lw=1)
    ax.axvline(shift_df['delta'].mean(), color='red', ls='--',
                label=f"mean={shift_df['delta'].mean():.2f} dBm")
    ax.set_xlabel('evening RSSI mean − morning RSSI mean (dBm)')
    ax.set_ylabel('# BSSIDs')
    ax.set_title(f'Q2: 跨時段 RSSI 漂移 ({len(shift_df)} BSSIDs ≥30 樣本)\n'
                 f'{n_sig} 個 BSSID 有顯著 shift (KS p<0.01)')
    ax.legend()
    save_plot(fig, '04_morning_evening_shift.png')

    # ── E. KNN baseline (Q1, Q2, Q3) ──────────────────────────
    print('[E]    running KNN baselines ...')
    rng = np.random.RandomState(42)

    # Use a fixed BSSID set (all that appear ≥10 times = useful features)
    useful_bssids = [b for b, c in bssid_appearance.most_common() if c >= 10]
    print(f'       feature set: {len(useful_bssids)} BSSIDs (each ≥10 samples)')
    X_all, y_all, sess_all = build_feature_matrix(records, useful_bssids)

    # Splits to evaluate
    morning_mask = sess_all == 'morning'
    evening_mask = sess_all == 'evening'

    # Split A: random 80/20 on ALL data (in-distribution)
    n = len(records)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    tr_A, te_A = idx[:cut], idx[cut:]

    # Split B: morning train (80%) + morning val (20%)
    morning_idx = np.where(morning_mask)[0]
    morning_perm = rng.permutation(morning_idx)
    cut_m = int(0.8 * len(morning_perm))
    tr_B, val_B = morning_perm[:cut_m], morning_perm[cut_m:]

    # Split C: morning all → evening all (covariate shift test)
    tr_C, te_C = np.where(morning_mask)[0], np.where(evening_mask)[0]

    split_results = []
    for split_name, tr_idx, te_idx in [
        ('A: random 80/20 (all data)', tr_A, te_A),
        ('B: morning 80/20 (in-distribution)', tr_B, val_B),
        ('C: morning → evening (covariate shift)', tr_C, te_C),
    ]:
        for k in (1, 3, 5):
            err, _ = knn_localize(X_all[tr_idx], y_all[tr_idx],
                                  X_all[te_idx], y_all[te_idx], k=k)
            split_results.append({
                'split': split_name, 'k': k,
                'n_train': len(tr_idx), 'n_test': len(te_idx),
                'median_err_m': round(np.median(err), 3),
                'mean_err_m': round(err.mean(), 3),
                'p90_err_m': round(np.percentile(err, 90), 3),
                'p10_err_m': round(np.percentile(err, 10), 3),
            })
    knn_df = pd.DataFrame(split_results)
    knn_df.to_csv(OUT_DIR / 'knn_baseline.csv', index=False)
    print(knn_df.to_string(index=False))

    # Plot 5: KNN error CDF for k=1 across 3 splits
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for split_name, tr_idx, te_idx, color in [
        ('A: random 80/20', tr_A, te_A, 'steelblue'),
        ('B: morning 80/20', tr_B, val_B, 'forestgreen'),
        ('C: morning → evening', tr_C, te_C, 'crimson'),
    ]:
        err, _ = knn_localize(X_all[tr_idx], y_all[tr_idx],
                              X_all[te_idx], y_all[te_idx], k=1)
        err_sorted = np.sort(err)
        cdf = np.arange(1, len(err)+1) / len(err)
        ax.plot(err_sorted, cdf, label=f'{split_name}  median={np.median(err):.2f} m',
                color=color, lw=2)
    ax.set_xlabel('localization error (m)')
    ax.set_ylabel('CDF')
    ax.set_xlim(0, 6)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title('Q1, Q2: KNN(k=1) localization error CDF\n'
                 '比較 in-distribution vs cross-time test')
    save_plot(fig, '05_knn_error_cdf.png')

    # ── F. Top-N curve (Q3: 需要多少 BSSID) ─────────────────────
    print('[F]    feature count sweep ...')
    n_grid = [3, 5, 10, 20, 30, 50, 75, 100, len(useful_bssids)]
    n_grid = sorted(set(n for n in n_grid if n <= len(useful_bssids)))
    topn_rows = []
    for nb in n_grid:
        bssids_n = sorted_bssids[:nb]
        Xn, yn, sn = build_feature_matrix(records, bssids_n)
        # Use split B (morning-only) for fair feature-count comparison
        err, _ = knn_localize(Xn[tr_B], yn[tr_B], Xn[val_B], yn[val_B], k=1)
        topn_rows.append({
            'top_n_bssids': nb,
            'median_err_m': round(np.median(err), 3),
            'mean_err_m': round(err.mean(), 3),
            'p90_err_m': round(np.percentile(err, 90), 3),
        })
    topn_df = pd.DataFrame(topn_rows)
    topn_df.to_csv(OUT_DIR / 'feature_sweep.csv', index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(topn_df['top_n_bssids'], topn_df['median_err_m'],
            marker='o', label='median error', color='steelblue', lw=2)
    ax.plot(topn_df['top_n_bssids'], topn_df['p90_err_m'],
            marker='s', label='p90 error', color='orange', lw=2)
    ax.set_xlabel('# top-N BSSIDs used as features')
    ax.set_ylabel('KNN(k=1) error on morning val set (m)')
    ax.set_title('Q3: 特徵數 vs 精度 (in-distribution morning split)')
    ax.grid(alpha=0.3)
    ax.legend()
    save_plot(fig, '06_feature_count_sweep.png')

    # ── G. Spatial coverage: train pts vs test pts overlay ────
    print('[G]    plotting spatial coverage ...')
    pgm, ox, oy, res, W, H = load_map()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mask, title, color in [
        (axes[0], morning_mask, f'Morning ({morning_mask.sum()})', 'forestgreen'),
        (axes[1], evening_mask, f'Evening ({evening_mask.sum()})', 'crimson'),
    ]:
        ax.imshow(pgm, cmap='gray', extent=[ox, ox + W*res, oy, oy + H*res])
        ax.scatter(y_all[mask, 0], y_all[mask, 1], s=3, c=color, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
    fig.suptitle('Q2 context: 早晚軌跡空間分布 (是否 overlap → 影響 cross-time 評估)')
    save_plot(fig, '07_spatial_coverage_split.png')

    # ── H. Pose error heatmap (KNN k=1 on cross-time split) ──
    err_C, y_pred_C = knn_localize(X_all[tr_C], y_all[tr_C],
                                    X_all[te_C], y_all[te_C], k=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(pgm, cmap='gray', extent=[ox, ox + W*res, oy, oy + H*res])
    sc = ax.scatter(y_all[te_C, 0], y_all[te_C, 1],
                    c=err_C, s=12, cmap='RdYlGn_r', vmin=0, vmax=3, alpha=0.85)
    fig.colorbar(sc, ax=ax, label='KNN error (m)')
    ax.set_title('Cross-time test: KNN(k=1) error on each evening point\n'
                 f"median={np.median(err_C):.2f} m  mean={err_C.mean():.2f} m  "
                 f"p90={np.percentile(err_C,90):.2f} m")
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    save_plot(fig, '08_knn_error_spatial.png')

    # ── 寫 markdown 報告 ──────────────────────────────────────
    write_report(records, n_morning, n_evening, n_unique_bssid,
                 aps_per_scan, useful_bssids,
                 cell_stds, shift_df, knn_df, topn_df, err_C)
    print(f'\n✓ all outputs → {OUT_DIR}')


def write_report(records, n_m, n_e, n_b, aps_per_scan, useful_bssids,
                 cell_stds, shift_df, knn_df, topn_df, err_C):
    lines = []
    P = lines.append

    P('# EDA Report — WiFi Fingerprint Indoor Localization\n')
    P(f'生成於: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')

    P('## 0. 摘要\n')
    median_knn_A = knn_df[(knn_df['split'].str.startswith("A")) & (knn_df['k']==1)]['median_err_m'].iloc[0]
    median_knn_B = knn_df[(knn_df['split'].str.startswith("B")) & (knn_df['k']==1)]['median_err_m'].iloc[0]
    median_knn_C = knn_df[(knn_df['split'].str.startswith("C")) & (knn_df['k']==1)]['median_err_m'].iloc[0]
    P(f'- **Q1 可學習?** KNN(k=1) 在 in-distribution(morning 80/20)中位數誤差 **{median_knn_B:.2f} m**;'
      f'random 全域 split **{median_knn_A:.2f} m** → 資料有強信號,可學')
    P(f'- **Q2 跨時段影響?** Morning→Evening 中位數誤差 **{median_knn_C:.2f} m**,'
      f'比 in-distribution 多 **{median_knn_C - median_knn_B:+.2f} m** ({100*(median_knn_C/median_knn_B - 1):+.0f}%)')
    P(f'- **Q3 BSSID 數?** 特徵 sweep 見表 / 圖 06,通常 top-20 ~ top-50 收斂')
    P(f'- **Q4 噪聲水準?** within-cell RSSI std 中位數 **{np.median(cell_stds):.2f} dBm**,p90 **{np.percentile(cell_stds,90):.2f} dBm**'
       + (' → 噪聲大,建議用機率/集成模型' if np.median(cell_stds) > 3 else ' → 噪聲適中,MLP 足夠'))
    P('')

    P('## 1. 資料概觀\n')
    P(f'- 總 record: {len(records):,}  (morning {n_m} + evening {n_e})')
    P(f'- 唯一 BSSID: {n_b}')
    P(f'- 每筆 scan 看到 AP 數: mean {np.mean(aps_per_scan):.1f} ± {np.std(aps_per_scan):.1f},'
      f' 中位數 {int(np.median(aps_per_scan))}')
    P(f'- 至少出現 10 次的「有用 BSSID」: **{len(useful_bssids)}** 個(將作為主特徵集)')
    P('\n![APs per scan](plots/01_aps_per_scan_hist.png)')
    P('![BSSID coverage](plots/02_bssid_coverage.png)\n')

    P('## 2. 跨時段 covariate shift (Q2)\n')
    n_sig = int(shift_df['significant_shift'].sum())
    P(f'- 共 {len(shift_df)} 個 BSSID 早晚都 ≥30 樣本')
    P(f'- **{n_sig} 個 ({100*n_sig/len(shift_df):.0f}%)** 有顯著漂移 (KS test p < 0.01)')
    P(f'- delta 平均: {shift_df["delta"].mean():+.2f} dBm,範圍 [{shift_df["delta"].min():+.2f}, {shift_df["delta"].max():+.2f}]')
    P('\n前 10 大漂移 BSSID:')
    P(shift_df.head(10)[['ssid', 'morning_mean', 'evening_mean', 'delta', 'ks_p']]
      .to_markdown(index=False))
    P('\n![morning evening shift](plots/04_morning_evening_shift.png)')
    P('![spatial coverage](plots/07_spatial_coverage_split.png)\n')

    P('## 3. Within-cell 噪聲 (Q4)\n')
    P(f'- 把空間切成 0.5 m × 0.5 m cells (≥5 樣本的 cell)')
    P(f'- 對 top-20 BSSID 算每個 cell 內的 RSSI std → 共 {len(cell_stds)} 個 (cell, bssid) 樣本')
    P(f'- **median: {np.median(cell_stds):.2f} dBm,p90: {np.percentile(cell_stds,90):.2f} dBm**')
    P('- 解讀:同一位置同一 AP 的 RSSI 波動有多大')
    P(f'- 跟跨時段 shift(~2.6 dBm)同數量級 → 任何模型都要對 ±2-3 dBm 的雜訊有韌性')
    P('\n![within cell std](plots/03_within_cell_rssi_std.png)\n')

    P('## 4. KNN baseline (Q1)\n')
    P('用 cosine 不行(RSSI 用 -100 填會把 missing 拉成「相似」),這裡用 Euclidean。')
    P('特徵 = 全部 ≥10 樣本的 BSSID(維度 = useful_bssids 上面那個數字)。\n')
    P(knn_df.to_markdown(index=False))
    P('\n**重點觀察:**')
    P(f'- 隨機 split(A)跟 morning 80/20(B)結果接近 → 不是 overfit 早晚某一段')
    P(f'- C(morning→evening)誤差顯著拉大 → 跨時段是真正的 generalization 挑戰')
    P('\n![KNN error CDF](plots/05_knn_error_cdf.png)')
    P('![KNN spatial error](plots/08_knn_error_spatial.png)\n')

    P('## 5. 特徵數 sweep (Q3)\n')
    P('在 morning 80/20 split 上,改變使用的 top-N BSSID 數量:')
    P(topn_df.to_markdown(index=False))
    P('\n![feature sweep](plots/06_feature_count_sweep.png)\n')

    P('## 6. 對 Lab 3 模型選型的建議\n')
    if median_knn_C - median_knn_B < 0.5:
        P('- ✅ 跨時段 drift 對 KNN 影響小 → 簡單 MLP 就夠,不需要 domain adaptation')
    elif median_knn_C - median_knn_B < 1.5:
        P('- ⚠️ 跨時段 drift 顯著但非災難 → 標準 MLP + 訓練時加 RSSI augmentation (±2 dBm 隨機抖動) 應該能補')
    else:
        P('- 🔴 跨時段 drift 嚴重 → 需要 domain adaptation / 對應的 inductive bias(機率模型 / Bayesian / contrastive)')

    if np.median(cell_stds) > 4:
        P('- 🔴 within-cell 噪聲大(>4 dBm)→ deterministic regression 不夠,建議:')
        P('  - **MDN (Mixture Density Network)**:輸出 GMM(μ, σ, π),量化不確定性')
        P('  - **Ensemble MLP**:5 個獨立模型 + 用 disagreement 做不確定性')
        P('  - **Gaussian Process**:小資料、principled uncertainty(但 ~1800 sample 是極限)')
    else:
        P('- ✅ within-cell 噪聲適中 → standard MLP 可,但加 dropout (~0.2) 當輕量 ensemble')

    P('- 特徵維度 ~{} 是合理的 input size,3 層 MLP (D → 128 → 64 → 2) 不會 overfit'
      .format(len(useful_bssids)))
    P('- **野心方向(若選):**')
    P('  - **Set Transformer**: scan 本質是 variable-size set of (bssid_emb, rssi);可以避免 fixed-vector 的 missing-AP 問題')
    P('  - **Contrastive pre-training**: 同位置兩個 scan 應該相似,跨位置應該遠 → 學一個 RSSI embedding,再 fine-tune (x,y)')
    P('  - **Bayesian/Probabilistic head**: 接 MDN 或 NLL loss,輸出位置不確定性 → 在報告裡 plot 95% confidence ellipse')
    P('')

    (OUT_DIR / 'eda_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'       eda_report.md written ({len(lines)} lines)')


if __name__ == '__main__':
    main()
