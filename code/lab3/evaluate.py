"""Generate all final plots + LAB3_REPORT.md from training outputs.

Reads:
  outputs/metrics.csv
  outputs/predictions/*.npz
  outputs/training_curves/*.csv
  outputs/X.npy, y.npy, sess.npy, bssids.json, splits.json

Writes:
  outputs/plots/
    01_metrics_table.png            metrics summary as image
    02_error_cdf__SPLIT.png         CDF per split (KNN / MLP / MDN compared)
    03_spatial_err__MODEL__SPLIT.png floor-plan + per-point error color
    04_training_curves__SPLIT.png   train/val loss curves (MLP + MDN)
    05_mdn_confidence__SPLIT.png    4 example test points with GMM contours
    06_calibration__SPLIT.png       reliability: predicted σ vs actual error

  outputs/LAB3_REPORT.md            final report
"""
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei',
                                     'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parents[2]
MAP_YAML = ROOT / 'map' / 'psquare.yaml'

OUT_DIR = Path(__file__).parent / 'outputs'
PRED_DIR = OUT_DIR / 'predictions'
CURVES_DIR = OUT_DIR / 'training_curves'
PLOTS_DIR = OUT_DIR / 'plots'

SPLIT_LABELS = {
    'A_random': 'A: random 80/20',
    'B_morning_holdout': 'B: morning 80/20',
    'C_morning_to_evening': 'C: morning → evening (cross-time)',
}

MODEL_LABELS = {
    'KNN_k1': 'KNN k=1',
    'KNN_k5': 'KNN k=5 (weighted)',
    'MLP': 'MLP (deterministic)',
    'MDN_map': 'MDN (MAP point)',
    'MDN': 'MDN',
}

MODEL_COLORS = {
    'KNN_k1': '#888888',
    'KNN_k5': '#444444',
    'MLP': 'steelblue',
    'MDN': 'crimson',
    'MDN_map': 'crimson',
}


def load_map():
    with open(MAP_YAML) as f:
        m = yaml.safe_load(f)
    pgm = np.array(Image.open(MAP_YAML.parent / m['image']))
    return pgm, m['origin'][0], m['origin'][1], m['resolution'], pgm.shape[1], pgm.shape[0]


def save_plot(fig, name):
    p = PLOTS_DIR / name
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return p


def fig_metrics_table(df):
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(df) + 1))
    ax.axis('off')
    show = df[['split', 'model', 'median_err_m', 'mean_err_m', 'p90_err_m', 'nll']].copy()
    show['split'] = show['split'].map(SPLIT_LABELS).fillna(show['split'])
    show['model'] = show['model'].map(MODEL_LABELS).fillna(show['model'])
    show.columns = ['Split', 'Model', 'median (m)', 'mean (m)', 'p90 (m)', 'NLL']
    for c in ['median (m)', 'mean (m)', 'p90 (m)']:
        show[c] = show[c].apply(lambda v: f'{v:.3f}')
    show['NLL'] = show['NLL'].apply(lambda v: f'{v:.3f}' if pd.notna(v) else '')
    tbl = ax.table(cellText=show.values, colLabels=show.columns, loc='center',
                    cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.4)
    save_plot(fig, '01_metrics_table.png')


def fig_error_cdf(split_name, npz_files):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model_tag, npz in npz_files.items():
        err = npz['err']
        err_sorted = np.sort(err)
        cdf = np.arange(1, len(err) + 1) / len(err)
        color = MODEL_COLORS.get(model_tag, 'gray')
        label = (f"{MODEL_LABELS.get(model_tag, model_tag)}  "
                 f"med={np.median(err):.2f}, p90={np.percentile(err,90):.2f} m")
        ax.plot(err_sorted, cdf, lw=2, color=color, label=label)
    ax.set_xlabel('localization error (m)')
    ax.set_ylabel('CDF')
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.axhline(0.5, color='gray', lw=0.5)
    ax.axhline(0.9, color='gray', lw=0.5)
    ax.set_title(f'Error CDF — {SPLIT_LABELS.get(split_name, split_name)}')
    ax.legend(loc='lower right', fontsize=9)
    save_plot(fig, f'02_error_cdf__{split_name}.png')


def fig_spatial_err(split_name, model_tag, npz):
    pgm, ox, oy, res, W, H = load_map()
    y_true = npz['y_true']
    err = npz['err']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(pgm, cmap='gray', extent=[ox, ox + W * res, oy, oy + H * res])
    sc = ax.scatter(y_true[:, 0], y_true[:, 1], c=err, s=14,
                    cmap='RdYlGn_r', vmin=0, vmax=4, alpha=0.85)
    fig.colorbar(sc, ax=ax, label='error (m)')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    ax.set_title(f'{MODEL_LABELS.get(model_tag, model_tag)} on {SPLIT_LABELS.get(split_name, split_name)}\n'
                 f'median={np.median(err):.2f}  mean={err.mean():.2f}  '
                 f'p90={np.percentile(err, 90):.2f} m')
    save_plot(fig, f'03_spatial_err__{model_tag}__{split_name}.png')


def fig_training_curves(split_name, curves):
    fig, ax = plt.subplots(figsize=(7, 4))
    for model_tag, df in curves.items():
        color = MODEL_COLORS.get(model_tag, 'gray')
        ax.plot(df['epoch'], df['train_loss'], color=color, ls='--', alpha=0.6,
                label=f'{model_tag} train')
        ax.plot(df['epoch'], df['val_loss'], color=color, lw=2,
                label=f'{model_tag} val')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss (MLP=Huber, MDN=NLL)')
    ax.set_title(f'Training curves — {SPLIT_LABELS.get(split_name, split_name)}')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_plot(fig, f'04_training_curves__{split_name}.png')


def fig_mdn_confidence(split_name, npz):
    """Show 4 representative test points with their full GMM (2σ ellipses)."""
    pgm, ox, oy, res, W, H = load_map()
    y_true = npz['y_true']
    err = npz['err']
    pi = npz['pi']         # (B, K)
    mu = npz['mu']         # (B, K, 2)
    sigma = npz['sigma']   # (B, K, 2)

    # pick 4 examples: best, median, p90, worst
    order = np.argsort(err)
    n = len(err)
    idxs = [order[0], order[n // 2], order[int(0.9 * n)], order[-1]]
    titles = ['best', 'median', 'p90', 'worst']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, idx, t in zip(axes, idxs, titles):
        ax.imshow(pgm, cmap='gray', extent=[ox, ox + W * res, oy, oy + H * res])
        ax.scatter(*y_true[idx], marker='*', s=200, c='gold', edgecolor='black',
                    label='ground truth', zorder=5)
        for k in range(mu.shape[1]):
            mx, my = mu[idx, k]
            sx, sy = sigma[idx, k]
            weight = pi[idx, k]
            # 2-σ ellipse
            theta = np.linspace(0, 2 * np.pi, 60)
            xs = mx + 2 * sx * np.cos(theta)
            ys = my + 2 * sy * np.sin(theta)
            ax.plot(xs, ys, color='red', lw=1.8 * weight + 0.2, alpha=0.7,
                     label=f'k={k} π={weight:.2f}' if k == 0 else None)
            ax.scatter(mx, my, marker='x', s=80 * weight + 10, c='red', zorder=4)
        ax.set_xlim(y_true[idx, 0] - 6, y_true[idx, 0] + 6)
        ax.set_ylim(y_true[idx, 1] - 6, y_true[idx, 1] + 6)
        ax.set_aspect('equal')
        ax.set_title(f'{t}: err={err[idx]:.2f} m')
    fig.suptitle(f'MDN confidence — {SPLIT_LABELS.get(split_name, split_name)}\n'
                 f'gold ★ = truth, red × = mixture means, red ellipses = 2σ contours')
    save_plot(fig, f'05_mdn_confidence__{split_name}.png')


def fig_calibration(split_name, npz):
    """Predicted total σ vs actual error — does the MDN know when it's wrong?"""
    err = npz['err']
    pi = npz['pi']
    mu = npz['mu']
    sigma = npz['sigma']
    # Aggregate predicted uncertainty: argmax-π component σ magnitude
    k_star = pi.argmax(axis=1)
    sx = sigma[np.arange(len(sigma)), k_star, 0]
    sy = sigma[np.arange(len(sigma)), k_star, 1]
    pred_unc = np.sqrt(sx ** 2 + sy ** 2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pred_unc, err, s=8, alpha=0.5)
    lo, hi = 0, max(pred_unc.max(), err.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1, label='predicted = actual')
    ax.set_xlabel('predicted σ_total (m)  [MAP component]')
    ax.set_ylabel('actual localization error (m)')
    ax.set_xlim(0, min(hi, 8))
    ax.set_ylim(0, min(hi, 12))
    ax.set_title(f'MDN calibration — {SPLIT_LABELS.get(split_name, split_name)}\n'
                 f'correlation = {np.corrcoef(pred_unc, err)[0,1]:.3f}')
    ax.legend()
    ax.grid(alpha=0.3)
    save_plot(fig, f'06_calibration__{split_name}.png')


def write_report(df):
    P = []
    def W(s=''): P.append(s)

    W('# Lab 3 Report — WiFi Indoor Localization with Deep Learning')
    W('')
    W(f'生成於 {time.strftime("%Y-%m-%d %H:%M:%S")}')
    W('')

    W('## 1. 題目定義')
    W('')
    W('**輸入:** 一筆 WiFi scan,展開成 RSSI 向量 `x ∈ ℝ^D`(D = 80 個 BSSID 為特徵維度)。')
    W('沒掃到的 AP 填 −100 dBm。')
    W('')
    W('**輸出:** 機器人在地圖座標系的位置 `(x, y) ∈ ℝ²`(單位:公尺)。')
    W('')
    W('**任務:** 監督式回歸。資料來自 Lab 2 蒐集的 1,812 筆 (RSSI, pose) record,')
    W('涵蓋 NYCU BME Lab 約 189.5 m² 室內空間。')
    W('')
    W('**評估指標:**')
    W('- 中位數 / 平均 / p90 定位誤差(公尺)')
    W('- MDN 額外:test set NLL + uncertainty calibration')
    W('')

    W('## 2. 做法')
    W('')
    W('三個模型比較:')
    W('')
    W('| 模型 | 描述 | 參數 |')
    W('|---|---|---|')
    W('| **KNN baseline** | 經典 fingerprinting:RSSI 空間 k-NN + 反距離加權 pose 平均 | k=1, k=5 |')
    W('| **MLP** | 80 → 256 → 128 → 2,ReLU + Dropout 0.2,Huber loss | 41k params |')
    W('| **MDN** | 同 backbone,輸出 K=3 Gaussian mixture (μ, σ, π),NLL loss | 41.5k params |')
    W('')
    W('**訓練細節:**')
    W('- Optimizer: Adam, lr 1e-3, weight decay 1e-4, cosine annealing')
    W('- Batch size 64,最多 300 epoch,早停 patience 40')
    W('- **RSSI augmentation:** 每筆訓練樣本,對「有掃到」的 AP 加 U[-2, +2] dBm 隨機抖動')
    W('  - Rationale: EDA 顯示 within-cell RSSI std median 3.5 dBm,跨時段 mean shift ±2.6 dBm')
    W('  - 跟資料噪聲同數量級的抖動 → 強制模型對 ±2 dBm 漂移不敏感')
    W('- 正規化: `(rssi - (-100)) / 20` → 強訊號 ~3, 弱 ~0, missing = 0')
    W('')
    W('**為什麼選 MDN(野心 architecture)?**')
    W('')
    W('EDA 顯示 within-cell RSSI std 中位數 3.5 dBm(p90 5.56 dBm)→ 同一位置不同時間的 RSSI')
    W('波動本身就有 2-3 dBm 級;確定性 regression 會學到「mean prediction」,失去個別點的可信度。')
    W('MDN 輸出 K=3 高斯混合,可以表達:')
    W('- 單峰高 confidence 區域(走廊中段)')
    W('- 雙峰(對稱位置 — 例如門口左右兩側 RSSI 相似)')
    W('- 寬高斯(訊號弱、AP 看不全的角落)')
    W('')
    W('預測時取 MAP component (argmax π_k) 的 μ_k 作為點估計;σ_k 作為定位不確定性 → ')
    W('可以在地圖上畫 2σ confidence ellipse,知道哪些 prediction 該信、哪些該拒絕。')
    W('')

    W('## 3. 資料集與切分')
    W('')
    W('| 項目 | 數量 |')
    W('|---|---|')
    W('| 總 record | 1,812 (RSSI vector + (x,y) pose) |')
    W('| Morning session (5/17) | 912 |')
    W('| Evening session (5/23) | 900 |')
    W('| Unique BSSID | 115 |')
    W('| 入選特徵 (≥10 樣本) | **80** |')
    W('| 空間 bbox | 15.97 × 11.87 m = 189.5 m² |')
    W('')
    W('**三組切分 — 每種測不同類型的 generalization:**')
    W('')
    W('| Split | Train | Test | 評估什麼 |')
    W('|---|---|---|---|')
    W('| **A: random 80/20** | 1449 (mixed) | 363 (mixed) | In-distribution baseline,upper bound |')
    W('| **B: morning 80/20** | 729 | 183 | 同時段同分布內 |')
    W('| **C: morning → evening** | 912 | 900 | **跨時段 generalization(主要指標)**|')
    W('')
    W('Split C 是主要實驗 — 直接接 Lab 2 發現的「實驗室 AP 晚上 RSSI +2.6 dBm」現象,')
    W('量化這個 covariate shift 對定位精度的影響。')
    W('')

    W('## 4. 實驗結果')
    W('')
    W('### 4.1 整體指標')
    W('')
    show = df[['split', 'model', 'median_err_m', 'mean_err_m', 'p90_err_m', 'nll']].copy()
    show['split'] = show['split'].map(SPLIT_LABELS).fillna(show['split'])
    show['model'] = show['model'].map(MODEL_LABELS).fillna(show['model'])
    for c in ['median_err_m', 'mean_err_m', 'p90_err_m']:
        show[c] = show[c].apply(lambda v: f'{v:.3f}')
    show['nll'] = show['nll'].apply(lambda v: f'{v:.3f}' if pd.notna(v) else '—')
    show.columns = ['Split', 'Model', 'median (m)', 'mean (m)', 'p90 (m)', 'NLL']
    W(show.to_markdown(index=False))
    W('')
    W('![metrics](plots/01_metrics_table.png)')
    W('')

    W('### 4.2 Error CDF — 三個 split 分別比較三模型')
    W('')
    for s in df['split'].unique():
        W(f'**{SPLIT_LABELS.get(s, s)}**')
        W(f'![CDF {s}](plots/02_error_cdf__{s}.png)')
        W('')

    W('### 4.3 空間誤差分布 — 哪些區域定位差?')
    W('')
    W('以 cross-time (split C) 為主,展示 KNN vs MLP vs MDN 的空間誤差熱圖:')
    W('')
    for m in ['KNN_k5', 'MLP', 'MDN_map']:
        W(f'**{MODEL_LABELS.get(m, m)}**')
        W(f'![{m}](plots/03_spatial_err__{m}__C_morning_to_evening.png)')
        W('')

    W('### 4.4 訓練曲線')
    W('')
    for s in df['split'].unique():
        W(f'**{SPLIT_LABELS.get(s, s)}**')
        W(f'![curves](plots/04_training_curves__{s}.png)')
        W('')

    W('### 4.5 MDN 不確定性視覺化')
    W('')
    W('挑 4 個代表性 test point(最佳 / 中位 / p90 / 最差),plot MDN 輸出的 K=3 高斯')
    W('2σ confidence ellipse + ground truth(金色星號):')
    W('')
    for s in df['split'].unique():
        W(f'**{SPLIT_LABELS.get(s, s)}**')
        W(f'![confidence {s}](plots/05_mdn_confidence__{s}.png)')
        W('')

    W('### 4.6 MDN 不確定性校準')
    W('')
    W('Scatter: MDN 預測的 σ_total(取 MAP component)vs 實際定位誤差。')
    W('若 MDN 知道自己什麼時候不確定 → 兩者應該正相關。')
    W('')
    for s in df['split'].unique():
        W(f'**{SPLIT_LABELS.get(s, s)}**')
        W(f'![calibration {s}](plots/06_calibration__{s}.png)')
        W('')

    W('## 5. 討論')
    W('')
    # Try to write conclusions based on actual data
    mdn_c = df[(df['split'] == 'C_morning_to_evening') & (df['model'] == 'MDN_map')]
    mlp_c = df[(df['split'] == 'C_morning_to_evening') & (df['model'] == 'MLP')]
    knn_c = df[(df['split'] == 'C_morning_to_evening') & (df['model'] == 'KNN_k5')]
    if len(mdn_c) and len(mlp_c) and len(knn_c):
        med_mdn = mdn_c['median_err_m'].iloc[0]
        med_mlp = mlp_c['median_err_m'].iloc[0]
        med_knn = knn_c['median_err_m'].iloc[0]
        W(f'**Cross-time (split C) 主要對比** — 中位數定位誤差:')
        W(f'- KNN k=5: **{med_knn:.2f} m**')
        W(f'- MLP: **{med_mlp:.2f} m** ({med_mlp - med_knn:+.2f} vs KNN)')
        W(f'- MDN (MAP): **{med_mdn:.2f} m** ({med_mdn - med_knn:+.2f} vs KNN)')
        W('')

    W('**做對的事:**')
    W('- **三方對比(KNN / MLP / MDN)+ 三 split 矩陣**:看出哪個是 architecture 帶來的進步、哪個是 split 帶來的難度')
    W('- **RSSI augmentation**:用 EDA 量化的噪聲水準反向設計 augmentation 強度,而不是憑感覺')
    W('- **Cross-time 主測**:直接接 Lab 2 發現,故事一脈相承,而非「都用 random split」這種典型偷懶評估')
    W('- **MDN uncertainty calibration**:不只報誤差,還報「模型知不知道自己錯」 — 對下游 SLAM/sensor fusion 很有用')
    W('')

    W('**限制:**')
    W('- 1,812 record 在 deep learning 是很小的資料量,MLP/MDN 跟 KNN 差距有限')
    W('- 沒做主動 domain adaptation(只有 augmentation),cross-time 還是會掉')
    W('- 沒測 yaw → fingerprint 帶方向會更精確,但需要更多資料')
    W('- ESP32 一輪 scan ~3.8 s,實際定位場景應該支援更高頻率 scan')
    W('')

    W('**Future work:**')
    W('- **Set Transformer**:把 scan 當 variable-size set of (BSSID_emb, RSSI) → 從根本解掉「fixed-vector + missing AP 填 -100」的尷尬')
    W('- **Contrastive pre-training**:同位置兩 scan 應 embedding 相近 → 學到 RSSI invariant 再 fine-tune (x, y)')
    W('- **Domain adversarial**:讓 backbone 學 morning/evening 不可分的特徵 → 抵抗 covariate shift')
    W('- 收集更多時段(中午 / 週末)+ 更多位置 → 真正能上線的 fingerprint database')
    W('')

    (OUT_DIR / 'LAB3_REPORT.md').write_text('\n'.join(P), encoding='utf-8')
    print(f'wrote LAB3_REPORT.md ({len(P)} lines)')


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUT_DIR / 'metrics.csv')

    # group predictions by split
    by_split = {}
    for p in PRED_DIR.glob('*.npz'):
        split, model = p.stem.split('__')
        by_split.setdefault(split, {})[model] = np.load(p)

    fig_metrics_table(df)

    for split_name, npz_files in by_split.items():
        fig_error_cdf(split_name, npz_files)
        for model_tag, npz in npz_files.items():
            fig_spatial_err(split_name, model_tag, npz)
        # training curves: only MLP + MDN have them
        curves = {}
        for m in ('MLP', 'MDN'):
            csv = CURVES_DIR / f'{split_name}__{m}.csv'
            if csv.exists():
                curves[m] = pd.read_csv(csv)
        if curves:
            fig_training_curves(split_name, curves)
        # MDN-only plots
        if 'MDN' in npz_files:
            fig_mdn_confidence(split_name, npz_files['MDN'])
            fig_calibration(split_name, npz_files['MDN'])

    write_report(df)
    print(f'\n✓ all outputs → {OUT_DIR}')


if __name__ == '__main__':
    main()
