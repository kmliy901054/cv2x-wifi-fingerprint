# Lab 3 模型演進史 — 從 1.57 m 到 0.79 m

> 一張總流程圖 + 每個歷代模型的架構快照 + 設計決策樹

---

## Part 1:總演進流程

```
                              KNN k=5  (sklearn baseline)
                              1.568 m
                                  │
                                  │ 加學習能力
                                  ▼
                              MLP (53k)
                              1.34 m
                                  │
                                  │ 多模態 posterior
                                  ▼
                              MDN  (56k, K=3 Gaussian)
                              1.20 m
                                  │
                                  │ 告訴模型「-100 是 padding 不是真實值」
                                  ▼
                              MaskedMDN (76k)
                              1.20 m
                                  │
                                  │ 改用「集合」表示變長 AP scan
                                  ▼
                              Set Transformer MDN (208k)
                              1.093 m
                                  │
                                  │ 加大模型 + 強正則化
                                  ▼
                              Big Set Transformer (619k)
                              1.10 m  (5-seed ens: 1.083)
                                  │
                                  │ 攻擊「空間覆蓋稀疏」
                                  ▼
                              + 5000 GP synth
                              0.906 m  (5-seed ens: 0.889) ← 大躍進
                                  │
                                  │
            ┌─────────────────────┼────────────────────┐
            ▼                     ▼                    ▼
        ✗ C-Mixup            Heatmap + mask        (Enhanced features
        0.942 m              0.836 m (5-ens)        — combo saturate)
        (干擾,雙重           換掉 MDN head,用
        augmentation)         40×33 grid + free-mask
                                  │
                                  │ 加 coarse head 抑制 outlier
                                  ▼
                              Cascade 2-level (882k)
                              0.778 m (best single)
                              0.793 m (5-ens geom)  ★ 冠軍
                                  │
        ┌─────────────────────────┼─────────────────────┐
        ▼                         ▼                     ▼
   ✗ CNN xattn (1.15M)       ✗ A+B combo          ✗ 激進路線
   0.907 m                   0.886 m              Phase 1: 3-cascade
   (map 是常數,無新訊息)     (A 拖累 B)          + 0.25m fine + 12k
                                                  synth + 10-seed
                                                  0.800 m (+0.9%)
                                                  Phase 2: Diffusion
                                                  1.821 m (+130%)
```

### 階段標記

| 階段 | 主要技術 | 收益 |
|------|---------|------|
| **A. 表徵升級** | KNN → MLP → MDN → Set Transformer | -30% |
| **B. 容量擴大** | Big config + ensemble | -1% |
| **C. 資料增強** | GP synth | -18% ⭐ |
| **D. 輸出端改造** | MDN → Heatmap → Cascade | -11% ⭐ |
| **E. 進一步擴大(失敗)** | 3-cascade、Diffusion、CNN xattn | 0% 或退步 |

---

## Part 2:歷代模型架構快照

### 1. KNN k=5 — Classical Fingerprinting (1.568 m)

```
  test scan X_q
       │
       ▼
  ┌──────────────────────────────────┐
  │  歐式距離搜尋 K=5 個最近鄰       │
  │  (over 1,449 training fingerprint)│
  └──────────────────────────────────┘
       │
       ▼
  weights = 1 / (dist + ε),歸一化
       │
       ▼
  pred = Σ_k w_k × y_k  (加權平均)
```
- 完全無「學習」
- 推論時間隨資料量線性增加
- **下限基準**

---

### 2. MLP — Vanilla Regressor (53k, 1.34 m)

```
  X (80-dim RSSI vector, -100 padded)
       │
       ▼
  Linear(80→256) → ReLU → Dropout(0.2)
       │
       ▼
  Linear(256→128) → ReLU → Dropout(0.2)
       │
       ▼
  Linear(128→2)
       │
       ▼
  pred (x, y) ∈ ℝ²
       │
  Loss = SmoothL1((x,y), gt)
```
- 第一個 NN,學到「RSSI 模式 → 位置」的非線性映射
- 缺點:單一點輸出,**無法處理多模態 posterior**

---

### 3. MDN — Mixture Density Network (56k, 1.20 m)

```
  X (80-dim)
       │
       ▼
  Linear(80→256) → ReLU → Dropout
       │
       ▼
  Linear(256→128) → ReLU → Dropout
       │
       ▼ (shared 128-dim backbone)
       │
  ┌────┼────┐
  ▼    ▼    ▼
  pi_h mu_h logsig_h
  →3   →6   →6
  │    │    │
  log  μ    σ
  π   (3,2) (3,2)
  (3)
       │
  Loss = -log Σ_k π_k · N(y | μ_k, σ_k²)   (NLL)
       │
  Predict: argmax_k π_k → μ_k  (MAP)
```
- 輸出**機率分佈**而非單點
- 處理走廊兩端 RSSI 相似的問題
- 副產物:每筆預測自帶 uncertainty

---

### 4. MaskedMDN (76k, 1.20 m)

```
  X (80-dim)        mask (80-dim, 0/1)
       │                  │
       └────── concat ────┘
                 │
                 ▼ (160-dim)
  Linear(160→256) → ReLU → Dropout
       │
       ▼ ... 同 MDN backbone
       ▼
  log_π, μ, log_σ → NLL loss
```
- 額外告訴模型「哪些 AP 是真的偵測到」
- 防止 -100 padding 被當訊號學

---

### 5. Set Transformer MDN — 變長集合輸入 (208k, 1.093 m)

```
  scan = N 個 (BSSID, RSSI) pairs
  pad to M=50,生成 (idx, val, mask)
       │
       ▼
  bssid_idx (B,50) ── Embedding(81, 32) ──→ (B, 50, 32)
                                              │
  rssi (B,50) ─────────────── concat ─────────┤
                                              ▼
                                    (B, 50, 33)
                                              │
                                    Linear(33→128)
                                              ▼
                                    (B, 50, 128) × mask
                                              │
                              SAB × 2 (4 heads, LN)
                                              ▼
                                    PMA (1 seed query)
                                              ▼
                                       Dropout
                                              ▼
                                       (B, 128)
                                              │
                                    MDN head (K=3)
                                              ▼
                                      pred (x, y)
```
- 第一次處理**變長集合輸入**
- self-attention 學 AP 間關聯
- PMA pooling **學自己關注哪些 AP**

---

### 6. Big Set Transformer MDN (619k, 1.10 m / ens 1.083)

```
  ↓ 同 Set Transformer 但 model_dim=192,
    num_sab=3, K=5, dropout=0.3, weight_decay=1e-3,
    jitter=4 dBm

  (B,50,48) Embedding  ─→ (B,50,49) concat RSSI
       │
       ▼
  Linear(49→192) ─→ (B,50,192) × mask
       │
       ▼
  ┌─────────┐
  │ SAB #1  │ ─ 4 heads, LN
  └─────────┘
       ▼
  ┌─────────┐
  │ SAB #2  │
  └─────────┘
       ▼
  ┌─────────┐
  │ SAB #3  │
  └─────────┘
       ▼
  PMA (1 seed query) → (B, 192)
       │
       ▼  Dropout(0.3)
       ▼
  ┌────────┬────────┬────────────┐
  │ π head │ μ head │ logσ head  │
  │ 192→5  │ 192→10 │ 192→10     │
  └────────┴────────┴────────────┘
       │
  K=5 Gaussian mixture → NLL
```
- 加大 + 加深 + 強正則化
- **5-seed ensemble** 平均消除隨機性

---

### 7. + GP Synthetic Data (no architecture change, **0.906/0.889**)

```
  data augmentation pipeline (offline):

  for each BSSID b (80 個):
    收集 (x, y, rssi) tuples from training
    fit sklearn GaussianProcessRegressor (RBF + WhiteKernel)
    → GP_b(x, y) → estimated RSSI + uncertainty

  KNN detection classifier:
    對位置 (x, y),最近 10 個 train scans 中 BSSID b 偵測率
    → P(detected | x, y, b)

  for 5000 個 free-cell positions:
    對每個 BSSID 用 KNN-prob 決定是否偵測
    若偵測,用 GP_b 取 RSSI
    → 一筆假 scan

  → 5000 合成 scans + 1449 真實 → 6449 訓練樣本
```

**架構不變,但訓練資料變 4.4×。** 模型同 Big Set Transformer.

---

### 8. Heatmap + Free-cell Mask (864k, 0.836 ens-geom)

```
  ┌─────────────────────────────────┐
  │ encoder 完全同 Big Set Trans     │
  │ (Embedding, 3×SAB, PMA, Dropout) │
  │ → (B, 192)                       │
  └─────────────────────────────────┘
                 │
                 ▼ MDN head 整個拿掉
                 ▼
        ┌──────────────────────┐
        │  Linear(192 → 1320)  │  ← 40×33 grid
        │  logits over cells   │     0.4 m / cell
        └──────────────────────┘
                 │
       ┌─────────┴──────────┐
       │ (訓練)             │ (推論)
       ▼                    ▼
  CE(soft target,         masked = logits + log_free_mask
     Gaussian σ=0.5m)      prob = softmax(masked)
                           prob /= prob.sum()
                           (x,y) = Σ prob × cell_xy
                                   ↑ sub-cell expectation
```
- 把 MDN 假設 5 個 Gaussian 換成 **non-parametric distribution over grid**
- **free-cell mask** 從 psquare.pgm 算出,**~30% 不可達區直接歸零**

---

### 9. ⭐ Cascade 2-level Heatmap (882k, **0.793 ens-geom 冠軍**)

```
  encoder 同 Big Set Trans
  → (B, 192) → Dropout(0.3)
                 │
        ┌────────┴────────┐
        ▼                 ▼
  ┌──────────┐    ┌────────────┐
  │ coarse   │    │  fine      │
  │ Linear   │    │  Linear    │
  │ 192→90   │    │  192→1320  │
  │ (10×9)   │    │  (40×33)   │
  │ 1.6m cell│    │  0.4m cell │
  └──────────┘    └────────────┘
        │                 │
        ▼                 ▼
  softmax+mask     softmax+mask
        │                 │
        ▼                 │
  coarse_prob[parent_c]   │     ← lookup by fine→coarse mapping
        │                 │
        └────── × ────────┘     ← cascade gate
                 │
                 ▼
        renormalize
                 │
                 ▼
        Σ prob × cell_xy
                 ▼
            pred (x, y)

  Loss = 0.5·CE(fine_soft, σ=0.4) +
         0.3·CE(coarse_soft, σ=1.0) +
         0.2·SmoothL1(E[xy], y_true)
```
- coarse head 學「我在哪個大區」
- fine head 被 coarse 機率「**蓋台**」 → 抑制走廊頭尾混淆的 outlier
- p90 從 2.67 → 2.56 m
- **5 seed 都 < 0.86,訓練穩定**

---

### 10. ✗ CNN Cross-Attention (1.15M, 0.907)

```
  ┌────────────────────────────────────┐
  │ Set Transformer encoder → (B, 192) │
  └────────────────────────────────────┘
                 │
  ┌──────────────┴──────────────────┐
  │                                 │
  │  MapEncoder (一次性,offline):  │
  │     psquare.pgm                 │
  │     → resize to 56×64           │
  │     → CNN (3 stride-2 convs)    │
  │     → 56 spatial tokens (192-d) │
  │     + 2D positional encoding    │
  │     ─→ (1, 56, 192)             │
  │     → broadcast (B, 56, 192)    │
  └──────────────┬──────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────┐
  │  Cross-Attention                │
  │  query: (B, 1, 192) (pooled)    │
  │  k/v:   (B, 56, 192) (map)      │
  │  → MAB                          │
  │  output: (B, 192) refined       │
  └─────────────────────────────────┘
                 │
                 ▼
        Heatmap head (192→1320)

  期望:模型「看見」地圖幾何
  實際:map 是 global 常數,cross-attn 只
        學「query→固定 keys 的查表」,沒帶來
        per-sample 新訊息,純粹多 capacity 而已。
        **退步 +8.5%**
```

---

### 11. ✗ Phase 1 — 3-level Cascade + 12k synth + 10-seed (1.38M, 0.800)

```
  encoder 同
       │
       ▼ (B, 192)
       │
  ┌────┼──────────────────────┐
  ▼    ▼                      ▼
  ┌──────┐  ┌──────────┐  ┌────────────┐
  │coarse│  │ medium   │  │  fine      │
  │192→90│  │192→594   │  │ 192→3328   │
  │10×9  │  │27×22     │  │ 64×52      │
  │1.6m  │  │0.6m      │  │ 0.25m      │
  └──────┘  └──────────┘  └────────────┘
     │          │              │
     ▼          ▼              ▼
   各自 softmax + mask
     │          │              │
     │          │              │
     └─→×←──────┤              │   ← coarse 蓋 medium
                medium_gated   │
                  │            │
                  └─→×←────────┘   ← medium_gated 蓋 fine
                                      fine_gated_3level
                       │
                       ▼
                   Σ × fine_xy
                       ▼
                   pred (x, y)

  + 12000 GP synth (vs 5000)
  + 10 seeds ensemble

  期望:更細的 grid + 更多 synth + 更多 seed → 大進步
  實際:fine head 192×3328=638k 對 1.4k 真實樣本過大;
        12k synth (synth/real=8:1) 稀釋真實訊號。
        **微輸 2-level Cascade (+0.9%)**
        ← 「不是越多越好」的教訓
```

---

### 12. ✗ Diffusion Head (940k, 1.821)

```
  encoder 同 → (B, 192) cond
                 │
                 ▼
  ┌──────────────────────────────────┐
  │ Conditional Diffusion Head       │
  │                                  │
  │ Train:                           │
  │   y_0 = true (x,y) / 5.0  (norm) │
  │   t ~ Uniform[1, 100]            │
  │   ε ~ N(0, I)                    │
  │   y_t = √(α̅_t)·y_0 + √(1-α̅_t)·ε │
  │   ε̂_θ(y_t, t, cond) = predict ε  │
  │   Loss = MSE(ε̂_θ, ε)             │
  │                                  │
  │ Inference (DDIM 25 steps × 8):   │
  │   y_T ~ N(0, I)                  │
  │   loop t = T → 1:                │
  │     ε̂ = predict(y_t, t, cond)    │
  │     y_0_hat = (y_t − √(1-α̅)·ε̂)  │
  │              / √(α̅)              │
  │     y_{t−1} = √(α̅_next)·y_0_hat  │
  │             + √(1-α̅_next)·ε̂     │
  │   snap to nearest free cell      │
  └──────────────────────────────────┘
                 │
                 ▼
          pred (x, y)

  期望:連續輸出,更彈性的 posterior
  實際:1.4k 真實樣本對 diffusion 學分佈太少;
        走廊兩端時 diffusion 傾向預測「中間」;
        失去 free-cell mask 在訓練時的引導力。
        **災難級失敗 +130%**
```

---

## Part 3:設計決策樹

```
                  WiFi RSSI → (x, y)
                          │
              ┌───────────┴────────────┐
              ▼                        ▼
       「無學習基準」              「學一個 model」
            KNN                          │
         (1.568)                ┌────────┴─────────┐
                                ▼                  ▼
                          固定維度向量         變長集合輸入
                                │                  │
                       MLP/MDN/MaskedMDN     Set Transformer
                          (1.20-1.34)            (1.09)
                                                  │
                                          ┌───────┴────────┐
                                          ▼                ▼
                                    加大正則化            加資料
                                    (Big config)        GP synth
                                       (1.08)            ──────┐
                                          │                    │
                                          └────── 結合 ────────┤
                                                               ▼
                                                          (0.889) ← 之前最佳
                                                               │
                                          ┌────────────────────┴────────────────────┐
                                          ▼                                         ▼
                                  改輸入端(失敗)                            改輸出端
                                  C-Mixup, Enhanced                          MDN→Heatmap
                                  (0.94, saturated)                         (0.836)
                                                                                │
                                                                ┌───────────────┼───────────────┐
                                                                ▼               ▼               ▼
                                                          加 coarse head    加 CNN map     加更多東西
                                                          ⭐ Cascade        ✗ xattn        ✗ Phase 1
                                                          0.793 冠軍       0.907          0.800
                                                                                          ✗ Diffusion
                                                                                          1.821

                                              -------- 物理極限線 ~0.5 m --------
                                                          (16% outlier 來自 RSSI
                                                          symmetric ambiguity,
                                                          需 IMU/multi-scan 才能再降)
```

### 每個分支點的「為什麼這樣選」

| 分支點 | 選了什麼 | 為什麼 |
|--------|---------|--------|
| MLP → MDN | 加 mixture density | 走廊兩端 RSSI 像,單點輸出無法處理 |
| MDN → MaskedMDN | 加 presence mask | -100 padding 被當訊號污染 |
| MaskedMDN → Set Trans | 變長集合表示 | 固定 80 維浪費容量在 padding |
| Set Trans → Big | 加大 + 正則化 | 容量不足 |
| Big → +synth | 合成資料填空間 | 75000 自由格 vs 1.4k 真實點 → 覆蓋稀疏 |
| +synth → C-Mixup | 試另一種 aug | 結果:同瓶頸,雙重 aug 互相干擾 ✗ |
| +synth → Heatmap | 改輸出端 | MDN K=5 Gaussian 套不住走廊型 posterior |
| Heatmap → Cascade | 加 coarse head | p90 outlier 需要先驗濾鏡 |
| Cascade → 各種失敗 | 試更多東西 | 證實 2-level Cascade 已是 sweet spot |

---

## Part 4:結論

**最終冠軍 Cascade 2-level (882k) @ Split A median 0.793 m**

從 KNN baseline 1.568 → 最終 0.793 m,**相對改進 49%**。

**4 個有效改進**(疊加,各貢獻 ~10% 以上):
1. Set Transformer encoder(處理變長輸入)
2. MDN → Heatmap head(打破 K-Gaussian 假設 + free-cell mask)
3. Coarse-to-Fine Cascade(coarse 蓋 fine,抑制 outlier)
4. GP synth + 5-seed geometric-median ensemble

**5 個失敗實驗**(對報告同樣有價值,證明了瓶頸位置):
- C-Mixup(雙重 augmentation 干擾)
- CNN floor-plan cross-attention(map 是 global 常數)
- A+B combo(A 拖累 B)
- 3-level Cascade(過大 fine head 過擬合)
- Diffusion head(失去 free-cell prior)

**錯誤分析**:Cascade 27% 預測已落在 AMCL noise floor (0.3 m) 之內;16% outlier 都是 RSSI symmetric ambiguity,**改架構救不了**。

要繼續壓低需要外部新訊息(IMU、multi-scan 時序),不是模型問題。

---

延伸閱讀:
- 完整模型細節 + 訓練配方 → [ARCHITECTURE.md](ARCHITECTURE.md)
- 報告主文 → [LAB3_REPORT.md](LAB3_REPORT.md)
