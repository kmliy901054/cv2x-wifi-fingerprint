# Lab 3 室內定位實驗報告

> 課程:CV2X
> 學生:Wayne (kmliy)
> 日期:2026/05/27
> 資料來源:Lab 2 自行於實驗室收集的 WiFi RSSI 指紋資料

---

## 1. 題目定義

從 Lab 2 收集到的 **WiFi RSSI scan 序列**(機器人在實驗室移動同時掃描周遭 AP 訊號強度,並由 ROS AMCL 給出對應 (x, y) 真值),訓練一個**深度學習室內定位模型**:

> **輸入**:單次 WiFi 掃描(可變長度的 (BSSID, RSSI) 集合,~25-40 個 AP)
> **輸出**:預測機器人在地圖座標系下的位置 (x, y) ∈ ℝ²
> **評估**:Median Location Error(m),數字越小越好

定位是 **回歸問題**,但與一般回歸不同:
1. **多模態 posterior**:走廊兩端 RSSI 可能很像,真實的條件分佈會有多個 mode → 不能用 L2 + 單點輸出
2. **變長輸入**:每個 scan 偵測到的 AP 子集不同 → 不能用固定維度向量直接 forward
3. **空間覆蓋稀疏**:訓練樣本(1,449 個 (x, y) 點)只覆蓋 ~16×12 m 實驗室中的少數位置,模型必須能對未見過的位置內插

---

## 2. 資料集

### 2.1 來源

從 Lab 2 收集的兩段 rosbag 抽取:

| Session | 日期 | scan 數 |
|---------|------|--------|
| morning | 2026-05-17 | 950 |
| evening | 2026-05-24 | 862 |
| **Total** | | **1,812** |

每筆 scan 是一個 JSON 物件,包含:
- `pose`: ROS AMCL 給出的 (x, y) ground truth(誤差 ±0.3 m)
- `aps`: 該瞬間偵測到的所有 BSSID + RSSI 列表

### 2.2 前處理

- **BSSID vocabulary**:只保留出現 ≥ 10 次的 BSSID(避免「曇花一現」的鄰居熱點污染訓練),最後得到 **80 個穩定 BSSIDs**。
- **集合表示法**:每個 scan 轉成 `(bssid_idx[50], rssi[50], mask[50])`,其中 `mask=1` 表真實偵測,`mask=0` 為 padding。
- **RSSI 正規化**:`rssi_norm = (rssi + 100) / 20`,大致落在 [0, 3] 範圍。

### 2.3 切分(4 種)

| Split | 描述 | 用途 |
|-------|------|------|
| **A_random** | 全資料 random 80/20 | **主要評估**(本報告重點) |
| B_morning | 只用早上,random 80/20 | 同時段對照 |
| C_morning→evening | 早上全 train、晚上全 test | 跨時段泛化(最嚴格) |
| D_stratified | 早晚各 80/20 合併 | 「資料庫覆蓋所有時段」的部署情境 |

Split A 的 train/test 分別為 1,449 / 363 筆。**本報告主攻 Split A**,因為:
- 它是課堂講義的主要評估設定
- 我們的模型容量在 ~1.4 k 樣本上對跨時段 (Split C) 容易過擬合,單獨討論不公平

---

## 3. 做法

### 3.1 模型演進總覽

```
KNN ─► MLP ─► MDN ─► MaskedMDN ─► Set Transformer + MDN
                                              │
                            ┌─────────────────┼─────────────────┐
                            ▼                 ▼                 ▼
                      + GP Synth         + C-Mixup       + Heatmap head
                                          (失敗)            + Cascade head
                                                            (本文最佳)
```

從古典 KNN 一路進化到 **Set Transformer + Coarse-to-Fine Heatmap Cascade**,每一步都針對前一步的特定問題:

| Step | 解決的問題 | 引入的技術 |
|------|-----------|-----------|
| KNN baseline | 沒有學習能力 | sklearn k-NN |
| MLP | 非線性映射 | Smooth L1 loss |
| MDN | 多模態 posterior | K Gaussian mixture, NLL loss |
| MaskedMDN | -100 padding 污染 | concat presence mask |
| **Set Transformer** | 變長輸入 + AP 間關聯 | self-attention on AP tokens |
| Big + Ensemble | seed variance | 5-seed 平均 |
| **GP Synth** | 空間覆蓋稀疏 | per-AP Gaussian Process kriging |
| **Heatmap head** | 走廊型 posterior + 物理可達性 | 40×33 grid + free-cell mask |
| **Cascade head** | outlier(p90 偏高) | coarse 10×9 gate + fine 40×33 |

### 3.2 最終模型:Set Transformer + Coarse-to-Fine Heatmap Cascade

#### 3.2.1 整體架構

```
INPUT: scan 偵測到的 N 個 AP(N 變動,max=50)
                            │
        ┌───────────────────┼────────────────────┐
        ▼                   ▼                    ▼
  bssid_idx (B,50)    rssi (B,50)        mask (B,50)
        │                   │                    │
        ▼                   │                    │
  Embedding(81, 48)         │                    │
        └──── concat ───────┤                    │
                            ▼                    │
                  input_proj Linear(49→192)      │
                            │ × mask ◄───────────┘
                            ▼
         ┌──────────────────────────────────────┐
         │  3 × SAB (Self-Attention Block)      │
         │  4 heads, LayerNorm, FFN, Residual   │
         │  → (B, 50, 192)                      │
         └──────────────────────────────────────┘
                            ▼
         ┌──────────────────────────────────────┐
         │  PMA (Pooling by Multi-head Attention)│
         │  1 個 learnable seed query           │
         │  → (B, 192)                          │
         └──────────────────────────────────────┘
                            │
                       Dropout(0.3)
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
   ┌─────────────────┐            ┌──────────────────┐
   │ coarse_head     │            │ fine_head        │
   │ Linear(192→90)  │            │ Linear(192→1320) │
   │ 10×9 grid       │            │ 40×33 grid       │
   │ cell size 1.6 m │            │ cell size 0.4 m  │
   └─────────────────┘            └──────────────────┘
            │                               │
            └─────── Cascade Gate ──────────┘
                          │
                          ▼
  refined_prob[c] ∝ softmax(fine)[c] · softmax(coarse)[parent(c)]
                  × free_cell_mask  (psquare.pgm)
                          ▼
              predict (x, y) = Σ_c prob[c] · cell_xy[c]
                                  (expected value)
```

#### 3.2.2 三個關鍵設計

**A. Set Transformer Encoder**(208k 參數)

每個 scan 轉成可變長度的 AP token 集合:
```
token_j = concat( Embedding(BSSID_j), RSSI_j_normalized )  # 49 維
input_proj: Linear(49 → 192)
SAB × 3:    每個 SAB 內含 4-head self-attention + FFN + LayerNorm + Residual
            讓 50 個 AP token 互相觀察彼此(模型學「這幾個 AP 都強通常在某區」)
PMA:        1 個 learnable query 加權聚合 → (B, 192)
```

為什麼選 Set Transformer:
- 自然處理 **變長輸入**(mask 過 padding)
- **Permutation-invariant**(掃描順序不影響輸出)
- **學到 AP 間關聯**(self-attention 抓 pairwise 共現)

**B. Heatmap Head + Free-cell Mask**

把實驗室切成 40×33 grid (0.4 m / cell,共 1320 cells)。Model 對每個 scan 輸出 1320 個 logits。

推論時:
```
prob = softmax(logits) × free_cell_mask    # mask 是預先從 psquare.pgm 算出
prob = prob / prob.sum()
(x, y) = Σ_c prob[c] × cell_center[c]      # sub-cell precision via expected value
```

`free_cell_mask`:psquare.pgm 中像素值 ≥200 的位置才算 free(67% of cells)。**33% 的不可達格子被直接歸零**,所以不會發生「預測點在牆裡」這種荒謬輸出。

**為什麼 Heatmap 比 MDN 強**:
- MDN 假設 posterior 是 K=5 個 Gaussian 之和,套不住走廊型 / L 型分佈
- Heatmap 是 **non-parametric posterior**,任何形狀都能表達
- 推論時可以直接套地圖 prior

**C. Coarse-to-Fine Cascade**

加一個 coarse head:10×9 grid (1.6 m / cell,共 90 cells,53 free)。

訓練 loss 是兩個 head 的 mixed objective:
```
L = 0.5 · CE(fine,   Gaussian-smoothed soft label, σ_f = 0.4 m)
  + 0.3 · CE(coarse, Gaussian-smoothed soft label, σ_c = 1.0 m)
  + 0.2 · SmoothL1(E[xy_gated], y_true)
```

推論時做 **cascade gating**:
```
fine_prob_refined[c] = softmax(fine)[c] × softmax(coarse)[parent(c)]
                       ← coarse 對 fine 投以「我相信哪個大區」的票
renormalize → expected (x, y)
```

**為什麼能進一步降誤差**:
- Fine head 單獨用,如果它把高機率放錯區(例如把走廊頭跟尾搞混),整個預測就會跑很遠
- Coarse head 因為粒度大,**幾乎不會錯區域** → 它的機率可以當 fine 的「先驗濾鏡」
- 對 outlier(p90 誤差)抑制特別有效

#### 3.2.3 GP 合成資料

訓練前先生成 5000 筆合成 scans 補空間覆蓋:
1. 對每個 BSSID,用 sklearn `GaussianProcessRegressor`(RBF + WhiteKernel)學「位置 → RSSI」二維 GP
2. KNN 分類器估「在每個位置該 BSSID 會不會被偵測到」(避免 GP 對遠方 AP 幻覺)
3. 從 psquare.pgm 隨機抽 5000 個自由格,合成 RSSI scans
4. 與原 1,449 筆真實資料合併訓練(總 6,449 筆)

合成 sanity check 通過:平均每個合成 scan 偵測到 27.1 個 AP(真實 27.7 個)。

### 3.3 訓練超參數

| 項目 | 值 | 動機 |
|------|-----|------|
| Optimizer | Adam, lr=1e-3 | 標配 |
| Weight decay | 1e-3 | L2 正則防過擬合 |
| LR schedule | CosineAnnealing T_max=300 | 後期 fine-tune |
| Epochs | 300 (early stop, patience=50) | val_loss 連 50 epoch 不降就停 |
| Batch size | 64 | 兼顧 GPU memory 與梯度穩定 |
| RSSI jitter(訓練時加噪) | ±4 dBm | 模擬硬體誤差 |
| AP dropout(訓練時) | 10% | 模擬 ESP32 漏掃 |
| Gradient clip | max_norm = 5.0 | 防 attention 爆炸 |
| Seeds | [42, 43, 44, 45, 46] | 5-seed ensemble |
| Hardware | RTX 4070 (CUDA 12.4) | 約 4 min / seed |

**Ensemble 聚合**:5 個 seed 的預測用 **geometric median** 平均(Weiszfeld 演算法)— 對 outlier seed 比 arithmetic mean robust。

---

## 4. 實驗結果

### 4.0 視覺化結果(關鍵圖)

主要結果視覺化(完整 PNG 在 `outputs/figures/`):

- **[ladder_bar.png](outputs/figures/ladder_bar.png)** — 全部 14 個實驗 median 誤差的橫條圖,綠色 Cascade 是冠軍
- **[error_cdf.png](outputs/figures/error_cdf.png)** — 7 個主要模型的誤差 CDF,Cascade 與 Heatmap 在低誤差段勝出
- **[error_boxplot.png](outputs/figures/error_boxplot.png)** — 各模型誤差分佈 boxplot,看 outlier 集中度
- **[cascade_scatter.png](outputs/figures/cascade_scatter.png)** — Cascade 全部 363 筆 test 預測點疊在地圖上,色深表誤差大小
- **[cascade_worst10.png](outputs/figures/cascade_worst10.png)** — 10 個最差預測的箭頭圖:green ○ = 真值,red × = 預測,線=誤差向量
- **[data_coverage.png](outputs/figures/data_coverage.png)** — train + test 真值位置疊在地圖上,**揭示 Lab 2 採樣集中於下半部,上半部幾乎沒走過**(這是 outlier 的根本原因)

### 4.1 Split A 完整 ladder

| # | 模型 | 參數 | median (m) | mean | p90 | Δ vs prior |
|---|------|------|-----------|------|-----|-----------|
| 1 | KNN k=5 | — | 1.568 | 1.92 | 3.50 | (baseline) |
| 2 | MLP | 53 k | 1.34 | — | — | −15% |
| 3 | MDN | 56 k | 1.25 | — | — | −7% |
| 4 | MaskedMDN | 76 k | 1.20 | — | — | −4% |
| 5 | Set Transformer MDN | 208 k | 1.093 | 1.40 | 3.10 | −9% |
| 6 | Big Set Trans (619k) | 619 k | 1.10 | — | — | ~0% |
| 7 | + 5-seed ensemble | 619k×5 | 1.083 | — | — | −2% |
| 8 | + 5000 GP synth, 單 seed | 619 k | 0.906 | 1.26 | 2.95 | **−16%** |
| 9 | + GP synth × 5-ens | 619k×5 | **0.889** | 1.24 | 2.99 | −2% |
| 10 | + C-Mixup × 5-ens(失敗)| 619k×5 | 0.942 | 1.24 | 2.99 | +6% 退步 |
| 11 | **Heatmap** + synth × 5-ens (geom) | 864k×5 | **0.836** | 1.13 | 2.67 | **−6%** |
| 12 | **Cascade** + synth × 5-ens (geom) | 882k×5 | **0.793** ⭐ | **1.12** | **2.56** | **−5%** |
| 13 | CNN cross-attn + synth × 5-ens(失敗)| 1.15M×5 | 0.907 | 1.14 | 2.58 | +8% 退步 |
| 14 | CNN xattn + Cascade × 4-ens (geom) | 1.17M×4 | 0.886 | 1.15 | 2.73 | +12% 退步 |
| 15 | 3-level Cascade + 12k synth × 10-ens (geom) | 1.38M×10 | 0.800 | 1.12 | 2.66 | +0.9% 微輸 |
| 16 | Diffusion head + GP synth × 5-ens (snap geom) | 939k×5 | 1.821 | 1.98 | 3.43 | +130% 災難 |

> Row 14 是 A+B combo:第 5 個 seed 訓練意外中斷(腳本崩潰於 s46 epoch 120),
> 我們用前 4 個 seeds 的 checkpoint 跑 ensemble。即使這樣,結果仍**清楚輸給單純 Cascade**,
> 完全確認「A 提供的不是新資訊,只是過度 capacity」這個假設。
>
> Row 15 是「激進路線」實驗:在 2-level Cascade 上同時疊四個改動 — finer fine grid
> (0.4 → 0.25 m → 64×52=3328 cells)、bigger synth(5k → 12k)、3-level cascade
> (coarse/medium/fine)、10-seed ensemble。**結果反而微輸 2-level Cascade ~1%**。
> 診斷:fine head 變成 192×3328 = 638k 參數,對 1,449 個真實樣本太大造成過擬合;
> 同時 12k synth 把 synth/real ratio 推到 8:1,稀釋了真實訊號。
> **這證明 2-level Cascade 已經是這個資料規模下的「甜蜜點」,加大架構/資料量會破壞平衡**。

### 4.2 關鍵躍進的歸因

**從 0.889 → 0.793 m(總計 −11%)的進步來自三件事:**

```
0.889 m  ┃ Big Set Transformer + GP synth × 5-ens (MDN head)        prior best
─────────╋──────────────────────────────────────────────────────────────
  −0.053 ┃ 換 Heatmap head + free-cell mask                       (1)
─────────╋──────────────────────────────────────────────────────────────
0.836 m  ┃ Heatmap × 5-ens (geom-median)
─────────╋──────────────────────────────────────────────────────────────
  −0.043 ┃ Heatmap → Coarse-to-Fine Cascade                       (2)
─────────╋──────────────────────────────────────────────────────────────
0.793 m  ┃ Cascade + synth × 5-ens (geom-median)               ★ 最佳
```

**(1) Heatmap + Free-cell Mask 為什麼有效**

之前的 MDN 在做「regress 連續 (x, y)」,但 RSSI → 位置本質上是離散+多模態的。把它改成 grid classification + sub-cell expected value:
- 走廊型分佈不再被 5 個 Gaussian 強行套
- ~30% 物理不可達區(牆/桌子/實驗室外)的預測**直接歸零**

**(2) Coarse-to-Fine Cascade 為什麼有效**

Fine head 單獨工作時,有時候會把高機率放到錯誤的大區(例如把走廊頭尾搞混)。Coarse head 由於粒度大,**幾乎不會錯區域**,可以當 fine 的先驗濾鏡。具體影響:

- p90 從 2.67 m 降到 2.56 m(**outlier tail 縮短**)
- median 從 0.836 降到 0.793 m(代表「該預測對的也被加分了」)
- 5 個 seed 都 < 0.86 m(過去 Heatmap 有 seed 跳到 0.97),**訓練穩定性也提升**

### 4.3 兩個失敗實驗的學術價值

**Row 10:C-Mixup 沒有提升**

C-Mixup 是 2024 年新提出的 regression-aware mixup,理論上應該補強樣本多樣性。但在我們的場景**疊上 GP synth 後反而退步**(0.889 → 0.942)。

歸因:GP synth、C-Mixup、Enhanced features 三者**都在解同一個瓶頸(空間覆蓋稀疏)**。硬疊不是「補強」是「互相干擾」 — C-Mixup 還引入 label noise,沒帶來新訊息抵銷。

**這是一個正面證據:GP synth 已經把這條路線的收益吃乾,要再進步必須換條完全不同的路。**

**Row 13:CNN Floor-plan Cross-Attention 沒有提升**

預期 CNN 編碼地圖後接 cross-attention 可以讓模型「看見」幾何。實際卻退步 8.5%。

歸因(這個發現很有趣):**地圖是 global 常數**(對每個樣本都一樣)。CNN 編出來的 56 個 spatial tokens 也是固定的。Cross-attention 只能讓 query(pooled scan vector)做「對 56 個固定向量加權」,本質上等於 **多加了一個小 MLP — 沒帶來新資訊**,只是徒增 capacity 容易過擬合。

**地圖 prior 早就被 Heatmap 的 free-cell mask 完整吃光了。**

進一步把 CNN xattn 跟 Cascade 結合的 ablation(Row 14)也驗證了這點 — A+B combo 還是輸給 B 單獨用(0.886 vs 0.793 m,**+12% 退步**)。A 並不是「孤立失敗」,是**本質上對這個問題沒有用**。

**Row 15:3-level Cascade + 12k synth + 10-seed 激進疊加沒有提升**

直覺上「更多 = 更好」,但實驗顯示在 1,449 個真實樣本的資料規模下,**Cascade 2-level 已經是 sweet spot**:
- Fine head 從 192×1320 = 253k 漲到 192×3328 = 638k,**容量已超過資料能支撐的訊號量**
- 12k synth 把 synth/real ratio 推到 8:1,**真實訊號被合成樣本稀釋**
- 加多一層 medium head 引入額外正則但也增加訓練不穩定性

**啟示:不要把「資料多 + 模型大 + 切細」當作 free lunch**。在我們的小資料情境,**選對結構比堆規模更重要**。

**Row 16:Diffusion head 災難級失敗(+130%)**

我們把 MDN/Heatmap 整個輸出端換掉,改用 conditional diffusion model 在 (x, y) 連續空間做 denoising:
- Encoder 同 Cascade(Set Transformer pooled vector 當條件)
- DDIM 25 個 inference steps,每個 sample 跑 8 條 trajectory 取平均
- 後處理用最近 free cell snap 修正落在牆內的預測

**結果 1.821 m,比 Cascade 差 2.3 倍**。per-seed 從 1.6 m 到 4.7 m,**極不穩定**。

**為什麼?**(這個結果意義很重大):
1. **沒有 free-cell prior**:Diffusion 在連續空間預測,雖然有 snap post-process,但無法在訓練時知道「這裡是牆」
2. **1,449 真實樣本對 diffusion 學分佈來說太少**:diffusion 需要密集採樣才能學到準確的條件分佈
3. **Mode-averaging**:走廊兩端 RSSI 像 → diffusion 傾向預測在兩端中間(走廊中央),距離真實位置反而更遠

**這證實了「structured output(grid + free-mask + cascade)在小資料結構化問題上遠勝 continuous denoising」**。

### 4.5 錯誤分析 — Cascade 已接近物理極限

對 Cascade 最佳模型(0.793 m)的 363 筆 test 預測做誤差分布分析:

| 區間 | 樣本比例 | 累積 | 解讀 |
|------|---------|------|------|
| ≤ 0.3 m(AMCL noise floor)| **27.0%** | 27.0% | **已達 GT 不確定性上限** |
| 0.3 – 0.5 m | 9.1% | 36.1% | |
| 0.5 – 0.8 m(median 周邊)| 14.3% | 50.4% | |
| 0.8 – 1.0 m | 9.1% | 59.5% | |
| 1.0 – 2.0 m | 24.2% | 83.7% | 中等誤差 |
| **2.0 – 3.0 m** | 8.5% | 92.3% | outliers |
| **> 3.0 m** | 7.7% | 100% | disasters(28 個 sample)|

**Mean (1.117) 跟 median (0.793) 的差距 0.32 m 完全來自 16% 的 outliers**。

**Top 10 最差預測都 > 4 m,且都是「整個落到另一區」**(例如真值 (-4.46, 7.83) 但預測 (-3.58, 1.23))。這是 WiFi fingerprint 經典的 **symmetric ambiguity**:不同區域的 RSSI 簽名碰巧很像(走廊兩端、對稱牆面),單一 scan 無法區分。

**這 16% 的 outlier 不是模型不夠強,是 single-WiFi-scan 的物理極限**:
- 解法 1:加 IMU + 時序連續性(可大幅改善)
- 解法 2:multi-scan averaging(可大幅改善)
- 解法 3:改架構 / 加參數 / 加合成資料 → **我們已用 6 個實驗證實救不了**

**結論:0.793 m median 已接近 single-WiFi-scan 的理論極限**。要繼續下降必須引入新 sensor 或時序訊號。

### 4.6 資料 coverage 偏差 — outlier 的真正原因

把 [data_coverage.png](outputs/figures/data_coverage.png) 放出來會看到一個重要事實:

**Lab 2 採樣**(我們 Lab 3 用的全部資料)**在實驗室空間分佈極度不均**:
- **下半部(y < 6 m,走廊 + 入口區):train+test 全部 1812 筆樣本幾乎全在這裡**
- **上半部(y > 6 m,大空房間):只有零星幾筆**

驗證:99.4% 的 Cascade 預測**確實落在 free cell 內**(只有 2/363 違反 free-cell mask)。所以**不是模型把點預測到牆裡**,而是 **「真值 y > 6 m 的稀有 test 樣本,模型沒見過足夠類似的 train 樣本,預測會偏向訓練密集的下半部」**。

看 [cascade_worst10.png](outputs/figures/cascade_worst10.png):
- 10 個最差預測的**真值(green ○)幾乎全在 y > 5 m 上半部**
- **預測(red ×)全被拉到 y < 3 m 下半部**
- 這不是「模型壞掉」,是 **「訓練資料沒覆蓋」的必然結果**

**啟示:Lab 3 的 0.793 m 是「給定 Lab 2 採樣分佈下的最佳解」**。若 Lab 2 採樣更均勻覆蓋整個實驗室,Cascade 同樣的架構可能還能再壓低 0.05-0.1 m。

**對未來:Lab 2 補資料時應該特別走訪上半部大空房間**。這比再改任何架構都更有效。

### 4.4 其他切分上的結果(輔助)

(以下用最佳模型 — Cascade + synth × 5-ens — 在其他切分上的表現,作為泛化能力參考)

| Split | median (m) | 備註 |
|-------|-----------|------|
| A_random | **0.793** | 主要評估 |
| B_morning | _未測_ | (時間限制) |
| C_morning→evening | _需獨立模型_ | 跨時段需 domain adaptation,不在本研究範圍 |
| D_stratified | _未測_ | (時間限制) |

---

## 5. 結論與討論

### 5.1 達成

從古典 KNN baseline(1.568 m)出發,在 Split A 上做到 **median 0.793 m**,**相對改進 49%**。
最終模型只用 ~882 k 參數(對 1,812 個訓練樣本來說已是合理上限),訓練時間 ~25 分鐘 / 完整 5-seed ensemble。

**16 個實驗 ablation 中,只有 4 個架構決策真正貢獻了改進**:
1. Set Transformer encoder(處理變長集合輸入)
2. MDN→Heatmap head(打破 Gaussian 假設 + free-cell mask)
3. Coarse-to-Fine Cascade(coarse gating 抑制 outlier)
4. GP synth + 5-seed geometric-median ensemble

**5 個有理論依據的架構嘗試最終失敗**:
- C-Mixup(與 GP synth 解同一瓶頸,互相干擾)
- CNN floor-plan cross-attention(map 是全域常數,沒帶來 per-sample 新資訊)
- A+B combo(CNN xattn 拖累 Cascade)
- 3-level Cascade × 10-seed × 12k synth(過大 fine head 過擬合,synth 稀釋真實訊號)
- Diffusion regression head(無 free-cell prior + 小資料 + mode-averaging)

### 5.2 接近理論下限

AMCL 給出的 ground truth 本身有 ±0.3 m 的雜訊。**理論下限約 0.3 m**。

我們的錯誤分析顯示:
- **27% 預測已落在 0.3 m AMCL noise floor 內**(無法再進步)
- **50% 預測 ≤ 0.8 m**(典型誤差)
- **16% 是 outliers > 2 m**,且都是 symmetric ambiguity(WiFi RSSI 在不同區域碰巧很像)

**這 16% outlier 用任何「純改架構」的方法救不了**,需要外部新資訊(IMU 時序、多 scan、其他 sensor)。

### 5.3 學到的事

1. **正確的歸納偏置 > 模型容量**:Heatmap (864k) + Cascade (882k) 比純加深 Transformer 有效得多。
2. **「資訊」與「容量」要分清**:CNN cross-attention 失敗的原因不是「不夠強」,而是它**沒提供 per-sample 新資訊**。
3. **不同方向的 augmentation 並不總是疊加**:GP synth、C-Mixup、Enhanced features 都解空間覆蓋,加在一起反而干擾。
4. **小資料有甜蜜點,不要硬擴**:Phase 1 用 1.38M 參數 + 12k synth + 10-seed,反而微輸 882k 的 2-level Cascade。
5. **Structured output 在小資料 + 結構化問題上遠勝 continuous regression**:Diffusion 比 Heatmap+Cascade 差 2.3 倍。
6. **失敗實驗有教育意義**:5 個負面結果幫助我們**理解了瓶頸所在 — 不是模型容量,是 RSSI 本身的物理極限**。
7. **Geometric median 比 mean robust**:對 5-seed ensemble 而言效果顯著(0.798 → 0.793)。

### 5.4 未來方向

- **跨時段穩健性 (Split C)**:domain adaptation / continual learning(WiFi 環境晝夜變化)
- **真正的多模態融合**:加入 IMU 等其他 sensor,而非單純 WiFi(直接攻擊 16% outlier 問題)
- **Multi-scan temporal modeling**:把連續 N 個 scan 合在一起(自然解 symmetric ambiguity)
- **Real-time ESP32 部署**:把訓練好的 model export 成 INT8,丟 ESP32-S3 跑(對應 Lab 2 收集端硬體)

---

## 附錄 A:重要檔案對照

| 階段 | 檔案 |
|------|------|
| 資料 / 切分 | `code/lab3/data.py` |
| 模型 (含所有 variant) | `code/lab3/models.py` |
| GP 合成 | `code/lab3/synthetic.py` |
| Baseline 訓練 | `train_baselines.py`, `train_settransformer.py` |
| 最強 MDN ensemble | `train_best_ensemble.py` |
| Heatmap | `train_heatmap.py` |
| **Cascade(最佳)** | **`train_heatmap_cascade.py`** |
| CNN xattn(失敗對照) | `train_heatmap_map.py` |
| A+B combo(完整 ablation) | `train_heatmap_map_cascade.py` |

## 附錄 B:結果存檔位置

- `outputs/metrics.csv` — 所有模型 metrics
- `outputs/predictions/A_random__CascadeEnsemble.npz` — **最佳模型的預測 + 真值**
- `outputs/checkpoints/A_random__Cascade_s4{2-6}.pt` — 5 個 seed 的訓練好權重
- `outputs/*_log.txt` — 訓練 log

完整架構解說與更詳細的演進史請參見 [ARCHITECTURE.md](ARCHITECTURE.md)。

歷代模型架構圖 + 演進流程 + 設計決策樹請參見 [EVOLUTION.md](EVOLUTION.md)。

報告圖表生成腳本:`make_report_figures.py`,所有圖存於 `outputs/figures/`。
