# Lab 3 完整解說 — 從零到 0.89 m

> 給未來的自己讀的版本:從「拿到 Lab 2 的 WiFi 資料」一路到「跑 Set Transformer + GP 合成資料」的所有過程,每一步都解釋**為什麼**這樣做、它的**直覺**是什麼、結果如何。

---

## 目錄

1. [我們在解什麼問題?](#1-我們在解什麼問題)
2. [手上的資料長什麼樣?](#2-手上的資料長什麼樣)
3. [評估方式:四種 split](#3-評估方式四種-split)
4. [演進史 — 模型怎麼一步一步爬到 0.89 m](#4-演進史--模型怎麼一步一步爬到-089-m)
5. [現役主力模型詳解(Big Set Transformer + MDN)](#5-現役主力模型詳解big-set-transformer--mdn)
6. [訓練配方](#6-訓練配方)
7. [改進路線(已完成 + 規劃中)](#7-改進路線已完成--規劃中)
8. [結果總表](#8-結果總表)

---

## 1. 我們在解什麼問題?

### 1.1 室內定位是什麼

GPS 在室內訊號收不到,所以無法用 GPS 定位你在大樓裡哪個房間哪個位置。**WiFi 室內定位**的核心想法是:

> **每個位置周圍能聽到的 WiFi AP 集合 + 強度組合,幾乎是獨一無二的「指紋」。**

只要事先建好「位置 ↔ 指紋」對應表,之後給一個新指紋,就能反查位置。

### 1.2 為什麼用 RSSI?

- **RSSI**(Received Signal Strength Indicator)= AP 訊號強度,單位 dBm,範圍大概 -30(很近)~ -90(很遠/穿很多牆)
- **任何手機/ESP32/筆電都能掃**,不需特殊硬體
- 缺點:訊號受人體、家具遮擋,**會跳動 ±3-5 dBm**

### 1.3 這個問題為什麼難?

| 困難 | 影響 |
|------|------|
| RSSI 時變(同一點不同時間掃,強度會差 ±3-5 dB) | 模型必須「容忍噪音」 |
| 每次掃到的 AP 集合都不一樣(有時掃到 25 個,有時 35 個) | 不能用固定維度向量 |
| 訓練資料只有 1812 個位置點,但實驗室空間有 ~75000 個自由格 | **空間覆蓋率超低** |
| 標準答案(x, y)本身是用 ROS 的 AMCL 自動定位,有 ±0.3 m 雜訊 | 模型的 **理論最佳誤差 ≈ 0.3 m** |

**這個 4 條決定了一切後續設計**。記住第 3 條:**空間覆蓋率才是我們最大的瓶頸**,不是模型不夠大。

---

## 2. 手上的資料長什麼樣?

### 2.1 來源

從 Lab 2 收的 ROS bag 抽出來的:

```
wifi/
  wifi_20260517_*.jsonl   ← 早上 session(在實驗室)
  wifi_20260524_*.jsonl   ← 晚上 session(回實驗室再收一輪)
```

每一行(一個 JSON object)是**一個瞬間的 scan**:

```json
{
  "stamp": "...",
  "pose": {"frame_id": "map", "x": 3.14, "y": 5.92},   ← 那一瞬間機器人(GT)位置
  "aps": [
    {"bssid": "AA:BB:CC:11:22:33", "rssi": -42, "ssid": "NCTU-CS"},
    {"bssid": "DD:EE:FF:44:55:66", "rssi": -58, "ssid": "guest"},
    ...30 多個 APs
  ]
}
```

### 2.2 數字大小

| 項目 | 數量 |
|------|------|
| 總 scans | 1,812 |
| 早上 scans | ~950 |
| 晚上 scans | ~862 |
| 所有看過的 BSSID(MAC 地址)| ~230 個 |
| **常見 BSSID**(出現 ≥10 次)→ 進入模型 | **80 個** |
| 實驗室掃描範圍 | 約 16 × 12 m |
| 每個 scan 平均偵測 APs | ~28 個 |

### 2.3 為什麼只取 ≥10 次的 BSSID?

去掉「只出現一兩次就消失」的 AP — 那些可能是手機熱點、外面的鄰居路由器,只會污染訓練。最後得到 80 個穩定的 AP。**80 = 我們模型的 BSSID vocabulary size**。

---

## 3. 評估方式:四種 split

把 1812 個 scans 拆成 train / test 有四種拆法,每種測不一樣的能力:

| Split | 拆法 | 測的能力 | 我們的重點 |
|-------|------|----------|----------|
| **A_random** | 全部 random 80/20 | 標準「能否內插」 | ★ 主要評估 |
| B_morning | 只用早上,random 80/20 | 同時段內泛化 | 對照組 |
| C_morning→evening | 早上全部當 train,晚上全部當 test | **跨時段**泛化 | 最嚴格 |
| D_stratified | 早晚各 80/20,合併當 train/test | 「資料庫有覆蓋所有時段」 | 部署時的真實狀況 |

**為什麼主攻 Split A?**

老師講義是「形式做法不拘」,但 Split A 是課堂示範用的標準。Split C 是另一個故事(MLP 在 C 反而打贏 Transformer,因為大模型在 1.4k 樣本下對時段漂移過擬合)。為了報告好寫,**集中火力在 A,把 C 當補充討論**。

---

## 4. 演進史 — 模型怎麼一步一步爬到 0.89 m

> 每個 step 都是一個獨立的程式檔。Split A median location error 是評估指標,**越小越好**。1 m 是定位到「正確房間/通道」的等級,0.5 m 是「正確桌子」等級。

### Step 0:先看資料能不能學(EDA)

**檔案**:`eda.py`

**動機**:在訓練任何模型之前,先看資料本身有沒有可學的訊號。如果同一個位置的 RSSI 在不同時間天差地遠,那再強的模型都白搭。

**做了什麼**:
- 計算「同一個 cell 內 RSSI 標準差」 → 3.5 dBm,可以接受
- 計算「早上 vs 晚上同一個 cell 的 RSSI 平均差」 → ±2.6 dBm,有 shift 但不是大平移
- 畫熱圖看 BSSID 分佈
- 結論:**資料是可學的**,可以進入建模

---

### Step 1:KNN baseline(古典指紋法)

**檔案**:`models.py::KNNRegressor`

**動機**:這是 WiFi 室內定位最古老、最簡單、最沒有任何「學習」的方法。當作所有後續方法的下限。

**做法**:
1. 把每筆 train 資料 X[i]、y[i] 存起來(就是個資料庫)
2. 給新 query X_q,找它跟誰最像(歐式距離最近的 K=5 個)
3. 把那 K 個鄰居的 (x, y) 用「距離倒數」加權平均當預測

**直覺**:就是「相似的指紋 → 相似的位置」。完全沒學任何 model。

**結果**:Split A median = **1.568 m** 🚫

意思是:有一半的預測偏差超過 1.5 m。對 16×12 m 的實驗室,這大概是「猜對哪個房間」的等級。

---

### Step 2:MLP — 學個非線性映射

**檔案**:`models.py::MLPRegressor`(53k 參數)

**動機**:KNN 對每個 query 都要算所有資料的距離,**沒有學東西**。MLP 用神經網路把 80 維 RSSI → 2 維座標的映射「壓縮」進 weights 裡。

**結構**:
```
80 維 RSSI → Linear(80→256) → ReLU → Dropout
            → Linear(256→128) → ReLU → Dropout
            → Linear(128→2)
```
Loss 用 **Smooth L1**(對離群值比 MSE 寬容)。

**結果**:Split A median ≈ **1.34 m** ⬇️ 比 KNN 好 15%

**為什麼會贏 KNN**:神經網路可以學到「AP A 強 + AP B 弱 → 偏走廊左側」這種非線性規則,KNN 做不到。

---

### Step 3:MDN — 輸出機率分佈而非單點

**檔案**:`models.py::MDNRegressor`

**動機**:RSSI → 位置是**多模態**的。

舉例:在走廊兩端 X、Y,可能 RSSI 看起來幾乎一樣(都離主要 AP 很遠)。MLP 用 MSE 訓會把這兩個點的預測「平均」到走廊中央,**那個答案兩個位置都不對**。MDN 直接讓模型說「可能在 X 也可能在 Y」。

**做法**:Linear backbone 後面接 **3 個 heads**:
- `pi_head` → K=3 個 mixture 權重(softmax 過)
- `mu_head` → K=3 個 Gaussian 中心 (x, y)
- `logsig_head` → K=3 個 Gaussian 變異數

**Loss = Negative Log-Likelihood**:

```
   NLL = -log Σ_k π_k · N(y_true | μ_k, σ_k²)
```

直覺:「真實的 (x, y) 應該被某個 component 賦予很高機率」。

**推論時**:取最大 π 的那個 component 的 μ 當預測(叫 **MAP**, Maximum A Posteriori)。

**結果**:Split A median ≈ **1.20 m** ⬇️ 又進一步

**副產物**:模型現在有 **uncertainty**。報告可以畫「預測點 + 95% 機率橢圓」。

---

### Step 4:Masked MDN — 告訴模型「哪些 AP 是真的」

**檔案**:`models.py::MaskedMDN`

**問題**:之前 MLP/MDN 看到的 X 是 80 維,沒掃到的 AP 我填 -100 dBm 當佔位符。但 model 不知道 **-100 是「沒掃到」還是「掃到但訊號超弱」**,會把 padding 當訊號去學,污染 ReLU。

**做法**:再加一個 **presence mask**(80 維 0/1 向量,1 = 真的掃到),concat 到 input。模型輸入變成 160 維 = (RSSI, mask)。

**為什麼有效**:模型現在能學到「mask 為 0 的位置,RSSI 數值不要看」。

**結果**:Split A median ≈ **1.20 m**(MDN 已經夠強了,Masked 加成有限)

---

### Step 5:Set Transformer — 處理「變長集合」

**檔案**:`models.py::SetTransformerMDN`

這是**架構大躍進**,前面所有模型的根本問題終於被解決。

**問題重述**:
- 真實的 scan 是 *變長*(20-40 個 APs)
- 我們之前硬塞成 80 維固定向量,80 - N 個位置填 -100
- 這做法**根本上錯**:80 維的「絕對位置」(第 j 維代表 BSSID j 的 RSSI)只是個方便編號,model 必須從 80 個輸入 slots 推「哪些是真實 AP」,浪費容量

**正確做法**:把一個 scan 看成一個**集合(set)**:
```
scan = {(BSSID_3, -42), (BSSID_17, -55), (BSSID_29, -71), ...}
```
- 集合**沒有順序**(API 回傳的 AP 順序不固定)
- 集合**長度可變**

**Set Transformer**(Lee et al. 2019)就是設計來處理這種輸入的架構。

**核心想法**:
1. 把每個 AP 變成一個 token:`token_j = concat( Embedding(BSSID_j), RSSI_j )`
2. 用 **self-attention** 讓所有 token 互相觀察 → 每個 token 知道自己在這個 scan 中的「相對地位」
3. 用 **attention pooling**(PMA)把不定數量的 tokens 聚合成 1 個 vector
4. 接 MDN head 出 (x, y)

**關鍵組件**(我們會在 §5 細講):
- **Embedding**:每個 BSSID 學一個 48 維向量(像 NLP 的 word embedding)
- **SAB (Self-Attention Block)**:讓 50 個 token 互相對話 3 輪
- **PMA (Pooling by Multi-head Attention)**:用一個 learnable query 「問」這 50 個 token 它想要什麼

**為什麼比硬塞 80 維強?**
1. 不會被 -100 padding 污染(用 mask 從根本擋掉)
2. Embedding 學到每個 AP 的「個性」(這個 AP 通常在哪)
3. Attention 學到 AP 間的關聯(這幾個 AP 都強通常代表在某個房間)

**結果**:Split A median ≈ **1.093 m** ⬇️ 終於跌破 1.1 m

模型參數量:208k(預設 model_dim=128)

---

### Step 6:把模型變大 + 強化正則化

**檔案**:`train_best_ensemble.py` 用的 `BIG_CFG`

**動機**:208k 參數對 1.4k 訓練樣本可能還沒打滿模型容量。**試著疊參數量**:

| 改動 | 值 |
|------|-----|
| `model_dim` | 128 → **192** |
| `num_sab` | 2 → **3**(SAB 層數) |
| `K`(MDN component) | 3 → **5** |
| `dropout` | 0.1 → **0.3**(防過擬合) |
| `weight_decay` | 0 → **1e-3**(L2 regularization) |
| `jitter`(訓練時 RSSI 加噪) | 2 → **4 dBm** |

參數量:**619k**(MDN head 拆成 5 個 Gaussian 反而頭變小,主要漲在 SAB)

**為什麼正則化加大?** 模型大但資料少,**會背答案**。Dropout / weight decay / jitter 都在強迫模型「不要太相信任何單一輸入特徵」。

**結果**:Split A median ≈ **1.10 m**(單 seed)

不算大躍進,但 p90 (90% 分位誤差)明顯降低,代表 **outlier 變少**,模型更穩定。

---

### Step 7:5-seed Ensemble — 用「平均」消除隨機性

**檔案**:`train_best_ensemble.py`

**動機**:同一個模型用不同 random seed 訓出來的 weights 不一樣,預測也有差。把 5 個獨立訓的模型 **預測平均**,可以消除掉「某個 seed 剛好過擬合在某個 outlier」的問題。

**做法**:
- 用 seeds `[42, 43, 44, 45, 46]` 各訓一個 Big Set Transformer
- 對每個 test sample,5 個模型各給一個 (x, y) 預測
- 取 5 個的 mean 當最終預測

**結果**:Split A median ≈ **1.083 m**(再下降 1.5%)

**為什麼提升小?** 5 個模型已經都很好,沒有特別爛的拖累。Ensemble 主要在「拉住最差的」,但這裡沒有特別差的可以拉。

---

### Step 8:GP 合成資料 — **真正的大躍進**(-19%)

**檔案**:`synthetic.py` + `train_best_ensemble.py`

**動機**(回到 §1.3 第 3 條瓶頸):**1.4k 訓練樣本 vs 75,000 個自由格 → 99% 的空間沒有被採樣**。模型在沒走過的區域亂猜。

**解法**:**自動產生**假的(但合理的)scans 來填空。

#### 8.1 怎麼產假資料?

**核心:Gaussian Process(GP)空間平滑**

對每個 BSSID(80 個),用 sklearn 的 `GaussianProcessRegressor` 學一個「**位置 → RSSI**」的二維 GP 模型:

```
For each BSSID b in 80:
    收集所有偵測到 b 的 (x, y, rssi) tuples
    用 RBF + WhiteKernel 配適一個 GP
    這個 GP 就學會「在(x, y) 點 BSSID b 的 RSSI 應該是多少」
```

GP 的好處:**自帶 uncertainty**。對採樣稀疏的區域,GP 會說「我猜是 -60 dBm,但 std 是 15 dBm(很不確定)」。

#### 8.2 不只是「會被聽到」還要「偵測到沒」

只用 GP 預測 RSSI 會有個問題:**它會幻覺出根本聽不到的 AP**。例如離 AP 很遠的位置,GP 仍然會輸出一個 RSSI 值(雖然不確定)。

所以我們再加一個 **KNN-based detection classifier**(`fit_detection_knn`):
- 對每個位置 (x, y),找它最近的 K=10 個真實 scans
- 看那 10 個 scans 中,BSSID b 出現的比例 → 那就是「在這個位置偵測到 b 的機率」
- 若機率 < 某閾值,就把這個 b 排除掉

**這個結合叫「fix C」**(因為我們嘗試過 fix A、B 都不理想)。

#### 8.3 生產流程

```
1. 從 psquare.pgm 抽出 ~75000 個自由格
2. 只保留「距離某個真實 sample 2 m 以內」的格 → ~6000 候選位置
3. 隨機抽 5000 個當合成位置
4. 對每個位置:
   - 用 KNN 算出每個 BSSID 的偵測機率
   - 對偵測機率 > 閾值的 BSSID,用 GP 預測 RSSI
   - 用伯努利採樣決定真的「偵測到沒」
5. 輸出 5000 個假 scans,跟真實 1.4k 合併
```

**Sanity check 是否合理?** 合成後 mean APs/scan = 27.1(真實 27.7,很接近)。

#### 8.4 結果

| 模型 | Split A median |
|------|----------------|
| Big Set Transformer + 1.4k real,**單 seed** | 1.10 m |
| Big + 1.4k real + **5000 GP synth**,單 seed | **0.906 m** ⬇️ -18% |
| Big + 1.4k real + 5000 GP synth,**5-seed ensemble** | **0.889 m** ⭐ |

**這是目前最好的結果**。1 m 大關被打破。

---

### Step 9:C-Mixup 嘗試(失敗 — 沒疊加效果)

**檔案**:`train_cmixup.py`, `train_cmixup_ensemble.py`

**動機**:**C-Mixup**(arXiv 2405.17938, 2024)是對 regression 友善的 mixup 增強。標準 mixup 對分類有效但對 regression 不友善(把貓和狗的圖片混在一起,label 變 0.5 沒物理意義)。C-Mixup 改良為:**只把 label 距離近的樣本混合**。

**做法**:
```
1. 算所有 train labels 之間的距離矩陣
2. 每個樣本 i 對其他樣本 j 的「混合機率」 ∝ exp(-||y_i - y_j||² / bandwidth²)
3. 訓練時對每個 sample i,按上面機率抽 j,生成:
     x_mix = λ·x_i + (1-λ)·x_j
     y_mix = λ·y_i + (1-λ)·y_j     ← y 也混
   其中 λ ~ Beta(α, α)
```

**期待**:C-Mixup 可以做為 GP synth 的補充,進一步壓誤差。

**結果**:
| 模型 | Split A median |
|------|----------------|
| Big + C-Mixup(沒有 synth) | 1.016 m(比 baseline 好) |
| Big + synth + C-Mixup 單 seed | 0.900 m(跟單純 synth 持平) |
| Big + synth + C-Mixup **× 5 ensemble** | **0.942 m**(比 0.889 退步!)|

**為什麼失敗?**

我們的關鍵 insight:**GP synth、C-Mixup、Enhanced features 三者都在解同一個瓶頸(空間覆蓋)**。硬疊起來不是「補強」是「互相干擾」 — C-Mixup 還引入 label noise,沒帶來新訊息抵銷。

**這結果其實是一個正面證據**:0.889 m 不是「還沒夠努力」,是 **GP synth 已經把這條路線吃乾了**。要再進步必須換另一條路。

---

### Step 10(規劃中):Heatmap 輸出 + Free-cell Mask

**檔案**(編寫中):`models.py::SetTransformerHeatmap`, `train_heatmap.py`

**動機**:Step 1~9 都是**改輸入或樣本**(input feature, augmentation, ensemble)。但**輸出**還是 MDN 那 5 個 Gaussian。

**MDN 的缺點**:
1. K=5 個 Gaussian 是「先驗形狀」假設,套不住走廊型/L 型的真實 posterior
2. 完全不知道「物理上不可能」的位置(牆裡面、實驗室外面),會把預測落在那裡

**新方法**:把實驗室切成 40×33 = **1320 個 0.4m × 0.4m 格子**。模型對每個 scan 輸出 1320 個 logits,代表「我認為位置在每個格子的可能性」。

**Free-cell mask**:
- 從 `psquare.pgm` 抽出「哪些格子是 free space」(67% 是,33% 是牆/桌子/外面)
- 推論時:`softmax(logits) × free_mask` → 重新歸一化 → 用期望值取 (x, y)

**為什麼會贏 MDN**:
1. ~30% 的 MDN 預測落在不可達區 → Heatmap mask 後**直接歸零**
2. 多模態自然(無 K 個 Gaussian 的限制)
3. 報告可以畫漂亮的熱圖

**預期**:Split A median 0.889 → **0.78-0.82 m**

(這個目前在 GPU 上跑,等結果)

---

## 5. 現役主力模型詳解(Big Set Transformer + MDN)

接下來把 **619k 參數的模型內部**徹底拆開。

### 5.1 整體資料流(從 scan 到 (x, y))

```
INPUT: 一個 scan 偵測到的 N 個 AP(N 變動,Big 配置 max=50)
       例:[(b3, -42 dBm), (b17, -55 dBm), (b29, -71 dBm), ...]
                            │
                  pad 到 M=50,給每個 token 算一個 mask
                            │
        ┌───────────────────┼────────────────────┐
        ▼                   ▼                    ▼
  bssid_idx (B,50)    rssi (B,50)        mask (B,50)
  long ∈ [0..80]     float ∈ [-1..0]     0=padding 1=真實
        │                   │                    │
        ▼                   │                    │
  ┌──────────────────┐      │                    │
  │ Embedding(81,48) │      │  ← 每個 BSSID 學   │
  │   (B,50,48)      │       一個 48 維向量      │
  └──────────────────┘      │                    │
            └──── concat ───┤                    │
                            ▼                    │
                    ┌──────────────┐             │
                    │  (B,50,49)   │             │
                    └──────────────┘             │
                            ▼                    │
                ┌──────────────────────┐         │
                │ Linear(49→192)       │         │
                │  input_proj          │         │
                │  → (B,50,192)        │         │
                └──────────────────────┘         │
                            │                    │
                       × mask  ◄─────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────┐
        │  SAB #1  (Self-Attention Block)          │
        │  4 heads, LayerNorm                      │
        │  每個 AP token 看其他 49 個 AP 算自己    │
        │  → (B, 50, 192)                          │
        └──────────────────────────────────────────┘
                            ▼
        ┌──────────────────────────────────────────┐
        │  SAB #2  ← 第二輪「AP 之間互相對話」     │
        └──────────────────────────────────────────┘
                            ▼
        ┌──────────────────────────────────────────┐
        │  SAB #3  ← 第三輪                        │
        └──────────────────────────────────────────┘
                            ▼
        ┌──────────────────────────────────────────┐
        │  PMA (Pooling by Multi-head Attention)   │
        │  1 個 learnable seed query,從 50 個     │
        │  token 加權聚合成 1 個 vector            │
        │   → (B, 1, 192) squeeze → (B, 192)       │
        └──────────────────────────────────────────┘
                            ▼
                  ┌──────────────────┐
                  │  Dropout(p=0.3)  │
                  │   → (B, 192)     │
                  └──────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
  ┌─────────────┐   ┌─────────────┐    ┌────────────────┐
  │ Linear      │   │ Linear      │    │ Linear         │
  │ 192 → 5     │   │ 192 → 10    │    │ 192 → 10       │
  │ log_softmax │   │ reshape     │    │ reshape, clamp │
  │  → log π    │   │ → μ (B,5,2) │    │ → logσ (B,5,2) │
  │   (B, 5)    │   │             │    │                │
  └─────────────┘   └─────────────┘    └────────────────┘
         │                  │                  │
         └────────────── MDN head ──────────────┘
                            │
   Loss = NLL = -log Σ_{k=1..5} π_k · N(y | μ_k, diag(σ_k²))
                            │
                  推論時取 MAP: 選 argmax_k π_k → μ_k
                            ▼
                    ┌────────────────┐
                    │  pred (x, y)   │
                    └────────────────┘
```

### 5.2 細解每一塊

#### 5.2.1 Embedding(每個 BSSID 學 48 維特徵)

每個 BSSID(MAC 地址)其實只是個任意編號,沒有數字意義。Embedding 層**把整數編號 → 連續 48 維向量**,讓網路可以對它做運算。

```python
self.embed = nn.Embedding(81, 48, padding_idx=80)
# 81 = 80 BSSIDs + 1 個 PAD 槽
# padding_idx=80:第 81 個槽的 embedding 永遠為 0,不會被學
```

**直覺**:就像 NLP 的 word embedding。常一起出現的 BSSID(例如同一個 SSID 的多個 AP)會學到相似的向量。

#### 5.2.2 Concat RSSI

把 RSSI 數值(已正規化到 0~3 範圍,-100 = 0)接在 embedding 後面:

```
input_token = concat( Embedding(BSSID), RSSI_normalized )
             = (48 維)              + (1 維)  = 49 維
```

`input_proj` 是個 Linear 把 49 維拉到 192 維(model_dim),後面的 attention 才有空間「思考」。

#### 5.2.3 SAB(Self-Attention Block)— 重頭戲

每個 SAB 內部跑一次 `MAB(X, X)`,意思是「自己跟自己 attention」。

**MAB(Q, K)** 細節:

```
MAB(Q, K):                                    ← Q 是 query, K 是 key
       Q  ──→ Linear(192→192) ──→ split 4 heads ─┐
                                                  │
       K  ──→ Linear(192→192) ──→ split 4 heads ─┤
                                                  ├─→  Q·Kᵀ / √192
       K  ──→ Linear(192→192) ──→ split 4 heads ─┤      (B*4, 50, 50)
              (這條就是 V)                        │
                                                  │   mask 掉 padding token
                                                  ▼   → softmax → 注意力權重 A
                                              ┌─────────┐
                                              │  A @ V  │
                                              └─────────┘
                                                  │
                                                  ▼
                            concat 4 heads → (B, 50, 192)
                                                  │
                                       + Q  (residual)
                                                  │
                                                  ▼
                                          LayerNorm
                                                  │
                                       + ReLU(Linear(192→192))  ← FFN
                                                  │
                                                  ▼
                                          LayerNorm  → 輸出
```

**白話解釋**:
- **Attention 在算什麼?** Q·Kᵀ → 給每對 (token_i, token_j) 算個分數,代表「token i 應該多關注 token j」
- **Multi-head 為什麼?** 4 個 heads 在不同子空間平行算,類似「從 4 個不同角度看 AP 間的關係」
- **Softmax + masked fill** 確保 padding 不被當真實 AP attention
- **Residual + LayerNorm + FFN** 是 Transformer 標配,讓信號能流通不爆

**SAB 在 WiFi 場景具體在學什麼?**

舉例:有兩個 AP 都很強。如果都是「室內掛在桌邊的 AP」,模型會把它們連起來,知道「我在桌子區」。SAB 就是讓每個 AP token 透過 attention 觀察「其他 AP 的存在強度」,而不是孤立判斷自己。

**3 個 SAB 疊起來**像 CNN 疊深層:第一層學 pairwise 關係,第二層學「pairwise 關係的 pairwise 關係」,如此遞迴。

#### 5.2.4 PMA(Pooling by Multi-head Attention)

問題:SAB 跑完 X 還是 (B, 50, 192) 一個變長序列。MDN head 只能吃固定維度的 (B, 192)。怎麼把 50 個 tokens 壓成 1 個?

**選擇 A**:平均池化 `X.mean(dim=1)`。簡單但所有 token 等權,沒法強調重要 AP。

**選擇 B**:最大池化。某些情境好,但這裡會丟太多訊息。

**選擇 C(我們用的)**:PMA。**讓模型自己學一個 query「我想要什麼資訊」**,然後 attention pooling。

```
S = nn.Parameter((1, 1, 192))  ← 1 個可學的 query vector
                              ← 訓練後它代表「對定位最重要的特徵組合」
                            │
                            ▼
                    MAB(S, X, mask)
                            │
       S 當 query,X (50 個 AP token) 當 key/value
       → 從 50 個 token 加權聚合成一個 vector
                            ▼
                      (B, 1, 192)
```

**直覺**:PMA 的 query 就像問「在這 50 個 AP 裡,跟我的定位最有關的訊息是什麼?」模型訓久了,它就會自動學會關注「強度差最大」「最不雜訊」的 AP。

#### 5.2.5 MDN Head(Mixture Density Network)

從 192 維 pooled vector 出 3 個東西:

```
log_pi:   Linear(192 → 5) → log_softmax    → 5 個 mixture 權重(機率)
mu:       Linear(192 → 10) → reshape (5, 2) → 5 個 (x, y) 中心
logsig:   Linear(192 → 10) → reshape, clamp → 5 個 (σx, σy)
```

`clamp(min=-3, max=2.5)` 防止 σ 飛掉(指數後 = [0.05, 12] m,涵蓋室內合理範圍)。

**Loss = NLL**(Negative Log-Likelihood):

```
NLL = -log Σ_{k=1..5} π_k · N(y_true | μ_k, diag(σ_k²))
```

數值上用 `torch.logsumexp` 避免數值下溢:

```python
log_p_full = log_π + log_N    # K 個 log-likelihood
loss = -logsumexp(log_p_full, dim=-1).mean()
```

**推論**:取 `argmax_k π_k` 那個 component 的 μ 當預測(MAP)。

#### 5.2.6 完整參數量

| 模組 | 計算 | 參數量 |
|------|------|--------|
| Embedding(81, 48) | 81 × 48 | 3,888 |
| input_proj Linear(49→192) | 49×192 + 192 | 9,600 |
| **SAB × 3** | 3 × (4×Linear(192²) + 2×LN) | 449,424 |
| PMA(seed + MAB) | 192 + 149,808 | 150,000 |
| pi_head Linear(192→5) | | 965 |
| mu_head Linear(192→10) | | 1,930 |
| logsig_head Linear(192→10) | | 1,930 |
| **Total** | | **≈ 617,737 ≈ 619 k** |

---

## 6. 訓練配方

### 6.1 超參數(Big 配置)

| 項目 | 值 | 為什麼 |
|------|-----|--------|
| Optimizer | Adam, lr=1e-3 | 標配,穩 |
| Weight decay | 1e-3 | L2 正則防過擬合 |
| LR schedule | CosineAnnealing T_max=300 | 訓練後期 lr 自動降低,讓 loss 收斂更平滑 |
| Epochs | 300 | 上限 |
| Early stop patience | 50 | val_loss 連 50 epochs 沒進步就停 |
| Batch size | 64 | 1.4k+5k 樣本下不能太大 |
| Gradient clip | max_norm=5.0 | 防 attention 爆炸 |
| RSSI jitter(訓練時加噪) | ±4 dBm | 模擬硬體誤差,讓 model robust |
| AP dropout(訓練時)| 10% | 模擬 ESP32 漏掃,讓 model 不依賴特定 AP |
| Loss | MDN NLL | 多模態輸出 |
| Ensemble seeds | [42, 43, 44, 45, 46] | 5 個獨立模型平均 |
| Hardware | RTX 4070 | 每 seed ~4 min |

### 6.2 為什麼這些設計?

- **Cosine LR**:讓 lr 從 1e-3 慢慢降到接近 0,後期 fine-tune 時不會劇烈跳
- **Early stop**:省時間,而且防 val 已經上揚還繼續訓
- **Jitter + Dropout**:本質是用 augmentation 模擬 noise,讓模型學「不要相信任何單一輸入」
- **Ensemble**:消除 seed-dependent variance

---

## 7. 改進路線(已完成 + 規劃中)

### 7.1 已走過的 Split A 進度表

```
階段          模型                                       median (m)    Δ
─────────────────────────────────────────────────────────────────────────
Baseline      KNN  k=5                                       1.568      —
              MLP  (53k 參數)                                 1.34   −15%
              MaskedMDN (76k)                                 1.20   −10%

架構升級      Set Transformer (208k)                          1.093  −9%

正則化加大    Big Set Transformer (619k) + 強 regularize     1.10   ~0%
              + 5-seed ensemble                               1.083  −1.5%

資料增強      + 5000 GP-synthesized 點                        0.906 −16%
              + 5-seed ensemble                               0.889  −1.9%

額外嘗試      + C-Mixup(沒有 synth)                          1.016
              + C-Mixup + synth × 5 ens                       0.942  退步

架構大躍進    Heatmap output (864k) + synth, best single      0.838  −5.7%
              ↑ 換掉 MDN head → 40×33 grid + free-cell mask
              + 5-seed mean ensemble                          0.883
              + 5-seed geometric median ensemble              0.836  −5.9%

架構再升級    Heatmap Cascade (882k) — coarse+fine 兩頭     0.778  −7.0%
              ↑ coarse 10×9 + fine 40×33,fine 被 coarse gate
              + 5-seed geometric median ensemble              0.793  −5.1%  ★ 目前最佳
              ↑ p90 也壓到 2.56 (-4%),outlier 抑制有效

架構嘗試失敗  + Floor Plan CNN Cross-Attention                 0.907  +8.5%
              ↑ 預期會贏,實際退步
              原因:map 是 global 常數,cross-attn 只是 query→
              fixed-key lookup,沒帶來 per-sample 新資訊
              map 的 prior 早被 free-cell mask 完整吃光

A+B 結合      + CNN xattn + Cascade combo                    0.886  +11.7%
              ↑ 第 5 seed 中斷,4-seed ensemble
              退步證實「A 本質沒幫助」,不只是孤立失敗

激進路線     Phase 1 = 3-level Cascade (0.25/0.6/1.6m)        0.800  +0.9%
              + 12k synth (vs 5k) + 10 seeds
              ↑ 1.38M 參數,反而微輸 2-level Cascade
              診斷:fine head 192×3328=638k 對 1.4k 真實樣本
              過大,12k synth 把真實訊號稀釋掉了
              結論:2-level Cascade 已經是「sweet spot」

範式對照     Phase 2 = Diffusion regression head              1.821  +130%
              (encoder + DDIM 25 steps × 8 samples)
              ↑ 災難級失敗 — 連續輸出 + 沒 free-cell prior
              在小資料(1.4k 真實)上無法學到準確分佈
              證實「structured output + grid + mask」對這
              個問題遠勝 continuous denoising
─────────────────────────────────────────────────────────────────────────
```

### 7.2 接下來的計畫:Heatmap + Free-cell Mask

```
                  Big Set Transformer + MDN + GP synth
                          (0.889 m 卡關)
                                  │
                                  ▼
              ┌─────────────────────────────────────────┐
              │ 瓶頸診斷:預測落在不可達區(牆/桌/室外)│
              │ → 模型完全不知道地圖長怎樣              │
              └─────────────────────────────────────────┘
                                  │
                                  ▼
              ┌──────────────────────────────────┐
              │ 改變輸出端,讓模型「看見」地圖   │
              │ 1320 個 grid cells + free-mask   │
              └──────────────────────────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
        改 model.py 的 head                  改 train + 後處理
        SetTransformerMDN → SetTransformerHeatmap
        pi/mu/logsig head 拿掉                Loss:
        換成 Linear(192 → 40×33 = 1320)        α·CE(soft target) +
                                                (1-α)·SmoothL1(E[xy], y)
                                              推論:
                                                prob = softmax(logits) × free_mask
                                                prob = prob / prob.sum()
                                                (x,y) = Σ prob · cell_xy
                                  │
                                  ▼
                預期 0.89 → 0.78 ~ 0.82 m
                (現有 ~30% 預測落在不可達區,mask 後直接歸零)


   若還想再上升 → 加完整 Floor Plan CNN Cross-Attention(方向 1 完整版)
                 CNN backbone 編碼 psquare.pgm → spatial features
                 Set Transformer pooled vector ↔ CNN feature map cross-attention
                 預期 0.78 → 0.74 m (但實作 + 調參 2-3 天,風險中等)
```

### 7.3 升級後的目標架構圖

```
        bssid_idx (B,50)   rssi (B,50)   mask (B,50)
              │                │              │
              ▼                ▼              │
        ┌─────────────────────────────┐       │
        │   Set Transformer Encoder   │ ◄─────┘
        │   (Embedding + 3×SAB + PMA) │
        │      → (B, 192)             │   ← 完全保留現有 encoder
        └─────────────────────────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  Dropout(0.3)    │
              └──────────────────┘
                       │
                       ▼
              ┌──────────────────────────┐
              │  Linear(192 → 1320)      │  ← 改這裡
              │  logits over 40×33 grid  │
              │  → (B, 1320)             │
              └──────────────────────────┘
                       │
                       ▼ (訓練)               ▼ (推論)
        ┌───────────────────────┐   ┌────────────────────────────┐
        │ CrossEntropy on soft  │   │ prob = softmax(logits)     │
        │ Gaussian-smoothed     │   │ prob = prob × free_mask    │  ← psquare.pgm
        │ target (label_sigma   │   │ prob = prob / prob.sum()   │     決定哪些
        │ = 0.5 m)              │   │ (x,y) = Σ prob · cell_xy   │     格子是
        │                       │   │ (連續座標,sub-cell)       │     free
        │ + auxiliary SmoothL1  │   │                            │
        │  on expected (x, y)   │   │                            │
        └───────────────────────┘   └────────────────────────────┘
```

**為什麼比 MDN 強**
1. MDN 的 K=5 Gaussian 是「先驗形狀」假設;走廊型/L 型分佈套不住
2. Heatmap 是真正的 non-parametric posterior
3. Free-cell mask 讓「物理不可達」的預測歸零 → 免費贏

**為什麼不直接做 CNN cross-attention 完整版**
- Encoder 不動,Lab 2 的 RSSI 表示法保留 → 風險小
- 半天可以實作完;cross-attention 至少 2-3 天
- 90% 的收益用 20% 的時間

---

## 8. 結果總表

### 8.1 Split A(主戰場)

| 模型 | 參數 | median (m) | mean | p90 | 備註 |
|------|------|-----------|------|-----|------|
| KNN k=5 | — | 1.568 | 1.92 | ~3.5 | 古典 baseline |
| MLP | 53k | 1.34 | — | — | 第一個 NN |
| MDN | 56k | 1.25 | — | — | 加機率輸出 |
| MaskedMDN | 76k | 1.20 | — | — | 加 presence mask |
| Set Transformer MDN | 208k | 1.093 | 1.40 | 3.10 | 變長集合輸入 |
| Big Set Transformer | 619k | 1.10 | — | — | 加正則化 |
| Big × 5-ensemble | 619k×5 | 1.083 | — | — | 平均 5 個 |
| **Big + 5000 GP synth(單)** | 619k | **0.906** | 1.26 | 2.95 | **大躍進** |
| **Big + GP synth × 5 ens** | 619k×5 | **0.889** | 1.24 | 2.99 | **★ 現任最佳** |
| Big + C-Mixup | 619k | 1.016 | — | — | 沒 synth |
| Big + synth + C-Mixup 單 | 619k | 0.900 | — | — | 持平 |
| Big + synth + C-Mixup × 5 ens | 619k×5 | 0.942 | 1.24 | 2.99 | 干擾 |
| Heatmap + synth, best single (s45) | 864k | 0.838 | 1.15 | 2.67 | 架構升級 |
| Heatmap + synth × 5-ens (mean) | 864k×5 | 0.883 | 1.13 | 2.67 | 被 s43 拉走 |
| Heatmap + synth × 5-ens (geom-median) | 864k×5 | 0.836 | 1.13 | 2.76 | |
| Cascade + synth best single (s45) | 882k | 0.778 | 1.18 | 2.65 | 兩頭 head |
| Cascade + synth × 5-ens (mean) | 882k×5 | 0.798 | 1.11 | 2.61 | |
| **Cascade + synth × 5-ens (geom-median)** | 882k×5 | **0.793** | 1.12 | 2.56 | **★★ 目前最佳** |
| CNN xattn + synth × 5-ens (geom)(失敗) | 1.15M×5 | 0.907 | 1.14 | 2.58 | A 沒效益 |

### 8.2 對應檔案

| 階段 | 主要檔案 |
|------|---------|
| EDA | `eda.py` |
| Baselines | `train_baselines.py` |
| Set Transformer | `train_settransformer.py`, `models.py::SetTransformerMDN` |
| Big + ensemble | `train_best_ensemble.py` |
| GP synth | `synthetic.py` |
| Enhanced features | `train_enhanced.py`, `data.py::build_set_input_v2` |
| C-Mixup | `train_cmixup.py`, `train_cmixup_ensemble.py` |
| TTA(沒幫助) | `tta.py` |
| **Heatmap 升級** | `train_heatmap.py`, `models.py::SetTransformerHeatmap` |

### 8.3 結果都存哪

- `outputs/metrics.csv` — 所有模型的 metrics
- `outputs/predictions/*.npz` — 每個 ensemble 的 5-seed 平均預測
- `outputs/checkpoints/*.pt` — 訓練好的 weights
- `outputs/figures/*.png` — 散佈圖、heatmap 圖
- `outputs/*_log.txt` — 每次訓練的詳細 log

---

## 附錄 A:幾個常見疑問

**Q1: 我為什麼要用 -100 當 missing 而不是 0?**
0 dBm 是「貼著 AP」的真實有效值。-100 比真實最弱(~-90)還弱,模型一看就知道是 padding。

**Q2: 為什麼 max_aps=50 而不是 80?**
雖然 vocab 有 80,但實際每個 scan 偵測到 25-40 個,50 是足夠的上限。少數超過 50 的會被截斷(保留最強的)。

**Q3: AMCL 的 GT 誤差 ±0.3 m,意思是我們再強也只能到 0.3 m?**
對。這是 **理論下限**。實際上 0.5 m 已經是「同一張桌子上」精度,對任何實用場景都夠了。

**Q4: 為什麼 Split C(早→晚)結果差?**
WiFi 環境在不同時段有變化(下班後人少了,訊號路徑改變;有些 AP 晚上關了)。這是 well-known indoor loc 問題,通常要靠「持續更新指紋庫」或「對抗式 domain adaptation」解。我們在這份 lab 不深入。

**Q5: 5 個 seed 為什麼是 [42, 43, 44, 45, 46]?**
42 是經典 random seed(《銀河便車指南》梗)。連續 5 個方便記。其實任何 5 個獨立的 seed 都行,結果差不多。

**Q6: 為什麼不用 BERT/GPT 那種預訓練?**
我們資料量太小(1.4k 標籤、1.4k unlabeled scans)。pretrain 至少要幾萬到幾百萬樣本才有意義。
