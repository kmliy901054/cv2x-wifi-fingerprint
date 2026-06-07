# Lab 3 — 深度學習室內定位

用 [Lab 2](../lab2/) 蒐集的指紋資料集(1,812 筆 (RSSI, pose)、189.5 m²)訓練
室內定位模型。從經典 KNN(中位誤差 1.57 m)一路做到 coarse-to-fine cascade,
中位誤差 **0.752 m**(嚴格巢狀交叉驗證 ~0.94 m)。最後接 ESP32 在實驗室即時跑。

## 文件導覽

| 文件 | 內容 |
|---|---|
| [LAB3_REPORT.md](LAB3_REPORT.md) | 課程報告(題目 / 做法 / 切分 / 結果) |
| [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | 完整技術報告:所有方法 + 8 個失敗實驗 + 驗證 |
| [EVOLUTION.md](EVOLUTION.md) | 模型演進史 1.57 → 0.75 m,含走過的失敗路線 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 各模型細節與訓練配方 |
| [DEMO.md](DEMO.md) | 即時 demo 操作(matplotlib 或 ROS 2 + RViz) |
| [outputs/slides/](outputs/slides/) | 簡報 `lab3_presentation.pptx`(33 張)+ 講稿 |
| [.claude/skills/run-lab3/](.claude/skills/run-lab3/) | 一鍵 reproduce / demo / screenshot 的 driver + skill |

## 成果(Split A 隨機 80/20 測試,363 筆)

| 模型 | 中位誤差 | 重點 |
|---|---:|---|
| KNN k=5 | 1.568 m | 經典 fingerprinting 基準線 |
| Set Transformer MDN | 1.093 m | 變長集合輸入 |
| + GP 合成資料 | 0.906 m | 填補空間覆蓋缺口(單一最大進步)|
| Heatmap + free-mask ×5 | 0.883 m | 分類取代回歸 |
| Cascade ×5-ens | 0.793 m | 粗網格守門細網格 |
| **Cascade-aggressive ×5-ens** | **0.752 m** | 調損失權重 ── 冠軍 |

> 表中為標準 train→test 數字;嚴格的巢狀五折交叉驗證(每折重訓、僅用 4/5 資料)
> 為 ~0.94 m。驗證方法見 [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) §6。

用已 commit 的權重重現數字(CPU 即可):

```bash
python load_best_model.py                   # tuned 5-seed → median 0.760 m
python load_best_model.py --variant baseline   # → 0.793 m
python .claude/skills/run-lab3/driver.py smoke # reproduce + demo + 截圖一次跑完
```

## 程式結構

```
data.py, models.py            資料管線 + 所有模型定義
train_*.py                    演進史上每個實驗各一支訓練腳本(含失敗實驗)
load_best_model.py            從已 commit 權重重現(tuned 0.760 / baseline 0.793 m)
honest_nested_cv.py           巢狀 5-fold CV(每折重訓,~0.94 m)
honest_validation.py, honest_5fold*.py   交叉驗證 / 模型選擇檢查
evaluate.py, tta.py           評估 + test-time augmentation
synthetic.py                  GP-kriging 合成資料生成

esp32_localizer_ros.py        即時 ROS 2 定位節點
lab3_demo.launch.py           啟動節點 + RViz
esp32_localizer.rviz          RViz 版面
run_at_lab.sh                 一鍵 live / replay 包裝
gt_error_node.py              在 RViz 點真實位置,即時印誤差
realtime_demo.py              matplotlib 版即時 demo

make_report_figures.py        結果圖(CDF、ladder、scatter、可靠度…)
make_arch_figures.py          各模型架構方塊圖
make_gating_figure.py         「cascade 為何贏」的 gating 對比圖

outputs/
  checkpoints/                權重(只 commit 勝出的 5-seed cascade)
  predictions/                各模型各 split 的測試預測 (.npz)
  figures/                    報告 + 簡報用圖
  plots/, plots_synthetic/    EDA + 合成資料診斷圖
  slides/                     簡報
  metrics.csv                 所有實驗的指標
```

> 資料來源在 repo 根目錄(與 Lab 2 共用):[`../map/`](../map/)、[`../wifi/`](../wifi/)。
> `data.py` 直接讀 `../wifi/` 的 jsonl。

## 即時 demo,一行啟動

```bash
sudo chmod 666 /dev/ttyACM0    # 每次開機一次,讓使用者能讀 ESP32
./run_at_lab.sh                # live ESP32 + RViz;或 ./run_at_lab.sh replay
```

細節與排錯見 [DEMO.md](DEMO.md)。
