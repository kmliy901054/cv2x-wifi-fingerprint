# `trajectories/` — 表格化的軌跡 / 指紋(共用資料集)

[`../wifi/`](../wifi/) 原始 jsonl 用 `lab2/code/jsonl_to_csv.py` 攤平成 CSV,
方便直接餵給工具或試算表。內容與 jsonl 相同,只是換成扁平格式。

| 檔案 | 列數 | 一列代表 | 欄位 |
|---|---:|---|---|
| `trajectories_slim.csv` | 1,513 | 一個位置點 | `path_id, x, y, z, r, g, b` |
| `trajectories_wide.csv` | 50,196 | 一筆 AP 偵測 | `path_id, x, y, z, r, g, b, yaw, ssid, bssid, rssi, channel, encryption, stamp_sec, stamp_nanos` |

- **slim**：學長指定的精簡格式(位置 + 每條 path 的 HSV 顏色),給軌跡疊圖用。
- **wide**：每個 (scan × 掃到的 AP) 展開成一列(50,196 = 所有 scan 的 AP 偵測總數),
  保留 wifi 欄位,給後處理 / EDA 用。
- `path_id` 把連續軌跡切成 30 秒一段(`jsonl_to_csv.py --split-by-time 30`),
  早上 113 段 + 晚上 108 段 = 221 段。

> Lab 3 訓練是直接讀 [`../wifi/`](../wifi/) 的 jsonl,不是這裡的 CSV;
> 這些 CSV 主要是 Lab 2 的交付格式與視覺化來源。
