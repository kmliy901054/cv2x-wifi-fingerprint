# `map/` — 樓層平面圖(共用)

NYCU BME Lab 的 2D occupancy grid,由 **Lab 2** 的 SLAM(Cartographer)建立,
**Lab 2 與 Lab 3 都會用到**(Lab 2 拿來定位收資料,Lab 3 拿來當 free-cell mask
與視覺化底圖)。

| 檔案 | 說明 |
|---|---|
| `psquare.pgm` | occupancy grid 點陣圖(0=占用/牆、254=自由、~205=未知) |
| `psquare.yaml` | ROS map metadata(見下） |
| `psquare.png` | PNG 版本,給簡報/報告插圖用 |

`psquare.yaml`:

```yaml
image: psquare.pgm
resolution: 0.050000          # 每像素 0.05 m
origin: [-8.462866, -4.640395, 0.0]   # 左下角在 map frame 的座標
occupied_thresh: 0.65
free_thresh: 0.196
```

座標換算:`world_xy = origin + pixel_xy * resolution`(注意 PGM row 0 在影像頂端,
ROS origin 在左下角,畫圖時需 `np.flipud`)。bbox 約 15.97 × 11.87 m = 189.5 m²。
