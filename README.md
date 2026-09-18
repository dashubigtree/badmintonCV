# Badminton CV

以 YOLO Pose、ByteTrack 與球場校正為基礎的羽球影片分析工具。專案可在原始影片上標示球員與人體關鍵點，並將球員位置投影到鳥瞰球場；新版追蹤器會將不穩定的 ByteTrack ID 穩定為固定的 P1–P4 身分。

## 功能

- 場地 ROI 標註與人物篩選。
- YOLO Pose 人體偵測、骨架與腳踝落地點估計。
- 影像像素到球場公尺座標的仿射／Homography 校正。
- 鳥瞰 minimap 與球員移動軌跡。
- 固定球員身分追蹤：遠近半場限制、常速度預測、raw ByteTrack ID 擁有權保護與位置跳點修正。
- 輸出標註影片與逐幀 CSV；CSV 同時含球場位置及 Pose 關鍵點的像素座標／信心值。

## 安裝

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

請依硬體為 Ultralytics 指定推論裝置，例如 Apple Silicon 使用 `--device mps`，CPU 使用 `--device cpu`。

## 快速開始：固定 P1–P4 身分追蹤

先準備影片及校正檔後，執行：

```bash
python -m scripts.track_players_identity \
  --video data/input/test1_short.mp4 \
  --court-calibration configs/test1_court_calibration.json \
  --output data/output/test1_short_identity.mp4 \
  --match-format doubles \
  --model yolo11s-pose.pt \
  --tracker bytetrack.yaml \
  --device mps
```

輸出檔案：

- `data/output/test1_short_identity.mp4`：人框、骨架、固定 P# 色彩與 minimap。
- `data/output/test1_short_identity.csv`：逐幀偵測與短暫遮擋的預測位置。

`--match-format singles` 使用遠場 P1 與近場 P3；`doubles` 使用 P1/P2（遠場）與 P3/P4（近場）。P# 是影片初始時依半場與左右位置建立的邏輯身分，並非人臉或姓名辨識。

常用調整參數：

| 參數 | 預設值 | 用途 |
| --- | ---: | --- |
| `--conf` | `0.25` | 人物偵測置信度門檻。漏偵測多時可小幅降低。 |
| `--court-margin-m` | `1.0` | 允許球場線外仍納入追蹤的距離。 |
| `--side-tolerance-m` | `0.65` | 球網兩側身分配對的容忍範圍。 |
| `--max-speed-m-per-frame` | `0.30` | 判定單幀位置跳點的速度上限。 |
| `--identity-timeout-frames` | `150` | 球員消失多久後釋放其 P# 槽位。 |
| `--predict-display-frames` | `12` | 遮擋期間維持顯示預測位置的幀數。 |

## CSV 欄位

每一列以 `status` 區分：

- `detected`：該幀有實際偵測。`raw_x_m/raw_y_m` 是投影後的原始球場座標；`corrected_*` 是跳點修正後座標；`stable_*` 是 EMA 平滑後、供 minimap 使用的座標。
- `predicted`：該球員短暫遮擋，以常速度模型估計的位置；沒有 raw tracker ID 或 Pose 資料。

`match_mode` 說明該 P# 的連接方式：`new_player`、`strict`、`relaxed_reacquire` 或 `motion_prediction`。`ground_status=corrected_motion_outlier` 表示量測移動過快，已改用預測位置。

Pose 欄位以 `*_x_px`、`*_y_px`、`*_conf` 命名，包含頭部、肩、肘、腕、髖、膝與腳踝。它們是影像像素座標，不能直接套用地面 Homography；只有腳部落地點才適合轉為球場公尺座標。

## 校正流程

### 1. 建立 ROI

```bash
python scripts/annotate_court_roi.py \
  --image data/frames/test1_first_frame.jpg \
  --output configs/test1_court_roi.json \
  --preview data/frames/roi_preview.jpg
```

### 2. 建立球場校正檔

推薦使用四點 far/net 校正：

```bash
python scripts/annotate_court_keypoints.py \
  --image data/frames/test1_first_frame.jpg \
  --output configs/test1_court_calibration.json \
  --preview data/frames/test1_court_keypoint_calibration_preview.jpg
```

依序標示：

1. `far_left_doubles_long_service`：遠端雙打發球線與左雙打邊線交點。
2. `far_right_doubles_long_service`：遠端雙打發球線與右雙打邊線交點。
3. `net_right_post`：網子右側立柱／右雙打邊線對齊點。
4. `net_left_post`：網子左側立柱／左雙打邊線對齊點。

若畫面只能辨識三個點，使用：

```bash
python scripts/annotate_court_keypoints.py \
  --image data/frames/test1_first_frame.jpg \
  --output configs/test1_court_calibration.json \
  --preview data/frames/test1_court_keypoint_calibration_preview.jpg \
  --mode three-point-far
```

## 舊版 minimap 追蹤器

`scripts.track_players.py` 保留作為 ROI、Pose 與基本 minimap 的追蹤器：

```bash
python scripts/track_players.py \
  --video data/input/test1.mp4 \
  --roi configs/test1_court_roi.json \
  --output data/output/test1_pose_minimap.mp4 \
  --model yolo11s-pose.pt \
  --tracker bytetrack.yaml \
  --device mps \
  --minimal-overlay \
  --minimap \
  --court-calibration configs/test1_court_calibration.json
```

此版本以最近位置重連 slot，並可用 `--stale-trail-frames`、`--minimap-match-distance`、`--minimap-merge-distance` 與 `--minimap-hold-frames` 調整。需要跨遮擋維持 P1–P4 時，請優先使用 `track_players_identity.py`。

## 影片與 Git LFS

`data/` 由 `.gitignore` 排除，避免將原始影片、模型輸出或暫存資料意外加入 Git。若要提供可重現的範例影片，建議只挑一支短片並使用 Git LFS：

```bash
git lfs track "data/input/test1_short.mp4"
git add .gitattributes
git add -f data/input/test1_short.mp4
git commit -m "Add sample video via Git LFS"
git push
```

請確認 GitHub 帳號的 LFS 儲存與下載額度足以容納檔案；不要將整個大型 `data/` 目錄直接加入版本控制。

## 測試

```bash
python -m unittest discover -s tests -v
```
