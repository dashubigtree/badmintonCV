# 羽毛球影片人物與場地標註 MVP 新手操作手冊

版本：v0.2  
對象：第一次接觸影片人物標註、場地標註、AI 影像辨識的新手  
目標平台：MacBook Pro M2 Max  
目標任務：用一支羽毛球影片，先確認能不能只標出目標球場上的四位球員，並排除隔壁場次的人。

## 0. 先講結論

你現在要做的事情，可以先想成三個動作：

1. 告訴電腦影片在哪裡。
2. 告訴電腦哪一塊畫面才是你要看的羽球場。
3. 讓電腦把那塊球場裡的球員框起來，輸出一支新影片。

第一版不追求完美，也不需要你一開始就懂所有 AI 名詞。第一版只要能回答這個問題：

> 電腦能不能大致穩定地標出我指定球場上的四個人？

如果答案是「大致可以」，後面再慢慢加強準確率、姿勢、球路、擊球判斷。

## 1. 你會看到哪些東西

完成後，理想情況下會產生一支新影片。影片裡會看到：

- 目標球場被畫出一個多邊形範圍。
- 場上的球員身上有框線。
- 每位球員旁邊有一個 ID，例如 `ID 1`、`ID 2`。
- 隔壁球場的人即使出現在畫面裡，也不會被當成目標球員標出。

如果你看到有些框偶爾消失、ID 偶爾跳掉，第一版可以接受。這份手冊的目的，是先跑出第一個可觀察結果。

## 2. 先認識三個關鍵名詞

### 2.1 人物偵測

人物偵測就是讓電腦在每一張畫面中找出「哪裡有人」。

你可以把它想成：

> 電腦幫每個人畫一個長方形框。

第一版會用 YOLO 這類模型來做這件事。

### 2.2 人物追蹤

影片不是一張圖片，而是一秒很多張圖片。人物追蹤就是讓電腦知道：

> 這一秒的這個人，跟下一秒的那個人，是不是同一個人？

追蹤成功時，同一位球員會維持同一個 ID。第一版會先用 ByteTrack 或 BoT-SORT。

### 2.3 場地 ROI

ROI 是 Region of Interest，意思是「我只關心的區域」。

在這個專案裡，ROI 就是你要看的那一面羽球場。因為影片可能拍到隔壁球場，所以我們要先畫出目標球場範圍，告訴電腦：

> 只有這個範圍裡的人才算目標球員，其他地方的人先忽略。

這就是避免誤判隔壁場球員的關鍵。

## 3. 第一版先不要做的事

為了不要一開始就太複雜，第一版先不做：

- 不先訓練自己的模型。
- 不先分析羽球飛行軌跡。
- 不先判斷擊球動作。
- 不先計算跑動距離。
- 不先做自動辨識球場線。

第一版只做一件事：

> 在你指定的球場範圍內，標出四位場上球員。

## 4. 建議資料夾長相

後續專案建議整理成這樣：

```text
Badminton CV/
├── data/
│   ├── input/
│   │   └── sample.mp4
│   ├── frames/
│   │   └── first_frame.jpg
│   └── output/
│       └── sample_tracked.mp4
├── configs/
│   └── court_roi.json
├── scripts/
│   ├── extract_first_frame.py
│   ├── track_players.py
│   └── draw_roi_preview.py
├── AGENTS.md
├── player_tracking_mvp_manual.md
└── README.md
```

新手可以先只記得這三個位置：

| 位置 | 用途 |
| --- | --- |
| `data/input/` | 放原始羽球影片 |
| `configs/court_roi.json` | 放你畫好的球場範圍 |
| `data/output/` | 放輸出的標註影片 |

## 5. 你要準備什麼

### 5.1 一支羽球影片

建議先選一支短影片測試，最好是 10 到 60 秒。影片越短，第一次測試越不容易等很久。

建議條件：

- 鏡頭固定，畫面不要一直晃。
- 目標球場完整出現在畫面中。
- 盡量選 1080p 或畫質清楚的影片。
- 如果畫面同時有多個球場，也沒關係，正好可以測試 ROI 過濾效果。

### 5.2 一台可以跑 Python 的電腦

你目前的 MacBook Pro M2 Max 足夠做第一版測試。

### 5.3 一點耐心

第一次做影片標註，通常不是一次就完美。最常見的流程是：

1. 先跑一次。
2. 發現場地範圍畫太大或太小。
3. 調整 ROI。
4. 再跑一次。

這是正常的，不是你做錯。

## 6. 第一次執行前的環境準備

以下指令之後會在終端機執行。新手可以先把它理解成：

> 幫這個專案準備一個獨立的 Python 工作環境。

建立虛擬環境：

```bash
python3 -m venv .venv
```

啟用虛擬環境：

```bash
source .venv/bin/activate
```

升級安裝工具：

```bash
python -m pip install --upgrade pip
```

安裝需要的套件：

```bash
pip install ultralytics supervision opencv-python numpy
```

檢查 M2 Max 的 GPU 加速是否可用：

```bash
python - <<'PY'
import torch
print("MPS available:", torch.backends.mps.is_available())
PY
```

如果看到：

```text
MPS available: True
```

表示可以使用 Apple Silicon 的加速。若看到 `False`，第一版仍可先用 CPU 跑，只是速度會比較慢。

## 7. 第一步：放入影片

把你要測試的影片放到：

```text
data/input/
```

建議先把檔名改簡單一點，例如：

```text
data/input/sample.mp4
```

為什麼建議改名：

- 中文長檔名不是不能用，但指令比較容易打錯。
- 新手第一版先降低干擾，等流程熟了再處理批次影片。

## 8. 第二步：擷取影片第一張畫面

我們需要先從影片抓一張圖，因為你要在這張圖上畫出「目標球場」。

預期輸出：

```text
data/frames/first_frame.jpg
```

預期指令：

```bash
python scripts/extract_first_frame.py \
  --video data/input/sample.mp4 \
  --output data/frames/first_frame.jpg
```

這一步成功後，你應該會看到一張圖片。這張圖片就是影片的其中一幀。

成功判斷：

- `data/frames/first_frame.jpg` 存在。
- 打開圖片後，看得到羽球場。
- 圖片比例看起來沒有變形。

## 9. 第三步：畫出你要看的球場範圍

這一步是整個 MVP 最重要的地方。

你要在 `first_frame.jpg` 上標出目標球場範圍。這個範圍可以是一個四邊形，也可以是比較貼合畫面的多邊形。

### 9.1 要畫哪裡

請畫出你要關注的那一面球場，盡量包含：

- 前場。
- 後場。
- 左右邊線。
- 球員可能踩到的界線附近。

建議稍微比實際球場大一點點，避免球員踩在線上時被排除。

但不要大到包進隔壁場，否則隔壁場球員可能又會被算進來。

### 9.2 為什麼不是直接叫 AI 自己知道哪個場地

因為第一版我們還沒有訓練「自動認球場」模型。對新手來說，最穩定的方法是：

> 你先手動畫一次球場範圍，AI 只負責在這個範圍內找人。

這樣可以大幅降低誤判隔壁場的機率。

### 9.3 ROI 設定檔長什麼樣

你可以用專案提供的互動式標註工具來畫。先確認第 8 步已經產生：

```text
data/frames/first_frame.jpg
```

然後執行：

```bash
python scripts/annotate_court_roi.py \
  --image data/frames/first_frame.jpg \
  --output configs/court_roi.json \
  --preview data/frames/roi_preview.jpg
```

執行後會跳出一個圖片視窗。你要做的是：

1. 用滑鼠左鍵沿著目標球場邊界依序點選。
2. 至少點 3 個點，建議先點球場四個角。
3. 點錯可以按 `U` 回到上一點。
4. 想重畫可以按 `R` 清空。
5. 確認範圍正確後按 `Enter` 儲存。
6. 想取消可以按 `Q` 或 `Esc`。

儲存後會得到一組座標，存成：

```text
configs/court_roi.json
```

範例：

```json
{
  "court_polygon": [
    [320, 180],
    [1580, 180],
    [1880, 1040],
    [80, 1040]
  ],
  "anchor": "bottom_center"
}
```

你不需要一開始理解每個數字。先知道：

- 每一組 `[x, y]` 是圖片上的一個點。
- 多個點連起來就是球場範圍。
- `bottom_center` 代表用人物框的底部中心，也就是接近腳的位置，判斷這個人是不是在球場內。

## 10. 第四步：先檢查 ROI 有沒有畫對

在真正跑人物追蹤前，建議先產生一張 ROI 預覽圖。

預期指令：

```bash
python scripts/draw_roi_preview.py \
  --image data/frames/first_frame.jpg \
  --roi configs/court_roi.json \
  --output data/frames/roi_preview.jpg
```

成功後打開：

```text
data/frames/roi_preview.jpg
```

你應該會看到球場上被畫出一個多邊形。

檢查重點：

| 你看到的狀況 | 下一步 |
| --- | --- |
| 多邊形剛好包住目標球場 | 可以進下一步 |
| 多邊形包到隔壁場 | 把 ROI 畫小一點 |
| 多邊形沒有包住球員活動範圍 | 把 ROI 畫大一點 |
| 多邊形位置整個歪掉 | 檢查是不是圖片尺寸或座標用錯 |

## 10.5 選配：建立右上角平面場地校正

如果你想在影片右上角顯示一個平面羽球場，並把球員位置投影到小地圖上，需要先建立一份場地校正檔。

執行：

```bash
python scripts/annotate_court_calibration.py \
  --image data/frames/first_frame.jpg \
  --output configs/court_calibration.json \
  --preview data/frames/court_calibration_preview.jpg
```

跳出圖片視窗後，請依照你希望右上角小地圖呈現的平面方向，點選目標球場四個角：

1. `top_left`
2. `top_right`
3. `bottom_right`
4. `bottom_left`

按 `Enter` 後會產生：

```text
configs/court_calibration.json
data/frames/court_calibration_preview.jpg
```

這份校正檔和 ROI 不同。ROI 是用來排除隔壁場球員；場地校正是用來把畫面中的球員位置轉成平面球場位置。

### 10.5.1 場地底角被裁切時：最佳四點校正

如果畫面像常見手機拍攝那樣，右下角和左下角球場邊界被切掉，不要硬點四個角。優先改用四個較容易出現在畫面中的遠端半場點：

1. `far_left_doubles_long_service`：遠端雙打發球線和左雙打邊線的交點。
2. `far_right_doubles_long_service`：遠端雙打發球線和右雙打邊線的交點。
3. `net_right_post`：網子右側立柱點，或網線和右雙打邊線對齊的位置。
4. `net_left_post`：網子左側立柱點，或網線和左雙打邊線對齊的位置。

執行：

```bash
python scripts/annotate_court_keypoints.py \
  --image data/frames/first_frame.jpg \
  --output configs/court_calibration.json \
  --preview data/frames/court_keypoint_calibration_preview.jpg
```

這個方法會產生以遠端半場為基準的 homography。它比三點校正穩，也比硬點畫面外底角合理；但因為近端半場仍是外推，若要做精準距離、速度或跑動米數分析，仍建議未來補更多標準場地點。

如果你真的只看得到三個點，可以退回三點備援模式：

```bash
python scripts/annotate_court_keypoints.py \
  --image data/frames/first_frame.jpg \
  --output configs/court_calibration.json \
  --preview data/frames/court_keypoint_calibration_preview.jpg \
  --mode three-point-far
```

## 11. 第五步：執行人物偵測與追蹤

這一步會讀取原始影片，輸出一支新影片。

如果只要畫人物方框，使用一般偵測模型：

```bash
python scripts/track_players.py \
  --video data/input/sample.mp4 \
  --roi configs/court_roi.json \
  --output data/output/sample_tracked.mp4 \
  --model yolo11s.pt \
  --tracker bytetrack.yaml \
  --device mps
```

如果要同時畫出人體關節點，改用 pose 模型：

```bash
python scripts/track_players.py \
  --video data/input/sample.mp4 \
  --roi configs/court_roi.json \
  --output data/output/sample_tracked_pose.mp4 \
  --model yolo11s-pose.pt \
  --tracker bytetrack.yaml \
  --device mps
```

如果覺得畫面上的 ID、信心分數、關節文字太干擾，可以加上簡潔模式，只保留方框、關節點、骨架線與 ROI：

```bash
python scripts/track_players.py \
  --video data/input/sample.mp4 \
  --roi configs/court_roi.json \
  --output data/output/sample_tracked_pose_minimal.mp4 \
  --model yolo11s-pose.pt \
  --tracker bytetrack.yaml \
  --device mps \
  --minimal-overlay
```

如果要在右上角顯示平面球場與球員移動尾跡，加入 `--minimap` 和場地校正檔：

```bash
python scripts/track_players.py \
  --video data/input/sample.mp4 \
  --roi configs/court_roi.json \
  --output data/output/sample_tracked_pose_minimap.mp4 \
  --model yolo11s-pose.pt \
  --tracker bytetrack.yaml \
  --device mps \
  --minimap \
  --court-calibration configs/court_calibration.json \
  --max-minimap-players 4 \
  --stale-trail-frames 12 \
  --minimap-match-distance 1.8 \
  --minimap-merge-distance 0.45 \
  --minimap-hold-frames 6
```

如果 `mps` 不能跑，改成：

```bash
python scripts/track_players.py \
  --video data/input/sample.mp4 \
  --roi configs/court_roi.json \
  --output data/output/sample_tracked_pose.mp4 \
  --model yolo11s-pose.pt \
  --tracker bytetrack.yaml \
  --device cpu
```

### 11.1 這個指令在做什麼

| 參數 | 白話意思 |
| --- | --- |
| `--video` | 原始影片在哪裡 |
| `--roi` | 球場範圍設定檔在哪裡 |
| `--output` | 新的標註影片要輸出到哪裡 |
| `--model` | 用哪個 AI 模型找人；要畫關節點請用 `yolo11s-pose.pt` 這類 pose 模型 |
| `--tracker` | 用哪個方法維持人物 ID |
| `--device` | 用 M2 Max GPU 或 CPU 跑 |
| `--pose-conf` | 關節點信心門檻，預設 `0.30`，數字越高越嚴格 |
| `--minimal-overlay` | 簡潔模式，不顯示 ID、信心分數、關節文字或左上角統計 |
| `--minimap` | 在右上角畫平面羽球場與球員移動軌跡 |
| `--court-calibration` | 場地校正檔，使用 `annotate_court_calibration.py` 產生 |
| `--trail-length` | 小地圖保留幾幀的移動尾跡，預設 `90` |
| `--max-minimap-players` | 小地圖最多顯示幾位球員，雙打建議 `4` |
| `--stale-trail-frames` | 球員短暫消失後，保留幾幀等待接回同一個小地圖 ID |
| `--minimap-match-distance` | 新 track ID 距離舊位置多近才接回同一個小地圖 ID，單位約為場地公尺 |
| `--minimap-merge-distance` | 同一幀中，小地圖座標距離太近的偵測會先合併，遠端球員被重複標示時可調高 |
| `--minimap-hold-frames` | 球員短暫漏偵測時，小地圖繼續用淡色顯示最後位置幾幀 |

如果遠端球員常常被標成好幾個點，通常是遠端人物太小、遮擋、或姿態點不穩造成的重複偵測。先試：

```bash
--trail-length 5 --stale-trail-frames 24 --minimap-match-distance 1.0 --minimap-merge-distance 0.6 --minimap-hold-frames 8
```

如果兩個不同球員靠很近卻被合成一個點，把 `--minimap-merge-distance` 降到 `0.3` 到 `0.4`。

如果遠端後場球員常常整個消失，先把 `--conf` 從 `0.5` 降到 `0.35` 到 `0.4`，並把 `--minimap-hold-frames` 調到 `8` 到 `12`。如果消失後回來但 ID 接不回，再把 `--stale-trail-frames` 調到 `30` 到 `45`。

### 11.2 第一次建議用哪個模型

只看人物方框時，第一版建議先用：

```text
yolo11s.pt
```

要看關節點時，第一版建議先用：

```text
yolo11s-pose.pt
```

這會標示頭、左右肩、左右手肘、左右手腕、左右髖、左右膝蓋、左右腳踝。

如果你只是想快速測流程，可以用：

```text
yolo11n.pt
```

如果漏人明顯，再改用：

```text
yolo11m.pt
```

簡單記法：

| 模型 | 速度 | 準確度 | 何時用 |
| --- | --- | --- | --- |
| `yolo11n.pt` | 最快 | 較低 | 只想快速測流程 |
| `yolo11s.pt` | 快 | 中等 | 第一版建議 |
| `yolo11m.pt` | 較慢 | 較高 | 漏人時再試 |

## 12. 第六步：看輸出影片

輸出影片位置：

```text
data/output/sample_tracked.mp4
```

請用一般影片播放器打開它，然後檢查：

- 目標球場是否有被畫出範圍。
- 場上四位球員是否大多數時間都有框。
- 隔壁場球員是否沒有被框出。
- 同一位球員的 ID 是否大致穩定。
- 有沒有某一位球員一直消失。

第一版不要用「每一秒都完美」當標準。請先用下面的方式判斷：

| 結果 | 判斷 |
| --- | --- |
| 四位球員大多數時間都有框，隔壁場沒有被標 | MVP 成功 |
| 四位球員有框，但 ID 偶爾交換 | 可接受，後續調 tracker |
| 常常只剩兩三個人 | 需要調模型或影片品質 |
| 隔壁場一直被標 | 需要重畫 ROI |
| 完全沒有框 | 需要檢查影片路徑、模型、套件安裝 |

## 13. 第一次測試紀錄表

建議每次測試都記錄，不然很容易忘記哪個設定比較好。

| 項目 | 紀錄 |
| --- | --- |
| 測試日期 |  |
| 影片檔名 |  |
| 影片長度 |  |
| 使用模型 |  |
| 使用 tracker |  |
| 是否使用 `mps` |  |
| ROI 是否正確包住目標球場 | 是 / 否 |
| 是否大多數時間標出四位球員 | 是 / 否 |
| 是否誤標隔壁場球員 | 是 / 否 |
| ID 是否穩定 | 穩定 / 偶爾交換 / 經常交換 |
| 下一次要調整什麼 |  |

## 14. 常見狀況

### 14.1 隔壁場球員也被標出來

最可能原因：

- ROI 畫太大，包到隔壁場。
- 球員框的底部中心點剛好落在 ROI 裡。

新手優先處理方式：

1. 回到 ROI 預覽圖。
2. 把球場範圍縮小一點。
3. 再跑一次。

### 14.2 場內球員沒有被標到

最可能原因：

- 球員太小或太模糊。
- 模型太小。
- ROI 沒有包到球員腳的位置。

新手優先處理方式：

1. 先確認 ROI 是否包住整個球員活動範圍。
2. 把模型從 `yolo11n.pt` 改成 `yolo11s.pt`。
3. 還是不行，再試 `yolo11m.pt`。

### 14.3 ID 一直換來換去

最可能原因：

- 兩位球員交錯遮擋。
- 球員衣服顏色接近。
- 追蹤器設定不適合這段影片。

新手優先處理方式：

1. 先接受第一版有少量 ID 交換。
2. 若交換很嚴重，從 ByteTrack 改成 BoT-SORT。
3. 後續再加入 ReID 或站位邏輯。

### 14.4 跑得很慢

最可能原因：

- 影片太長。
- 模型太大。
- 沒有使用 MPS。

新手優先處理方式：

1. 先用 10 到 30 秒短片。
2. 模型先用 `yolo11n.pt` 或 `yolo11s.pt`。
3. 確認 `--device mps` 是否可用。

### 14.5 完全看不懂錯誤訊息

這很正常。請先保留三個資訊：

- 你執行的完整指令。
- 終端機出現的錯誤訊息。
- 你使用的影片檔名。

有這三個資訊，就比較容易定位問題。

## 15. 什麼時候算第一版成功

當你能產生一支新影片，並且符合以下條件，就算第一版成功：

- 看得到目標球場 ROI。
- 目標球場內大多數時間有四位球員框線。
- 隔壁場球員大多數時間沒有被標出。
- 輸出影片可以正常播放。

第一版成功後，就可以進入第二階段：

- 改善 ID 穩定度。
- 加入人體姿態。
- 自動辨識球場線。
- 加入羽球軌跡。
- 建立自己的訓練資料。

## 16. 如果第一版失敗，先不要急著訓練模型

很多新手會以為結果不好就要立刻訓練模型。其實第一版通常先調這些：

1. ROI 是否畫對。
2. 影片是否太長、太糊或角度太差。
3. 模型大小是否太小。
4. tracker 是否適合。
5. 是否真的只保留 ROI 裡的人。

等這些都確認過，仍然不穩，再考慮自訂訓練。

## 17. 下一步建議

這份手冊目前是操作流程說明。接下來建議依序完成：

1. 建立 `data/input/`、`data/frames/`、`data/output/`、`configs/`。
2. 選一支短影片當 `sample.mp4`。
3. 寫或確認 `extract_first_frame.py`。
4. 寫或確認 `draw_roi_preview.py`。
5. 寫或確認 `track_players.py`。
6. 跑第一支標註影片。
7. 用第 13 節紀錄第一次結果。

不用一次做到完美。先讓第一支影片跑起來，才會知道真正需要改善的是 ROI、模型、追蹤器，還是影片本身。
