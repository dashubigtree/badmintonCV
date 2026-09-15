# BadmintonCV — 專案指引

## 專案概述

用電腦視覺辨識羽毛球比賽影片中的球員與場地，目標是逐步建立「上手揮拍能力分級框架」。目前處於 MVP 階段：

1. 場地 ROI / 四點校準標註（`annotate_court_roi.py`、`annotate_court_keypoints.py`、`annotate_court_calibration.py`）
2. 球員偵測與追蹤（YOLO + ByteTrack，`track_players.py`）
3. 姿勢關鍵點標註（yolo11s-pose 模型）
4. 場地小地圖疊加與身分穩定化（`court_minimap.py`、`minimap_identity.py`）

完整操作步驟見 [player_tracking_mvp_manual.md](player_tracking_mvp_manual.md)（新手手冊）與 [README.md](README.md)（指令範例）。

## 目錄結構

| 路徑 | 用途 |
|------|------|
| `scripts/` | 主要 pipeline 程式（標註、追蹤、小地圖、工具函式） |
| `configs/` | 場地 ROI / 校準座標 JSON（小檔案，會進 git） |
| `data/input` `data/frames` `data/output` | 原始影片、擷取幀、產出影片（大檔案，**不進 git**，見 `.gitignore`） |
| `tests/` | 單元測試 |

## 開發規則

- **每次改動後必須建立對應 git commit**，方便追蹤、比對、回滾。commit 只包含本次任務相關變更，不要連帶加入未經要求的檔案。
- **每次改動後必須有測試或驗證**：程式碼改動要新增/更新 `tests/` 對應測試；文件或設定改動至少要讀回檔案、跑一次指令確認可用。交付時說明已執行的驗證方式；若無法測試要講原因。
- 改動超過一個檔案前，先列出會動到哪些檔案。
- 大改動前先 `git add -A && git commit -m "checkpoint before changes"` 存檔點。
- 刪除檔案用 `trash`，不要用 `rm`。
- `*.pt`（模型權重）與 `data/`（影片/圖片）已加入 `.gitignore`；新增其他大型檔案類型（如新的資料格式）前，先確認是否也要排除，不要直接 commit。
- Python 依賴見 `requirements.txt`；目前沒有虛擬環境，執行前建議先建立 venv 再 `pip install -r requirements.txt`。
- 實驗性/一次性腳本放獨立資料夾（例如 `experiments/`），不要混進 `scripts/` 主 pipeline。

## 備註

- 本專案同時對應論文研究方向（見 `~/2026` vault 的 `01-Projects/羽毛球影像偵測分級系統`），程式碼與研究文件分開管理，不互相同步。
- Git 目前為本地 repo，尚未設定 remote。
