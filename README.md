

python scripts/annotate_court_roi.py --image data/frames/test1_first_frame.jpg --output configs/court_roi.json --preview data/frames/roi_preview.jpg

python scripts/track_players.py --video data/input/test1.mp4 --roi configs/court_roi.json --output data/output/test1_sample_tracked.mp4 --model yolo11s.pt --tracker bytetrack.yaml --device mps

# 關節標記版本
python scripts/track_players.py --video data/input/test2.mp4 --roi configs/test2_court_roi.json --output data/output/test2_sample_tracked_pose.mp4 --model yolo11s-pose.pt --tracker bytetrack.yaml --device mps

# 場地校準來標示人物在球場上相對位置
python scripts/annotate_court_calibration.py --image data/frames/test1_first_frame.jpg --output configs/test1_court_calibration.json --preview data/frames/test1_court_calibration_preview.jpg

python scripts/track_players.py --video data/input/test1.mp4 --roi configs/test1_court_roi.json --output data/output/test1_sample_tracked_pose_minimap.mp4 --model yolo11s-pose.pt --tracker bytetrack.yaml --device mps --minimap --court-calibration configs/test1_court_calibration.json

python scripts/track_players.py --video data/input/test1.mp4 --roi configs/test1_court_roi.json --output data/output/test1_sample_tracked_pose_minimap_minimal.mp4 --model yolo11s-pose.pt --tracker bytetrack.yaml --device mps --minimal-overlay --minimap --court-calibration configs/test1_court_calibration.json （遮蔽畫面上的數據點）

## 升級版
python scripts/annotate_court_keypoints.py --image data/frames/test1_first_frame.jpg --output configs/test1_court_calibration.json --preview data/frames/test1_court_keypoint_calibration_preview.jpg (四個點版本)

1. far_left_doubles_long_service
遠端雙打發球線 × 左雙打邊線

2. far_right_doubles_long_service
遠端雙打發球線 × 右雙打邊線

3. net_right_post
網子右側立柱點，或網線和右雙打邊線對齊的位置

4. net_left_post
網子左側立柱點，或網線和左雙打邊線對齊的位置

python scripts/annotate_court_keypoints.py --image data/frames/test1_first_frame.jpg --output configs/test1_court_calibration.json --preview data/frames/test1_court_keypoint_calibration_preview.jpg --mode three-point-far （三個點版本）

1. far_center_back_doubles_service
遠端中線 × 遠端雙打發球線的交點
也就是你前面說「中線和後方雙打發球線的交點」。

2. net_left_post
網子左側立柱點，或網線和左雙打邊線對齊的位置。

3. net_right_post
網子右側立柱點，或網線和右雙打邊線對齊的位置。

# track player
python scripts/track_players.py --video data/input/test1.mp4 --roi configs/test1_court_roi.json --output data/output/test1_sample_tracked_pose_minimap.mp4 --model yolo11s-pose.pt --tracker bytetrack.yaml --device mps --minimal-overlay --minimap --court-calibration configs/test1_court_calibration.json


# 人物閃爍和多重點問題
python scripts/track_players.py --video data/input/test1_short.mp4 --roi configs/test1_court_roi.json --output data/output/test1_short_sample_tracked_pose_minimap_stable.mp4 --model yolo11s-pose.pt --tracker bytetrack.yaml --device mps --minimal-overlay --minimap --court-calibration configs/test1_court_calibration.json --max-minimap-players 4 --stale-trail-frames 36 --minimap-match-distance 1.2 --minimap-merge-distance 0.6 --minimap-hold-frames 10 --trail-length 5 --conf 0.35 --pose-conf 0.25

調整方向：
- 還是常消失：--conf 0.35 再降到 0.30
- 消失幾幀可以接受但不要閃：--minimap-hold-frames 10 調到 12 或 15
- 消失後回來接不回 ID：--stale-trail-frames 36 調到 45
- 遠端兩個人被合成一個點：--minimap-merge-distance 0.6 降到 0.4