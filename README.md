# AI Nhận Diện Biển Báo & Giữ Làn Đường Cho Robot Tự Hành (Webots R2025a)

**Nhóm Nanana:** Nguyễn Quốc Bảo · Cát Minh Khoa · Ngô Quang Minh

Hệ thống mô phỏng robot/xe tự hành 4 bánh Ackermann trên **Webots R2025a** kết hợp:
- **Nhận diện 8 biển báo giao thông Việt Nam (QCVN 41:2019/BGTVT):** `stop`, `turn_left`, `turn_right`, `go_straight`, `no_left_turn`, `no_right_turn`, `speed_limit_30`, `speed_limit_50` bằng **YOLOv11n** (chạy CUDA FP16 trên GPU NVIDIA).
- **Giữ làn đường (Lane Keeping):** OpenCV (HLS/HSV + Sobel X + Bird's Eye View + Sliding Window + Polynomial Fitting) + bộ điều khiển **PID** chống bão hòa tích phân.
- **Ra quyết định (Behavior FSM):** Lọc nhiễu theo chuỗi khung hình (debouncing), khoảng cách kích hoạt theo chiều cao bounding box, cooldown tránh dừng lặp và cờ cấm rẽ.
- **Giám sát & đánh giá định lượng:** Dashboard OpenCV thời gian thực (Camera + BEV + Telemetry) + log CSV & biểu đồ KPI (`Mean |CTE|`, `% In-Lane`, `FPS`, `mAP@50`).

## 1. Cấu trúc thư mục
- `config/robot_config.yaml`: Toàn bộ tham số ROI, PID, YOLO, tốc độ và FSM.
- `webots_project/`: World mô phỏng `autonomous_track.wbt`, PROTO biển báo Việt Nam `VietnamTrafficSign.proto`, texture biển báo và 2 controller (`autonomous_driver`, `dataset_collector`).
- `src/`: Các module lõi (`perception/`, `control/`, `ui/`, `utils/`).
- `training/`: Sinh texture biển báo, chuẩn bị dataset, huấn luyện & đánh giá YOLOv11n.
- `evaluation/`: Phân tích file CSV telemetry và xuất biểu đồ cho báo cáo.
- `tests/`: Unit test tự động bằng `pytest`.

## 2. Chạy nhanh trên Windows
### Bước 1: Kích hoạt môi trường ảo Python 3.11
```powershell
cd D:\Projects\AI&Robotics
.\.venv\Scripts\Activate.ps1
```

### Bước 2: Chạy mô phỏng trên Webots R2025a
1. Mở **Webots R2025a** (`D:\Webots\Webots.lnk` hoặc gõ `webots` trong PowerShell mới).
2. Chọn **File → Open World...** → mở `D:\Projects\AI&Robotics\webots_project\worlds\autonomous_track.wbt`.
3. Bấm **Play (▶)** — cửa sổ **Live HUD** sẽ tự động mở và robot bắt đầu tự hành giữ làn + xử lý biển báo.
   - Phím **`M`**: Chuyển giữa chế độ Tự hành (`AUTONOMOUS`) và Lái tay (`MANUAL` bằng phím mũi tên + Space).
   - Phím **`R`**: Reset bộ lọc làn đường và PID.

### Bước 3: Thu thập thêm ảnh thật từ Webots & Fine-tune lại YOLO (khuyên dùng để tối ưu tối đa)
1. Trong Scene Tree của Webots, chọn `BmwX5 ("autonomous_robot")` → đổi field `controller` từ `"autonomous_driver"` sang `"dataset_collector"`.
2. Cho chạy 1–2 phút: ảnh 3D và nhãn YOLO tự động lưu vào `datasets/raw_webots/`.
3. Đổi `controller` lại thành `"autonomous_driver"`, rồi chạy:
```powershell
python training/prepare_dataset.py --synthetic-per-class 200
python training/train_yolo.py --epochs 60 --batch 32
python training/evaluate_model.py
```

### Bước 4: Xuất biểu đồ đánh giá sau khi chạy mô phỏng
```powershell
python evaluation/analyze_run.py
```
Biểu đồ PNG (CTE, tốc độ, quỹ đạo GPS, độ trễ xử lý) được lưu trong thư mục `logs/`.

## 3. Kiểm thử tự động
```powershell
pytest tests/ -v
```
