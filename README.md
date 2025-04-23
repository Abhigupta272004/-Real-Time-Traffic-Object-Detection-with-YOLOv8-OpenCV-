# 🚦 Real-Time Traffic Object Detection using YOLOv8

This project implements a real-time object detection system using [Ultralytics YOLOv8](https://docs.ultralytics.com/) and OpenCV to detect and count vehicles and people in video feeds.

---

## 🎯 Features

- 🔍 **YOLOv8-based object detection** (person, car, truck, bus, motorcycle)
- 🎨 **Theme toggle** – dark or light visualization
- 📊 **Real-time analytics** – total object count and per-class count
- 🎥 **Custom video input** – run detection on any video file
- ⏸️ **Pause/Resume control** with keyboard
- 📈 **FPS display** – measure real-time performance

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **YOLOv8 (Ultralytics)**
- **OpenCV**
- **PyTorch**

---

## 📂 Directory Structure

```
📁 object-detection-yolov8/
│
├── detection.py           # Main script
├── yolov8n.pt             # Pre-trained YOLOv8 model
└── README.md              # Project description
```

---

## ▶️ How to Run

1. **Install dependencies**:
```bash
pip install ultralytics opencv-python torch
```

2. **Download the YOLOv8 model** (if not present):
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
```

3. **Run the detection**:
```bash
python detection.py
```

*To use a different video, change the file path in the last line of the script:*
```python
run_video_detection("your_video.mp4", theme="dark")
```

---

## ⌨️ Controls

| Key | Action          |
|-----|-----------------|
| q   | Quit            |
| p   | Pause/Resume    |

---

## 📸 Sample Output

> ![FEF482D2-0525-43B2-A641-AF9181C750F0_1_105_c](https://github.com/user-attachments/assets/c9a290ef-e839-444c-bbb0-8b6f0d52403d)


---



## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)

