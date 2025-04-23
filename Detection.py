from ultralytics import YOLO
import torch
import cv2
import time
from collections import defaultdict

def run_video_detection(video_path, model_path="yolov8n.pt", delay=0.001, theme="dark"):
    device = "mps" if torch.has_mps else "cpu"
    print(f"Using device: {device}")

    model = YOLO(model_path).to(device)
    cap = cv2.VideoCapture(video_path)

    frame_width = 1280
    frame_height = 720
    paused = False
    prev_time = time.time()

    class_colors = {
        "person": (255, 0, 255),
        "car": (255, 100, 0),
        "bus": (0, 140, 255),
        "motorcycle": (0, 255, 100),
        "truck": (0, 0, 255),
    }

    if theme == "dark":
        bg_color = (137,171,227)
        text_color = (251, 234, 235)
    else:
        bg_color = (100, 140, 140)
        text_color = (0, 0, 0)

    cv2.namedWindow("Traffic Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traffic Detection", frame_width, frame_height)

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (frame_width, frame_height))
            results = model(frame, device=device)[0]

            class_counts = defaultdict(int)

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                label = model.names[class_id]
                class_counts[label] += 1

                color = class_colors.get(label, (255, 255, 0))

                label_text = f"{label}"
                (label_width, label_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                label_bg_x1 = x1
                label_bg_y1 = y1 - label_height - 12
                label_bg_x2 = x1 + label_width + 10
                label_bg_y2 = y1

                cv2.rectangle(frame, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, label_text, (label_bg_x1 + 5, label_bg_y2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            info_start_y = 20
            info_height = 40 + 30 * len(class_counts)
            cv2.rectangle(frame, (20, info_start_y), (330, info_start_y + info_height), bg_color, -1)

            total_count = sum(class_counts.values())
            cv2.putText(frame, f"Objects: {total_count} |  FPS: {fps:.1f} ",
                        (20, info_start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

            y_offset = info_start_y + 60
            for cls_name, count in sorted(class_counts.items()):
                cv2.putText(frame, f"{cls_name}: {count}", (40, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                y_offset += 25

            cv2.imshow("Traffic Detection", frame)
            time.sleep(delay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Video stopped by user.")
            break
        elif key == ord("p"):
            paused = not paused
            print("Paused." if paused else "Resumed.")

    cap.release()
    cv2.destroyAllWindows()
    print("Video capture and windows released.")

run_video_detection("video4.mp4", theme="dark")
