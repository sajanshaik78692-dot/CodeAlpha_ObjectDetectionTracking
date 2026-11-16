import cv2
import time
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict
from ultralytics import YOLO

# ==========================================
# 1. INPUT & MODEL PRECISION SELECTOR
# ==========================================
def setup_session():
    print("================================================================")
    print("   CODEALPHA - HIGH-ACCURACY AI OBJECT DETECTION & TRACKING     ")
    print("================================================================")
    print("Select Input Source:")
    print("  1. Live Webcam (Default)")
    print("  2. Browse Video File (.mp4, .avi, .mkv, .mov)")
    src_choice = input("Enter choice (1 or 2) [Default=1]: ").strip()

    source = 0
    source_label = "Live Webcam"
    if src_choice == "2":
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(
            title="Select Video File for High-Accuracy Scanning",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov *.wmv"), ("All Files", "*.*")]
        )
        root.destroy()
        if file_path:
            source = file_path
            source_label = file_path
        else:
            print("[INFO] No file selected. Defaulting to Live Webcam...")

    print("\nSelect Detection Accuracy Mode:")
    print("  1. Balanced High-Accuracy (YOLO26s @ 960px + BoT-SORT) [Recommended]")
    print("  2. Ultra-Precision Deep Scan (YOLO26m @ 960px + BoT-SORT)")
    print("  3. High-Speed Edge Mode (YOLO26n @ 640px + BoT-SORT)")
    mode_choice = input("Enter choice (1, 2, or 3) [Default=1]: ").strip()

    if mode_choice == "2":
        return source, source_label, "yolo26m.pt", 960, "Ultra-Precision (YOLO26m)"
    elif mode_choice == "3":
        return source, source_label, "yolo26n.pt", 640, "High-Speed (YOLO26n)"
    return source, source_label, "yolo26s.pt", 960, "High-Accuracy (YOLO26s)"

# ==========================================
# 2. TACTICAL SCANNER BOUNDING BOX & HUD
# ==========================================
def draw_scanner_box(img, x1, y1, x2, y2, color, label):
    # Subtle translucent fill inside bounding box
    sub_img = img[y1:y2, x1:x2]
    if sub_img.size > 0:
        rect = sub_img.copy()
        rect[:] = color
        cv2.addWeighted(rect, 0.12, sub_img, 0.88, 0, sub_img)

    # Outer frame + precision corner brackets
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    line_len = min(24, int((x2 - x1) * 0.25), int((y2 - y1) * 0.25))
    t = 3
    cv2.line(img, (x1, y1), (x1 + line_len, y1), color, t)
    cv2.line(img, (x1, y1), (x1, y1 + line_len), color, t)
    cv2.line(img, (x2, y1), (x2 - line_len, y1), color, t)
    cv2.line(img, (x2, y1), (x2, y1 + line_len), color, t)
    cv2.line(img, (x1, y2), (x1 + line_len, y2), color, t)
    cv2.line(img, (x1, y2), (x1, y2 - line_len), color, t)
    cv2.line(img, (x2, y2), (x2 - line_len, y2), color, t)
    cv2.line(img, (x2, y2), (x2, y2 - line_len), color, t)

    # Center crosshair dot
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(img, (cx, cy), 3, (255, 255, 255), -1)

    # Label badge
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    y_top = max(y1 - h - 10, 44)
    cv2.rectangle(img, (x1, y_top), (x1 + w + 10, y_top + h + 8), color, -1)
    cv2.putText(img, label, (x1 + 5, y_top + h + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

# ==========================================
# 3. MAIN DETECTION & BOTSORT TRACKING LOOP
# ==========================================
def main():
    source, source_label, model_weights, scan_imgsz, mode_label = setup_session()

    print(f"\n[INFO] Loading {mode_label} weights ({model_weights})...")
    model = YOLO(model_weights)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source_label}")
        return

    conf_thresh = 0.40
    track_history = defaultdict(list)
    unique_ids_seen = set()
    prev_time = time.time()
    paused = False
    recording = False
    video_writer = None
    annotated_frame = None

    print("\n[LIVE CONTROLS]")
    print("  [+] / [-]      : Increase / Decrease Detection Confidence")
    print("  [P] / [Space]  : Pause / Resume Frame")
    print("  [R]            : Start / Stop MP4 Video Recording")
    print("  [S]            : Save High-Res Snapshot")
    print("  [Q] / [Esc]    : Quit\n")

    window_name = f"CodeAlpha AI Scanner - {mode_label} + BoT-SORT"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while cap.isOpened():
        if not paused:
            success, frame = cap.read()
            if not success:
                print("[INFO] End of video stream.")
                break

            # Normalize display resolution while preserving aspect ratio
            fh, fw = frame.shape[:2]
            if fw > 1280:
                scale = 1280 / fw
                frame = cv2.resize(frame, (1280, int(fh * scale)))

            # High-accuracy YOLO26 + BoT-SORT tracking pass
            results = model.track(
                frame,
                persist=True,
                tracker="botsort.yaml",
                imgsz=scan_imgsz,
                conf=conf_thresh,
                iou=0.50,
                agnostic_nms=True,
                verbose=False
            )

            annotated_frame = frame.copy()
            active_counts = defaultdict(int)

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu().numpy()

                for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = model.names[cls_id].capitalize()
                    active_counts[class_name] += 1
                    unique_ids_seen.add(track_id)

                    color = (
                        int((track_id * 83 + 60) % 235),
                        int((track_id * 149 + 80) % 235),
                        int((track_id * 211 + 100) % 235)
                    )

                    label = f"ID:{track_id} | {class_name} ({conf:.0%})"
                    draw_scanner_box(annotated_frame, x1, y1, x2, y2, color, label)

                    # Smooth trajectory trail
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    track = track_history[track_id]
                    track.append((cx, cy))
                    if len(track) > 35:
                        track.pop(0)

                    for i in range(1, len(track)):
                        thickness = max(1, int((i / len(track)) * 3))
                        cv2.line(annotated_frame, track[i - 1], track[i], color, thickness)

            # FPS calculation
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
            prev_time = curr_time

            # Top Telemetry HUD
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (annotated_frame.shape[1], 42), (17, 17, 27), -1)
            cv2.addWeighted(overlay, 0.82, annotated_frame, 0.18, 0, annotated_frame)

            counts_str = " | ".join([f"{k}: {v}" for k, v in active_counts.items()]) if active_counts else "Scanning..."
            hud_text = f"FPS: {fps:.1f} | Conf: {conf_thresh:.2f} | Unique IDs: {len(unique_ids_seen)} | Active: {counts_str}"
            cv2.putText(annotated_frame, hud_text, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (166, 227, 161), 2)

            if recording:
                cv2.circle(annotated_frame, (annotated_frame.shape[1] - 90, 21), 7, (0, 0, 255), -1)
                cv2.putText(annotated_frame, "REC", (annotated_frame.shape[1] - 75, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if video_writer is not None:
                    video_writer.write(annotated_frame)

        if annotated_frame is not None:
            display_frame = annotated_frame.copy()
            if paused:
                cv2.putText(display_frame, "PAUSED - Press [P] to Resume", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2)
            cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord("p"), ord(" ")):
            paused = not paused
        elif key in (ord("+"), ord("=")):
            conf_thresh = min(0.85, round(conf_thresh + 0.05, 2))
            print(f"[CONF] Confidence threshold -> {conf_thresh:.2f}")
        elif key in (ord("-"), ord("_")):
            conf_thresh = max(0.15, round(conf_thresh - 0.05, 2))
            print(f"[CONF] Confidence threshold -> {conf_thresh:.2f}")
        elif key == ord("s") and annotated_frame is not None:
            cv2.imwrite("detection_snapshot.jpg", annotated_frame)
            print("[SAVED] High-res snapshot saved -> detection_snapshot.jpg")
        elif key == ord("r") and annotated_frame is not None:
            if not recording:
                rh, rw = annotated_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter("tracked_output.mp4", fourcc, 20.0, (rw, rh))
                recording = True
                print("[REC] Recording started -> tracked_output.mp4")
            else:
                recording = False
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                print("[REC] Recording saved -> tracked_output.mp4")

    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()