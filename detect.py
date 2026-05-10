"""
Hazardous Zone Intrusion Detection System
==========================================
Real-time human detection in restricted areas using YOLOv8.

Usage:
    # Live webcam
    python detect.py

    # From a video file
    python detect.py --source path/to/video.mp4

    # Custom confidence threshold and hazard zone
    python detect.py --confidence 0.6 --zone 100 100 540 380

    # Save the annotated output video
    python detect.py --save

Author: M. Adhitya | IITRAM Ahmedabad | Computer Vision Project
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "best.pt"
DEFAULT_SOURCE     = 0            # 0 = default webcam; pass a file path for video
DEFAULT_CONFIDENCE = 0.50         # minimum confidence to count as a detection
ALERT_COOLDOWN_SEC = 3            # seconds between repeated alerts for the same event
LOG_FILE           = "intrusion_log.csv"

# Colours (BGR)
COLOR_SAFE     = (0,   200,  0)   # green  — person outside hazard zone
COLOR_DANGER   = (0,   0,   255)  # red    — person inside hazard zone
COLOR_ZONE     = (0,   165, 255)  # orange — hazard zone boundary
COLOR_HUD      = (200, 200, 200)  # light grey — HUD text
COLOR_ALERT    = (0,   0,   255)  # red    — ALERT banner


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hazardous Zone Intrusion Detection System — YOLOv8"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Path to YOLOv8 weights file (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--source", default=DEFAULT_SOURCE,
        help="Video source: 0 for webcam, or path to a video file (default: 0)"
    )
    parser.add_argument(
        "--confidence", type=float, default=DEFAULT_CONFIDENCE,
        help=f"Minimum detection confidence 0-1 (default: {DEFAULT_CONFIDENCE})"
    )
    parser.add_argument(
        "--zone", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help=(
            "Hazard zone as top-left / bottom-right pixel coordinates. "
            "Example: --zone 100 100 540 380. "
            "If omitted, the entire frame is treated as the hazard zone."
        )
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save the annotated output video to disk."
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Disable CSV intrusion logging."
    )
    return parser.parse_args()


def is_inside_zone(box_xyxy: list[float], zone: tuple[int, int, int, int]) -> bool:
    """
    Return True if the centre of the bounding box falls inside the hazard zone.
    box_xyxy  : [x1, y1, x2, y2] from YOLO result
    zone      : (zx1, zy1, zx2, zy2) pixel coordinates
    """
    cx = (box_xyxy[0] + box_xyxy[2]) / 2
    cy = (box_xyxy[1] + box_xyxy[3]) / 2
    zx1, zy1, zx2, zy2 = zone
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


def draw_hud(
    frame,
    fps: float,
    total_detections: int,
    alert_active: bool,
    zone: tuple | None,
) -> None:
    """Overlay FPS, detection count, zone, and alert banner onto frame."""
    h, w = frame.shape[:2]

    # Hazard zone rectangle
    if zone:
        zx1, zy1, zx2, zy2 = zone
        overlay = frame.copy()
        cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), COLOR_ZONE, -1)
        cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)   # semi-transparent fill
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), COLOR_ZONE, 2)
        cv2.putText(
            frame, "HAZARD ZONE", (zx1 + 6, zy1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ZONE, 1, cv2.LINE_AA
        )

    # Top-left HUD
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_HUD, 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"Detections: {total_detections}", (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_HUD, 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"), (12, 84),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_HUD, 1, cv2.LINE_AA
    )

    # Alert banner (bottom of frame)
    if alert_active:
        banner_y = h - 50
        cv2.rectangle(frame, (0, banner_y), (w, h), (0, 0, 180), -1)
        cv2.putText(
            frame,
            "⚠  INTRUSION DETECTED — RESTRICTED AREA",
            (20, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA
        )


def log_event(writer, confidence: float, zone_breach: bool) -> None:
    """Append one detection event row to the CSV log."""
    writer.writerow({
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
        "confidence":   f"{confidence:.3f}",
        "zone_breach":  zone_breach,
    })


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Model ───────────────────────────────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Make sure best.pt is in the same folder as detect.py."
        )
    model = YOLO(str(model_path))
    print(f"[INFO] Loaded model: {model_path}")

    # ── Video source ─────────────────────────────────────────────────────────
    source = args.source
    try:
        source = int(source)   # webcam index
    except (ValueError, TypeError):
        pass                   # keep as string path

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[INFO] Source: {source}  |  {frame_w}x{frame_h} @ {src_fps:.1f} FPS")

    # ── Hazard zone ──────────────────────────────────────────────────────────
    if args.zone:
        zone = tuple(args.zone)          # (x1, y1, x2, y2)
    else:
        zone = (0, 0, frame_w, frame_h)  # full frame
    print(f"[INFO] Hazard zone: {zone}")

    # ── Output video writer ──────────────────────────────────────────────────
    writer_video = None
    if args.save:
        out_path = Path("output_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_video = cv2.VideoWriter(str(out_path), fourcc, src_fps, (frame_w, frame_h))
        print(f"[INFO] Saving output to: {out_path}")

    # ── CSV log ──────────────────────────────────────────────────────────────
    log_file = None
    csv_writer = None
    if not args.no_log:
        log_file = open(LOG_FILE, "a", newline="")
        fieldnames = ["timestamp", "confidence", "zone_breach"]
        csv_writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        if log_file.tell() == 0:
            csv_writer.writeheader()
        print(f"[INFO] Logging intrusions to: {LOG_FILE}")

    # ── Detection state ──────────────────────────────────────────────────────
    last_alert_time = 0.0
    total_detections = 0
    fps_start = time.time()
    fps_counter = 0
    current_fps = 0.0

    print("[INFO] Starting detection. Press Q to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream.")
                break

            # ── YOLO inference ───────────────────────────────────────────────
            results = model.predict(
                frame,
                conf=args.confidence,
                classes=[0],          # class 0 = person (COCO)
                verbose=False,
            )

            alert_this_frame = False
            boxes = results[0].boxes

            for box in boxes:
                xyxy = box.xyxy[0].tolist()    # [x1, y1, x2, y2]
                conf = float(box.conf[0])
                total_detections += 1

                in_zone = is_inside_zone(xyxy, zone)

                # ── Draw bounding box ────────────────────────────────────────
                color = COLOR_DANGER if in_zone else COLOR_SAFE
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"Person {'⚠' if in_zone else '✓'}  {conf:.0%}"
                (lw, lh), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                )
                cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
                cv2.putText(
                    frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
                )

                # ── Alert + log ──────────────────────────────────────────────
                if in_zone:
                    alert_this_frame = True
                    now = time.time()
                    if now - last_alert_time >= ALERT_COOLDOWN_SEC:
                        ts = datetime.now().strftime("%H:%M:%S")
                        print(
                            f"[ALERT] {ts}  — Person detected in hazard zone  "
                            f"(confidence: {conf:.0%})"
                        )
                        last_alert_time = now
                    if csv_writer:
                        log_event(csv_writer, conf, zone_breach=True)
                elif csv_writer:
                    log_event(csv_writer, conf, zone_breach=False)

            # ── FPS calculation ───────────────────────────────────────────────
            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.time()

            # ── HUD overlay ───────────────────────────────────────────────────
            draw_hud(frame, current_fps, total_detections, alert_this_frame, zone)

            # ── Display + save ────────────────────────────────────────────────
            cv2.imshow("Hazardous Zone Intrusion Detection", frame)
            if writer_video:
                writer_video.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] User quit.")
                break

    finally:
        cap.release()
        if writer_video:
            writer_video.release()
        if log_file:
            log_file.close()
        cv2.destroyAllWindows()
        print(f"\n[DONE] Total detections this session: {total_detections}")


if __name__ == "__main__":
    main()
