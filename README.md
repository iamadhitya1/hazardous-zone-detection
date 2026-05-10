<div align="center">

# Hazardous Zone Intrusion Detection System

**Real-time human detection in restricted areas using YOLOv8 and OpenCV.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge&logo=pytorch&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*B.Tech Computer Engineering — Computer Vision Course Project | IITRAM Ahmedabad | M. Adhitya*

</div>

---

## Overview

Industrial accidents in high-risk zones are frequently caused by unauthorized human entry. Traditional sensor-based monitoring is expensive and prone to false alarms. This project implements a **real-time, vision-based intrusion detection system** that uses the YOLOv8-Nano deep learning model to detect human presence in a live video feed — no expensive hardware required.

When a person enters the defined **hazard zone**, the system:
- Draws a red bounding box with confidence score on screen
- Prints a timestamped console alert
- Logs the event to a CSV file for audit trail

---

## Features

- **Zone-based detection** — Define a custom pixel-boundary hazard zone. Persons inside trigger alerts; persons outside are tracked safely (green box)
- **Alert cooldown** — Avoids spam alerts. A new console alert fires at most once every 3 seconds per event
- **Intrusion log** — Every detection is appended to `intrusion_log.csv` with timestamp, confidence, and whether it was a zone breach
- **Live HUD** — Real-time FPS counter, detection count, and timestamp overlaid on the video feed
- **Save output** — Optional flag to save the fully annotated video to disk
- **Flexible source** — Runs on a webcam, a pre-recorded video file, or any OpenCV-compatible stream
- **Configurable confidence** — Adjust the minimum detection threshold via CLI flag

---

## Demo

The system processes live video frame-by-frame. Each frame shows:

```
┌─────────────────────────────────────────┐
│ FPS: 28.4        [timestamp]            │
│ Detections: 7                           │
│                                         │
│   ┌─────────────────┐                   │
│   │ Person ⚠  87%   │  ← RED = in zone │
│   └─────────────────┘                   │
│                                         │
│ [HAZARD ZONE boundary visible]          │
│                                         │
│ ⚠ INTRUSION DETECTED — RESTRICTED AREA │
└─────────────────────────────────────────┘
```

---

## Project Structure

```
CV/
├── detect.py            ← Main inference script (run this)
├── best.pt              ← Trained YOLOv8-Nano model weights
├── requirements.txt     ← Python dependencies
├── intrusion_log.csv    ← Auto-generated detection log
└── README.md
```

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/iamadhitya1/hazardous-zone-detection.git
cd hazardous-zone-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

That's it. No other setup required.

---

## Usage

**Run on live webcam (default)**
```bash
python detect.py
```

**Run on a saved video file**
```bash
python detect.py --source path/to/video.mp4
```

**Define a custom hazard zone** (top-left x y, bottom-right x y)
```bash
python detect.py --zone 100 100 540 380
```

**Raise the confidence threshold**
```bash
python detect.py --confidence 0.70
```

**Save the annotated output video**
```bash
python detect.py --save
```

**All options combined**
```bash
python detect.py --source 0 --confidence 0.65 --zone 80 80 560 400 --save
```

Press `Q` to stop the detection at any time.

---

## CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `best.pt` | Path to YOLOv8 weights file |
| `--source` | `0` | Video source: `0` for webcam, or file path |
| `--confidence` | `0.50` | Minimum detection confidence (0–1) |
| `--zone X1 Y1 X2 Y2` | Full frame | Hazard zone boundary in pixels |
| `--save` | Off | Save annotated output as `output_annotated.mp4` |
| `--no-log` | Off | Disable CSV intrusion logging |

---

## How It Works

### Detection Pipeline

```
Live Frame → YOLOv8-Nano Inference → Filter: class=Person, conf≥threshold
    → For each detection:
         Is centre point inside hazard zone?
         YES → Red box + Console Alert + CSV log entry
         NO  → Green box + CSV log entry (no alert)
    → Draw HUD (FPS, count, timestamp, zone, alert banner)
    → Display / Save frame
```

### Model

The system uses **YOLOv8-Nano** — a single-stage object detector that predicts bounding boxes and class probabilities in one forward pass. Unlike two-stage detectors (R-CNN family), this enables real-time performance on standard hardware.

Transfer learning from the **COCO dataset** is used, with inference filtered to `class 0 (Person)` for focused, low-latency detection.

### Zone Intrusion Logic

The centre point `(cx, cy)` of each bounding box is checked against the hazard zone rectangle `(x1, y1, x2, y2)`. This is more reliable than checking box corners — a person partially entering from the edge still triggers when their centre crosses the boundary.

---

## Results

Tested on live webcam in a standard indoor environment:

| Metric | Result |
|--------|--------|
| Detection confidence | Typically > 80% |
| Real-time performance | ✅ on standard CPU/GPU |
| Partial occlusion handling | ✅ Robust |
| False positives (empty frame) | Minimal |

---

## Improvements Over Baseline

The original implementation was a single YOLO CLI command:
```bash
yolo predict model=best.pt source=0 show=True
```

This project adds:

- Custom hazard **zone boundary** with per-person inside/outside classification
- Colour-coded bounding boxes (red = danger, green = safe)
- **Alert cooldown** to prevent console flooding
- **CSV intrusion log** with timestamps for post-incident audit
- Live **FPS counter** and detection count overlay
- **Argparse CLI** for full runtime configurability (source, confidence, zone, save)
- Clean modular code structure with typed function signatures

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Detection model | YOLOv8-Nano (Ultralytics) |
| Framework | PyTorch |
| Video processing | OpenCV |
| Training dataset | COCO (transfer learning) |
| Training platform | Google Colab (NVIDIA T4 GPU) |
| Language | Python 3.10+ |

---

## Author

**M. Adhitya** — B.Tech Computer Engineering, IITRAM Ahmedabad (2023–27)
Enrollment: 231049012001

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/loveadhitya/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/iamadhitya1)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
