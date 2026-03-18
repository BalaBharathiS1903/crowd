# Crowd Density Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object_Detection-orange?style=flat-square)
![React](https://img.shields.io/badge/React-Frontend-61DAFB?style=flat-square&logo=react)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Vercel-black?style=flat-square&logo=vercel)](https://crowd-flax.vercel.app)

> Real-time crowd density estimation using YOLOv8 object detection — detects and counts people in images or video streams and classifies the crowd level as Low, Medium, or High.

🌐 **Live Demo**: https://crowd-flax.vercel.app

---

## What It Does

This system uses the **YOLOv8s** pre-trained model to detect individual people in a given image or video frame. Based on the detected count, it classifies the crowd density into three levels:

| Density Level | Person Count | Use Case |
|--------------|-------------|----------|
| 🟢 Low       | 0 – 10      | Safe gatherings, open spaces |
| 🟡 Medium    | 11 – 30     | Moderate crowds, events |
| 🔴 High      | 31+         | Dense crowds, safety alerts |

---

## Tech Stack

| Layer      | Technology                        |
|------------|-----------------------------------|
| Detection  | YOLOv8s (Ultralytics)             |
| Language   | Python 3.10                       |
| Frontend   | React.js                          |
| Deployment | Vercel                            |
| Model File | yolov8s.pt (pre-trained COCO)     |

---

## Project Structure

```
crowd/
├── crowd/              # Core Python detection module
├── yolov8s.pt          # Pre-trained YOLOv8s model weights
├── capture.jpg         # Sample test image
├── package-lock.json   # Frontend dependencies
├── .gitignore
└── README.md
```

---

## How It Works

```
Input Image / Video Frame
        │
        ▼
YOLOv8s Object Detection
  └── Detects all "person" class bounding boxes
        │
        ▼
Person Count Aggregation
  └── Counts total detected persons
        │
        ▼
Density Classification
  └── Low / Medium / High
        │
        ▼
Output: Annotated image + density label
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/BalaBharathiS1903/crowd.git
cd crowd
pip install ultralytics opencv-python
```

### Run Detection

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")
results = model("capture.jpg")

persons = sum(1 for r in results for b in r.boxes if int(b.cls) == 0)

if persons <= 10:
    density = "🟢 Low"
elif persons <= 30:
    density = "🟡 Medium"
else:
    density = "🔴 High"

print(f"Detected: {persons} people | Density: {density}")
```

---

## Live Demo

The project includes a React frontend deployed on Vercel:

🔗 **https://crowd-flax.vercel.app**

---

## Sample Output

| Input | Detected Persons | Density Level |
|-------|-----------------|--------------|
| capture.jpg (sample) | See live demo | Classified automatically |

---

## Use Cases

- 🏟️ Event crowd monitoring
- 🚉 Public transport safety
- 🏫 School/campus safety systems
- 🏙️ Smart city surveillance
- 🚨 Emergency crowd management

---

## Author

**Bala Bharathi S**
- 💼 LinkedIn: https://www.linkedin.com/in/balabharathi617
- 🐙 GitHub: https://github.com/BalaBharathiS1903
- 🌐 Portfolio: https://balabharathis1903.github.io/portfolioBalaBharathiS/

---

## License

Educational project © 2025 Bala Bharathi S
