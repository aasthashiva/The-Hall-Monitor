# The-Hall-Monitor

# The Hall Monitor
### Multi-Modal Occupancy Detection System

> Three independent sensing channels. One fused estimate. Real-time library occupancy — without relying on any single point of failure.

**Status: Ongoing** — Software pipeline complete and validated. Physical hardware integration (Raspberry Pi + load cell sensors + WiFi sniffer) in progress.

---

## The Problem

Libraries, labs, and study halls have no reliable way to communicate real-time occupancy. Students walk in to find no seats. Staff have no utilisation data. Existing solutions rely on a single sensor type — cameras alone, or WiFi alone — which fail silently when conditions are poor.

---

## The Solution

The Hall Monitor fuses three independent sensing channels into a single weighted occupancy estimate:

| Channel | Technology | Role |
|---|---|---|
| Seat Sensors | Load cell + Linear Regression | Ground truth — physical occupancy per seat |
| Camera | YOLOv8 object detection | Visual corroboration — people in frame |
| WiFi Probes | MAC sniffing + Linear Regression | Supplementary — captures standing/common area occupancy |

No single channel is trusted blindly. If one fails, the other two carry the estimate.

---

## Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT SOURCES                            │
├───────────────────┬─────────────────────┬───────────────────────┤
│   Load Cell       │   Camera Feed        │   WiFi Interface     │
│   (IoT Sensors)   │   (Image/Stream)     │   (MAC Sniffing)     │
└────────┬──────────┴──────────┬──────────┴──────────┬────────────┘
         │                     │                      │
         ▼                     ▼                      ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│ Voltage Reading │  │ YOLOv8 Inference│  │ Feature Extraction  │
│ per seat        │  │ conf=0.3        │  │ - unique MACs       │
│                 │  │                 │  │ - avg RSSI          │
│                 │  │                 │  │ - probe rate        │
└────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘
         │                    │                       │
         ▼                    ▼                       ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│ Linear          │  │ Bounding Box    │  │ Linear Regression   │
│ Regression      │  │ Count           │  │ Model               │
│ weight > 20000g │  │                 │  │                     │
│ → seat occupied │  │                 │  │                     │
└────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘
         │                    │                       │
         ▼                    ▼                       ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│ sensor_count    │  │ yolo_count      │  │ wifi_count          │
│ (ground truth)  │  │ (corroboration) │  │ (supplementary)     │
│ weight = 0.55   │  │ weight = 0.25   │  │ weight = 0.20       │
└────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘
         │                    │                       │
         └────────────────────┼───────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Weighted Fusion      │
                 │                        │
                 │  fused = dot(weights,  │
                 │          [yolo,        │
                 │           sensor,      │
                 │           wifi])       │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Occupancy Report     │
                 │                        │
                 │  X people — XX%        │
                 │  OPEN / BUSY / FULL    │
                 └────────────────────────┘
```

---

## Model Performance

| Channel | Metric | Value |
|---|---|---|
| YOLOv8 Camera | mAP@50 | 0.353 |
| YOLOv8 Camera | Detection Accuracy (validation) | 91.8% (45/49 people) |
| WiFi Probe Model | R² | 0.91 |
| WiFi Probe Model | MAE | 1.49 people |
| Sensor Model | MAE | 143.94g |

---

## Sample Output

```
==================================================
   The Hall Monitor — Multi-Modal Occupancy Detection
==================================================

[1/3] Camera Channel (YOLOv8)...
      Image                  : sample_images/aastha1.jpeg
      People detected        : 45

[2/3] Seat Sensor Channel (Load Cell)...
      Seats scanned          : 50
      Occupied seats         : 39

[3/3] WiFi Probe Channel...
      Live reading           : MACs=90, RSSI=-50, probe_rate=4.2
      Estimated occupancy    : 46 people

==================================================
       The Hall Monitor — Library Occupancy Report
==================================================
  Seat Sensors  (ground truth) :  39 people
  Camera        (YOLOv8)       :  45 people
  WiFi Probes   (estimate)     :  46 people
--------------------------------------------------
  Fused Occupancy Estimate     :  42 people
  Capacity Usage               : 84%
  Status                       : BUSY    — Limited seats available
==================================================
```

---

## Project Structure

```
hall-monitor/
├── demo.py                  # Full fusion pipeline — run this
├── sample_images/
│   └── aastha1.jpeg         # Sample test image
├── weights/
│   └── best.pt              # YOLOv8 trained weights (see Setup)
├── models/
│   ├── sensor_model.pkl     # Trained load cell regression model
│   └── wifi_occupancy_model.pkl  # Trained WiFi regression model
└── README.md
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/aasthashiva/hall-monitor
cd hall-monitor
```

**2. Install dependencies**
```bash
pip install ultralytics scikit-learn pandas joblib numpy
```

**3. Download model files**

Weights and models are too large for GitHub. Download from [Google Drive →](#) and place as shown in the project structure above.

**4. Run**
```bash
python3 demo.py
```

---

## Tech Stack

- **YOLOv8** (Ultralytics) — object detection
- **Scikit-learn** — linear regression models
- **Python** — pipeline, fusion logic
- **Roboflow** — dataset sourcing and annotation
- **Google Colab** — training environment
- **Raspberry Pi** *(planned)* — edge deployment

---

## Deployment Architecture (Planned)

The system is designed for deployment on a Raspberry Pi unit per room:
- Camera module → YOLOv8 inference at the edge
- Load cell sensors per seat → voltage readings via GPIO
- WiFi interface in monitor mode → passive MAC sniffing
- Fused output → dashboard or display board

Target environments: libraries, lecture halls, restaurants, stadiums.

---

