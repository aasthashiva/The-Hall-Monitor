"""
LibPulse — Multi-Modal Library Occupancy Detection System
demo.py

Runs the full fusion pipeline using actual trained models.

Requirements:
    pip install ultralytics scikit-learn pandas joblib numpy

Folder structure expected:
    your-repo/
    ├── demo.py
    ├── weights/
    │   └── best.pt                  ← copy from Drive: runs/library_crowd_low_compute_v2-3/weights/best.pt
    ├── models/
    │   ├── sensor_model.pkl         ← copy from Drive: library_crowd_project/sensor_model.pkl
    │   └── wifi_occupancy_model.pkl ← copy from Drive: library_crowd_project/wifi_occupancy_model.pkl
    └── sample_images/
        └── aastha1.jpeg             ← copy from Drive or /content/

Usage:
    python demo.py
    python demo.py --image sample_images/your_image.jpg
"""

import numpy as np
import pandas as pd
import joblib
import argparse
import random
import os
from ultralytics import YOLO

# ─── ARGUMENT PARSING ────────────────────────────────────────────
parser = argparse.ArgumentParser(description="LibPulse Demo")
parser.add_argument(
    "--image",
    type=str,
    default="sample_images/aastha1.jpeg",
    help="Path to test image for YOLO inference"
)
args = parser.parse_args()

# ─── PATH CONFIGURATION ──────────────────────────────────────────
YOLO_WEIGHTS  = "weights/best.pt"
SENSOR_MODEL  = "models/sensor_model.pkl"
WIFI_MODEL    = "models/wifi_occupancy_model.pkl"
TEST_IMAGE    = args.image
MAX_CAPACITY  = 50

# ─── SANITY CHECK ────────────────────────────────────────────────
missing = []
for path in [YOLO_WEIGHTS, SENSOR_MODEL, WIFI_MODEL, TEST_IMAGE]:
    if not os.path.exists(path):
        missing.append(path)

if missing:
    print("\n[ERROR] Missing files:")
    for f in missing:
        print(f"        {f}")
    print("\nSee folder structure in the docstring at the top of this file.")
    exit(1)

print("=" * 50)
print("   The Hall Monitor — Multi-Modal Occupancy Detection")
print("=" * 50)

# ─── CHANNEL 1: YOLO CAMERA ──────────────────────────────────────
print("\n[1/3] Camera Channel (YOLOv8)...")

yolo_model = YOLO(YOLO_WEIGHTS)
results    = yolo_model.predict(source=TEST_IMAGE, conf=0.3, save=False, verbose=False)
yolo_count = len(results[0].boxes)

print(f"      Image                  : {TEST_IMAGE}")
print(f"      People detected        : {yolo_count}")

# ─── CHANNEL 2: LOAD CELL SEAT SENSORS ───────────────────────────
print("\n[2/3] Seat Sensor Channel (Load Cell)...")

sensor_model = joblib.load(SENSOR_MODEL)

random.seed(42)  # fixed seed — reproducible output across runs
people_count = 0
for _ in range(MAX_CAPACITY):
    voltage = random.uniform(0.0, 3.3)
    weight  = sensor_model.predict(pd.DataFrame({'Voltage_V': [voltage]}))[0]
    if weight > 20000:
        people_count += 1

sensor_count = people_count
print(f"      Seats scanned          : {MAX_CAPACITY}")
print(f"      Occupied seats         : {sensor_count}")

# ─── CHANNEL 3: WIFI PROBE MODEL ─────────────────────────────────
print("\n[3/3] WiFi Probe Channel...")

wifi_model   = joblib.load(WIFI_MODEL)
live_reading = [[90, -50, 4.2]]    # [unique_macs, avg_rssi, probe_rate]
wifi_count   = round(wifi_model.predict(live_reading)[0])

print(f"      Live reading           : MACs=90, RSSI=-50, probe_rate=4.2")
print(f"      Estimated occupancy    : {wifi_count} people")

# ─── FUSION ──────────────────────────────────────────────────────
# Sensor = ground truth (highest weight) — directly measures physical occupancy
# Camera = visual corroboration
# WiFi   = supplementary (covers standing/common area people)

weights    = np.array([0.25, 0.55, 0.20])   # camera, sensor, wifi
raw_counts = np.array([yolo_count, sensor_count, wifi_count])
fused      = int(round(np.dot(weights, raw_counts)))

occupancy_pct = min(100, round((fused / MAX_CAPACITY) * 100))

if occupancy_pct >= 90:
    status = "FULL    — Library is at capacity"
elif occupancy_pct >= 60:
    status = "BUSY    — Limited seats available"
else:
    status = "OPEN    — Plenty of seats available"

print("\n" + "=" * 50)
print("       The Hall Monitor — Library Occupancy Report")
print("=" * 50)
print(f"  Seat Sensors  (ground truth) : {sensor_count:>3} people")
print(f"  Camera        (YOLOv8)       : {yolo_count:>3} people")
print(f"  WiFi Probes   (estimate)     : {wifi_count:>3} people")
print("-" * 50)
print(f"  Fused Occupancy Estimate     : {fused:>3} people")
print(f"  Capacity Usage               : {occupancy_pct}%")
print(f"  Status                       : {status}")
print("=" * 50)
