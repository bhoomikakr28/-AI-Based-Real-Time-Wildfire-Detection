"""
Fine-tune YOLOv8 on a fire/smoke dataset.

Before running:
  1. Download a YOLO-format fire dataset from Roboflow:
     https://universe.roboflow.com/school-tvtyg/fire-and-smoke-detection
  2. Extract it so the folder layout matches fire_dataset.yaml
  3. Run:  python 2_model/train_yolo.py
"""

from ultralytics import YOLO

# ── Model ──────────────────────────────────────────────────────────────────────
# yolov8n = nano (fastest, great for drone edge devices)
# alternatives: yolov8s, yolov8m for more accuracy
model = YOLO("yolov8n.pt")

# ── Training ───────────────────────────────────────────────────────────────────
results = model.train(
    data="fire_dataset.yaml",        # dataset config (see below)
    epochs=50,
    imgsz=640,
    batch=16,
    name="wildfire_yolo",
    project="2_model/saved",
    patience=10,                     # early stopping
    device=0,                        # GPU 0; use 'cpu' if no GPU
    verbose=True,
)

print("\n✅  YOLO training complete!")
print(f"Best weights: 2_model/saved/wildfire_yolo/weights/best.pt")
