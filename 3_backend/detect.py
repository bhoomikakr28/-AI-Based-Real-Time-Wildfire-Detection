import torch
from torchvision import models, transforms
from torch import nn
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── CNN MODEL ──────────────────────────────────────────────────
cnn = models.resnet18(weights=None)
cnn.fc = nn.Linear(cnn.fc.in_features, 2)
CNN_PATH = Path("../2_model/saved/cnn_wildfire.pth")
if CNN_PATH.exists():
    cnn.load_state_dict(torch.load(CNN_PATH, map_location=device))
    print("✅ CNN model loaded")
else:
    print("⚠️ CNN model not found")
cnn.eval().to(device)

# ── YOLO MODEL ─────────────────────────────────────────────────
# Try custom weights first, fall back to pretrained YOLOv8
YOLO_PATH = Path("../2_model/saved/wildfire_yolo_weights.pt")
yolo = None
yolo_mode = "none"

if YOLO_PATH.exists():
    yolo = YOLO(str(YOLO_PATH))
    yolo_mode = "custom"
    print("✅ Custom YOLO weights loaded")
else:
    try:
        # Use pretrained YOLOv8 — it detects fire AND smoke natively
        yolo = YOLO("yolov8n.pt")
        yolo_mode = "pretrained"
        print("✅ Pretrained YOLOv8 loaded (fire + smoke detection)")
    except Exception as e:
        print(f"⚠️ YOLO not available: {e}")

# Classes that indicate fire/smoke in pretrained YOLO
# Pretrained YOLO classes that relate to fire
FIRE_SMOKE_CLASSES = {
    "fire", "smoke", "flame", "wildfire",
    "fire hydrant",  # sometimes misidentified — filter these
}

# Classes to EXCLUDE (common false positives from pretrained model)
EXCLUDE_CLASSES = {"fire hydrant", "person", "car", "truck", "bus"}

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def run_yolo_detection(pil_img, cnn_label):
    """Run YOLO and return fire/smoke boxes."""
    boxes = []
    if yolo is None:
        return boxes

    if yolo_mode == "custom":
        # Custom model — try multiple thresholds
        for threshold in [0.1, 0.05, 0.01]:
            results = yolo(pil_img, verbose=False, conf=threshold, iou=0.3)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    b_conf = float(box.conf[0])
                    cls    = int(box.cls[0])
                    name   = r.names[cls].lower()
                    boxes.append({
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2),
                        "confidence": round(b_conf, 3),
                        "label": "fire" if "fire" in name or "flame" in name else "smoke"
                    })
            if boxes:
                break

    elif yolo_mode == "pretrained":
        # Pretrained model — run on FIRE images and look for hot/orange regions
        # Strategy: run YOLO with ALL classes, then also run color-based fire detection
        results = yolo(pil_img, verbose=False, conf=0.15, iou=0.4)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                b_conf = float(box.conf[0])
                cls    = int(box.cls[0])
                name   = r.names[cls].lower()

                # Only keep fire/smoke related detections
                if any(kw in name for kw in ["fire", "smoke", "flame"]) and name not in EXCLUDE_CLASSES:
                    label = "fire" if any(kw in name for kw in ["fire", "flame"]) else "smoke"
                    boxes.append({
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2),
                        "confidence": round(b_conf, 3),
                        "label": label
                    })

        # If CNN says fire but YOLO found nothing → use color-based fire detection
        if not boxes and cnn_label == "fire":
            boxes = detect_fire_by_color(pil_img)

    return boxes


def detect_fire_by_color(pil_img):
    """
    Detect fire regions using HSV color analysis.
    Fire pixels are orange/red/yellow — reliable heuristic.
    """
    boxes = []
    try:
        # Convert PIL to OpenCV
        img_cv = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        hsv    = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

        # Fire color ranges in HSV
        # Orange-red fire
        lower1 = np.array([0,   120, 120])
        upper1 = np.array([20,  255, 255])
        # Yellow-orange fire
        lower2 = np.array([20,  120, 120])
        upper2 = np.array([35,  255, 255])
        # Bright red fire
        lower3 = np.array([160, 120, 120])
        upper3 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask3 = cv2.inRange(hsv, lower3, upper3)
        mask  = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = img_cv.shape[:2]
        min_area = (w * h) * 0.005  # at least 0.5% of image

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Expand box slightly
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)
            conf = min(0.95, area / (w * h) * 10 + 0.5)
            boxes.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": round(conf, 3),
                "label": "fire"
            })

        # Sort by area descending, keep top 5
        boxes.sort(key=lambda b: (b["x2"]-b["x1"]) * (b["y2"]-b["y1"]), reverse=True)
        boxes = boxes[:5]

    except Exception as e:
        print(f"Color detection error: {e}")

    return boxes


def get_cnn_attention_box(pil_img, confidence):
    """
    Divide image into 3×3 grid, score each cell with CNN.
    Returns box around highest-scoring fire region.
    """
    try:
        w, h = pil_img.size
        cell_w = w // 3
        cell_h = h // 3
        scored = []

        for row in range(3):
            for col in range(3):
                x1   = col * cell_w
                y1   = row * cell_h
                x2   = x1 + cell_w
                y2   = y1 + cell_h
                crop = pil_img.crop((x1, y1, x2, y2))
                t    = tfm(crop.convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    score = torch.softmax(cnn(t), 1)[0][0].item()  # fire probability
                scored.append((score, x1, y1, x2, y2))

        # Sort by fire score, keep top 2 cells
        scored.sort(reverse=True)
        result_boxes = []
        for score, x1, y1, x2, y2 in scored[:2]:
            if score > 0.55:
                result_boxes.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": round(score, 3),
                    "label": "fire"
                })
        return result_boxes

    except Exception as e:
        print(f"CNN attention error: {e}")
        return []


def predict_image(pil_img):
    """
    Main prediction function.
    1. CNN classifies full image → fire / no_fire
    2. YOLO detects smoke bounding boxes
    3. For fire images: ALWAYS run color-based fire detection
       and COMBINE with YOLO smoke boxes
    4. If still no boxes → CNN attention grid
    """
    rgb = pil_img.convert("RGB")

    # ── Step 1: CNN Classification ──────────────────────────
    t = tfm(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(cnn(t), 1)[0]

    label = ["fire", "no_fire"][probs.argmax()]
    conf  = round(probs.max().item(), 3)

    # ── Step 2: YOLO Bounding Boxes (smoke etc.) ────────────
    yolo_boxes = run_yolo_detection(rgb, label)

    # ── Step 3: Color-based FIRE detection (always run on fire images) ──
    fire_boxes = []
    if label == "fire":
        fire_boxes = detect_fire_by_color(rgb)

    # ── Step 4: Combine — fire boxes first, then smoke boxes ──
    boxes = fire_boxes + yolo_boxes

    # Remove duplicate/overlapping boxes (keep highest conf)
    boxes = deduplicate_boxes(boxes)

    # ── Step 5: Final fallback — CNN attention grid ──────────
    if not boxes and label == "fire":
        boxes = get_cnn_attention_box(rgb, conf)

    return {
        "label":      label,
        "confidence": conf,
        "boxes":      boxes
    }


def deduplicate_boxes(boxes, iou_threshold=0.4):
    """Remove overlapping boxes keeping the highest confidence one."""
    if len(boxes) <= 1:
        return boxes

    # Sort by confidence descending
    boxes = sorted(boxes, key=lambda b: b["confidence"], reverse=True)
    kept  = []

    for box in boxes:
        overlap = False
        for kept_box in kept:
            if compute_iou(box, kept_box) > iou_threshold:
                overlap = True
                break
        if not overlap:
            kept.append(box)

    return kept[:8]  # max 8 boxes


def compute_iou(a, b):
    """Compute Intersection over Union of two boxes."""
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (area_a + area_b - inter)