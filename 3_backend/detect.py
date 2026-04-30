import torch
from torchvision import models, transforms
from torch import nn
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"

cnn = models.resnet18(weights=None)
cnn.fc = nn.Linear(cnn.fc.in_features, 2)

CNN_PATH = Path("../2_model/saved/cnn_wildfire.pth")
if CNN_PATH.exists():
    cnn.load_state_dict(torch.load(CNN_PATH, map_location=device))
    print("✅ CNN model loaded")
else:
    print("⚠️ CNN model not found")

cnn.eval().to(device)

YOLO_PATH = Path("../2_model/saved/wildfire_yolo_weights.pt")
yolo = None
if YOLO_PATH.exists():
    yolo = YOLO(str(YOLO_PATH))
    print("✅ YOLO model loaded")
else:
    print("⚠️ YOLO weights not found, using CNN only")

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict_image(pil_img):
    t = tfm(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(cnn(t), 1)[0]
    label = ["fire", "no_fire"][probs.argmax()]
    boxes = []
    if yolo:
        results = yolo(pil_img, verbose=False)
        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                name = r.names[cls]
                boxes.append({
                    "x1": round(x1), "y1": round(y1),
                    "x2": round(x2), "y2": round(y2),
                    "confidence": round(conf, 3),
                    "label": name
                })
    return {
        "label": label,
        "confidence": round(probs.max().item(), 3),
        "boxes": boxes
    }