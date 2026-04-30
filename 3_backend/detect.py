import torch
from torchvision import models, transforms
from torch import nn
from pathlib import Path
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# CNN model
cnn = models.resnet18(weights=None)
cnn.fc = nn.Linear(cnn.fc.in_features, 2)

CNN_PATH = Path("../2_model/saved/cnn_wildfire.pth")
if CNN_PATH.exists():
    cnn.load_state_dict(torch.load(CNN_PATH, map_location=device))
    print("✅ CNN model loaded")
else:
    print("⚠️ CNN model not found, using random weights")

cnn.eval().to(device)

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
    return {"label": label, "confidence": round(probs.max().item(), 3)}