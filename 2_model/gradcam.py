"""
Grad-CAM visualisation for the wildfire CNN.

Usage:
    python 2_model/gradcam.py --image path/to/image.jpg
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ── Config ─────────────────────────────────────────────────────────────────────
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["fire", "no_fire"]
MODEL_PATH = "2_model/saved/cnn_wildfire.pth"

# ── Transforms ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model


def generate_gradcam(image_path: str, model: nn.Module) -> tuple[str, float, str]:
    """
    Returns
    -------
    predicted_class : str
    confidence      : float  (0–100)
    output_path     : str    (path to the saved heatmap overlay)
    """
    features: list = []
    grads: list    = []

    # Register hooks on last conv block
    handle_fwd = model.layer4.register_forward_hook(
        lambda _m, _i, o: features.append(o)
    )
    handle_bwd = model.layer4.register_full_backward_hook(
        lambda _m, _i, o: grads.append(o[0])
    )

    # Forward pass
    img_pil = Image.open(image_path).convert("RGB")
    x = transform(img_pil).unsqueeze(0).to(DEVICE)
    x.requires_grad_()

    output     = model(x)
    pred_class = int(output.argmax(1))
    confidence = float(torch.softmax(output, 1)[0, pred_class]) * 100

    # Backward pass
    model.zero_grad()
    output[0, pred_class].backward()

    # Build heatmap
    pooled_grads = grads[0].mean(dim=[0, 2, 3])
    activation   = features[0][0].detach()
    for i in range(activation.shape[0]):
        activation[i] *= pooled_grads[i]

    heatmap = activation.mean(0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap_color = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )

    # Overlay
    original = cv2.cvtColor(
        np.array(img_pil.resize((224, 224))), cv2.COLOR_RGB2BGR
    )
    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    out_path = image_path.rsplit(".", 1)[0] + "_gradcam.jpg"
    cv2.imwrite(out_path, overlay)

    # Clean up hooks
    handle_fwd.remove()
    handle_bwd.remove()

    print(f"Prediction : {CLASSES[pred_class]} ({confidence:.1f}%)")
    print(f"Grad-CAM   : {out_path}")
    return CLASSES[pred_class], confidence, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for a wildfire image.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    args = parser.parse_args()

    net = load_model()
    generate_gradcam(args.image, net)
