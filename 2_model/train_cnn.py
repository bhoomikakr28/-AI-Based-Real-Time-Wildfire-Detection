import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Data transforms ────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Datasets & loaders ─────────────────────────────────────────────────────────
train_data   = datasets.ImageFolder("1_data/processed/train", transform=transform)
val_data     = datasets.ImageFolder("1_data/processed/val",   transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False, num_workers=0)

print(f"Classes: {train_data.classes}")   # ['fire', 'no_fire']
print(f"Train samples: {len(train_data)} | Val samples: {len(val_data)}")

# ── Model: ResNet-18 fine-tuned ─────────────────────────────────────────────────
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(512, 2)   # fire / no_fire
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ── Training loop ──────────────────────────────────────────────────────────────
best_acc = 0.0
EPOCHS   = 15

for epoch in range(1, EPOCHS + 1):
    # -- train --
    model.train()
    correct = total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        correct += (outputs.argmax(1) == labels).sum().item()
        total   += labels.size(0)

    train_acc = correct / total * 100

    # -- validate --
    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total   += labels.size(0)

    val_acc = val_correct / val_total * 100
    print(f"Epoch {epoch:02d}/{EPOCHS} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")
    scheduler.step()

    # -- save best --
    if val_acc > best_acc:
        best_acc = val_acc
        os.makedirs("2_model/saved", exist_ok=True)
        torch.save(model.state_dict(), "2_model/saved/cnn_wildfire.pth")
        print(f"  ✅  Best model saved ({val_acc:.1f}%)")

print(f"\n🎯  Best validation accuracy: {best_acc:.1f}%")
