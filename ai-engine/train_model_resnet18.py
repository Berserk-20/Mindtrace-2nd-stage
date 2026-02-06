
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 80
IMG_SIZE = 48
NUM_CLASSES = 7
EARLY_STOP_PATIENCE = 12

TRAIN_DIR = "dataset/train"
TEST_DIR  = "dataset/test"

print("Device:", DEVICE)

# ----------------------------------------------------------
# DATA AUGMENTATION
# ----------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(
        degrees=10,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------------------------------------
# DATASETS
# ----------------------------------------------------------
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset  = datasets.ImageFolder(TEST_DIR, transform=test_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True
)

class_names = train_dataset.classes
print("Classes:", class_names)

# ----------------------------------------------------------
# MODEL – RESNET18
# ----------------------------------------------------------
class EmotionResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.fc = nn.Linear(
            self.model.fc.in_features, num_classes
        )

    def forward(self, x):
        return self.model(x)

model = EmotionResNet18(NUM_CLASSES).to(DEVICE)

# ----------------------------------------------------------
# CLASS-WEIGHTED FOCAL LOSS
# ----------------------------------------------------------
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_dataset.targets),
    y=train_dataset.targets
)

class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.CrossEntropyLoss(
            weight=self.alpha, reduction="none"
        )(logits, targets)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

criterion = FocalLoss(alpha=class_weights)

# ----------------------------------------------------------
# OPTIMIZER, SCHEDULER, AMP
# ----------------------------------------------------------
optimizer = optim.AdamW(
    model.parameters(), lr=3e-4, weight_decay=1e-4
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=4
)

scaler = GradScaler()

# ----------------------------------------------------------
# TRAINING LOOP
# ----------------------------------------------------------
best_val_acc = 0
early_stop = 0

for epoch in range(EPOCHS):
    model.train()
    correct, total, train_loss = 0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # ---------------- VALIDATION ----------------
    model.eval()
    correct, total, val_loss = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    scheduler.step(val_acc)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        early_stop = 0
        torch.save(model.state_dict(), "best_emotion_model.pth")
    else:
        early_stop += 1
        if early_stop >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break

# ----------------------------------------------------------
# CONFUSION MATRIX + F1
# ----------------------------------------------------------
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ----------------------------------------------------------
# TORCHSCRIPT EXPORT (FPS BOOST)
# ----------------------------------------------------------
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save("emotion_resnet18_ts.pt")

print("\nTraining complete.")
print("Best Validation Accuracy:", best_val_acc)
print("TorchScript model saved for high FPS inference.")
