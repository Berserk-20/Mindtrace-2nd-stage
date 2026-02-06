import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
DATASET_DIR = "dataset1"
BATCH_SIZE = 64
EPOCHS = 60                 
IMG_SIZE = 96
NUM_CLASSES = 7
LR = 5e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------
# TRANSFORMS
# ----------------------------------------------------------
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(
        degrees=10,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------------------------------------
# MODEL
# ----------------------------------------------------------
class EmotionResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        self.model.fc = nn.Linear(
            self.model.fc.in_features, num_classes
        )

    def forward(self, x):
        return self.model(x)

# ----------------------------------------------------------
# TRAINING FUNCTION
# ----------------------------------------------------------
def train():
    print("Device:", DEVICE)

    # ---------------- DATASET ----------------
    train_dataset = datasets.ImageFolder(
        os.path.join(DATASET_DIR, "train"),
        transform=train_tfms
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(DATASET_DIR, "test"),
        transform=val_tfms
    )

    print("Classes:", train_dataset.classes)

    # ⚠️ Windows-safe
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,      
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,       
        pin_memory=True
    )

    # ---------------- CLASS WEIGHTS ----------------
    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(
        class_weights, dtype=torch.float
    ).to(DEVICE)

    # ---------------- MODEL ----------------
    model = EmotionResNet18(NUM_CLASSES).to(DEVICE)

    # Freeze early layers
    for param in model.model.parameters():
        param.requires_grad = False

    for param in model.model.layer3.parameters():
        param.requires_grad = True

    for param in model.model.layer4.parameters():
        param.requires_grad = True

    for param in model.model.fc.parameters():
        param.requires_grad = True

    # ---------------- LOSS / OPT ----------------
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-4
    )

    scaler = torch.amp.GradScaler("cuda")

    best_val_acc = 0.0

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0

        for images, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS}"
        ):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            with torch.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100

        # ---------------- VALIDATION ----------------
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = correct / total * 100

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_rafdb.pth")

    # ---------------- FINAL REPORT ----------------
    print("\nBest Validation Accuracy:", best_val_acc)
    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=train_dataset.classes
        )
    )

    # ---------------- TORCHSCRIPT ----------------
    model.load_state_dict(torch.load("best_model_rafdb.pth"))
    model.eval()

    example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    traced = torch.jit.trace(model, example)
    traced.save("emotion_resnet18_rafdb.pt")

    print("Training complete. TorchScript model saved.")

# ----------------------------------------------------------
# ENTRY POINT (MANDATORY ON WINDOWS)
# ----------------------------------------------------------
if __name__ == "__main__":
    train()
