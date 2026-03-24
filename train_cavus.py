#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
import random
import time
import copy
import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ===========================================
# 0. 基本設定
# ===========================================
RAW_DATA_DIR = "CVUS_RAW"          # PNG 已分類為 2CH / 4CH（但會有多層資料夾）
DATA_DIR = "CVUS"                  # 自動產生 train / val
TRAIN_RATIO = 0.8
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================
# 1. 建立資料夾結構
# ===========================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

for phase in ["train", "val"]:
    for cls in ["2CH", "4CH"]:
        ensure_dir(os.path.join(DATA_DIR, phase, cls))


# ===========================================
# 2. 遞迴掃描 RAW → 建立資料列表
# ===========================================
all_data = []

print("Scanning raw data (recursive)...")

for label in ["2CH", "4CH"]:
    class_dir = os.path.join(RAW_DATA_DIR, label)

    # ⭐ 遞迴找所有 PNG
    pattern = os.path.join(class_dir, "**", "*.png")
    png_files = glob.glob(pattern, recursive=True)

    for f in png_files:
        all_data.append((f, label))

print(f"Found {len(all_data)} PNG files.")
if len(all_data) == 0:
    raise RuntimeError("❌ 無 PNG 檔案！請確認 CVUS_RAW/2CH 與 4CH 資料結構是否正確。")


# ===========================================
# 3. 切 train / val
# ===========================================
random.shuffle(all_data)
train_count = int(len(all_data) * TRAIN_RATIO)

train_set = all_data[:train_count]
val_set = all_data[train_count:]


# ===========================================
# 4. 複製圖片到資料夾
# ===========================================
def copy_data(data_list, phase):
    total = len(data_list)
    for i, (src, label) in enumerate(data_list):
        dst = os.path.join(DATA_DIR, phase, label, f"{phase}_{i}.png")
        shutil.copy(src, dst)

        if i % 200 == 0:
            print(f"[{phase}] {i}/{total} done.")

print("Copying TRAIN data...")
copy_data(train_set, "train")

print("Copying VAL data...")
copy_data(val_set, "val")

print("Data ready!")


# ===========================================
# 5. 建立 DataLoader
# ===========================================
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
}

image_datasets = {
    x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
    for x in ["train", "val"]
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                  shuffle=(x == "train"), num_workers=2)
    for x in ["train", "val"]
}

dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
class_names = image_datasets["train"].classes
print("Classes:", class_names)
print("Train size:", dataset_sizes["train"])
print("Val size:", dataset_sizes["val"])


# ===========================================
# 6. DenseNet121 模型
# ===========================================
print("Building model...")
model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 2)  # 2CH / 4CH

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


# ===========================================
# 7. 訓練流程
# ===========================================
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\nBest Val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model


# ===========================================
# 8. 開始訓練
# ===========================================
model = train_model(model, criterion, optimizer, scheduler, EPOCHS)

torch.save(model.state_dict(), "densenet121_cvus_20251201.pth")
print("\nModel saved → densenet121_cvus_20251201.pth")







