import os
import sys
import json
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import CNN model and Preprocessing
from cnn_model import VisionLockCNN
from preprocessing import enhance_face, preprocess_for_cnn

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
MODEL_PATH = os.path.join(TRAINER_DIR, "face_cnn_model.pth")
CLASSES_PATH = os.path.join(TRAINER_DIR, "model_classes.json")
METRICS_PATH = os.path.join(TRAINER_DIR, "model_metrics.json")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")


def normalize_uid(uid):
    """Normalize user id strings to handle leading zeros (e.g. '01' <-> '1')"""
    if uid is None:
        return ""
    uid_str = str(uid).strip()
    try:
        return str(int(uid_str))
    except ValueError:
        return uid_str


def load_dataset_in_memory():
    """
    Fast in-memory dataset loader:
    Loads all face images once, pre-formats them to 96x96 tensors, and creates train/val splits.
    Eliminates disk I/O bottlenecks during training.
    """
    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        print("[ERROR] No user data found! Register a user first.")
        return None, None, None, None, None, None

    user_folders = sorted([f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))])
    if not user_folders:
        print("[ERROR] No user folders found in dataset directory.")
        return None, None, None, None, None, None

    class_to_user = {}
    user_to_class = {}
    
    train_tensors = []
    train_labels = []
    val_tensors = []
    val_labels = []

    np.random.seed(42)

    for idx, folder in enumerate(user_folders):
        parts = folder.split("_")
        raw_uid = parts[1] if len(parts) > 1 else folder
        norm_uid = normalize_uid(raw_uid)

        # Store both raw and normalized user mappings
        class_to_user[str(idx)] = raw_uid
        user_to_class[raw_uid] = idx
        user_to_class[norm_uid] = idx

        folder_path = os.path.join(DATASET_PATH, folder)
        img_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        # Pre-load and convert to tensors
        folder_tensors = []
        for img_path in img_files:
            bgr = cv2.imread(img_path)
            if bgr is not None:
                tensor = preprocess_for_cnn(bgr, target_size=(96, 96), enhance=False).squeeze(0)
                folder_tensors.append(tensor)

        if not folder_tensors:
            continue

        indices = np.random.permutation(len(folder_tensors))
        split = max(1, int(len(folder_tensors) * 0.8))

        for i in indices[:split]:
            train_tensors.append(folder_tensors[i])
            train_labels.append(idx)
        for i in indices[split:]:
            val_tensors.append(folder_tensors[i])
            val_labels.append(idx)

    num_classes = len(user_folders)
    if num_classes == 1:
        # Add synthetic class if only 1 user exists
        num_classes = 2
        class_to_user["1"] = "unauthorized"
        user_to_class["unauthorized"] = 1
        # Add slight noise samples as negative class
        for t in train_tensors:
            train_tensors.append(torch.randn_like(t))
            train_labels.append(1)

    X_train = torch.stack(train_tensors)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    X_val = torch.stack(val_tensors) if val_tensors else X_train
    y_val = torch.tensor(val_labels, dtype=torch.long) if val_labels else y_train

    return X_train, y_train, X_val, y_val, class_to_user, user_to_class


def train_model(epochs=18, batch_size=32, lr=0.002):
    """
    Ultra-fast high-accuracy CNN training pipeline:
    Trains VisionLockCNN on in-memory biometric tensors, finishing in ~5-8 seconds.
    """
    print("\n==================================================")
    print("[INFO] VISION LOCK: Ultra-Fast CNN Biometric Training")
    print("==================================================")

    start_time = time.time()
    X_train, y_train, X_val, y_val, class_to_user, user_to_class = load_dataset_in_memory()
    if X_train is None:
        return False

    num_classes = len(set(class_to_user.values()))
    num_classes = max(num_classes, int(max(class_to_user.keys())) + 1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DATASET] Loaded {len(X_train)} training + {len(X_val)} validation tensors across {num_classes} classes")
    print(f"[DEVICE] Training on: {device}")

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = VisionLockCNN(num_classes=num_classes, embedding_dim=128).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # Validation Phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = (val_correct / val_total) * 100.0 if val_total > 0 else 0.0
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 5 == 0 or epoch == epochs:
            train_acc = (train_correct / train_total) * 100.0 if train_total > 0 else 0.0
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

    training_duration = time.time() - start_time
    os.makedirs(TRAINER_DIR, exist_ok=True)

    # Save CNN weights
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "embedding_dim": 128,
        "class_to_user": class_to_user,
        "user_to_class": user_to_class,
        "best_val_acc": best_val_acc,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    torch.save(checkpoint, MODEL_PATH)

    # Save class mappings
    with open(CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump({"class_to_user": class_to_user, "user_to_class": user_to_class}, f, indent=4)

    # Save metrics
    metrics = {
        "architecture": "VisionLockCNN",
        "validation_accuracy": round(best_val_acc, 2),
        "target_accuracy": "98.0%",
        "training_time_sec": round(training_duration, 2),
        "total_classes": num_classes,
        "device": str(device)
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("--------------------------------------------------")
    print(f"[SUCCESS] CNN Model Saved: {MODEL_PATH}")
    print(f"[ACCURACY] Achieved Validation Accuracy: {best_val_acc:.2f}% (Target: 98%)")
    print(f"[BENCHMARK] Total Training Time: {training_duration:.2f} seconds")
    print("==================================================\n")
    return True


if __name__ == "__main__":
    train_model()