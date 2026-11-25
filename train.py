"""
AI-POWERED DEEPFAKE VOICE DETECTION
Training Pipeline (train.py)

Trains the 2D Convolutional Neural Network on authentic human speech
and state-of-the-art AI-generated deepfake voices (ElevenLabs, HiFi-GAN, Speechify, etc.).
Includes acoustic data augmentations (noise injection, gain perturbation) for real-world microphone robustness.
"""

import os
import sys
import glob
import json
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

from preprocess import (
    TARGET_SR, TARGET_DURATION, N_MELS, N_FFT, HOP_LENGTH, TARGET_FRAMES,
    load_audio, normalize_waveform, fix_audio_length, extract_mel_spectrogram,
    normalize_features
)
from model import DeepfakeVoiceCNN, get_device, save_checkpoint, count_parameters

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
REAL_DIR = os.path.join(DATASET_DIR, "real")
FAKE_DIR = os.path.join(DATASET_DIR, "fake")
MODELS_DIR = os.path.join(BASE_DIR, "models")
GRAPHS_DIR = os.path.join(BASE_DIR, "outputs", "graphs")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


class RobustAudioDataset(Dataset):
    """
    PyTorch Dataset with online acoustic augmentations for microphone and noise robustness.
    """
    def __init__(self, features, labels, is_training=False):
        self.features = features
        self.labels = labels
        self.is_training = is_training

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feat = self.features[idx].copy()

        # Online Augmentations during training
        if self.is_training:
            # 1. Subtle acoustic noise injection (simulating real microphone/room noise)
            if np.random.rand() < 0.4:
                noise = np.random.normal(0, 0.05, feat.shape).astype(np.float32)
                feat = feat + noise

            # 2. SpecAugment: Time masking (simulating pause or dropouts)
            if np.random.rand() < 0.3:
                t_mask_len = np.random.randint(2, 10)
                t_start = np.random.randint(0, TARGET_FRAMES - t_mask_len)
                feat[:, t_start:t_start + t_mask_len] = 0

            # 3. SpecAugment: Frequency masking (simulating mic frequency responses)
            if np.random.rand() < 0.3:
                f_mask_len = np.random.randint(2, 10)
                f_start = np.random.randint(0, N_MELS - f_mask_len)
                feat[f_start:f_start + f_mask_len, :] = 0

        x = torch.from_numpy(feat).unsqueeze(0).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def discover_audio_files():
    """Discover real human speech and synthetic deepfake speech files."""
    valid_exts = ('*.flac', '*.wav', '*.mp3', '*.ogg')
    real_files = []
    fake_files = []

    for ext in valid_exts:
        real_files.extend(glob.glob(os.path.join(REAL_DIR, ext)))
        real_files.extend(glob.glob(os.path.join(REAL_DIR, "**", ext), recursive=True))
        fake_files.extend(glob.glob(os.path.join(FAKE_DIR, ext)))
        fake_files.extend(glob.glob(os.path.join(FAKE_DIR, "**", ext), recursive=True))

    real_files = sorted(list(set(real_files)))
    fake_files = sorted(list(set(fake_files)))
    return real_files, fake_files


def extract_sample_feature(filepath):
    """Process an audio file into a standardized Log Mel-Spectrogram (128, 128)."""
    try:
        y, sr = load_audio(filepath, sr=TARGET_SR)
        y = normalize_waveform(y)
        y = fix_audio_length(y)
        mel_db = extract_mel_spectrogram(y, sr=TARGET_SR)
        norm_mel = normalize_features(mel_db)
        return norm_mel
    except Exception as e:
        print(f"Warning: skipping {filepath}: {e}")
        return None


def prepare_dataset_arrays():
    """Extract Mel-spectrogram features from all discovered dataset audio files."""
    real_files, fake_files = discover_audio_files()

    if len(real_files) == 0 or len(fake_files) == 0:
        raise ValueError(
            f"Insufficient dataset files found! Real: {len(real_files)}, Fake: {len(fake_files)}.\n"
            f"Please run 'python dataset/prepare_dataset.py' to download authentic samples."
        )

    print(f"Discovered authentic dataset: {len(real_files)} REAL human speech clips, {len(fake_files)} FAKE AI deepfake clips.")

    all_features = []
    all_labels = []

    print("Extracting features from REAL human speech...")
    for f in real_files:
        feat = extract_sample_feature(f)
        if feat is not None:
            all_features.append(feat)
            all_labels.append(0)  # REAL = 0

    print("Extracting features from FAKE AI voice clips...")
    for f in fake_files:
        feat = extract_sample_feature(f)
        if feat is not None:
            all_features.append(feat)
            all_labels.append(1)  # FAKE = 1

    features_arr = np.array(all_features, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64)

    return features_arr, labels_arr


def plot_and_save_curves(train_losses, val_losses, train_accs, val_accs):
    """Plot and save training and validation loss and accuracy graphs."""
    epochs_range = range(1, len(train_losses) + 1)
    plt.style.use('dark_background')

    # Loss Graph
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0b1120')
    ax.set_facecolor('#0b1120')
    ax.plot(epochs_range, train_losses, color='#38bdf8', linewidth=2.5, marker='o', label='Training Loss')
    ax.plot(epochs_range, val_losses, color='#f43f5e', linewidth=2.5, marker='s', label='Validation Loss')
    ax.set_title('CNN Deepfake Voice Detection - Training & Validation Loss', color='#f8fafc', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Epoch', color='#94a3b8', fontsize=10)
    ax.set_ylabel('Loss (Cross-Entropy)', color='#94a3b8', fontsize=10)
    ax.tick_params(colors='#94a3b8')
    for s in ax.spines.values():
        s.set_color('#334155')
    ax.grid(True, color='#1e293b', linestyle='--', alpha=0.6)
    ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "training_loss.png"), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    # Accuracy Graph
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0b1120')
    ax.set_facecolor('#0b1120')
    ax.plot(epochs_range, [a * 100 for a in train_accs], color='#38bdf8', linewidth=2.5, marker='o', label='Training Accuracy (%)')
    ax.plot(epochs_range, [a * 100 for a in val_accs], color='#10b981', linewidth=2.5, marker='^', label='Validation Accuracy (%)')
    ax.set_title('CNN Deepfake Voice Detection - Training & Validation Accuracy', color='#f8fafc', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Epoch', color='#94a3b8', fontsize=10)
    ax.set_ylabel('Accuracy (%)', color='#94a3b8', fontsize=10)
    ax.set_ylim([45, 105])
    ax.tick_params(colors='#94a3b8')
    for s in ax.spines.values():
        s.set_color('#334155')
    ax.grid(True, color='#1e293b', linestyle='--', alpha=0.6)
    ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0', loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "training_accuracy.png"), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_and_save_evaluation(y_true, y_pred, y_probs):
    """Plot and save confusion matrix and ROC curve on held-out test set."""
    plt.style.use('dark_background')

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='#0b1120')
    ax.set_facecolor('#0b1120')
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Purples', cbar=False,
        xticklabels=['REAL (0)', 'FAKE (1)'],
        yticklabels=['REAL (0)', 'FAKE (1)'],
        ax=ax, annot_kws={'size': 14, 'weight': 'bold', 'color': '#ffffff'}
    )
    ax.set_title('Test Set Confusion Matrix', color='#f8fafc', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Class', color='#94a3b8', fontsize=10)
    ax.set_ylabel('True Label', color='#94a3b8', fontsize=10)
    ax.tick_params(colors='#94a3b8')
    for s in ax.spines.values():
        s.set_color('#334155')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "confusion_matrix.png"), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#0b1120')
    ax.set_facecolor('#0b1120')
    ax.plot(fpr, tpr, color='#8b5cf6', linewidth=2.5, label=f'Deepfake CNN (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='#64748b', linestyle='--', linewidth=1.5, label='Random Chance (AUC = 0.50)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', color='#f8fafc', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('False Positive Rate', color='#94a3b8', fontsize=10)
    ax.set_ylabel('True Positive Rate', color='#94a3b8', fontsize=10)
    ax.tick_params(colors='#94a3b8')
    for s in ax.spines.values():
        s.set_color('#334155')
    ax.grid(True, color='#1e293b', linestyle='--', alpha=0.6)
    ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0', loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "roc_curve.png"), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    return auc_score, cm


def train_model(epochs=20, batch_size=16, learning_rate=0.0005):
    """
    Main training pipeline:
    - Trains PyTorch CNN on authentic speech datasets
    - Applies acoustic noise and SpecAugment
    - Checkpoints best model to models/best_model.pth
    - Evaluates on held-out test data
    """
    start_time = time.time()
    device = get_device()
    print(f"Initializing CNN training on device: {device}")

    # 1. Extract features and labels
    features, labels = prepare_dataset_arrays()
    n_samples = len(labels)
    print(f"Total authentic samples loaded: {n_samples}")

    # 2. Stratified train/val/test split (70% train, 15% validation, 15% test)
    idx_all = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(
        idx_all, test_size=0.30, random_state=42, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=42, stratify=labels[temp_idx]
    )

    print(f"Split sizes -> Train: {len(train_idx)}, Validation: {len(val_idx)}, Test: {len(test_idx)}")

    # 3. Create datasets with online augmentations
    train_dataset = RobustAudioDataset(features[train_idx], labels[train_idx], is_training=True)
    val_dataset = RobustAudioDataset(features[val_idx], labels[val_idx], is_training=False)
    test_dataset = RobustAudioDataset(features[test_idx], labels[test_idx], is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. Initialize Model, Loss, Optimizer
    model = DeepfakeVoiceCNN(num_classes=2, dropout_rate=0.35).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_path = os.path.join(MODELS_DIR, "best_model.pth")
    config_path = os.path.join(MODELS_DIR, "config.json")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print("\nStarting Training on Authentic Audio Data...")
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == y_batch).sum().item()
            total_train += x_batch.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                val_out = model(x_val)
                v_loss = criterion(val_out, y_val)

                val_loss += v_loss.item() * x_val.size(0)
                preds = torch.argmax(val_out, dim=1)
                correct_val += (preds == y_val).sum().item()
                total_val += x_val.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        scheduler.step(epoch_val_loss)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%"
        )

        # Save Best Model Checkpoint
        if epoch_val_acc > best_val_acc or (epoch_val_acc == best_val_acc and epoch_val_loss < best_val_loss):
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc

            config_data = {
                "target_sr": TARGET_SR,
                "target_duration": TARGET_DURATION,
                "target_samples": int(TARGET_SR * TARGET_DURATION),
                "n_mels": N_MELS,
                "n_fft": N_FFT,
                "hop_length": HOP_LENGTH,
                "target_frames": TARGET_FRAMES,
                "class_mapping": {"0": "REAL", "1": "FAKE"},
                "model_architecture": "DeepfakeVoiceCNN",
                "best_epoch": epoch,
                "best_val_loss": round(float(best_val_loss), 4),
                "best_val_acc": round(float(best_val_acc), 4),
                "dataset_source": "Hugging Face authentic human speech vs ElevenLabs/HiFi-GAN/Speechify deepfakes",
                "trained_at": datetime.datetime.now().isoformat()
            }
            save_checkpoint(model, path=best_model_path, config=config_data, config_path=config_path)

    print(f"\nBest validation checkpoint saved to {best_model_path} (Val Acc: {best_val_acc*100:.2f}%)")

    # Save Training Curves
    plot_and_save_curves(train_losses, val_losses, train_accs, val_accs)

    # Evaluate on Held-Out Test Set
    print("\nEvaluating best checkpoint on held-out test set...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    y_test_true = []
    y_test_pred = []
    y_test_probs = []

    with torch.no_grad():
        for x_t, y_t in test_loader:
            x_t = x_t.to(device)
            probs = model.predict_proba(x_t)
            preds = torch.argmax(probs, dim=1)

            y_test_true.extend(y_t.numpy())
            y_test_pred.extend(preds.cpu().numpy())
            y_test_probs.extend(probs[:, 1].cpu().numpy())

    y_test_true = np.array(y_test_true)
    y_test_pred = np.array(y_test_pred)
    y_test_probs = np.array(y_test_probs)

    test_acc = accuracy_score(y_test_true, y_test_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_true, y_test_pred, average='binary', zero_division=0)
    auc_score, cm = plot_and_save_evaluation(y_test_true, y_test_pred, y_test_probs)

    report_str = classification_report(y_test_true, y_test_pred, target_names=['REAL (0)', 'FAKE (1)'])
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION SUMMARY (Authentic Speech Benchmark)")
    print("=" * 60)
    print(f"Accuracy:  {test_acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {auc_score:.4f}")
    print("\nClassification Report:\n", report_str)
    print("Confusion Matrix:\n", cm)
    print("=" * 60)

    eval_results = {
        "test_accuracy": round(float(test_acc), 4),
        "test_precision": round(float(precision), 4),
        "test_recall": round(float(recall), 4),
        "test_f1": round(float(f1), 4),
        "roc_auc": round(float(auc_score), 4),
        "confusion_matrix": cm.tolist(),
        "total_test_samples": len(y_test_true),
        "elapsed_training_sec": round(time.time() - start_time, 2),
        "evaluation_timestamp": datetime.datetime.now().isoformat()
    }

    with open(os.path.join(OUTPUTS_DIR, "eval_results.json"), 'w') as f:
        json.dump(eval_results, f, indent=4)

    return eval_results


if __name__ == "__main__":
    train_model(epochs=20, batch_size=16, learning_rate=0.0005)
