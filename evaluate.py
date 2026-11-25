"""
AI-POWERED DEEPFAKE VOICE DETECTION
Standalone Model Evaluation Module (evaluate.py)

This module evaluates the trained CNN model (models/best_model.pth)
against a specified test set or the held-out audio files in dataset/real and dataset/fake.
It outputs:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrix and ROC curve plots in outputs/graphs/
- Structured JSON evaluation metrics in outputs/eval_results.json
"""

import os
import sys
import glob
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

from preprocess import (
    TARGET_SR, load_audio, normalize_waveform, fix_audio_length,
    extract_mel_spectrogram, normalize_features
)
from model import DeepfakeVoiceCNN, get_device, load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
GRAPHS_DIR = os.path.join(BASE_DIR, "outputs", "graphs")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


def evaluate_dataset(model_path="models/best_model.pth", config_path="models/config.json"):
    """
    Load trained model and evaluate on the available dataset files.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at '{model_path}'. Please run 'python train.py' first.")

    # Load configuration
    mean = None
    std = None
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
            mean = cfg.get("feature_mean")
            std = cfg.get("feature_std")

    device = get_device()
    model = load_model(model_path, device=device)

    real_dir = os.path.join(BASE_DIR, "dataset", "real")
    fake_dir = os.path.join(BASE_DIR, "dataset", "fake")

    valid_exts = ('*.wav', '*.mp3', '*.flac', '*.ogg')
    real_files = []
    fake_files = []
    for ext in valid_exts:
        real_files.extend(glob.glob(os.path.join(real_dir, ext)))
        fake_files.extend(glob.glob(os.path.join(fake_dir, ext)))

    if not real_files or not fake_files:
        raise ValueError("Dataset files missing in 'dataset/real/' or 'dataset/fake/'.")

    all_files = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]

    y_true = []
    y_pred = []
    y_probs = []

    print(f"Evaluating {len(all_files)} audio samples using model {model_path} on {device}...")

    with torch.no_grad():
        for filepath, label in all_files:
            try:
                y, sr = load_audio(filepath, sr=TARGET_SR)
                y = normalize_waveform(y)
                y = fix_audio_length(y)
                mel_db = extract_mel_spectrogram(y, sr=TARGET_SR)
                norm_mel = normalize_features(mel_db, mean=mean, std=std)

                tensor = torch.from_numpy(norm_mel).unsqueeze(0).unsqueeze(0).float().to(device)
                probs = model.predict_proba(tensor).cpu().numpy()[0]
                pred = int(np.argmax(probs))

                y_true.append(label)
                y_pred.append(pred)
                y_probs.append(float(probs[1]))  # Probability of FAKE
            except Exception as e:
                print(f"Skipping {filepath}: {e}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    auc = roc_auc_score(y_true, y_probs)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 60)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 60)
    print(f"Total Evaluated Samples: {len(y_true)}")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=['REAL', 'FAKE']))
    print("=" * 60)

    # Save outputs
    eval_dict = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "total_samples": len(y_true),
        "confusion_matrix": cm.tolist()
    }
    with open(os.path.join(OUTPUTS_DIR, "eval_results.json"), 'w') as f:
        json.dump(eval_dict, f, indent=4)

    return eval_dict


if __name__ == "__main__":
    evaluate_dataset()
