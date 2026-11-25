"""
AI-POWERED DEEPFAKE VOICE DETECTION
Prediction & Inference Module (predict.py)

This module implements:
1. predict_audio(audio_path, model_path, config_path):
   - Validates audio file
   - Performs identical preprocessing as used during training
   - Loads PyTorch CNN model checkpoint (with dynamic cache reload on model update)
   - Executes torch.no_grad() inference
   - Computes Softmax probabilities for REAL and FAKE classes
   - Returns structured prediction dictionary with waveform and spectrogram data URLs
2. Safe handling for missing model checkpoint with clear informative messages.
"""

import os
import json
import torch
import numpy as np

from preprocess import (
    preprocess_audio_pipeline,
    generate_waveform_plot,
    generate_spectrogram_plot,
    AudioProcessingError,
    TARGET_SR
)
from model import DeepfakeVoiceCNN, get_device, count_parameters

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "models", "config.json")

# In-memory cached model instance for low-latency web serving
_CACHED_MODEL = None
_CACHED_CONFIG = None
_CACHED_MTIME = None


def get_cached_model(model_path=DEFAULT_MODEL_PATH, config_path=DEFAULT_CONFIG_PATH):
    """
    Retrieve or reload the PyTorch CNN model whenever checkpoint file is updated.
    """
    global _CACHED_MODEL, _CACHED_CONFIG, _CACHED_MTIME

    if not os.path.exists(model_path):
        return None, None

    current_mtime = os.path.getmtime(model_path)
    device = get_device()

    if _CACHED_MODEL is None or _CACHED_MTIME != current_mtime:
        try:
            model = DeepfakeVoiceCNN(num_classes=2)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            _CACHED_MODEL = model
            _CACHED_MTIME = current_mtime
        except Exception as e:
            raise RuntimeError(f"Error loading trained model checkpoint from {model_path}: {e}")

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                _CACHED_CONFIG = json.load(f)
        except Exception:
            _CACHED_CONFIG = {}

    return _CACHED_MODEL, _CACHED_CONFIG


def predict_audio(audio_path, model_path=DEFAULT_MODEL_PATH, config_path=DEFAULT_CONFIG_PATH):
    """
    Predict whether an audio file is REAL VOICE or FAKE VOICE.

    Args:
        audio_path (str): Path to audio file.
        model_path (str): Path to model weights (.pth).
        config_path (str): Path to training config (.json).

    Returns:
        dict: Structured prediction outcome.
    """
    # 1. Check if model exists
    if not os.path.exists(model_path):
        return {
            "status": "error",
            "error_type": "model_missing",
            "message": (
                "Trained CNN model checkpoint not found at 'models/best_model.pth'. "
                "Please train the model first by running 'python train.py' in the terminal."
            )
        }

    # 2. Check file existence
    if not os.path.exists(audio_path):
        return {
            "status": "error",
            "error_type": "file_not_found",
            "message": f"Audio file not found: {audio_path}"
        }

    # 3. Load model and config
    try:
        model, config = get_cached_model(model_path, config_path)
    except Exception as e:
        return {
            "status": "error",
            "error_type": "model_load_error",
            "message": str(e)
        }

    device = get_device()

    # 4. Audio Preprocessing
    try:
        tensor, raw_waveform, duration_sec, mel_db = preprocess_audio_pipeline(audio_path)
    except AudioProcessingError as ape:
        return {
            "status": "error",
            "error_type": "audio_invalid",
            "message": str(ape)
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": "preprocessing_error",
            "message": f"Preprocessing failed: {str(e)}"
        }

    # 5. Deep Learning CNN Inference
    try:
        tensor = tensor.to(device)
        with torch.no_grad():
            probabilities = model.predict_proba(tensor).cpu().numpy()[0]

        real_prob = float(probabilities[0])
        fake_prob = float(probabilities[1])

        # Classification verdict
        predicted_idx = int(np.argmax(probabilities))
        if predicted_idx == 0:
            predicted_class = "REAL VOICE"
            confidence = real_prob * 100.0
            badge_color = "success"
            verdict_description = "Genuine Human Speech detected. Acoustic harmonic features, formant dynamics, and pitch continuity are consistent with natural biological vocal tract resonance."
        else:
            predicted_class = "FAKE VOICE"
            confidence = fake_prob * 100.0
            badge_color = "danger"
            verdict_description = "Synthetic / AI-Generated Deepfake Voice detected. Unnatural spectral anomalies, vocoder artifacts, or synthetic phase discontinuities were identified."

        # 6. Generate Dark-Themed Visual Graphs
        waveform_uri = generate_waveform_plot(raw_waveform, sr=TARGET_SR)
        spectrogram_uri = generate_spectrogram_plot(mel_db)

        return {
            "status": "success",
            "predicted_class": predicted_class,
            "class_index": predicted_idx,
            "badge_color": badge_color,
            "verdict_description": verdict_description,
            "real_probability": round(real_prob, 4),
            "fake_probability": round(fake_prob, 4),
            "real_percentage": round(real_prob * 100.0, 2),
            "fake_percentage": round(fake_prob * 100.0, 2),
            "confidence": round(confidence, 2),
            "audio_duration": round(duration_sec, 2),
            "audio_filename": os.path.basename(audio_path),
            "waveform_plot": waveform_uri,
            "spectrogram_plot": spectrogram_uri,
            "model_info": {
                "architecture": "DeepfakeVoiceCNN (4-Block Conv2D + BatchNorm + Dropout)",
                "parameters": count_parameters(model),
                "device": str(device).upper(),
                "sample_rate": TARGET_SR,
                "input_shape": "(1, 128, 128)"
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": "inference_error",
            "message": f"Inference execution failed: {str(e)}"
        }
