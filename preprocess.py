"""
AI-POWERED DEEPFAKE VOICE DETECTION
Audio Preprocessing Pipeline (preprocess.py)

This module handles:
1. Audio loading across WAV, MP3, FLAC, and OGG formats using Librosa & SoundFile.
2. Mono conversion and high-fidelity resampling to 16,000 Hz.
3. Voice Activity Detection (VAD) / silence trimming to focus on active vocal harmonics.
4. Consistent amplitude normalization with headroom.
5. Fixed-duration temporal framing (4.0 seconds = 64,000 samples) using speech tiling.
6. Log Mel-Spectrogram feature extraction (128 mels x 128 frames) with 80 dB dynamic range.
7. Volume and microphone invariant per-instance feature standardization.
8. Conversion to PyTorch tensors (Batch, 1, 128, 128).
9. Visual plotting utilities for waveforms and spectrograms.
"""

import os
import io
import json
import base64
import numpy as np
import librosa
import soundfile as sf
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Default Audio Parameters
TARGET_SR = 16000          # 16 kHz sample rate standard for speech
TARGET_DURATION = 4.0      # 4 seconds fixed temporal window
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)  # 64,000 samples
N_MELS = 128               # 128 Mel frequency bins
N_FFT = 1024               # FFT window size
HOP_LENGTH = 500           # Hop length
TARGET_FRAMES = 128        # Fixed temporal frames (128 x 128 representation)

SUPPORTED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg'}


class AudioProcessingError(Exception):
    """Custom exception raised for unrecoverable audio loading and processing errors."""
    pass


def load_audio(file_path_or_bytes, sr=TARGET_SR):
    """
    Load an audio file or bytes buffer into a mono floating-point numpy array at TARGET_SR.
    Includes silence trimming to isolate genuine vocal speech activity.
    """
    try:
        y, orig_sr = librosa.load(file_path_or_bytes, sr=sr, mono=True)
    except Exception as e:
        raise AudioProcessingError(f"Failed to decode audio file: {str(e)}")

    if y is None or len(y) == 0:
        raise AudioProcessingError("The provided audio file is empty or contains no readable audio data.")

    # Check for completely silent/zeroed audio
    if np.all(y == 0) or np.isnan(y).any() or np.isinf(y).any():
        y = np.nan_to_num(y)
        if np.max(np.abs(y)) < 1e-5:
            raise AudioProcessingError("The audio file contains only silence (amplitude below threshold).")

    # Trim leading and trailing silence so the model analyzes actual speech phonemes
    try:
        y_trimmed, _ = librosa.effects.trim(y, top_db=28)
        # Only keep trimmed version if it retains at least 0.4 seconds of audio
        if len(y_trimmed) >= int(sr * 0.4):
            y = y_trimmed
    except Exception:
        pass

    return y.astype(np.float32), sr


def normalize_waveform(y):
    """
    Apply peak amplitude normalization to keep waveform in range [-1.0, 1.0].
    """
    max_val = np.max(np.abs(y))
    if max_val > 1e-5:
        y = y / max_val
    return y


def fix_audio_length(y, target_samples=TARGET_SAMPLES):
    """
    Ensure the audio waveform exactly matches the target sample count (4.0s = 64,000 samples).
    If short: tiles/repeats the speech to maintain continuous vocal formant structure.
    If long: center-truncates to capture the core active speech segment.
    """
    n_samples = len(y)

    if n_samples == target_samples:
        return y
    elif n_samples < target_samples:
        # Tile/repeat the speech to fill the duration seamlessly without artificial silence steps
        repeat_factor = int(np.ceil(target_samples / n_samples))
        tiled = np.tile(y, repeat_factor)
        return tiled[:target_samples]
    else:
        # Center truncate
        start = (n_samples - target_samples) // 2
        return y[start:start + target_samples]


def extract_mel_spectrogram(y, sr=TARGET_SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, target_frames=TARGET_FRAMES):
    """
    Extract a Log-Mel Spectrogram from the normalized waveform.
    Returns a 2D numpy array of shape (n_mels, target_frames) = (128, 128).
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=20,
        fmax=8000,
        power=2.0
    )

    # Convert to decibel scale with 80 dB dynamic range cap
    mel_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)

    # Ensure temporal dimension matches target_frames exactly
    cur_frames = mel_db.shape[1]
    if cur_frames < target_frames:
        pad_width = target_frames - cur_frames
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='edge')
    elif cur_frames > target_frames:
        mel_db = mel_db[:, :target_frames]

    return mel_db.astype(np.float32)


def normalize_features(mel_db, mean=None, std=None):
    """
    Standardize Mel spectrogram features.
    Uses per-spectrogram z-score normalization to ensure gain and microphone invariance.
    """
    m = np.mean(mel_db)
    s = np.std(mel_db)
    normalized = (mel_db - m) / (s + 1e-6)
    return normalized.astype(np.float32)


def preprocess_audio_pipeline(file_path_or_bytes, mean=None, std=None):
    """
    Complete end-to-end preprocessing pipeline:
    1. Load audio (resampled to 16 kHz mono with silence trimming)
    2. Normalize waveform amplitude
    3. Tile/Truncate to 4.0 seconds (64,000 samples)
    4. Compute Log Mel-Spectrogram (128 x 128)
    5. Standardize features (gain-invariant)
    6. Convert to PyTorch Tensor of shape (1, 1, 128, 128)

    Returns:
        tensor (torch.FloatTensor): Model input tensor of shape (1, 1, 128, 128).
        raw_waveform (np.ndarray): Audio waveform for visualization.
        duration_sec (float): Effective duration in seconds.
        mel_db_raw (np.ndarray): Log Mel-Spectrogram for visual plotting.
    """
    raw_y, orig_sr = load_audio(file_path_or_bytes, sr=TARGET_SR)
    duration_sec = float(len(raw_y)) / float(orig_sr)

    # Waveform normalization
    y_norm = normalize_waveform(raw_y)

    # Fixed length framing
    y_fixed = fix_audio_length(y_norm, target_samples=TARGET_SAMPLES)

    # Log Mel Spectrogram
    mel_db = extract_mel_spectrogram(y_fixed, sr=TARGET_SR)

    # Feature normalization
    norm_mel = normalize_features(mel_db, mean=mean, std=std)

    # Convert to PyTorch tensor: (batch_size=1, channels=1, n_mels=128, frames=128)
    tensor = torch.from_numpy(norm_mel).unsqueeze(0).unsqueeze(0).float()

    return tensor, raw_y, duration_sec, mel_db


def generate_waveform_plot(raw_waveform, sr=TARGET_SR):
    """
    Generates a dark-themed cyber waveform plot and returns it as a Base64-encoded PNG data URI.
    """
    fig, ax = plt.subplots(figsize=(8, 2.5), facecolor='#0b1120')
    ax.set_facecolor('#0b1120')

    time_axis = np.linspace(0, len(raw_waveform) / sr, num=len(raw_waveform))

    # Downsample for faster plotting if large
    if len(raw_waveform) > 10000:
        step = len(raw_waveform) // 5000
        time_axis = time_axis[::step]
        raw_waveform = raw_waveform[::step]

    ax.plot(time_axis, raw_waveform, color='#38bdf8', linewidth=0.8, alpha=0.9)
    ax.fill_between(time_axis, raw_waveform, color='#38bdf8', alpha=0.25)

    ax.set_title('Raw Audio Waveform Amplitude', color='#e2e8f0', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Time (seconds)', color='#94a3b8', fontsize=9)
    ax.set_ylabel('Amplitude', color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#334155')
    ax.grid(True, color='#1e293b', linestyle='--', alpha=0.6)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_b64}"


def generate_spectrogram_plot(mel_db):
    """
    Generates a dark-themed Mel Spectrogram plot and returns it as a Base64-encoded PNG data URI.
    """
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor='#0b1120')
    ax.set_facecolor('#0b1120')

    im = ax.imshow(mel_db, origin='lower', aspect='auto', cmap='magma')
    ax.set_title('Mel-Frequency Spectrogram (dB Scale)', color='#e2e8f0', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Time Frames (Temporal Bins)', color='#94a3b8', fontsize=9)
    ax.set_ylabel('Mel Frequency Bins (0 - 8000 Hz)', color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#334155')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='#94a3b8', labelsize=8)
    cbar.set_label('Energy (dB)', color='#94a3b8', fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_b64}"
