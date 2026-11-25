# AI-POWERED DEEPFAKE VOICE DETECTION (AIDL)

[![Python 3.11](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Librosa](https://img.shields.io/badge/Librosa-0.10%2B-orange.svg)](https://librosa.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-green.svg)](https://palletsprojects.com/p/flask/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An academic Artificial Intelligence and Deep Learning (AIDL) engineering platform for detecting synthesized, AI-cloned, vocoded, and manipulated human speech recordings.

The system features **TWO unified detection modalities** powered by the **SAME trained 2D Convolutional Neural Network (CNN)** and identical audio signal preprocessing pipelines:
1. **Upload Audio Detection:** Supports WAV, MP3, FLAC, and OGG formats with interactive drag-and-drop file inspection and pre-loaded benchmark demonstration samples.
2. **Live Microphone Voice Detection:** Real-time in-browser speech recording utilizing the **Web Audio API** and client-side PCM 16-bit WAV encoding with an animated oscilloscope visualizer.

---

## Architecture & Detection Pipeline

```
+-------------------------------------------------------------------------+
|                              AUDIO INPUT                                |
|          [ Pre-recorded Audio File ]  OR  [ Live Microphone Stream ]     |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  ACOUSTIC SIGNAL PREPROCESSING (preprocess.py)          |
|  * Resample to 16,000 Hz Mono                                           |
|  * Peak Amplitude Normalization                                         |
|  * Fixed Temporal Framing (4.0s = 64,000 samples)                       |
|  * 128 Mel-Frequency Filterbank Log Power-to-dB (128 x 128 Matrix)      |
|  * Z-Score Normalization using Training Statistics                      |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                   PYTORCH 2D-CNN ARCHITECTURE (model.py)                |
|  Input: (Batch, 1, 128, 128)                                            |
|  * Block 1: Conv2d(1->32)  -> BatchNorm2d -> ReLU -> MaxPool2d -> Drop  |
|  * Block 2: Conv2d(32->64) -> BatchNorm2d -> ReLU -> MaxPool2d -> Drop  |
|  * Block 3: Conv2d(64->128)-> BatchNorm2d -> ReLU -> MaxPool2d -> Drop  |
|  * Block 4: Conv2d(128->256)->BatchNorm2d -> ReLU -> AdaptiveAvgPool2d  |
|  * Dense:   Linear(4096->256) -> BatchNorm1d -> ReLU -> Dropout(0.4)    |
|             Linear(256->64)   -> ReLU -> Dropout(0.25) -> Linear(64->2) |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                          SOFTMAX INFERENCE & VERDICT                     |
|  * Class 0: REAL VOICE (Genuine Human Speech)                           |
|  * Class 1: FAKE VOICE (Synthesized / Neural Vocoder Spoof)             |
|  * Probabilities, Confidence Metric, Mel-Spectrogram, Waveform Plot     |
|  * Downloadable Forensic Audit Report (ReportLab PDF)                   |
+-------------------------------------------------------------------------+
```

---

## Directory Structure

```
deepfake_voice_detection/
|-- app.py                     # Flask web server, API endpoints, route controllers
|-- model.py                   # PyTorch CNN neural network architecture
|-- preprocess.py              # Audio loading, resampling, Log Mel-spectrogram pipeline
|-- train.py                   # Model training, grouped validation, checkpointing
|-- predict.py                 # Unified inference engine for files and live audio
|-- evaluate.py                # Standalone test-set evaluation and metrics calculation
|-- report_generator.py        # ReportLab PDF forensic audit report generator
|-- requirements.txt           # Python dependencies
|-- README.md                  # Complete documentation
|
|-- templates/                 # Glassmorphic cyber UI templates
|   |-- base.html              # Core navigation layout, header, footer, loading overlay
|   |-- index.html             # Dashboard with project statistics and mode selection
|   |-- upload.html            # File upload dropzone, audio player, demo benchmark loader
|   |-- live.html              # Real-time microphone recording and oscilloscope visualizer
|   |-- result.html            # Forensic audit result, confidence bars, spectrogram plots
|   |-- about.html             # Academic AIDL paper documentation, methodology, graphs
|
|-- static/
|   |-- css/
|   |   |-- style.css          # Dark navy cybersecurity glassmorphism stylesheet
|   |-- js/
|   |   |-- script.js          # File dropzone and upload AJAX controller
|   |   |-- recorder.js        # Web Audio API microphone recorder and WAV encoder
|   |-- images/
|
|-- models/
|   |-- best_model.pth         # Trained PyTorch CNN checkpoint weights
|   |-- config.json            # Model parameters and feature standardization stats
|
|-- dataset/
|   |-- prepare_dataset.py     # Dataset generator and academic acquisition tool
|   |-- real/                  # Genuine human speech recordings
|   |-- fake/                  # Synthesized deepfake voice recordings
|
|-- outputs/
|   |-- eval_results.json      # Test metrics (accuracy, precision, recall, F1, AUC)
|   |-- graphs/                # Generated training loss/accuracy, ROC, confusion matrix
|   |-- reports/               # Generated forensic PDF audit reports
|
|-- uploads/                   # Temporary directory and preloaded benchmark samples
|   |-- demo_real_voice.wav    # Preloaded genuine speech test sample
|   |-- demo_deepfake_voice.wav# Preloaded synthesized voice test sample
|
`-- tests/
    |-- test_preprocess.py     # Preprocessing and audio validation unit tests
    |-- test_prediction.py     # CNN inference and PDF generation unit tests
    `-- test_flask_routes.py   # Flask endpoint HTTP integration tests
```

---

## Getting Started

### 1. Prerequisites
- Python 3.11+
- Windows PowerShell, macOS Terminal, or Linux Bash
- A microphone (optional, for live recording test)

### 2. Environment Setup (Windows PowerShell)

```powershell
# Open terminal in project directory
cd c:\Users\hp\Downloads\deep_fake_voice

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install required dependencies
python -m pip install -r requirements.txt
```

---

## Dataset Acquisition & Training

### 1. Academic Dataset Support
The platform is designed to ingest standard academic speech anti-spoofing benchmarks:
- **ASVspoof 2019 Logical Access (LA):** Place genuine files with `bonafide` labels into `dataset/real/` and vocoded `spoof` files into `dataset/fake/`.
- **WaveFake:** Place original LJSpeech recordings in `dataset/real/` and vocoded clips (MelGAN, HiFi-GAN, WaveGlow) into `dataset/fake/`.

### 2. Prepare Balanced Dataset
For instant training and testing without downloading 30 GB datasets:
```powershell
python dataset/prepare_dataset.py
```
This generates genuine acoustic speech samples and synthesized deepfake samples directly into `dataset/real/` and `dataset/fake/`, along with benchmark samples in `uploads/`.

### 3. Train the CNN Model
```powershell
python train.py
```
**Training Pipeline Outputs:**
- Best checkpoint saved to `models/best_model.pth`.
- Feature standardization stats saved to `models/config.json`.
- Training curves saved to `outputs/graphs/training_loss.png` and `training_accuracy.png`.
- Held-out evaluation plots saved to `outputs/graphs/confusion_matrix.png` and `roc_curve.png`.
- Summary metrics saved to `outputs/eval_results.json`.

---

## Running the Web Application

Start the Flask server:
```powershell
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### Usage Guide

1. **Dashboard (`/`):**
   - Review project metrics, test accuracy, model architecture, and technical pipeline summary.
2. **Upload Audio Detection (`/upload`):**
   - Drag and drop any `.wav`, `.mp3`, `.flac`, or `.ogg` file.
   - Alternatively, click **"Load Genuine Speech Sample"** or **"Load Deepfake Voice Sample"** to test pre-loaded benchmark files instantly.
   - Listen to the audio preview and click **"Analyze Audio with CNN"**.
3. **Live Microphone Voice Detection (`/live`):**
   - Click **"Start Recording"** and approve the browser microphone prompt.
   - Speak for 2 to 5 seconds. The real-time oscilloscope will render your vocal waveform.
   - Click **"Stop Recording"**. Preview your captured audio.
   - Click **"Analyze Recording with CNN"** to run the exact same deep-learning model.
4. **Results Page (`/result`):**
   - View classification verdict (**REAL VOICE** in glowing emerald or **FAKE VOICE** in glowing red).
   - Inspect derived Softmax probability breakdown and confidence score.
   - Examine the generated time-domain **Waveform Plot** and 128-bin **Mel-Spectrogram**.
   - Click **"Download Forensic Audit Report (PDF)"** to generate a signed forensic audit PDF.

---

## Running Unit & Integration Tests

The test suite validates signal processing, mathematical feature dimensions, CNN inference, PDF generation, and all Flask endpoints:

```powershell
# Run all unit and integration tests
python -m unittest discover tests
```

---

## Mathematical Specification

1. **Log Mel-Spectrogram Calculation:**
   Given discrete time audio $x[n]$ sampled at $f_s = 16{,}000\text{ Hz}$, the Short-Time Fourier Transform (STFT) with Hann window $w[n]$ of length $N_{\text{fft}} = 1024$ and hop size $R = 500$ is computed as:
   $$X(m, k) = \sum_{n=0}^{N_{\text{fft}}-1} x[n + mR] \cdot w[n] \cdot e^{-j \frac{2\pi}{N_{\text{fft}}} k n}$$
   Energy across $M = 128$ triangular Mel filters is mapped onto the perceptual Mel scale:
   $$m_{\text{mel}} = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$
   Converted to decibel scale:
   $$S_{\text{dB}}(m, b) = 10 \log_{10} \left( \frac{S(m, b)}{\max(S)} + 10^{-10} \right)$$

2. **Softmax Decision Boundary:**
   The final linear layer outputs raw logits $\mathbf{z} = [z_0, z_1]$. Class probabilities are obtained via Softmax:
   $$P(\text{Class } c \mid \mathbf{x}) = \frac{e^{z_c}}{e^{z_0} + e^{z_1}}, \quad c \in \{0, 1\}$$
   Where $c = 0$ corresponds to **REAL VOICE** and $c = 1$ denotes **FAKE VOICE**.

---

## Academic Disclaimer
This software is developed for research and educational purposes in Artificial Intelligence and Deep Learning (AIDL). Voice integrity verdicts are probabilistic estimations derived from acoustic features and neural vocoder artifacts.
