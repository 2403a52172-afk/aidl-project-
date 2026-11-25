"""
AI-POWERED DEEPFAKE VOICE DETECTION
Dataset Preparation & Acquisition Tool (dataset/prepare_dataset.py)

Downloads authentic real human voice recordings (from YouTube interviews/podcasts)
and authentic AI-synthesized deepfake voices (ElevenLabs, HiFi-GAN, Speechify, PlayHT, Lovo).
"""

import os
import sys
import glob
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import soundfile as sf
import librosa

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
REAL_DIR = os.path.join(DATASET_DIR, "real")
FAKE_DIR = os.path.join(DATASET_DIR, "fake")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

SAMPLE_RATE = 16000


def clean_existing_synthetic_samples():
    """Remove previous placeholder files to ensure dataset purity."""
    for pattern in ["real_spk*.wav", "fake_vocoder*.wav"]:
        for f in glob.glob(os.path.join(REAL_DIR, pattern)) + glob.glob(os.path.join(FAKE_DIR, pattern)):
            try:
                os.remove(f)
            except Exception:
                pass


def fetch_file_list(category):
    """Fetch available audio files from the Hugging Face dataset."""
    url = f"https://huggingface.co/api/datasets/garystafford/deepfake-audio-detection/tree/main/{category}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as resp:
        items = json.loads(resp.read().decode())
    return [x['path'] for x in items if x.get('type') == 'file' and x['path'].endswith('.flac')]


def download_single_file(item_path, dest_folder):
    """Download and save a single audio file."""
    fname = os.path.basename(item_path)
    dest_path = os.path.join(dest_folder, fname)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return dest_path

    url = f"https://huggingface.co/datasets/garystafford/deepfake-audio-detection/resolve/main/{item_path}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=20) as resp:
        content = resp.read()

    with open(dest_path, 'wb') as f:
        f.write(content)
    return dest_path


def download_authentic_dataset(target_count_per_class=120):
    """
    Downloads authentic human speech and authentic AI-generated speech.
    """
    os.makedirs(REAL_DIR, exist_ok=True)
    os.makedirs(FAKE_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    print("Cleaning older synthetic placeholders...")
    clean_existing_synthetic_samples()

    print(f"Connecting to authentic deepfake voice repository on Hugging Face...")
    real_paths = fetch_file_list('real')
    fake_paths = fetch_file_list('fake')

    print(f"Available in remote repository: {len(real_paths)} REAL, {len(fake_paths)} FAKE")

    # Select balanced diverse samples
    # For REAL: spread across various speakers
    selected_real = real_paths[:target_count_per_class]

    # For FAKE: balance across ElevenLabs (el_), HiFi-GAN (hg_), Speechify (sp_), PlayHT (po_), Lovo (lv_)
    el_fakes = [p for p in fake_paths if '/el_' in p]
    hg_fakes = [p for p in fake_paths if '/hg_' in p]
    sp_fakes = [p for p in fake_paths if '/sp_' in p]
    po_fakes = [p for p in fake_paths if '/po_' in p]
    lv_fakes = [p for p in fake_paths if '/lv_' in p]

    selected_fake = []
    # Interleave to ensure maximum generator diversity
    for group in [el_fakes, hg_fakes, sp_fakes, po_fakes, lv_fakes]:
        selected_fake.extend(group[:target_count_per_class // 5 + 5])
    selected_fake = selected_fake[:target_count_per_class]

    print(f"Downloading {len(selected_real)} genuine human voice clips into '{REAL_DIR}'...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_single_file, p, REAL_DIR) for p in selected_real]
        for f in as_completed(futures):
            f.result()

    print(f"Downloading {len(selected_fake)} AI-generated voice clips into '{FAKE_DIR}'...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_single_file, p, FAKE_DIR) for p in selected_fake]
        for f in as_completed(futures):
            f.result()

    # Create high-quality demo samples in uploads/
    demo_real_src = os.path.join(REAL_DIR, os.path.basename(selected_real[0]))
    demo_fake_src = os.path.join(FAKE_DIR, os.path.basename(selected_fake[0]))

    # Convert to standard 16kHz WAV for demo files
    y_real, sr_real = librosa.load(demo_real_src, sr=16000, mono=True)
    y_fake, sr_fake = librosa.load(demo_fake_src, sr=16000, mono=True)

    demo_real_path = os.path.join(UPLOADS_DIR, "demo_real_voice.wav")
    demo_fake_path = os.path.join(UPLOADS_DIR, "demo_deepfake_voice.wav")

    sf.write(demo_real_path, y_real, 16000)
    sf.write(demo_fake_path, y_fake, 16000)

    print(f"Prepared demonstration benchmarks:")
    print(f"  Real human speech: {demo_real_path}")
    print(f"  AI deepfake speech: {demo_fake_path}")
    print(f"Dataset preparation complete! Total: {len(selected_real)} REAL, {len(selected_fake)} FAKE.")


if __name__ == "__main__":
    download_authentic_dataset(target_count_per_class=120)
