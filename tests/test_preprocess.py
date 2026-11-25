"""
AI-POWERED DEEPFAKE VOICE DETECTION
Unit Tests: Audio Preprocessing (tests/test_preprocess.py)
"""

import os
import sys
import unittest
import numpy as np
import soundfile as sf
import torch

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess import (
    load_audio, normalize_waveform, fix_audio_length,
    extract_mel_spectrogram, normalize_features,
    preprocess_audio_pipeline, AudioProcessingError,
    TARGET_SR, TARGET_SAMPLES, N_MELS, TARGET_FRAMES
)


class TestAudioPreprocessing(unittest.TestCase):

    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_data")
        os.makedirs(self.test_dir, exist_ok=True)

        # Create valid test audio (2 seconds stereo at 44.1kHz to test resampling & mono conversion)
        sr = 44100
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        stereo_signal = np.stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
            0.5 * np.cos(2 * np.pi * 880 * t)
        ], axis=1)
        self.valid_wav = os.path.join(self.test_dir, "test_stereo.wav")
        sf.write(self.valid_wav, stereo_signal, sr)

        # Create empty audio file
        self.empty_wav = os.path.join(self.test_dir, "test_empty.wav")
        sf.write(self.empty_wav, np.array([], dtype=np.float32), TARGET_SR)

        # Create silent audio file
        self.silent_wav = os.path.join(self.test_dir, "test_silent.wav")
        sf.write(self.silent_wav, np.zeros(TARGET_SR, dtype=np.float32), TARGET_SR)

        # Create corrupted non-audio file
        self.corrupt_file = os.path.join(self.test_dir, "test_corrupt.wav")
        with open(self.corrupt_file, "wb") as f:
            f.write(b"NOT_A_VALID_RIFF_AUDIO_DATA_HEADER_CORRUPTED")

    def tearDown(self):
        # Clean up temporary test files
        for f in [self.valid_wav, self.empty_wav, self.silent_wav, self.corrupt_file]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass
        if os.path.exists(self.test_dir):
            try:
                os.rmdir(self.test_dir)
            except Exception:
                pass

    def test_load_audio_resampling_and_mono(self):
        """Verify audio loads as 1D mono and resamples to 16,000 Hz."""
        y, sr = load_audio(self.valid_wav, sr=TARGET_SR)
        self.assertEqual(sr, TARGET_SR)
        self.assertEqual(y.ndim, 1)
        self.assertGreater(len(y), 0)

    def test_waveform_normalization(self):
        """Verify peak amplitude normalization does not exceed 1.0."""
        raw = np.array([0.2, -0.8, 0.4, -1.5, 0.1], dtype=np.float32)
        norm = normalize_waveform(raw)
        self.assertAlmostEqual(np.max(np.abs(norm)), 1.0, places=4)

    def test_fix_audio_length_padding_and_truncation(self):
        """Verify audio is padded or truncated to exact TARGET_SAMPLES."""
        # Test shorter input (padding)
        short_y = np.ones(1000, dtype=np.float32)
        fixed_short = fix_audio_length(short_y, target_samples=TARGET_SAMPLES)
        self.assertEqual(len(fixed_short), TARGET_SAMPLES)

        # Test longer input (truncation)
        long_y = np.ones(TARGET_SAMPLES + 5000, dtype=np.float32)
        fixed_long = fix_audio_length(long_y, target_samples=TARGET_SAMPLES)
        self.assertEqual(len(fixed_long), TARGET_SAMPLES)

    def test_extract_mel_spectrogram_shape(self):
        """Verify Mel-spectrogram output dimensions match (N_MELS, TARGET_FRAMES)."""
        waveform = np.random.randn(TARGET_SAMPLES).astype(np.float32)
        mel_db = extract_mel_spectrogram(waveform, sr=TARGET_SR)
        self.assertEqual(mel_db.shape, (N_MELS, TARGET_FRAMES))
        self.assertEqual(mel_db.shape, (128, 128))

    def test_feature_normalization(self):
        """Verify feature normalization standardizes with mean/std."""
        data = np.random.randn(128, 128).astype(np.float32) * 15.0 - 40.0
        norm = normalize_features(data, mean=-40.0, std=15.0)
        self.assertAlmostEqual(np.mean(norm), 0.0, delta=0.5)

    def test_full_pipeline_output_tensor(self):
        """Verify full preprocessing pipeline outputs PyTorch tensor of shape (1, 1, 128, 128)."""
        tensor, raw_y, duration, mel_db = preprocess_audio_pipeline(self.valid_wav)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (1, 1, 128, 128))
        self.assertGreater(duration, 0.0)

    def test_error_on_empty_and_corrupted_audio(self):
        """Verify proper AudioProcessingError exceptions on corrupt, empty, and silent files."""
        with self.assertRaises(AudioProcessingError):
            load_audio(self.empty_wav)

        with self.assertRaises(AudioProcessingError):
            load_audio(self.silent_wav)

        with self.assertRaises(AudioProcessingError):
            load_audio(self.corrupt_file)


if __name__ == '__main__':
    unittest.main()
