"""
AI-POWERED DEEPFAKE VOICE DETECTION
Unit Tests: Model Inference & Prediction (tests/test_prediction.py)
"""

import os
import sys
import unittest

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import predict_audio, DEFAULT_MODEL_PATH
from report_generator import generate_pdf_report
from model import load_model, DeepfakeVoiceCNN, count_parameters


class TestPredictionPipeline(unittest.TestCase):

    def setUp(self):
        self.real_sample = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads", "demo_real_voice.wav")
        self.fake_sample = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads", "demo_deepfake_voice.wav")

    def test_model_loading_and_architecture(self):
        """Verify model checkpoint can be loaded and parameter count is correct."""
        if not os.path.exists(DEFAULT_MODEL_PATH):
            self.skipTest("Model weights not found. Run train.py first.")

        model = load_model(DEFAULT_MODEL_PATH)
        self.assertIsInstance(model, DeepfakeVoiceCNN)
        self.assertGreater(count_parameters(model), 1_000_000)

    def test_predict_real_audio(self):
        """Verify inference returns valid structured output for genuine voice."""
        if not os.path.exists(self.real_sample):
            self.skipTest("Demo real audio not found.")

        result = predict_audio(self.real_sample)
        self.assertEqual(result.get("status"), "success")
        self.assertIn("predicted_class", result)
        self.assertIn("confidence", result)
        self.assertIn("real_probability", result)
        self.assertIn("fake_probability", result)
        self.assertIn("waveform_plot", result)
        self.assertIn("spectrogram_plot", result)
        self.assertGreaterEqual(result["confidence"], 50.0)
        self.assertLessEqual(result["confidence"], 100.0)

    def test_predict_fake_audio(self):
        """Verify inference returns valid structured output for fake voice."""
        if not os.path.exists(self.fake_sample):
            self.skipTest("Demo fake audio not found.")

        result = predict_audio(self.fake_sample)
        self.assertEqual(result.get("status"), "success")
        self.assertIn("predicted_class", result)
        self.assertIn(result["predicted_class"], ["REAL VOICE", "FAKE VOICE"])

    def test_missing_model_handling(self):
        """Verify informative error handling when an invalid model path is specified."""
        result = predict_audio(self.real_sample, model_path="non_existent_model.pth")
        self.assertEqual(result.get("status"), "error")
        self.assertEqual(result.get("error_type"), "model_missing")

    def test_pdf_report_generation(self):
        """Verify that a PDF forensic report is successfully compiled from prediction data."""
        if not os.path.exists(self.real_sample):
            self.skipTest("Demo audio not found.")

        result = predict_audio(self.real_sample)
        if result.get("status") == "success":
            pdf_path = generate_pdf_report(result)
            self.assertTrue(os.path.exists(pdf_path))
            self.assertGreater(os.path.getsize(pdf_path), 1000)
            # Cleanup test report
            if os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                except Exception:
                    pass


if __name__ == '__main__':
    unittest.main()
