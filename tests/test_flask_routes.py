"""
AI-POWERED DEEPFAKE VOICE DETECTION
Flask Endpoint Integration Tests (tests/test_flask_routes.py)
"""

import os
import io
import sys
import json
import unittest
import numpy as np
import soundfile as sf

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from preprocess import TARGET_SR


class TestFlaskEndpoints(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

        # Generate a small valid test audio stream
        audio_buffer = io.BytesIO()
        t = np.linspace(0, 3.0, TARGET_SR * 3, endpoint=False)
        tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sf.write(audio_buffer, tone, TARGET_SR, format='WAV')
        audio_buffer.seek(0)
        self.valid_wav_bytes = audio_buffer.read()

    def test_get_routes(self):
        """Test that all UI pages load with HTTP 200."""
        routes = ['/', '/upload', '/live', '/about', '/result']
        for route in routes:
            resp = self.client.get(route)
            self.assertEqual(resp.status_code, 200, f"Route {route} failed")

    def test_health_check(self):
        """Test /health endpoint returns JSON with model status."""
        resp = self.client.get('/health')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("status", data)
        self.assertIn("pytorch_device", data)

    def test_upload_valid_audio(self):
        """Test POST /predict with a valid WAV file."""
        data = {
            'audio_file': (io.BytesIO(self.valid_wav_bytes), 'test_sample.wav')
        }
        resp = self.client.post('/predict', data=data, content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 200)
        json_data = resp.get_json()
        self.assertEqual(json_data.get('status'), 'success')
        self.assertIn(json_data.get('predicted_class'), ['REAL VOICE', 'FAKE VOICE'])

    def test_upload_unsupported_file(self):
        """Test POST /predict rejects unsupported extensions like .exe or .txt."""
        data = {
            'audio_file': (io.BytesIO(b"Hello text data"), 'malicious.txt')
        }
        resp = self.client.post('/predict', data=data, content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 400)
        json_data = resp.get_json()
        self.assertEqual(json_data.get('status'), 'error')

    def test_upload_empty_request(self):
        """Test POST /predict without audio_file parameter."""
        resp = self.client.post('/predict', data={})
        self.assertEqual(resp.status_code, 400)

    def test_predict_live_valid(self):
        """Test POST /predict-live receives live recorded wav blob."""
        data = {
            'audio_file': (io.BytesIO(self.valid_wav_bytes), 'live_rec.wav')
        }
        resp = self.client.post('/predict-live', data=data, content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 200)
        json_data = resp.get_json()
        self.assertEqual(json_data.get('status'), 'success')

    def test_predict_live_too_short(self):
        """Test POST /predict-live rejects tiny/empty recordings."""
        data = {
            'audio_file': (io.BytesIO(b"RIFFtiny"), 'live_rec.wav')
        }
        resp = self.client.post('/predict-live', data=data, content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 400)


if __name__ == '__main__':
    unittest.main()
