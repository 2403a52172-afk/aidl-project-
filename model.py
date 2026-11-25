"""
AI-POWERED DEEPFAKE VOICE DETECTION
Deep Learning Model Architecture (model.py)

This module defines:
1. DeepfakeVoiceCNN: PyTorch Convolutional Neural Network for binary classification.
   - Class 0: REAL VOICE (Genuine human speech)
   - Class 1: FAKE VOICE (Synthesized / AI-generated / Deepfake speech)
2. Model weight checkpointing, loading, and device management (CUDA / CPU).
3. Softmax prediction inference utility.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepfakeVoiceCNN(nn.Module):
    """
    Convolutional Neural Network tailored for 2D Audio Time-Frequency Representations (Mel Spectrograms).
    Input Shape: (Batch_Size, 1, 128, 128)
    Output Shape: (Batch_Size, 2) -> Raw Logits for [REAL, FAKE]
    """

    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(DeepfakeVoiceCNN, self).__init__()

        # Conv Block 1: (1, 128, 128) -> (32, 64, 64)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.15)
        )

        # Conv Block 2: (32, 64, 64) -> (64, 32, 32)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2)
        )

        # Conv Block 3: (64, 32, 32) -> (128, 16, 16)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25)
        )

        # Conv Block 4: (128, 16, 16) -> (256, 8, 8)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(p=dropout_rate)
        )

        # Dense Classifier Layers
        # 256 channels * 4 * 4 spatial dimension = 4096
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Forward pass returning unnormalized class logits.
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def predict_proba(self, x):
        """
        Forward pass applying Softmax to return class probabilities.
        """
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=1)
        return probabilities


def get_device():
    """
    Select CUDA GPU acceleration if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def count_parameters(model):
    """
    Return total trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, path="models/best_model.pth", config=None, config_path="models/config.json"):
    """
    Save trained model weights and preprocessing configuration.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

    if config is not None:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)


def load_model(path="models/best_model.pth", device=None):
    """
    Instantiate architecture and load trained PyTorch weights.
    """
    if device is None:
        device = get_device()

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Model training is required before inference.")

    model = DeepfakeVoiceCNN(num_classes=2)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Test model shape and parameter count
    dev = get_device()
    test_model = DeepfakeVoiceCNN().to(dev)
    dummy_input = torch.randn(4, 1, 128, 128).to(dev)
    out = test_model(dummy_input)
    probs = test_model.predict_proba(dummy_input)
    print(f"Device: {dev}")
    print(f"Model Parameters: {count_parameters(test_model):,}")
    print(f"Output Logits Shape: {out.shape}")
    print(f"Softmax Probabilities:\n{probs.detach().cpu().numpy()}")
