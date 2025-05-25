import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import argparse
from torchvision import transforms
import pandas as pd

# Constants
IMG_WIDTH = 200
IMG_HEIGHT = 50
MAX_LENGTH = 10  # Must match training script

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CSV and process EXACTLY like training script
df = pd.read_csv('lpr.csv')  # Must be same CSV used in training
df = df.drop("Unnamed: 0", axis=1)
df['images'] = "./cropped_lps/" + df['images'].astype(str)

# Process labels EXACTLY like training script
labels = df['labels'].values.tolist()
labels = [label.ljust(MAX_LENGTH) for label in labels]  # Same padding as training

# Generate character set EXACTLY like training script
characters = set(char for label in labels for char in label)
characters = sorted(list(characters))

# Create mappings EXACTLY like training script
char_to_num = {char: idx + 1 for idx, char in enumerate(characters)}
char_to_num['[PAD]'] = 0  # Padding token
num_to_char = {idx: char for char, idx in char_to_num.items()}

# Verify we have the expected number of classes
print(f"Number of classes: {len(char_to_num)}")  # Should match training script

# Define the transformations for images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize for grayscale images
])

# Dataset class for testing
class LicensePlateDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return 1  # Single image

    def __getitem__(self, idx):
        img = Image.open(self.image_path).convert('L')  # Grayscale
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        if self.transform:
            img = self.transform(img)
        return img

# Define the CRNN model (must match training exactly)
class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_dim=256):
        super(CRNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.rnn = nn.LSTM(128, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1, 128)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

def decode_prediction(pred):
    # Convert raw output to probabilities (softmax not applied in model)
    pred = torch.nn.functional.softmax(pred, dim=2)
    # Get most likely characters (batch, seq_len)
    pred = pred.argmax(2)  # shape: (batch, seq_len)
    pred = pred.cpu().numpy()
    
    # Decode each sequence in batch
    texts = []
    for sequence in pred:
        text = []
        for idx in sequence:
            if idx == 0:  # Skip padding
                continue
            text.append(num_to_char[idx])
        texts.append(''.join(text))
    
    return texts[0] if texts else ""


# Parse command-line arguments
parser = argparse.ArgumentParser(description='OCR model for license plate recognition')
parser.add_argument('--test', type=str, required=True, help='Path to the test image')
args = parser.parse_args()

# Initialize model with correct number of classes
model = CRNN(num_classes=len(char_to_num))  # This must match training

# Load model weights
try:
    model.load_state_dict(torch.load('ocr_model.pth', map_location=device))
    print("Model loaded successfully")
except RuntimeError as e:
    print(f"Error loading model: {e}")
    print(f"Expected {len(char_to_num)} classes but model has different size")
    exit(1)

# Move model to device
model = model.to(device)
model.eval()

# Test the image
test_dataset = LicensePlateDataset(args.test, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        
        # Print raw outputs for debugging
        print("Raw outputs shape:", outputs.shape)
        print("Output sample:", outputs[0, :5, :5])
        
        pred_text = decode_prediction(outputs)
        print(f"Predicted text: {pred_text}")