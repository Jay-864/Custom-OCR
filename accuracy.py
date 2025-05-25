# predict_accuracy_500_images.py

import os
import torch
import numpy as np
from PIL import Image
import pandas as pd
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Constants
IMG_WIDTH = 200
IMG_HEIGHT = 50
MAX_LENGTH = 10

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CSV
df = pd.read_csv('lpr.csv')
df = df.drop("Unnamed: 0", axis=1)
df['images'] = "./cropped_lps/" + df['images'].astype(str)

# Only first 500 samples
df = df.iloc[:500].reset_index(drop=True)

# Prepare labels
labels = df['labels'].values.tolist()
labels = [label.ljust(MAX_LENGTH) for label in labels]

# Character mappings
characters = set(char for label in labels for char in label)
characters = sorted(list(characters))
char_to_num = {char: idx + 1 for idx, char in enumerate(characters)}
char_to_num['[PAD]'] = 0
num_to_char = {idx: char for char, idx in char_to_num.items()}

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset Class
class LicensePlateDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['images']
        label = self.df.iloc[idx]['labels'].ljust(MAX_LENGTH)

        img = Image.open(img_path).convert('L')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))

        if self.transform:
            img = self.transform(img)

        label_encoded = torch.tensor([char_to_num[char] for char in label], dtype=torch.long)

        return img, label_encoded, label.strip()

# Model Class
class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_dim=256):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
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

# Decode prediction
def decode_prediction(pred):
    pred = torch.nn.functional.softmax(pred, dim=2)
    pred = pred.argmax(2)
    pred = pred.cpu().numpy()

    texts = []
    for sequence in pred:
        text = []
        for idx in sequence:
            if idx == 0:  # Padding
                continue
            text.append(num_to_char[idx])
        texts.append(''.join(text))

    return texts

# Load dataset
dataset = LicensePlateDataset(df, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Load model
model = CRNN(num_classes=len(char_to_num))
model.load_state_dict(torch.load('ocr_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Evaluation
correct = 0
incorrect = 0
wrong_predictions = []

with torch.no_grad():
    for images, labels_encoded, labels_str in tqdm(loader, desc="Evaluating on 500 Images"):
        images = images.to(device)
        outputs = model(images)

        pred_texts = decode_prediction(outputs)

        pred_text = pred_texts[0].strip()
        true_text = labels_str[0].strip()

        if pred_text == true_text:
            correct += 1
        else:
            incorrect += 1
            wrong_predictions.append((true_text, pred_text))

total = correct + incorrect
accuracy = correct / total * 100
print(f"\n✅ Accuracy on 500 images: {accuracy:.2f}% ({correct}/{total})")

# ---------------------------
# Plotting Section
# ---------------------------

# Bar plot: Correct vs Incorrect
plt.figure(figsize=(6, 4))
plt.bar(['Correct', 'Incorrect'], [correct, incorrect], color=['green', 'red'])
plt.title('Prediction Results (500 Images)')
plt.ylabel('Number of Samples')
plt.grid(True, axis='y')
plt.savefig('prediction_bar_plot_500.png')
plt.show()

# Pie chart: Percentage breakdown
plt.figure(figsize=(6, 6))
plt.pie([correct, incorrect], labels=['Correct', 'Incorrect'], autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
plt.title('Prediction Accuracy (500 Images)')
plt.savefig('prediction_pie_chart_500.png')
plt.show()

# Show some wrong predictions
print("\nSome Wrong Predictions (True → Predicted):")
for true, pred in wrong_predictions[:5]:  # Show first 5 wrong samples
    print(f"{true} → {pred}")
