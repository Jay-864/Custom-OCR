import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import os.path as osp

# Constants
IMG_WIDTH = 200
IMG_HEIGHT = 50
BATCH_SIZE = 32
MAX_LENGTH = 10  # Adjust this according to your dataset's max label length
PLOTS_DIR = 'plots'  # Directory to save plots

# Create plots directory if it doesn't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load CSV
df = pd.read_csv('lpr.csv')  # Adjust path if needed
df = df.drop("Unnamed: 0", axis=1)

# Add full path for images
df['images'] = "./cropped_lps/" + df['images'].astype(str)

# Extract image paths and labels
images = df['images'].values.tolist()
labels = df['labels'].values.tolist()

# Pad labels to MAX_LENGTH
labels = [label.ljust(MAX_LENGTH) for label in labels]

# Define the character set based on the labels
characters = set(char for label in labels for char in label)
characters = sorted(list(characters))

# Create char-to-num and num-to-char mappings
char_to_num = {char: idx + 1 for idx, char in enumerate(characters)}  # idx + 1 because 0 will be used for padding
char_to_num['[PAD]'] = 0  # Add padding token
num_to_char = {idx: char for char, idx in char_to_num.items()}

# Dataset class
class LicensePlateDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load and preprocess image
        img = Image.open(image_path).convert('L')  # Convert image to grayscale
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))  # Resize image

        if self.transform:
            img = self.transform(img)

        # Convert the label into a tensor of character indices
        label = torch.tensor([char_to_num[char] for char in label], dtype=torch.long)

        return img, label

# Train-validation split (90% train, 10% validation)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(images, labels, test_size=0.1, random_state=42)

# Define the transformations for images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize for grayscale images
])

# Create datasets and dataloaders
train_dataset = LicensePlateDataset(X_train, y_train, transform=transform)
valid_dataset = LicensePlateDataset(X_valid, y_valid, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the model
class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_dim=256):
        super(CRNN, self).__init__()
        
        # CNN part
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # RNN part
        self.rnn = nn.LSTM(128, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        
        # Fully connected layer to output class probabilities
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # CNN layers
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        # RNN part
        x = x.view(x.size(0), -1, 128)  # Flatten the image feature map
        
        # Pass through RNN
        x, _ = self.rnn(x)
        
        # Fully connected layer
        x = self.fc(x)
        
        return x

# Initialize the model
num_classes = len(char_to_num)  # Total number of characters + 1 for the padding
model = CRNN(num_classes=num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function (CTC Loss)
criterion = nn.CTCLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 50

# Initialize lists to store metrics
train_losses = []
val_losses = []
epoch_times = []

print("\n🚀 Starting Training Process")
print(f"📊 Dataset Info: {len(X_train)} training samples, {len(X_valid)} validation samples")
print(f"🔢 Number of character classes: {num_classes}")
print(f"🖥️  Using device: {device}\n")

for epoch in range(EPOCHS):
    epoch_start_time = time()
    
    # Training phase
    model.train()
    running_loss = 0.0
    train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]', leave=False)
    
    for batch_idx, (images, labels) in enumerate(train_loop):
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate CTC Loss
        outputs = outputs.permute(1, 0, 2)  # Change shape to (seq_len, batch, num_classes)
        
        input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long, device=device)
        target_lengths = torch.full((images.size(0),), labels.size(1), dtype=torch.long, device=device)
        
        loss = criterion(outputs, labels, input_lengths, target_lengths)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        train_loop.set_postfix(loss=loss.item())
    
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    valid_loss = 0.0
    val_loop = tqdm(valid_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]', leave=False)
    
    with torch.no_grad():
        for images, labels in val_loop:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)
            
            input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long, device=device)
            target_lengths = torch.full((images.size(0),), labels.size(1), dtype=torch.long, device=device)
            
            loss = criterion(outputs, labels, input_lengths, target_lengths)
            valid_loss += loss.item()
            val_loop.set_postfix(loss=loss.item())
    
    avg_val_loss = valid_loss / len(valid_loader)
    val_losses.append(avg_val_loss)
    
    epoch_time = time() - epoch_start_time
    epoch_times.append(epoch_time)
    
    # Print epoch summary
    print(f"\n✅ Epoch {epoch+1}/{EPOCHS} Complete")
    print(f"⏱️  Time: {epoch_time:.2f}s")
    print(f"📉 Train Loss: {avg_train_loss:.4f}")
    print(f"📊 Val Loss: {avg_val_loss:.4f}")
    
    # Save plots after each epoch
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot epoch times
    plt.subplot(1, 2, 2)
    plt.plot(epoch_times, marker='o', color='green')
    plt.title('Epoch Duration')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(osp.join(PLOTS_DIR, f'epoch_{epoch+1}_metrics.png'))
    plt.close()

# Save the trained model
torch.save(model.state_dict(), "ocr_model.pth")
print("\n🎉 Training Complete! Model saved as 'ocr_model.pth'")

# Final summary
print("\n📊 Training Summary:")
print(f"🏁 Final Training Loss: {train_losses[-1]:.4f}")
print(f"🏁 Final Validation Loss: {val_losses[-1]:.4f}")
print(f"⏱️  Average Epoch Time: {sum(epoch_times)/len(epoch_times):.2f}s")
print(f"🔥 Best Validation Loss: {min(val_losses):.4f} at epoch {val_losses.index(min(val_losses))+1}")

# Save final plots
plt.figure(figsize=(12, 5))

# Final loss plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='o')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Final epoch times plot
plt.subplot(1, 2, 2)
plt.plot(epoch_times, marker='o', color='green')
plt.title('Epoch Duration')
plt.xlabel('Epoch')
plt.ylabel('Time (s)')
plt.grid(True)

plt.tight_layout()
plt.savefig(osp.join(PLOTS_DIR, 'final_metrics.png'))
plt.close()