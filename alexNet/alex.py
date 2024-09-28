import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import time
import signal
import sys
import numpy as np
import argparse  # Import argparse for command-line argument parsing

# Define the AlexNet model
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.conv4(x)
        x = nn.ReLU()(x)
        x = self.conv5(x)
        x = nn.ReLU()(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Function to create and return a model based on selection
def create_model(model_name='alexnet', num_classes=10):
    if model_name.lower() == 'alexnet':
        return AlexNet(num_classes=num_classes)
    else:
        raise ValueError("Model not recognized. Available options: ['alexnet']")

# Function to handle saving the model on Ctrl+C
def signal_handler(sig, frame):
    print('\nSaving model weights...')
    torch.save(model.state_dict(), './models/saved_model.pth')
    print('Model saved as saved_model.pth')
    sys.exit(0)

# Set up signal handling
signal.signal(signal.SIGINT, signal_handler)

# Argument parser for model loading
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Path to the model checkpoint to load (e.g., saved_model.pth)')
args = parser.parse_args()

# Select model and create it
model_name = 'alexnet'
num_classes = 10
model = create_model(model_name=model_name, num_classes=num_classes)

# Load pre-trained model if provided
if args.model:
    try:
        model.load_state_dict(torch.load(args.model))
        print(f'Model loaded from {args.model}')
    except Exception as e:
        print(f'Error loading model: {e}')
        sys.exit(1)

# Data loading and transforming
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare to log training statistics
log_file = 'logs/training_log.csv'
with open(log_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Batch', 'Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy', 'Time (s)', 'Batches per Second', 'Learning Rate'])

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0  # Track the loss for this epoch
    running_corrects = 0  # Track correct predictions
    start_time = time.time()

    print(f"Start Epoch [{epoch + 1}/{num_epochs}]")
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch') as pbar:
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to('cpu'), labels.to('cpu')  # Move data to the device if available

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # Accumulate loss
            _, preds = torch.max(outputs, 1)  # Get predictions
            running_corrects += (preds == labels).sum().item()  # Count correct predictions
            
            # Update the progress bar
            pbar.set_postfix({'loss': running_loss / (batch_idx + 1), 'accuracy': running_corrects / ((batch_idx + 1) * images.size(0))})  # Update loss and accuracy in progress bar
            pbar.update(1)  # Increment progress bar

            # Log the loss and batch time
            elapsed_time = time.time() - start_time
            batches_per_second = (batch_idx + 1) / elapsed_time
            learning_rate = optimizer.param_groups[0]['lr']  # Get the current learning rate

            # Write to log file
            with open(log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, batch_idx + 1, loss.item(), running_corrects / ((batch_idx + 1) * images.size(0)), '', '', elapsed_time, batches_per_second, learning_rate])

    # Print loss and accuracy per epoch
    train_accuracy = running_corrects / len(train_loader.dataset)
    print(f"End Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Training Accuracy: {train_accuracy:.4f}")

    # Save the model checkpoint at the end of the epoch
    torch.save(model.state_dict(), f'alexnet_model_epoch_{epoch + 1}.pth')
    print(f'Model saved as alexnet_model_epoch_{epoch + 1}.pth')

# Save the final model checkpoint
torch.save(model.state_dict(), 'alexnet_final_model.pth')
print('Final model saved as alexnet_final_model.pth')