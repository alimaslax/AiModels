import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # Import matplotlib for image display

# Data loading and transforming
transform = transforms.Compose([
    transforms.Resize((227, 227)),  # Resize images to 227x227 as expected by AlexNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ImageNet
])

# Load an example dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# Print the number of images in the training dataset
print(f'Total number of images in the training dataset: {len(train_dataset)}')

# Function to display images with a start index
def show_images(dataset, num_images=10, start_index=0):
    """Display a grid of images from the dataset starting from the given index."""
    # Ensure start_index is within range
    if start_index < 0 or start_index >= len(dataset):
        raise IndexError("Start index is out of range.")

    # Initialize lists to hold images and labels
    images, labels = [], []

    # Get the specified number of images from the dataset starting at the start_index
    for i in range(start_index, min(start_index + num_images, len(dataset))):
        image, label = dataset[i]
        images.append(image)
        labels.append(label)

    # Create a grid of images
    num_images = len(images)
    num_cols = 5  # Set the number of columns you want in the grid
    num_rows = (num_images + num_cols - 1) // num_cols  # Calculate the number of rows needed

    plt.figure(figsize=(8, 6))  # Set the figure size to 800x600 pixels (8x6 inches at 100 DPI)
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')  # Hide axes
    plt.show()

# Call the function to display images starting from index 30
show_images(train_dataset, num_images=10, start_index=30)
