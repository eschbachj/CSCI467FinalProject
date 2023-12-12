import os
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Set seed for reproducibility
torch.manual_seed(42)

# Define data directories
train_data_dir = './test_files/train/images'
dev_data_dir = './segmentation_dataset/final_datasets/dev/images'
test_data_dir = './segmentation_dataset/final_datasets/test/images'

train_mask_dir = './test_files/train/masks'
dev_mask_dir = './segmentation_dataset/final_datasets/dev/masks'
test_mask_dir = './segmentation_dataset/final_datasets/test/masks'

# Function to load data
class CellSegmentationDataset(Dataset):
    def __init__(self, data_dir, mask_dir, transform=None):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [img for img in os.listdir(data_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.data_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask

# Define transforms
transform = transforms.Compose([
    transforms.Resize((904, 1224)),
    transforms.ToTensor(),
])

# Load training data
train_dataset = CellSegmentationDataset(train_data_dir, train_mask_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Load dev data
dev_dataset = CellSegmentationDataset(dev_data_dir, dev_mask_dir, transform=transform)
dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)

# Load test data
test_dataset = CellSegmentationDataset(test_data_dir, test_mask_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define U-Net model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)

        # Adjust the dimensions to match the input size
        x3 = nn.functional.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)

        return x3

# Instantiate the model and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, masks in tqdm(train_loader):
        inputs, masks = inputs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

def calculate_iou(predicted, target):
    intersection = torch.logical_and(predicted, target)
    union = torch.logical_or(predicted, target)
    iou = torch.sum(intersection) / torch.sum(union)
    return iou.item()


# Evaluate the model on the test set
model.eval()
test_iou_scores = []

output_dir = './segmentation_results'
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    for i, (inputs, masks) in enumerate(tqdm(test_loader)):
        inputs, masks = inputs.to(device), masks.to(device)
        outputs = model(inputs)
        predictions = torch.sigmoid(outputs)
        predictions = (predictions > 0.5).float()

        # Calculate IoU
        iou = calculate_iou(predictions, masks)
        test_iou_scores.append(iou)

        # Convert tensors to numpy arrays for visualization
        input_image = inputs[0].permute(1, 2, 0).cpu().numpy()
        correct_mask = masks[0].cpu().numpy()
        predicted_mask = predictions[0].cpu().numpy()

        # Ensure that correct_mask and predicted_mask have the same number of dimensions
        correct_mask = np.expand_dims(correct_mask, axis=-1)  # Ensure it has 3 dimensions
        predicted_mask = np.expand_dims(predicted_mask, axis=-1)  # Ensure it has 3 dimensions

        # Create subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot original image
        axs[0].imshow(input_image)
        axs[0].set_title('Original Image')

        # Plot correct mask
        axs[1].imshow(correct_mask.squeeze(), cmap='gray')
        axs[1].set_title('Correct Mask')

        # Plot predicted mask
        axs[2].imshow(predicted_mask.squeeze(), cmap='gray')
        axs[2].set_title('Predicted Mask\nIoU: {:.4f}'.format(iou))

        # Save the figure
        output_path = os.path.join(output_dir, f'result_{i}.png')
        plt.savefig(output_path)
        plt.close()

# Calculate and print average IoU score
average_iou = np.mean(test_iou_scores)
print("Average IoU Score:", average_iou)