import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torch.utils.data import Subset
import numpy as np
import pandas as pd
import model
import cv2
import os
import datetime
from torchvision.io import read_image

#instantate the convolution model
conv_model = model.ConvModel()
print(conv_model)

# list of parameters
params = list(conv_model.parameters())
print("length parameters: ", len(params))
total_params = sum(
    param.numel() for param in conv_model.parameters()
)
print("total parameters: ", total_params)
print("output_size: ", params[0].size(), "\n")

# test input for network
# x = torch.randn(size=(32, 3, 13, 13)) # 1 image with 3 channels and 13x13 pixels
# print("x:", x)
# print("x_prediction: ", (conv_model(x)))

# Loss function
loss_fn = nn.L1Loss()

# Optimizers specified in the torch.optim package
optimizer = torch.optim.Adam(conv_model.parameters(), lr=0.001)

def target_transform(target):
    target = torch.tensor(target, dtype=torch.float32)
    normalized_label  = target / 128
    return normalized_label


class ABSectionDataset(Dataset):
    def __init__(self, csv_file, image_dir_param, transform_func=None, target_transform_func=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        
        image = Image.open(img_path)
        
        label = self.data.iloc[idx, 1:3].values
        label = label.astype(np.float32)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

# Transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

csv_path = r"F:\Projekte\bell_repo\conv_netzwerk_dataset\train.csv"
data = pd.read_csv(csv_path)

image_dir = r"F:\Projekte\bell_repo\conv_netzwerk_dataset\train"

dataset = ABSectionDataset(csv_file=csv_path, image_dir_param=image_dir, transform_func=transform)

# print(dataset)

subset_indices = list(range(10000))
subset = Subset(dataset, subset_indices)

train_size = int(0.9 * len(subset))
val_size = len(subset) - train_size

train_dataset, val_dataset = random_split(subset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conv_model.to(device)

for epoch in range(num_epochs):
    conv_model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)

        # Forward pass
        outputs = conv_model(images)
        rescaled_outputs = outputs * 128
        loss = loss_fn(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # Validation (optional)
    conv_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = conv_model(images)
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

model_path = r".\saved-models"
torch.save(conv_model.state_dict(), model_path)