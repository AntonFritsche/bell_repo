import os
import numpy as np
import pandas as pd
import torch as torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torchvision import transforms
import time

import model

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
loss_fn = nn.MSELoss() # Mean Squared Error: error is squared
# loss_fn = nn.L1Loss() # Mean Absolute Error: error is just absolute

# Optimizers specified in the torch.optim package
optimizer = torch.optim.Adam(conv_model.parameters(), lr=0.001)

def target_transform_func(target):
    target = torch.tensor(target, dtype=torch.float32)
    normalized_label  = torch.div(target, 128) # rescale target to the range (-1; 1) for better data handling for the model
    return normalized_label

class ABSectionDataset(Dataset):
    def __init__(self, csv_file, image_dir_param, transform_func=None, target_transform_param=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform_func

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        
        image = Image.open(img_path)

        if image.mode != "L":
            image = image.convert("L")

        label = self.data.iloc[idx, 1:3].values.astype(np.float32)
        
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

csv_path = r"E:\Programmierung\Datein\Python\bell_repo\conv-network\data.csv"
data = pd.read_csv(csv_path)

image_dir = r"E:\Programmierung\Datein\Python\bell_repo\conv-network\train"

dataset = ABSectionDataset(csv_file=csv_path, image_dir_param=image_dir, transform_func=None, target_transform_param=target_transform_func)

# print(dataset)

subset_indices = list(range(14999))
subset = Subset(dataset, subset_indices)

train_size = int(0.9 * len(subset))
val_size = len(subset) - train_size

train_dataset, val_dataset = random_split(subset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

num_epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conv_model.to(device)
start_time = time.time()

for epoch in range(num_epochs):
    epoch_time = time.time()
    conv_model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        # print(f"images.shape: {images.shape}")
        # Forward pass
        # print(images[:1][:1])
        outputs = conv_model(images)

        rescaled_outputs = torch.mul(outputs, 128) # scale the outputs back to lab color space
        rescaled_targets = torch.mul(targets, 128)

        loss = loss_fn(rescaled_outputs, rescaled_targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # noinspection PyUnboundLocalVariable
    print(f"rescaled targets: {rescaled_targets[:1]}")
    # noinspection PyUnboundLocalVariable
    print(f"rescaled outputs: {rescaled_outputs[:1]}")
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # Validation (optional)
    conv_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = conv_model(images)

            rescaled_outputs_loss = torch.mul(outputs, 128) # scale the outputs back to lab color space
            rescaled_targets_loss = torch.mul(targets, 128)

            loss = loss_fn(rescaled_outputs_loss, rescaled_targets_loss)
            val_loss += loss.item()
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
    print(f"Epoch time: {time.time() - epoch_time:.2f} seconds")
    print("\n")
elapsed_time = time.time() - start_time
# noinspection PyUnboundLocalVariable
print(f"Training time: {elapsed_time:.2f} seconds")
model_path = r"E:\Programmierung\Datein\Python\bell_repo\conv-network\saved-models\conv_model_2.pth"
torch.save(conv_model.state_dict(), model_path)