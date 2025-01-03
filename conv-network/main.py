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
x = torch.randn(size=(32, 3, 13, 13)) # 1 image with 3 channels and 13x13 pixels
print("x:", x)
print("x_prediction: ", conv_model(x))

# Loss function
loss_fn = nn.L1Loss()

# Optimizers specified in the torch.optim package
optimizer = torch.optim.Adam(conv_model.parameters(), lr=0.001)

def target_transform(label):
    label_tensor = torch.tensor(label, dtype=torch.float)
    
    normalized_label = label_tensor / 128.0  # scaling by factor of 128
    return normalized_label

class ABSectionDataset(Dataset):
    def __init__(self, csv_file, image_dir_param, section_size=13, transform_func=None, target_transform_func=None):
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
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

csv_path = r"F:\Projekte\bell_repo\conv_netzwerk_dataset\train.csv"
data = pd.read_csv(csv_path)

image_dir = r"F:\Projekte\bell_repo\conv_netzwerk_dataset\train"

dataset = ABSectionDataset(csv_file=csv_path, image_dir_param=image_dir, section_size=13, transform_func=transform, target_transform_func=target_transform)

# print(dataset)

subset_indices = list(range(10000))
subset = Subset(dataset, subset_indices)

train_size = int(0.9 * len(subset))
val_size = len(subset) - train_size

train_dataset, val_dataset = random_split(subset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

def train_one_epoch(epoch_index, tb_writer_param):
    running_loss = 0.0
    last_loss = 0.0

    for i, data_train_loader in enumerate(train_loader):
        inputs, labels = data_train_loader
        # print("inputs: ", inputs)
        # print("labels: ", labels)

        optimizer.zero_grad()
        
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = inputs.to("cpu")
        outputs = conv_model(inputs)
        
        # Rescaling of the outputs and labels
        outputs = outputs * 128
        labels = labels * 128

        print("outputs: ", outputs[:1])
        print("labels: ", labels[:1])

        # print("outputs: ", outputs[:1])
        # print("labels: ", labels[:1])

        loss = loss_fn(outputs, labels)
        loss = loss.clone().detach().requires_grad_(True)
        loss.backward()

        optimizer.step()
        
        running_loss += loss.item()

        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print(f'  Batch {i + 1} loss: {last_loss:.4f}')
            
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer_param.add_scalar('Loss/train', last_loss, tb_x)
            
            running_loss = 0.0

    val_loss_train = validate_model(val_loader)
    return last_loss, val_loss_train

def validate_model(val_loader_param):
    conv_model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader_param:
            inputs, labels = inputs.to("cpu"), labels.to("cpu")
            outputs = conv_model(inputs)

            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            predicted = torch.argmax(outputs, dim=0)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader_param)
    accuracy = 100 * correct / total

    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
    conv_model.train()
    return val_loss

num_epochs = 20

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    tb_writer = SummaryWriter()

    train_loss, validation_loss = train_one_epoch(epoch, tb_writer)
    
    print(f'Epoch {epoch + 1} - Training Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}')