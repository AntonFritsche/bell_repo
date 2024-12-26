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
print("output_size: ", params[0].size())

x = torch.randn(size=(1, 13, 13))
print("x: ", conv_model(x))

# Loss function
loss_fn = nn.L1Loss()

# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(conv_model.parameters(), lr=0.001, momentum=0.9)

class ABSectionDataset(Dataset):
    def __init__(self, csv_file, image_dir, section_size=13, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.section_size = section_size
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = row['filename']
        section_id = row['Abschnitt_ID']
        a_value = row['A-Wert']
        b_value = row['B-Wert']

        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        num_sections = int(image.size[0] // self.section_size)
        row_idx = section_id // num_sections
        col_idx = section_id % num_sections

        left = col_idx * self.section_size
        upper = row_idx * self.section_size
        right = left + self.section_size
        lower = upper + self.section_size
        section = image.crop((left, upper, right, lower))

        if self.transform:
            section = self.transform(section)

        ab_values = torch.tensor([a_value, b_value], dtype=torch.float32)

        return section, ab_values

# Transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

csv_path = "train.csv"
data = pd.read_csv(csv_path)

image_dir = "train/"

dataset = ABSectionDataset(csv_file=csv_path, image_dir=image_dir, section_size=13, transform=transform)

subset_indices = list(range(1000))  # Indexliste für die ersten 1000
subset = Subset(dataset, subset_indices)

train_size = int(0.8 * len(subset))
val_size = len(subset) - train_size

train_dataset, val_dataset = random_split(subset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print(f'  Batch {i + 1} loss: {last_loss:.4f}')
            
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            
            running_loss = 0.0

    val_loss = validate_model(val_loader)
    return last_loss, val_loss

def validate_model(val_loader):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total

    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
    model.train()
    return val_loss

num_epochs = 100

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    tb_writer = SummaryWriter()

    train_loss, val_loss = train_one_epoch(epoch, tb_writer)
    
    print(f'Epoch {epoch + 1} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')