import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import sys
import time
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.data import Subset

training_num = 3 # number for iterations for the same hyperparameters for better mean
batch_sizes = [4, 8, 16, 64] # different batch_sizes
learning_rates = [0.1, 0.01, 0.001, 0.0001] # different learning rates
num_epochs = [10, 25, 50, 100]

module_path = os.path.abspath(os.path.join('..', 'conv-network'))

if module_path not in sys.path:
    sys.path.append(module_path)

# noinspection PyUnresolvedReferences
import model

conv_model = model.ConvModel(1, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 32, 32, 2)
print(conv_model)
loss_fn = torch.nn.MSELoss()

transform = transforms.Compose([
    transforms.ToTensor(),
])
# noinspection DuplicatedCode
def target_transform_func(target):
    target = torch.tensor(target, dtype=torch.float32)
    normalized_label  = torch.div(target, 128) # rescale target to the range (-1; 1) for better data handling for the model
    return normalized_label

class ABSectionDataset(Dataset):
    def __init__(self, csv_file, image_dir_param, transform_func=None, target_transform_param=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir_param
        self.transform = transform_func
        self.target_transform = target_transform_param

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.data.iloc[idx, 0])

        # noinspection PyTypeChecker
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        label = self.data.iloc[idx, 1:3].values.astype(np.float32)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# noinspection DuplicatedCode
csv_path = r"E:\Programmierung\Datein\Python\bell_repo\conv-network\data.csv"
data = pd.read_csv(csv_path)

image_dir = r"F:\Projekte\bell_repo\conv_netzwerk_dataset\train"

dataset = ABSectionDataset(csv_file=csv_path, image_dir_param=image_dir, transform_func=transform, target_transform_param=target_transform_func)

subset_indices = list(range(8000))
subset = Subset(dataset, subset_indices)

train_size = int(0.9 * len(subset))
val_size = len(subset) - train_size

train_dataset, val_dataset = random_split(subset, [train_size, val_size])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conv_model.to(device)
start_time = time.time()

# noinspection PyShadowingNames,PyUnboundLocalVariable
epochs = num_epochs[0]
training_loss = []
validation_loss = []
epoch_times = []

optimizer = torch.optim.Adam(conv_model.parameters(), lr=learning_rates[0]) # optimizer with the learning rate

train_loader = DataLoader(train_dataset, batch_size=batch_sizes[0], shuffle=True) # train loader with the batch size
val_loader = DataLoader(val_dataset, batch_size=batch_sizes[0], shuffle=False) # validation loader with the bath size

training_loss_ot = [] # list for training loss over time
validation_loss_ot = [] # list for validation loss over time

for epoch in range(epochs):
    epoch_time = time.time()
    conv_model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        # noinspection DuplicatedCode
        images, targets = images.to(device), targets.to(device)
        # print(f"images.shape: {images.shape}")
        # Forward pass
        # print(images[:1][:1])
        outputs = conv_model(images)

        rescaled_outputs = torch.mul(outputs, 128) # scale the outputs back to lab color space
        rescaled_targets = torch.mul(targets, 128) # scale the targets back to lab color space

        loss = loss_fn(rescaled_outputs, rescaled_targets)
        training_loss_ot.append(loss.item()) # loss.item() returns the value of tensor as Python number

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        actually_running_loss = running_loss / len(train_loader)
        training_loss.append(actually_running_loss)

    # noinspection PyUnboundLocalVariable
    print(f"rescaled targets: {[round(val, 4) for val in rescaled_targets[0].tolist()]}")
    # noinspection PyUnboundLocalVariable
    print(f"rescaled outputs: {[round(val, 4) for val in rescaled_outputs[0].tolist()]}")
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {actually_running_loss:.4f}")

    # Validation (optional)
    conv_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            # noinspection DuplicatedCode
            images, targets = images.to(device), targets.to(device)
            outputs = conv_model(images)

            rescaled_outputs_loss = torch.mul(outputs, 128) # scale the outputs back to lab color space
            rescaled_targets_loss = torch.mul(targets, 128)

            loss = loss_fn(rescaled_outputs_loss, rescaled_targets_loss)
            validation_loss_ot.append(loss.item())  # loss.item() returns the value of tensor as Python number
            val_loss += loss.item()

            actually_val_loss = val_loss / len(val_loader)
            validation_loss.append(actually_val_loss)

    epoch_time = time.time() - epoch_time
    epoch_times.append(epoch_time)

    print(f"Validation Loss: {actually_val_loss:.4f}")
    print(f"Epoch time: {time.time() - epoch_time:.2f} seconds")
    print("\n")
elapsed_time = time.time() - start_time
# noinspection PyUnboundLocalVariable
print(f"Training time: {elapsed_time:.2f} seconds")



fig_training, ax_training = plt.figure()
ax_training.set_title(f"{training_loss:.4f}")
plt.plot(training_loss, label="training")

fig_valid, ax_valid = plt.figure()
ax_valid.set_title(f"{validation_loss:.4f}")
plt.plot(validation_loss, label="valid")

fig_time, ax_time = plt.figure()
ax_time.set_title(f"{elapsed_time:.2f}")
plt.plot(time, label="time")