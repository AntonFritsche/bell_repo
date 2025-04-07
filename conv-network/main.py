import os
from time import time
import cv2
import numpy as np
import pandas as pd
import torch as torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.data import Subset
from torchvision import transforms
from torchsummary import summary
import model

#instantate the convolution model
conv_model = model.ConvModel(1, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 32, 32, 32, 32, 2)

# list of parameters
params = list(conv_model.parameters())
total_params = sum(
    param.numel() for param in conv_model.parameters()
)

summary(conv_model, (1, 13, 13))

# Loss function
loss_fn = nn.MSELoss() # Mean Squared Error: error is squared

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
        image = cv2.cvtColor(cv2.imread(img_path).astype(np.float32)/255.0, cv2.COLOR_BGR2Lab)
        image = cv2.extractChannel(image, 0)/100
        label = self.data.iloc[idx, 1:3].values.astype(np.float32)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

# noinspection DuplicatedCode
csv_path = r"E:\Programmierung\Datein\Python\bell_repo\conv-network\data.csv"
image_dir = r"F:\Projekte\bell_repo\conv_netzwerk_dataset\train"

dataset = ABSectionDataset(csv_file=csv_path, image_dir_param=image_dir, transform_func=transform, target_transform_param=target_transform_func)

subset_indices = list(range(12000))
subset = Subset(dataset, subset_indices)
train_size = int(0.9 * len(subset))
val_size = len(subset) - train_size
train_dataset, val_dataset = random_split(subset, [train_size, val_size])

print("length parameters: ", len(params))
print("total parameters: ", total_params)
print("output_size: ", params[0].size())

batch_sizes = [4, 8, 16, 32] # different batch_sizes
learning_rates = [0.1, 0.01, 0.001, 0.0001] # different learning rates
num_epochs = [10, 25, 50, 100] # different number of epochs

batch_size = batch_sizes[3]
num_workers = 0

# Optimizers specified in the torch.optim package
optimizer_adam = torch.optim.Adam(conv_model.parameters(), lr=learning_rates[2])
optimizer_SGD = torch.optim.SGD(conv_model.parameters(), lr=learning_rates[2])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

num_epoch = num_epochs[3]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conv_model.to(device)
start_time = time()
training_loss_ot = [] # list for training loss over time
validation_loss_ot = [] # list for validation loss over time

# track best model
best_val_loss = float("inf")
best_model_weights = None  # Store best state_dict()

for epoch in range(num_epoch):
    epoch_time = time()
    conv_model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        # noinspection DuplicatedCode
        images, targets = images.to(device), targets.to(device)
        outputs = conv_model(images)

        rescaled_outputs = torch.mul(outputs, 128) # scale the outputs back to lab color space
        rescaled_targets = torch.mul(targets, 128) # scale the targets back to lab color space

        loss = loss_fn(rescaled_outputs, rescaled_targets)
        training_loss_ot.append(loss.item()) # loss.item() returns the value of tensor as Python number
        # Backward pass and optimization
        optimizer_adam.zero_grad()
        loss.backward()
        optimizer_adam.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{num_epoch}")
    print(f"rescaled targets: {[round(val, 4) for val in rescaled_targets[0].tolist()]}")
    print(f"rescaled outputs: {[round(val, 4) for val in rescaled_outputs[0].tolist()]}")
    print(f"Training Loss: {running_loss/len(train_loader):.4f}")

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

    avg_val_loss = val_loss / len(val_loader)

    print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
    print(f"Epoch time: {time() - epoch_time:.2f} seconds")
    print("\n")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_weights = conv_model.state_dict().copy()

if best_model_weights is not None:
    conv_model.load_state_dict(best_model_weights)
    print(f"Loaded best model weights with val loss: {best_val_loss:.4f}")

elapsed_time = time() - start_time
# noinspection PyUnboundLocalVariable
print(f"Training time: {elapsed_time:.2f} seconds")
model_path = rf"saved-models/conv_model_leakyReLU_{10}.pth"
torch.save(conv_model, model_path)