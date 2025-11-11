import cv2
from PIL.ImageOps import grayscale
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import random

class Section_Dataset(Dataset):
    def __init__(self, data_directory: str, section_size: int, patches_per_image=300):
        self.image_paths = [os.path.join(data_directory, f) for f in os.listdir(data_directory)]
        self.section_size = section_size
        self.patches_per_image = patches_per_image
        self.num_patches = len(self.image_paths) * patches_per_image

    def __getitem__(self, index):
        img_idx = index // self.patches_per_image
        image = cv2.imread(self.image_paths[img_idx])
        image = cv2.resize(image, (500, 500))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        image[:, :, 0] = image[:, :, 0] / 255.0
        image[:, :, 1] = (image[:, :, 1] - 128.0) / 128.0
        image[:, :, 2] = (image[:, :, 2] - 128.0) / 128.0

        h, w, _ = image.shape
        r = random.randint(0, h - self.section_size)
        c = random.randint(0, w - self.section_size)

        center_x = r + self.section_size // 2
        center_y = c + self.section_size // 2

        grayscale = cv2.extractChannel(image, coi=0)
        data = grayscale[r:r+self.section_size, c:c+self.section_size]

        a = cv2.extractChannel(image, coi=1)[center_x, center_y]
        b = cv2.extractChannel(image, coi=2)[center_x, center_y]

        data = torch.from_numpy(data).float().unsqueeze(0)
        label = torch.tensor([a, b]).float()

        return data, label

    def __len__(self):
        return self.num_patches
