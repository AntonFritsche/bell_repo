import cv2
from torch.utils.data import Dataset
import os
import torch
import numpy as np

class RuntimeABSectionDataset(Dataset):
    def __init__(self,
                 data_directory: str,
                 section_size: int,
                 return_centers: bool = False,
                 ):
        super(RuntimeABSectionDataset, self).__init__()
        self.data_directory = data_directory
        self.section_size = section_size
        self.files = os.listdir(self.data_directory)
        self.return_centers = return_centers

        self.images = []
        self.image_shapes = []

        for fname in self.files:
            self.image_path = os.path.join(self.data_directory, fname)
            image = cv2.imread(self.image_path)
            image = cv2.resize(image, (500, 500))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0

            self.images.append(image)
            self.image_shapes.append(image.shape[:2])

        self.patch_positions = []
        for img_idx, shape in enumerate(self.image_shapes):
            rows = shape[0] - section_size + 1
            cols = shape[1] - section_size + 1
            for r in range(rows):
                for c in range(cols):
                    self.patch_positions.append((img_idx, r, c))

    def __getitem__(self, index):
        img_idx, row_idx, col_idx = self.patch_positions[index]
        image = self.images[img_idx]

        center_x = row_idx + self.section_size // 2
        center_y = col_idx + self.section_size // 2

        grayscale = cv2.extractChannel(image, coi=0)
        data = grayscale[row_idx:(row_idx + self.section_size), col_idx:(col_idx + self.section_size)]

        a = cv2.extractChannel(image, 1)[center_x, center_y]
        b = cv2.extractChannel(image, 2)[center_x, center_y]

        data = torch.from_numpy(data).float().unsqueeze(0)
        label = torch.tensor([a, b]).float()

        if self.return_centers:
            return data, label, center_x, center_y
        else:
            return data, label

    def __len__(self):
        return len(self.patch_positions)
