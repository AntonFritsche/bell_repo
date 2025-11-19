import os, torch
import numpy as np

from torch.utils.data import Dataset


class Section_Dataset(Dataset):
    def __init__(self, data_directory: str):
        self.image_paths = [os.path.join(data_directory, f) for f in os.listdir(data_directory)]

    def __getitem__(self, index):
        data = np.load(self.image_paths[index], allow_pickle=True).item()
        patch = torch.from_numpy(data["patch"]).unsqueeze(0).float()
        label = torch.from_numpy(data["label"]).float()
        return patch, label

    def __len__(self):
        return len(self.image_paths)
