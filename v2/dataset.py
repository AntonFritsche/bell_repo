import cv2
from torch.utils.data import Dataset
import os
import torch

class RuntimeABSectionDataset(Dataset):
    def __init__(self,
                 data_directory: str,
                 section_size: int,
                 return_centers: bool = False
                 ):
        super(RuntimeABSectionDataset, self).__init__()
        self.data_directory = data_directory
        self.section_size = section_size

        # todo: support more than one image
        files = os.listdir(self.data_directory)
        assert(len(files) == 1)
        self.image_path = os.path.join(self.data_directory, files[0])

        self.image = cv2.imread(self.image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        self.shape = self.image.shape[:2]

        self.return_centers = return_centers

    def __getitem__(self, index):
        row_idx = index // (self.shape[0] - self.section_size + 1)
        col_idx = index % (self.shape[0] - self.section_size + 1)

        center_x = row_idx + self.section_size // 2
        center_y = col_idx + self.section_size // 2

        grayscale = cv2.extractChannel(self.image, coi=0)
        data = grayscale[row_idx:(row_idx + self.section_size), col_idx:(col_idx + self.section_size)]

        a = cv2.extractChannel(self.image, 1)[center_x, center_y]
        b = cv2.extractChannel(self.image, 2)[center_x, center_y]

        data = torch.from_numpy(data).float().unsqueeze(0)
        label = torch.tensor([a, b]).float()

        if self.return_centers:
            return data, label, center_x, center_y

        return data, label

    def __len__(self):
        return (self.shape[0] - self.section_size + 1) * (self.shape[1] - self.section_size + 1)
