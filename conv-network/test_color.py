# BeLL Development Projekt

import os
import cv2
import matplotlib.pyplot as plt
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, random_split
# from torch.utils.data import Dataset
# from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter
# from PIL import Image
# from torch.utils.data import Subset
# import datetime
# from torchvision.io import read_image
# import pandas as pd
# from dataset import preprocess_image
import numpy as np
import torch as torch
from PIL import Image
from model import ConvModel
import shutil
import re
from numpy import asarray


# load model from path with input from model.py
conv_model = ConvModel(1, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 32, 32, 2)
model_path = r"saved-models/conv_model_leakyReLU_0.pth"
assert os.path.isfile(model_path), f"Model file not found at {model_path}"
state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
conv_model.eval()

temp_folder_images = r"F:\Projekte\bell_repo\conv_netzwerk_dataset\temp_folder_images"
temp_folder_rows = r"F:\Projekte\bell_repo\conv_netzwerk_dataset\temp_folder_rows"

def preprocess_image_rebuild():
    # shutil.rmtree(temp_folder_images, ignore_errors=True)
    # shutil.rmtree(temp_folder_rows, ignore_errors=True)
    os.makedirs(temp_folder_images)
    os.makedirs(temp_folder_rows)
    from dataset import preprocess_image

    if len(os.listdir(temp_folder_images)) == 0:
        preprocess_image(temp_folder_images, r"E:\Programmierung\Datein\Python\bell_repo\conv-network\cat.png")

def show_image(input_image):
    image = cv2.imread(input_image)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Reconstructed Image")
    plt.show()

def extract_numbers(filename):
    numbers = re.findall(r'\d+', filename)
    return tuple(map(int, numbers)) if numbers else (0,)

def create_pxl_from_preds(input_image, prediction):
    prediction_rescaled = torch.mul(prediction, 128)
    a, b = prediction_rescaled[0]
    a = a.detach().numpy()
    b = b.detach().numpy()

    l_channel = input_image[6, 6]

    a_channel = np.full((1, 1), a, dtype=np.float32)
    b_channel = np.full((1, 1), b, dtype=np.float32)
    l_channel = np.full((1, 1), l_channel, dtype=np.float32)

    l_channel = (l_channel / 255 * 100).astype(np.float32)
    a_channel = a_channel.astype(np.float32)
    b_channel = b_channel.astype(np.float32)

    image_pred = cv2.merge([l_channel, a_channel, b_channel])
    # print(image_pred[:1])

    return image_pred

# noinspection DuplicatedCode
def rebuild_image_pxl_row(
        start_calc: int,
        end_calc: int,
        num_sections_per_row=487,
) -> None:
    tensor_rows = torch.arange(start_calc, end_calc)
    list_rows = tensor_rows.tolist()
    add_index = 0
    for i in list_rows:
        idx = i

        image_files = [f for f in os.listdir(temp_folder_images) if
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        image_files.sort(key=extract_numbers)

        start_idx = idx * num_sections_per_row
        start_idx += add_index

        end_idx = start_idx + num_sections_per_row
        row_sections = image_files[start_idx:end_idx]
        print(f"start idx: {start_idx}, end idx: {end_idx}")

        row_images = []

        for index, section_file in enumerate(row_sections):
            section_path = os.path.join(temp_folder_images, section_file)

            section_image = cv2.imread(section_path, cv2.IMREAD_GRAYSCALE)
            if section_image is None:
                print(f"Fehler: Bild {section_path} konnte nicht gelesen werden.")
                break

            section_tensor = torch.from_numpy(section_image).float().unsqueeze(0).unsqueeze(0)

            section_pred = conv_model(section_tensor)
            if section_pred is None:
                print(f"Fehler: Modell liefert keine Ausgabe für {section_path}")
                break

            section_reconstructed = create_pxl_from_preds(section_image, section_pred)
            if section_reconstructed is None or section_reconstructed.size == 0:
                print(f"Fehler: Rekonstruktion für {section_path} fehlgeschlagen.")
                break

            row_images.append(section_reconstructed)

        row = np.hstack(row_images)

        row_path = os.path.join(temp_folder_rows, f"row_{idx}.png")
        cv2.imwrite(row_path, row)

        add_index += 1
        print(f"Row {idx} reconstructed and saved as {row_path}.")

# noinspection PyTypeChecker,DuplicatedCode
def rebuild_image_pxl(row_ordner, target_height=487):
    row_files = [f for f in os.listdir(row_ordner) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    row_files.sort(key=extract_numbers)
    print(row_files[:20])

    row_files = row_files[:target_height]

    all_rows = []

    for index, row_file in enumerate(row_files):
        row_path = os.path.join(row_ordner, row_file)
        row_image = cv2.imread(row_path).astype(np.float32)
        row_image /= 255
        row_image = cv2.cvtColor(row_image, cv2.COLOR_BGR2LAB)

        if row_image is None:
            print(f"Fehler: Konnte {row_path} nicht lesen.")
            continue

        all_rows.append(row_image)

    final_image = cv2.vconcat(all_rows)
    final_image = np.fliplr(final_image)
    final_image = cv2.rotate(final_image, cv2.ROTATE_90_CLOCKWISE)
    final_image = cv2.cvtColor(final_image, cv2.COLOR_LAB2BGR)
    final_image *= 255.0
    cv2.imwrite("image.png", final_image)

    print("Reconstructed image: image.png")
    show_image("image.png")

# preprocess_image_rebuild()
# rebuild_image_pxl_row(0, 488)
# rebuild_image_pxl(temp_folder_rows)