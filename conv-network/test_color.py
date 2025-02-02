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
from dataset import preprocess_image
from model import ConvModel
import shutil
import re


# load model from path with input from model.py
conv_model = ConvModel(1, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 32, 32, 2)
model_path = r"saved-models/conv_model_leakyReLU.pth"
assert os.path.isfile(model_path), f"Model file not found at {model_path}"
state_dict = torch.load(model_path, map_location='cpu')
# conv_model.load_state_dict(state_dict)
conv_model.eval()

temp_folder_images = r"F:\Projekte\bell_repo\conv_netzwerk_dataset\temp_folder_images"
temp_folder_rows = r"F:\Projekte\bell_repo\conv_netzwerk_dataset\temp_folder_rows"

def preprocess_image_rebuild():
    shutil.rmtree(temp_folder_images, ignore_errors=True)
    shutil.rmtree(temp_folder_rows, ignore_errors=True)
    os.makedirs(temp_folder_images)
    os.makedirs(temp_folder_rows)

    # use preprocess function to slice image into all possible 13x13 pixel image spaces
    if len(os.listdir(temp_folder_images)) == 0:
        preprocess_image(temp_folder_images, r"E:\Programmierung\Datein\Python\bell_repo\conv-network\cat.png")

def show_image(input_image):
    image = Image.open(input_image)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Reconstructed Image")
    plt.show()

def extract_numbers(filename):
    return [int(num) for num in re.findall(r'\d+', filename)]

# creates an LAB image from the predictions of the model and returns it
def create_image_from_predictions(input_image, prediction):
    prediction_rescaled = torch.mul(prediction, 128) # scale the outputs back to lab color space
    a, b = prediction_rescaled[0]
    a = a.detach().numpy() # converts a prediction into numpy arrays
    b = b.detach().numpy() # converts a prediction into numpy arrays
    # print(f"a: {a}, b: {b}")

    l_channel = input_image
    l_channel = l_channel

    a_channel = np.full_like(l_channel, a)
    b_channel = np.full_like(l_channel, b)

    image_pred = cv2.merge([l_channel, a_channel, b_channel])
    return image_pred

# rebuild the original image with the predicted colors of the network
# noinspection DuplicatedCode,PyTypeChecker
def rebuild_rows(
        image_to_predict: str,
        start_calc: int,
        end_calc: int,
        num_sections_per_row=487,
        target_row_width=500,
        section_size=13
) -> None:
    tensor_rows = torch.arange(start_calc, end_calc)  # Zeilennummern für die Rekonstruktion
    list_rows = tensor_rows.tolist()
    print(f"list_rows : {list_rows[:20]}")

    for i in list_rows:
        idx = i

        image_files = [f for f in os.listdir(temp_folder_images) if
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        image_files.sort(key=lambda image_file: extract_numbers(os.path.basename(image_file)))

        start_idx = idx * num_sections_per_row
        end_idx = start_idx + num_sections_per_row
        row_sections = image_files[start_idx:end_idx]

        row_images = []

        for section_file in row_sections:
            section_path = os.path.join(temp_folder_images, section_file)
            try:
                # print(f"Lade und verarbeite Bild: {section_path}")

                section_image = cv2.imread(section_path, cv2.IMREAD_GRAYSCALE)
                if section_image is None:
                    print(f"Fehler: Bild {section_path} konnte nicht gelesen werden.")
                    break

                section_tensor = torch.from_numpy(section_image).float().unsqueeze(0).unsqueeze(0)
                # print(f"Tensor erfolgreich erstellt für {section_path}")

                section_pred = conv_model(section_tensor)
                if section_pred is None:
                    print(f"Fehler: Modell liefert keine Ausgabe für {section_path}")
                    break

                section_reconstructed = create_image_from_predictions(section_image, section_pred)
                if section_reconstructed is None or section_reconstructed.size == 0:
                    print(f"Fehler: Rekonstruktion für {section_path} fehlgeschlagen.")
                    break

                row_images.append(section_reconstructed)

            except Exception as e:
                print(f"Unerwarteter Fehler beim Verarbeiten des Bildes {section_path}: {e}")
                break

        row = np.hstack(row_images)

        if row.shape[1] != target_row_width:
            row = cv2.resize(row, (target_row_width, section_size), interpolation=cv2.INTER_AREA)

        row_path = os.path.join(temp_folder_rows, f"row_{idx}.png")
        cv2.imwrite(row_path, row)

        print(f"Row {idx} reconstructed and saved as {row_path}.")


# noinspection DuplicatedCode
def rebuild_image(row_ordner):
    row_files = [f for f in os.listdir(row_ordner) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    row_files.sort(key=lambda image_file: extract_numbers(os.path.basename(image_file)))

    for index, x in enumerate(row_files):
        if index == 0:
            continue
        elif index == 1:
            image_row = cv2.imread(f"temp_folder_rows/row_{index - 1}.png")
            row_new = cv2.imread(f"temp_folder_rows/row_{index}.png")

            overlap_pixel1 = image_row[-1:, :, :]
            overlap_pixel2 = row_new[:1, :, :]
            average_overlap = (overlap_pixel1.astype(np.float64) + overlap_pixel2.astype(np.float64)) // 2
            average_overlap = average_overlap.astype(np.uint8)

            image_row = cv2.vconcat([image_row[:-1], average_overlap, row_new[1:]])

            cv2.imwrite(f"temp_folder_rows/image.png", image_row)
            os.remove(f"temp_folder_rows/row_{index}.png")
        else:
            image_row = cv2.imread(f"temp_folder_rows/image.png")
            row_new = cv2.imread(f"temp_folder_rows/row_{index}.png")

            overlap_pixel1 = image_row[-1:, :, :]
            overlap_pixel2 = row_new[:1, :, :]
            average_overlap = (overlap_pixel1.astype(np.float64) + overlap_pixel2.astype(np.float64)) // 2
            average_overlap = average_overlap.astype(np.uint8)

            image_row = cv2.vconcat([image_row[:-1], average_overlap, row_new[1:]])

            cv2.imwrite(f"temp_folder_rows/image.png", cv2.hconcat([image_row, row_new]))
            # os.remove(f"temp_folder_rows/row_{index}.png")

        print("Reconstructed image: image.png")
        show_image("temp_folder_rows/image.png")  # Jetzt erst den Ordner entfernen
        # shutil.rmtree(temp_folder_images, ignore_errors=True)
        # shutil.rmtree(temp_folder_rows, ignore_errors=True)

# preprocess_image_rebuild()

# rebuild_rows(r"E:\Programmierung\Datein\Python\bell_repo\conv-network\cat.png", 0, 100)
rebuild_rows(r"E:\Programmierung\Datein\Python\bell_repo\conv-network\cat.png", 100, 200)
# rebuild_rows(r"E:\Programmierung\Datein\Python\bell_repo\conv-network\cat.png", 200, 300)
# rebuild_rows(r"E:\Programmierung\Datein\Python\bell_repo\conv-network\cat.png", 300, 400)
# rebuild_rows(r"E:\Programmierung\Datein\Python\bell_repo\conv-network\cat.png", 400, 487)

# rebuild_image(r"temp_folder_rows")