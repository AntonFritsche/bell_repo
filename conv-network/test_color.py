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
from torchvision import transforms


# load model from path with input from model.py
conv_model = ConvModel(1, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 32, 32, 2)
model_path = r"saved-models/conv_model_leakyReLU_2.pth"
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
    numbers = re.findall(r'\d+', filename)
    return tuple(map(int, numbers)) if numbers else (0,)

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

def create_pxl_from_preds(input_image, prediction):
    prediction_rescaled = torch.mul(prediction, 128)
    a, b = prediction_rescaled[0]
    a = a.detach().numpy()
    b = b.detach().numpy()

    l_channel = input_image[6, 6]
    a_channel = np.full_like(l_channel, a)
    b_channel = np.full_like(l_channel, b)

    l_channel = np.expand_dims(l_channel, axis=-1)
    a_channel = np.expand_dims(a_channel, axis=-1)
    b_channel = np.expand_dims(b_channel, axis=-1)

    l_channel = l_channel.astype(np.float32)
    a_channel = a_channel.astype(np.float32)
    b_channel = b_channel.astype(np.float32)

    # print("Shapes:", l_channel.shape, a_channel.shape, b_channel.shape)
    image_pred = cv2.merge([l_channel, a_channel, b_channel])

    return image_pred

# rebuild the original image with the predicted colors of the network
# noinspection DuplicatedCode,PyTypeChecker,PyUnboundLocalVariable
def rebuild_rows(
        start_calc: int,
        end_calc: int,
        num_sections_per_row=487,
) -> None:
    tensor_rows = torch.arange(start_calc, end_calc)  # Zeilennummern für die Rekonstruktion
    list_rows = tensor_rows.tolist()
    # print(f"list_rows : {list_rows[:20]}")

    for i in list_rows:
        idx = i

        image_files = [f for f in os.listdir(temp_folder_images) if
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        image_files.sort(key=extract_numbers)


        start_idx = idx * num_sections_per_row
        end_idx = start_idx + num_sections_per_row
        row_sections = image_files[start_idx:end_idx]

        row_images = []

        for index, section_file in enumerate(row_sections):
            section_path = os.path.join(temp_folder_images, section_file)

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

            if index == 486:
                section_reconstructed_left = section_reconstructed[:, :12]
                section_reconstructed_right = section_reconstructed[:, -1:]
                # print(section_reconstructed.shape)
                row_images.append(section_reconstructed_left)
            else:
                section_reconstructed = section_reconstructed[:, :0]
                # print(section_reconstructed.shape)
                row_images.append(section_reconstructed)

        row = np.hstack(row_images)
        row = np.hstack((row, section_reconstructed_right))

        row_path = os.path.join(temp_folder_rows, f"row_{idx}.png")
        cv2.imwrite(row_path, row)

        print(f"Row {idx} reconstructed and saved as {row_path}.")


# noinspection DuplicatedCode,PyTypeChecker,PyUnboundLocalVariable
def rebuild_image(row_ordner, target_height=500):
    row_files = [f for f in os.listdir(row_ordner) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    row_files.sort(key=lambda image_file: extract_numbers(os.path.basename(image_file)))

    # Begrenze die Zeilen auf max. 500
    row_files = row_files[:target_height]

    all_rows = []

    for index, row_file in enumerate(row_files):
        row_path = os.path.join(row_ordner, row_file)
        row_image = cv2.imread(row_path)

        if row_image is None:
            print(f"Fehler: Konnte {row_path} nicht lesen.")
            continue

        if index == 486:
            row_image_up = row_image[:12, :]
            row_image_down = row_image[-1:, :]
            all_rows.append(row_image_up)
        else:
            row_image = row_image[:0, :]
            all_rows.append(row_image)

    final_image = cv2.vconcat(all_rows)
    final_image = cv2.vconcat((final_image, row_image_down))

    cv2.imwrite("image.png", final_image)
    print("Reconstructed image: image.png")
    show_image("image.png")
    # shutil.rmtree(temp_folder_images, ignore_errors=True)
    # shutil.rmtree(temp_folder_rows, ignore_errors=True)

# preprocess_image_rebuild()
# rebuild_rows(0, 488)
# rebuild_image(temp_folder_rows)


# noinspection DuplicatedCode
def rebuild_image_pxl_row(
        start_calc: int,
        end_calc: int,
        num_sections_per_row=487,
) -> None:
    tensor_rows = torch.arange(start_calc, end_calc)  # Zeilennummern für die Rekonstruktion
    list_rows = tensor_rows.tolist()
    # print(f"list_rows : {list_rows[:20]}")
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
            # print(section_tensor.shape)
            # print(f"Tensor erfolgreich erstellt für {section_path}")

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

    # Begrenze die Zeilen auf max. 500
    row_files = row_files[:target_height]

    all_rows = []

    for index, row_file in enumerate(row_files):
        row_path = os.path.join(row_ordner, row_file)
        row_image = cv2.imread(row_path)

        if row_image is None:
            print(f"Fehler: Konnte {row_path} nicht lesen.")
            continue

        all_rows.append(row_image)

    final_image = cv2.vconcat(all_rows)
    final_image = np.fliplr(final_image)

    cv2.imwrite("image.png", final_image)

    final_image = Image.open("image.png")
    final_image = final_image.rotate(90)
    final_image.save("image.png")

    print("Reconstructed image: image.png")
    show_image("image.png")
    # shutil.rmtree(temp_folder_images, ignore_errors=True)
    # shutil.rmtree(temp_folder_rows, ignore_errors=True)

# rebuild_image_pxl_row(0, 488)
rebuild_image_pxl(temp_folder_rows)
