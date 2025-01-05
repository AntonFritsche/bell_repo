# BeLL Projekt: Python Implementierung von These

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
import model
import cv2
import os
import matplotlib.pyplot as plt

model_path = r"E:\Programmierung\Datein\Python\bell_repo\conv-network\saved-models\conv_model_2.pth"
conv_model = model.ConvModel()
conv_model.load_state_dict(torch.load(model_path, weights_only=True))

def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title("Reconstructed Image")
    plt.show()

# creates an LAB image from the predictions of the model and returns it
def create_image_from_predictions(input_image, prediction):
    prediction_rescaled = np.divide(prediction, 128) # scale the outputs back to lab color space
    a, b = prediction_rescaled

    height, width = 13, 13
    lab_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2Lab)
    l_channel, _, _ = cv2.split(lab_image)

    image = np.zeros((height, width, 3), np.uint8)

    image[:, :, 0] = l_channel
    image[:, :, 1] = a
    image[:, :, 2] = b

    return image

# LAB-Farbraum normalisieren
# def normalize_lab(lab_image):
    # L-Kanal normalisieren auf [0, 1] (0–100 wird durch 100 geteilt)
    # l_normalized = lab_image[:, :, 0] / 100.0
    # a-Kanal normalisieren auf [-1, 1] ([-128, 127] wird durch 128 geteilt)
    # a_normalized = lab_image[:, :, 1] / 128.0
    # b-Kanal normalisieren auf [-1, 1] ([-128, 127] wird durch 128 geteilt)
    # b_normalized = lab_image[:, :, 2] / 128.0

    # Ergebnis zusammenfügen
    # normalized_lab = np.dstack((l_normalized, a_normalized, b_normalized))
    # normalized_lab = torch.tensor(normalized_lab, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    # normalized_lab = normalized_lab.detach().cpu().numpy()
    # print(f"numpy array shape: {input_tensor.shape}")

    # print(f"Normalized LAB shape: {normalized_lab.shape}")
    # if len(normalized_lab.shape) != 3 or normalized_lab.shape[2] != 3:
        # raise ValueError(f"Inconsistent LAB shape after normalization: {normalized_lab.shape}")
    # return normalized_lab


# rebuild the original image with the predicted colors of the network
def rebuild_image(
        image_folder: str,
        model_param: callable,
        create_image_from_predictions_func: callable,
        output_file: str = "reconstructed_image.jpg"
) -> None:
    canvas = None
    current_row = None
    images_per_row = 487
    row_count = 0

    for idx, image_name in enumerate(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        original_height, original_width = image.shape[:2]

        input_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) # right input dimension for the model
        # print(f"Input tensor shape: {input_tensor.shape}")
        print(f"first input tensor shape: {input_tensor.shape}") # shape: [1, 3, 13, 13]
        # input_tensor = normalize_lab(input_tensor)
        # print(f"Normalized LAB image shape: {input_tensor.shape}")
        input_tensor = input_tensor.clone().detach().float()
        print(f"second input tensor shape: {input_tensor.shape}") # shape: [1, 3, 13, 13]

        with torch.no_grad():
            output = model_param(input_tensor)
            if output is None or output.numel() == 0:
                print(f"Error: Model provided no output for {image_name}")
                continue

        scaled_output = torch.mul(output, 128) # scale the outputs back to lab color space
        print(scaled_output)
        if output is None or torch.isnan(scaled_output).any():
            print(f"Error: Invalid output from the model for {image_name}")
            continue

        pred_image = create_image_from_predictions_func(image, output.squeeze(0).cpu().numpy())
        print(f"third input tensor shape: {input_tensor.shape}") # shape: [1, 3, 13, 13]
        if pred_image is None:
            print(f"Error: Invalid image from function for {image_name}")
            continue

        show_image(pred_image)

        # Resize predicted image to match the original image size (original_width, original_height)
        pred_image_resized = cv2.resize(pred_image, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        print(f"fourth input tensor shape: {input_tensor.shape}") # shape: [1, 3, 13, 13]
        # print(f"Original image size: {image.shape}")
        # print(f"Predicted image size after resize: {pred_image_resized.shape}")

        show_image(pred_image_resized)
        if current_row is None:
            try:
                current_row = np.hstack((current_row, pred_image_resized))
            except ValueError as e:
                print(f"Error merging images at index {idx}: {e}")
                break
        else:
            current_row = np.hstack((current_row, pred_image_resized))

        print(f"Processed image {idx + 1}/{len(os.listdir(image_folder))}: {image_name}")

        if (idx + 1) % images_per_row == 0:
            if canvas is None:
                canvas = current_row
            else:
                canvas = np.vstack((canvas, current_row))

            current_row = None
            row_count += 1
            print(f"Completed row {row_count}...")

    if current_row is not None:
        if canvas is None:
            canvas = current_row
        else:
            canvas = np.vstack((canvas, current_row))

    cv2.imwrite(output_file, canvas)
    print(f"Reconstructed image saved to {output_file}")

rebuild_image(r"E:\Programmierung\Datein\Python\bell_repo\conv-network\prediction_cat", conv_model, create_image_from_predictions, output_file="reconstructed_image_cat.jpg")

