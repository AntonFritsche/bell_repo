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
from dataset import extract_numbers
from dataset import preprocess_image

model_path = r"saved-models/conv_model_leakyReLU.pth"
conv_model = torch.load(model_path, weights_only=False)

def show_image(input_image):
    plt.imshow(input_image, cmap='gray')
    plt.axis('off')
    plt.title("Reconstructed Image")
    plt.show()

# creates an LAB image from the predictions of the model and returns it
def create_image_from_predictions(input_image, prediction):
    prediction_rescaled = np.multiply(prediction, 128) # scale the outputs back to lab color space
    a, b = prediction_rescaled

    height, width = 13, 13
    lab_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2Lab)
    l_channel, _, _ = cv2.split(lab_image)

    image_pred = np.zeros((height, width, 3), np.uint8)

    image_pred[:, :, 0] = l_channel
    image_pred[:, :, 1] = a
    image_pred[:, :, 2] = b

    return image_pred

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
        image_to_predict: str,
        model_param: callable,
        create_image_from_predictions_func: callable,
        output_file: str
) -> None:
    images_per_row = 487
    row_count = 0

    temp_folder = "temp_folder"
    preprocess_image(temp_folder, image_to_predict)

    image_files = [f for f in os.listdir(temp_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    image_files.sort(key=lambda x: extract_numbers(os.path.basename(x)))

    for image_name in image_files:
        if not os.path.isfile("reconstructed_image.png"):
            image_path = os.path.join(image_to_predict, image_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

            original_height, original_width = image.shape[:2]
            print(f"Original image shape: {image.shape}")

            input_tensor = torch.tensor(image, dtype=torch.float32) # right input dimension for the model
            print(f"input tensor shape: {input_tensor.shape}") # shape: [1, 3, 13, 13]

            with torch.no_grad():
                output = model_param(input_tensor)
                if output is None or output.numel() == 0:
                    print(f"Error: Model provided no output for {image_name}")
                    continue
            print(f"output tensor shape: {output.shape}")
            scaled_output = torch.mul(output, 128) # scale the outputs back to lab color space
            print(scaled_output[:1])

            pred_image = create_image_from_predictions_func(image, output.squeeze(0).cpu().numpy())
            show_image(pred_image)

            # Resize predicted image to match the original image size (original_width, original_height)
            pred_image_resized = cv2.resize(pred_image, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

            # show_image(pred_image_resized)
            cv2.imwrite(os.path.join(temp_folder, image_name), pred_image_resized)
        else:
            image_path = os.path.join(image_to_predict, image_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

            original_height, original_width = image.shape[:2]
            print(f"Original image shape: {image.shape}")

            input_tensor = torch.tensor(image, dtype=torch.float32)  # right input dimension for the model
            print(f"input tensor shape: {input_tensor.shape}")  # shape: [1, 3, 13, 13]

            with torch.no_grad():
                output = model_param(input_tensor)
                if output is None or output.numel() == 0:
                    print(f"Error: Model provided no output for {image_name}")
                    continue
            print(f"output tensor shape: {output.shape}")
            scaled_output = torch.mul(output, 128)  # scale the outputs back to lab color space
            print(scaled_output[:1])

            pred_image = create_image_from_predictions_func(image, output.squeeze(0).cpu().numpy())
            show_image(pred_image)

            # Resize predicted image to match the original image size (original_width, original_height)
            pred_image_resized = cv2.resize(pred_image, (original_width, original_height),
                                            interpolation=cv2.INTER_LINEAR)

            # show_image(pred_image_resized)
            cv2.imwrite(os.path.join(temp_folder, image_name), pred_image_resized)


    print(f"Reconstructed image saved to {output_file}")

rebuild_image(r"E:\Programmierung\Datein\Python\bell_repo\conv-network\prediction_cat", conv_model, create_image_from_predictions, output_file="reconstructed_image.png")

# test image creation function
# image_path_test = "./train/sektion_0_0.png"
# image_test = cv2.imread(image_path_test)
# image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2LAB)
# image_test_pred = create_image_from_predictions(image_test, [100, 100])
# show_image(image_test_pred)

