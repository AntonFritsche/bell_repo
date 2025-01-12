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

# rebuild the original image with the predicted colors of the network
# noinspection DuplicatedCode
def rebuild_image(
        image_to_predict: str,
        model_param: callable,
        create_image_from_predictions_func: callable,
        output_file: str
) -> None:

    temp_folder_images = "temp_folder_images"
    temp_folder_rows = "temp_folder_rows"

    if not os.path.exists(temp_folder_images):
        os.makedirs(temp_folder_images)
    if not os.path.exists(temp_folder_rows):
        os.makedirs(temp_folder_rows)

    # use preprocess function to slice image into all 13x13 pixel image spaces
    preprocess_image(temp_folder_images, image_to_predict)

    list_rows =  [torch.arange(0, 488)] # a number for each row in the input image

    for i in list_rows:
        idx = i

        image_files = [f for f in os.listdir(temp_folder_images) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        image_files.sort(key=lambda x: extract_numbers(os.path.basename(x)))
        image_files = image_files[idx * 487:idx * 487 + 487]

        for index, image in enumerate(image_files):
            image_path = os.path.join(temp_folder_images, image)
            image_to_predict = cv2.imread(image_path)
            image_to_predict = cv2.cvtColor(image_to_predict, cv2.COLOR_LAB2BGR)
            image_pred = conv_model(image_to_predict)
            image_rebuild_pred = create_image_from_predictions_func(image_to_predict, image_pred)
            if index == 0:
                cv2.imwrite(f"temp_folder_rows/row_{idx}.png", image_rebuild_pred)
            else:
                row = cv2.imread(f"temp_folder_rows/row_{idx}.png")
                stacked_image = cv2.hconcat([row, image_rebuild_pred])
                cv2.imwrite(f"temp_folder_rows/row_{idx}.png", stacked_image)

    row_files = [f for f in os.listdir(temp_folder_rows) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    row_files.sort(key=lambda x: extract_numbers(os.path.basename(x)))

    for index, x in enumerate(row_files):
        if index == 0:
            continue
        elif index == 1:
            image_row = cv2.imread(f"temp_folder_rows/row_{index - 1}.png")
            row_new = cv2.imread(f"temp_folder_rows/row_{index}.png")

            image_row = cv2.hconcat([image_row, row_new])
            cv2.imwrite(f"temp_folder_rows/image.png", image_row)
            # os.remove(f"temp_folder_rows/row_{index}.png")
        else:
            image_row = cv2.imread(f"temp_folder_rows/image.png")
            row_new = cv2.imread(f"temp_folder_rows/row_{index}.png")

            cv2.imwrite(f"temp_folder_rows/image.png", cv2.hconcat([image_row, row_new]))
            # os.remove(f"temp_folder_rows/row_{index}.png")












    print(f"Reconstructed image saved to {output_file}")

# rebuild_image(r"E:\Programmierung\Datein\Python\bell_repo\conv-network\prediction_cat", conv_model, create_image_from_predictions, output_file="reconstructed_image.png")

# test image creation function
# image_path_test = "./train/sektion_0_0.png"
# image_test = cv2.imread(image_path_test)
# image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2LAB)
# image_test_pred = create_image_from_predictions(image_test, [100, 100])
# show_image(image_test_pred)

