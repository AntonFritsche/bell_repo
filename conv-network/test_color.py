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
import shutil
from model import ConvModel
from PIL import Image

# load model from path with input from model.py
conv_model = ConvModel()
model_path = r"saved-models/conv_model_leakyReLU.pth"
assert os.path.isfile(model_path), f"Model file not found at {model_path}"
state_dict = torch.load(model_path, map_location='cpu')
# conv_model.load_state_dict(state_dict)
conv_model.eval()

temp_folder_images = "temp_folder_images/"
temp_folder_rows = "temp_folder_rows/"

#shutil.rmtree(temp_folder_images, ignore_errors=True)
#shutil.rmtree(temp_folder_rows, ignore_errors=True)
#os.makedirs(temp_folder_images)
#os.makedirs(temp_folder_rows)

def show_image(input_image):
    image = Image.open(input_image)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Reconstructed Image")
    plt.show()

# creates an LAB image from the predictions of the model and returns it
def create_image_from_predictions(input_image, prediction):
    prediction_rescaled = np.multiply(prediction, 128) # scale the outputs back to lab color space
    a, b = prediction_rescaled

    height, width = 13, 13
    l_channel, _, _ = cv2.split(input_image)

    image_pred = np.zeros((height, width, 3), np.uint8)

    image_pred[:, :, 0] = l_channel
    image_pred[:, :, 1] = a
    image_pred[:, :, 2] = b

    return image_pred

# rebuild the original image with the predicted colors of the network
# noinspection DuplicatedCode,PyTypeChecker
def rebuild_image(
        image_to_predict: str,
        create_image_from_predictions_func: callable,
) -> None:

    temp_folder_images = "temp_folder_images/"
    temp_folder_rows = "temp_folder_rows/"

    # use preprocess function to slice image into all possible 13x13 pixel image spaces
    if len(os.listdir(temp_folder_images)) == 0:
        preprocess_image(temp_folder_images, image_to_predict)

    list_rows =  [torch.arange(0, 488)] # a number for each row in the input image

    for i in list_rows:
        idx = list_rows.index(i)
        # print(f"shape of i: {i.shape}")
        # print(f"shape of idx: {idx.shape}")

        image_files = [f for f in os.listdir(temp_folder_images) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        image_files.sort(key=lambda image_file: extract_numbers(os.path.basename(image_file)))
        index_part_one = idx * 487
        index_part_two = idx * 487 + 487
        image_files = image_files[index_part_one:index_part_two]

        for index, image in enumerate(image_files):
            image_path = os.path.join(temp_folder_images, image)

            # access image with opencv
            image_to_predict = cv2.imread(image_path)
            image_to_predict = cv2.cvtColor(image_to_predict, cv2.COLOR_RGB2LAB)
            image_to_predict = image_to_predict[:, :, 0]
            image_to_predict = torch.from_numpy(image_to_predict).float() # convert numpy image into torch tensor

            image_to_predict = image_to_predict.unsqueeze(0).unsqueeze(0)  # [1, 1, 13, 13]
            print(image_to_predict.shape)
            # image_to_predict = image_to_predict.permute(2, 0, 1).unsqueeze(0) # rearrange the dimension: [batch_size, channels, height, width]

            # predict image and create image with predicted values
            image_pred = conv_model(image_to_predict)
            image_rebuild_pred = create_image_from_predictions_func(image_to_predict, image_pred)

            if index == 0:
                cv2.imwrite(f"temp_folder_rows/row_{idx}.png", image_rebuild_pred)
            else:
                row = cv2.imread(f"temp_folder_rows/row_{idx}.png")

                overlap_pixel1 = row[:, -1:, :]
                overlap_pixel2 = image_rebuild_pred[:, :1, :]
                average_overlap = (overlap_pixel1.astype(np.float32) + overlap_pixel2.astype(np.float32)) // 2
                average_overlap = average_overlap.astype(np.uint8)

                stacked_image = cv2.hconcat([row[:, :-1], average_overlap, image_rebuild_pred[:, 1:]])
                cv2.imwrite(f"temp_folder_rows/row_{idx}.png", stacked_image)

    row_files = [f for f in os.listdir(temp_folder_rows) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    row_files.sort(key=lambda image_file: extract_numbers(os.path.basename(image_file)))

    for index, x in enumerate(row_files):
        if index == 0:
            continue
        elif index == 1:
            image_row = cv2.imread(f"temp_folder_rows/row_{index - 1}.png")
            row_new = cv2.imread(f"temp_folder_rows/row_{index}.png")

            overlap_pixel1 = image_row[-1:, :, :]
            overlap_pixel2 = row_new[:1, :, :]
            average_overlap = (overlap_pixel1.astype(np.float32) + overlap_pixel2.astype(np.float32)) // 2
            average_overlap = average_overlap.astype(np.uint8)

            image_row = cv2.vconcat([image_row[:-1], average_overlap, row_new[1:]])

            cv2.imwrite(f"temp_folder_rows/image.png", image_row)
            os.remove(f"temp_folder_rows/row_{index}.png")
        else:
            image_row = cv2.imread(f"temp_folder_rows/image.png")
            row_new = cv2.imread(f"temp_folder_rows/row_{index}.png")

            overlap_pixel1 = image_row[-1:, :, :]
            overlap_pixel2 = row_new[:1, :, :]
            average_overlap = (overlap_pixel1.astype(np.float32) + overlap_pixel2.astype(np.float32)) // 2
            average_overlap = average_overlap.astype(np.uint8)

            image_row = cv2.vconcat([image_row[:-1], average_overlap, row_new[1:]])

            cv2.imwrite(f"temp_folder_rows/image.png", cv2.hconcat([image_row, row_new]))
            os.remove(f"temp_folder_rows/row_{index}.png")

        print("Reconstructed image: image.png")
        show_image("temp_folder_rows/image.png")  # Jetzt erst den Ordner entfernen
        # shutil.rmtree(temp_folder_images, ignore_errors=True)
        # shutil.rmtree(temp_folder_rows, ignore_errors=True)

rebuild_image(r"E:\Programmierung\Datein\Python\bell_repo\conv-network\cat.png", conv_model)
