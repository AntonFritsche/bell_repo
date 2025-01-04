import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torch.utils.data import Subset
import numpy as np
import pandas as pd
import model
from dataset import preprocess_image
import cv2
import os
import datetime
from torchvision.io import read_image

conv_model = model.ConvModel()
conv_model.load_state_dict(torch.load("saved-models", weights_only=True))
print("test")
# creates an LAB image from the predictions of the model and returns it
def create_image_from_predictions(input_image, prediction):
    a, b = prediction

    height, width = 13, 13
    lab_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2Lab)
    l_channel, _, _ = cv2.split(lab_image)

    image = np.zeros((height, width, 3), np.uint8)

    image[:, :, 0] = l_channel
    image[:, :, 1] = a * 128
    image[:, :, 2] = b * 128

    return image




def rebuild_image(image_folder, model_param, create_image_from_predictions_func, output_file="reconstructed_image.jpg"):
    # Initialize variables for the final canvas
    canvas = None
    current_row = None
    images_per_row = 487
    row_count = 0

    # Loop through all images in the folder
    for idx, image_name in enumerate(os.listdir(image_folder)):
        # Load the image
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)

        # Use the model to predict and create the output image
        output = model_param(image)
        pred_image = create_image_from_predictions_func(image, output)

        # Add the predicted image to the current row
        if current_row is None:
            current_row = pred_image  # Start a new row
        else:
            current_row = np.hstack((current_row, pred_image))  # Append horizontally

        # When the row is complete, add it to the canvas
        if (idx + 1) % images_per_row == 0:
            if canvas is None:
                canvas = current_row  # First row initializes the canvas
            else:
                canvas = np.vstack((canvas, current_row))  # Append the row vertically

            # Reset the current row
            current_row = None
            row_count += 1
            print(f"Completed row {row_count}...")

    # Add any remaining row to the canvas
    if current_row is not None:
        if canvas is None:
            canvas = current_row
        else:
            canvas = np.vstack((canvas, current_row))

    # Save and display the final reconstructed image
    cv2.imwrite(output_file, canvas)
    print(f"Reconstructed image saved to {output_file}")

rebuild_image("prediction_cat", conv_model, create_image_from_predictions, output_file="reconstructed_image_cat.jpg")

