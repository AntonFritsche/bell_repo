# tuto: https://github.com/VidushiBhatia/U-Net-Implementation/blob/main/U_Net_for_Image_Segmentation_From_Scratch_Using_TensorFlow_v4.ipynb

import tensorflow as tf
import keras
# from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import Dropout 
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import Conv2DTranspose
# from tensorflow.keras.layers import concatenate
# from tensorflow.keras.losses import binary_crossentropy
# from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import cv2
    

def LoadData (path1, path2, path3):
    # Read the images folder like a list
    image_dataset_1 = os.listdir(path1)
    image_dataset_2 = os.listdir(path2)
    image_dataset_3 = os.listdir(path3)

    # create list for the images
    img_list = []

    img_list_1 = []
    img_list_2 = []
    img_list_3 = []

    # loop through the images in the list and add them to the img_list array
    for file in image_dataset_1:
        img_list.append(file)
        img_list_1.append(file)
    
    for file in image_dataset_2:
        img_list.append(file)
        img_list_2.append(file)

    for file in image_dataset_3:
        img_list.append(file)
        img_list_3.append(file)

    len1 = len(img_list_1)
    len2 = len(img_list_2)
    len3 = len(img_list_3)

    #return img_list array
    return img_list, len1, len2, len3

def PreprocessData(img, mask, target_shape_img, target_shape_mask, path1, path2, path3, len1, len2):
    """
    Processes the images and mask present in the shared list and path
    Returns a NumPy dataset with images as 3-D arrays of desired size
    Please note the masks in this dataset have only one channel
    """

    # Pull the relevant dimensions for image and mask
    m = len(img)                     # number of images
    i_h,i_w,i_c = target_shape_img   # pull height, width, and channels of image
    m_h,m_w,m_c = target_shape_mask  # pull height, width, and channels of mask
    
    # Define X and Y as number of images along with shape of one image
    X = np.zeros((m,i_h,i_w,i_c), dtype=np.float32)
    
    # Resize images and masks
    for file in img:
        # convert image into an array of desired shape (3 channels)
        index = img.index(file)
        if index < len1:
            path = os.path.join(path1, file)
        elif index >= len1 and index < (len1 + len2):
            path = os.path.join(path2, file)
        elif index >= (len1 + len2):
            path = os.path.join(path3, file)

        single_img = Image.open(path).convert('L')
        single_img = single_img.resize((i_h,i_w))
        single_img = np.reshape(single_img,(i_h,i_w,i_c)) 
        single_img = single_img/256.
        X[index] = single_img
        
    return X