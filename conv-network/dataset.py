import os
import shutil
import random
from PIL import Image
import csv
import re
import cv2

def create_datasets(source_folder, dest_folder, test_size=200):
    train_folder = os.path.join(dest_folder, 'train')
    test_folder = os.path.join(dest_folder, 'test')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    categories = [folder for folder in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, folder))]

    test_images = []
    train_images = []

    for category in categories:
        category_path = os.path.join(source_folder, category)
        images = [os.path.join(category_path, img) for img in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, img))]

        random.shuffle(images)

        test_images.extend(images[:min(test_size // len(categories), len(images))])

        train_images.extend(images[min(test_size // len(categories), len(images)):])

    for img in test_images:
        category_name = os.path.basename(os.path.dirname(img))
        dest_category_folder = os.path.join(test_folder, category_name)
        os.makedirs(dest_category_folder, exist_ok=True)
        shutil.copy(img, dest_category_folder)

    for img in train_images:
        category_name = os.path.basename(os.path.dirname(img))
        dest_category_folder = os.path.join(train_folder, category_name)
        os.makedirs(dest_category_folder, exist_ok=True)
        shutil.copy(img, dest_category_folder)

source_folder = "../Bilder_Kolorierung_dataset/natural_images_color"
conv_network_folder = "../conv-network"
# create_datasets(source_folder, conv_network_folder)

def convert_images_to_grayscale(folder_path):
    images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, img))]

    for image_path in images:
        try:
            with Image.open(image_path) as img:
                grayscale_img = img.convert("L")
                grayscale_img.save(image_path)
        except Exception as e:
            print(f"Fehler beim Konvertieren von {image_path}: {e}")

greyscale_folder = "./test"
# convert_images_to_grayscale(greyscale_folder)

def create_csv_from_dataset(folder_path, csv_path):
    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["filename", "label"])

        for category in os.listdir(folder_path):
            category_path = os.path.join(folder_path, category)
            if os.path.isdir(category_path):
                for idx, img_name in enumerate(os.listdir(category_path)):
                    if os.path.isfile(os.path.join(category_path, img_name)):
                        label = category
                        filename = f"{label}_{idx}"
                        csv_writer.writerow([filename, label])

train_csv_path = os.path.join(conv_network_folder, 'train.csv')
test_csv_path = os.path.join(conv_network_folder, 'test.csv')
# create_csv_from_dataset(os.path.join(conv_network_folder, 'train'), train_csv_path)
# create_csv_from_dataset(os.path.join(conv_network_folder, 'test'), test_csv_path)

def resize_images(folder_path, size=(500, 500)):
    images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) 
              if os.path.isfile(os.path.join(folder_path, img))]

    for image_path in images:
        try:
            with Image.open(image_path) as img:
                filename = os.path.basename(image_path)
                cleared_filename = re.sub(r'[^a-zA-Z0-9]', '', filename.split('.')[0])  # Unerwünschte Zeichen entfernen
                
                resized_image = img.resize(size)
                
                save_path = os.path.join(folder_path, f"{cleared_filename}.jpg")
                resized_image.save(save_path)
                print(f"Bild resized und gespeichert: {save_path}")
        except Exception as e:
            print(f"Fehler beim Resizen von {image_path}: {e}")

train_folder = "./train"
test_folder = "./test"
resize_images(train_folder)
resize_images(test_folder)