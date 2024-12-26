import os
import shutil
import random
from PIL import Image
import csv
import re
import cv2
import pandas as pd

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
# resize_images(train_folder)
# resize_images(test_folder)
# resize_images("../conv-network")

def rename_images(folder_path, folder_name):
    counter = 1
    
    images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) 
              if os.path.isfile(os.path.join(folder_path, img))]

    for image_path in images:
        try:
            with Image.open(image_path) as img:
                if folder_name == "train":
                    filename = os.path.basename(image_path)

                    clean_name = re.sub(r'\d+_', '', filename).lower()

                    if clean_name.startswith("car"):
                        image_name = f"car_{counter}"
                    elif clean_name.startswith("flower"):
                        image_name = f"flower_{counter}"
                    elif clean_name.startswith("dog"):
                        image_name = f"dog_{counter}"
                    elif clean_name.startswith("cat"):
                        image_name = f"cat_{counter}"
                    else:
                        image_name = f"unknown_{counter}"

                    save_path = os.path.join(folder_path, f"{image_name}.jpg")
                    img.save(save_path)
                    print(f"Bild umbenannt und gespeichert: {save_path}")
                    counter += 1
                elif folder_name == "test":
                    filename = os.path.basename(image_path)

                    clean_name = re.sub(r'\d+', '', filename).lower()

                    if clean_name.startswith("car"):
                        image_name = f"car_{counter}"
                    elif clean_name.startswith("flower"):
                        image_name = f"flower_{counter}"
                    elif clean_name.startswith("dog"):
                        image_name = f"dog_{counter}"
                    elif clean_name.startswith("cat"):
                        image_name = f"cat_{counter}"
                    elif clean_name.startswith("airplane"):
                        image_name = f"airplane_{counter}"
                    else:
                        image_name = f"unknown_{counter}"

                    save_path = os.path.join(folder_path, f"{image_name}.jpg")
                    img.save(save_path)
                    print(f"Bild umbenannt und gespeichert: {save_path}")
                    counter += 1
                else:
                    return print("Error: wrong folder name")
        except Exception as e:
            print(f"Fehler beim Bearbeiten von {image_path}: {e}")

image_train_folder = "./train"
image_test_folder = "./test"
# rename_images(image_train_folder, "train")
# rename_images(image_test_folder, "test")

def resize_images_in_folders(base_folder):
    target_size = (500, 500)
    folders = ['train', 'test']

    for folder in folders:
        folder_path = os.path.join(base_folder, folder)
        if not os.path.exists(folder_path):
            print(f"Ordner nicht gefunden: {folder_path}")
            continue

        print(f"\n--- Bearbeite Bilder im Ordner: {folder} ---")
        images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) 
                  if os.path.isfile(os.path.join(folder_path, img))]

        for image_path in images:
            try:
                with Image.open(image_path) as img:
                    print(f"Bild: {os.path.basename(image_path)}, Größe: {img.size}")

                    if img.size != target_size:
                        img_resized = img.resize(target_size, Image.ANTIALIAS)

                        img_resized.save(image_path)
                        print(f"Bild resized und gespeichert: {os.path.basename(image_path)}")
                    else:
                        print(f"Bild hat bereits die richtige Größe: {os.path.basename(image_path)}")

            except Exception as e:
                print(f"Fehler beim Bearbeiten von {image_path}: {e}")

base_foler_path = "E:/Programmierung/Datein/Python/bell_repo/conv-network"
# resize_images_in_folders(base_foler_path)

def rename_images_in_folder(folder_path):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]

    files.sort()

    for index, file_name in enumerate(files):
        old_path = os.path.join(folder_path, file_name)
        new_name = f"{index}{os.path.splitext(file_name)[1].lower()}"
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)

    print(f"Renamed {len(files)} files in folder: {folder_path}")

def rename_images_in_train_and_test():
    for folder in ["train", "test"]:
        if os.path.exists(folder):
            rename_images_in_folder(folder)
        else:
            print(f"Folder not found: {folder}")

# rename_images_in_train_and_test()


def preprocess_image(input_image, section_size=13, overlap=1):
    image = cv2.imread(input_image)
    if image is None:
        raise ValueError(f"Image at path '{input_image}' could not be read.")
    
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    img_height, img_width, _ = image_lab.shape
    
    step = section_size - overlap
    central_pixels = []

    for y in range(0, img_height, step):
        for x in range(0, img_width, step):
            x_end = min(x + section_size, img_width)
            y_end = min(y + section_size, img_height)
            
            section = image_lab[y:y_end, x:x_end]
            
            center_y = (y_end - y) // 2
            center_x = (x_end - x) // 2
            
            _, a_value, b_value = section[center_y, center_x]
            central_pixels.append([a_value, b_value])

    return central_pixels

def process_all_images(image_dir, csv_path, section_size=13, overlap=1):
    with open(csv_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        for image_name in os.listdir(image_dir):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(image_dir, image_name)
                
                try:
                    central_pixels = preprocess_image(image_path, section_size, overlap)
                    
                    for pixel in central_pixels:
                        csv_writer.writerow([image_name] + pixel)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

image_directory_train = "train/"
image_directory_test = "test/"
csv_file_path_train = "train.csv"
csv_file_path_test = "test.csv"

# process_all_images(image_directory_train, csv_file_path_train)
# process_all_images(image_directory_test, csv_file_path_test)

def add_section_id_to_csv(csv_file, output_csv_file):
    data = pd.read_csv(csv_file)
    
    section_id = 0
    
    section_ids = []
    
    for filename in data['filename'].unique():
        image_data = data[data['filename'] == filename]
        
        for _ in image_data.itertuples():
            section_ids.append(section_id)
            section_id += 1
    
    data['section_id'] = section_ids
    
    data.to_csv(output_csv_file, index=False)
    print(f"Section IDs wurden hinzugefügt. Neue CSV gespeichert als: {output_csv_file}")

csv_file_train = 'train.csv'
csv_file_test = 'test.csv'
output_csv_file_train = 'train_with_section_ids.csv'
output_csv_file_test = 'test_with_section_ids.csv'

# Funktion aufrufen
# add_section_id_to_csv(csv_file_train, output_csv_file_train)
# add_section_id_to_csv(csv_file_test, output_csv_file_test)