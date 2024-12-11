# import tensorflow as tf
# from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import Dropout 
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import Conv2DTranspose
# from tensorflow.keras.layers import concatenate
# from tensorflow.keras.losses import binary_crossentropy
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import numpy as np

import os
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import cv2

def mask_data(no_color_path, color_path, output_dir):
    # Ausgabepfade für maskierte Bilder
    no_color_mask_output = os.path.join(output_dir, "no_color_img_mask")
    color_mask_output = os.path.join(output_dir, "color_img_mask")

    # Verzeichnisse erstellen, falls sie nicht existieren
    os.makedirs(no_color_mask_output, exist_ok=True)
    os.makedirs(color_mask_output, exist_ok=True)

    # Bilder aus dem "no_color"-Ordner verarbeiten (Schwarz-Weiß-Bilder)
    print("Verarbeite Schwarz-Weiß-Bilder...")
    for root, dirs, files in os.walk(no_color_path):
        print(f"Durchlaufe Ordner: {root}")
        for file_name in files:
            print(f"Lade Bild: {file_name}")
            file_path = os.path.join(root, file_name)

            # Bild laden
            image = cv2.imread(file_path)
            
            # Überprüfen, ob das Bild korrekt geladen wurde
            if image is None:
                print(f"Fehler beim Laden des Bildes: {file_name}")
                continue  # Wenn das Bild nicht geladen werden kann, überspringen

            # Überprüfen, ob es sich um ein Graustufenbild handelt (1 Kanal)
            edges = cv2.Canny(image, threshold1=50, threshold2=150)
            
            # Überprüfen, ob die Kanten gut erkennbar sind
            if edges.sum() == 0:
                print(f"Keine Kanten gefunden in {file_name}. Überspringe das Bild.")
                continue  # Wenn keine Kanten gefunden werden, überspringen

            # Bild im entsprechenden Ordner speichern
            output_path = os.path.join(no_color_mask_output, file_name)
            cv2.imwrite(output_path, edges)
            print(f"Verarbeitet und gespeichert (Schwarz-Weiß): {output_path}")

    # Bilder aus dem "color"-Ordner verarbeiten (Farb-Bilder)
    print("Verarbeite Farbbilder...")
    for root, dirs, files in os.walk(color_path):
        print(f"Durchlaufe Ordner: {root}")
        for file_name in files:
            print(f"Lade Bild: {file_name}")
            file_path = os.path.join(root, file_name)

            # Bild laden
            image = cv2.imread(file_path)
            
            # Überprüfen, ob das Bild korrekt geladen wurde
            if image is None:
                print(f"Fehler beim Laden des Bildes: {file_name}")
                continue  # Wenn das Bild nicht geladen werden kann, überspringen

            # Überprüfen, ob es sich um ein Farbbild handelt (3 Kanäle)
            print(f"Farb-Bild gefunden: {file_name}")
            edges = cv2.Canny(image, threshold1=50, threshold2=150)
            # Überprüfen, ob die Kanten gut erkennbar sind
            if edges.sum() == 0:
                print(f"Keine Kanten gefunden in {file_name}. Überspringe das Bild.")
                continue  # Wenn keine Kanten gefunden werden, überspringen

            # Bild im entsprechenden Ordner speichern
            output_path = os.path.join(color_mask_output, file_name)
            cv2.imwrite(output_path, edges)
            print(f"Verarbeitet und gespeichert (Farbe): {output_path}")


no_color_folder = r"E:\Programmierung\Datein\Python\bell_repo\combined_images\no_color_img_comb"  # Ordner mit Schwarz-Weiß-Bildern
color_folder = r"E:\Programmierung\Datein\Python\bell_repo\combined_images\color_img_comb"  # Ordner mit Farbbildern
output_folder = r"E:\Programmierung\Datein\Python\bell_repo\combined_images"  # Ordner für die Ausgabedaten

mask_data(no_color_folder, color_folder, output_folder)

# test_image = cv2.imread(r'E:\Programmierung\Datein\Python\bell_repo\combined_images\no_color_img_comb\1.jpg')
# if test_image is None:
#     print("Fehler beim Laden des Bildes!")
# else:
#     print(f"Bild geladen mit Form: {test_image.shape}")