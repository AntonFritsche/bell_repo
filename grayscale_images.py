import tensorflow as tf
import os

def preprocess_images(path):
    with open(path, 'rb') as image_open:
        read_image = image_open.read()

    image_decode = tf.image.decode_jpeg(read_image, channels=3)

    gray_image = tf.image.rgb_to_grayscale(image_decode)

    gray_image_uint8 = tf.image.convert_image_dtype(gray_image, dtype=tf.uint8)

    return gray_image_uint8

def rgb_to_grayscale(directory):
    for i, name in enumerate(os.listdir(directory), start=1):
        file_path = os.path.join(directory, name)
        
        if not name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        new_image = preprocess_images(file_path)

        encoded_image = tf.image.encode_jpeg(new_image, quality=95)

        with open(file_path, 'wb') as f:
            f.write(encoded_image.numpy())
        print(f"Deleted: {file_path} in {directory} and created new picture number: {i}")

directory_1_natural_images = r"E:\Programmierung\Datein\Python\bell_repo\Bilder_Kolorierung_dataset\natural_images_no_color\airplane"
directory_2_natural_images = r"E:\Programmierung\Datein\Python\bell_repo\Bilder_Kolorierung_dataset\natural_images_no_color\car"
directory_3_natural_images = r"E:\Programmierung\Datein\Python\bell_repo\Bilder_Kolorierung_dataset\natural_images_no_color\cat"
directory_4_natural_images = r"E:\Programmierung\Datein\Python\bell_repo\Bilder_Kolorierung_dataset\natural_images_no_color\dog"
directory_5_natural_images = r"E:\Programmierung\Datein\Python\bell_repo\Bilder_Kolorierung_dataset\natural_images_no_color\flower"

directory_1_gender = r"E:\Programmierung\Datein\Python\bell_repo\Bilder_Kolorierung_dataset\gender_dataset_no_color\men"
directory_2_gender = r"E:\Programmierung\Datein\Python\bell_repo\Bilder_Kolorierung_dataset\gender_dataset_no_color\women"

directory_1_fashion = r"E:\Programmierung\Datein\Python\bell_repo\Bilder_Kolorierung_dataset\fashion_dataset_no_color\fashion-dataset\images"

rgb_to_grayscale(directory_1_natural_images)
rgb_to_grayscale(directory_2_natural_images)
rgb_to_grayscale(directory_3_natural_images)
rgb_to_grayscale(directory_4_natural_images)
rgb_to_grayscale(directory_5_natural_images)

rgb_to_grayscale(directory_1_gender)
rgb_to_grayscale(directory_2_gender)

rgb_to_grayscale(directory_1_fashion)
