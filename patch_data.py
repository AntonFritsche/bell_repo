import os
import numpy as np

import cv2 as cv
from tqdm import trange


def convert_to_8bit(dir: str):
    files = os.listdir(dir)

    for img_name in trange(len(files)):
        img_name = os.listdir(dir)[img_name]
        image = cv.imread(os.path.join(dir, img_name))
        image = image[..., :3]
        print(image.dtype)

def patch_data(dir: str, train_data: bool, sections_size: int, patches_per_image: int):
    if train_data:
        dst = f"./data/train_patches_{sections_size}"
    else:
        dst = f"./data/val_patches_{sections_size}"
    os.makedirs(dst, exist_ok=True)

    for img_name in trange(len(os.listdir(dir))):
        img_name = os.listdir(dir)[img_name]
        img = cv.imread(os.path.join(dir, img_name))
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        if lab.shape[-1] == 4:
            lab = img[..., :3]

        # print(f"L: {np.max(lab[:, :, 0])} min {np.min(lab[:, :, 0])}; A: {np.max(lab[:, :, 1])} min {np.min(lab[:, :, 1])}; B: {np.max(lab[:, :, 2])} min {np.min(lab[:, :, 2])}")

        # Separate channels
        L = lab[:, :, 0] / 255
        a = lab[:, :, 1] / 255
        b = lab[:, :, 2] / 255

        # print(f"L: {np.max(L)} min {np.min(L)}; A: {np.max(a)} min {np.min(a)}; B: {np.max(b)} min {np.min(b)}")

        scaled_image = np.stack([L, a, b], axis=-1)
        h, w, _ = scaled_image.shape

        for i in range(patches_per_image):
            r = np.random.randint(0, h - sections_size)
            c = np.random.randint(0, w - sections_size)

            center_x = r + sections_size // 2
            center_y = c + sections_size // 2

            patch = scaled_image[r:r+sections_size, c:c+sections_size, 0]
            a = scaled_image[center_x, center_y, 1]
            b = scaled_image[center_x, center_y, 2]

            np.save(os.path.join(dst, f"{img_name}_{i}.npy"), {
                "patch": patch,
                "label": np.array([a, b])
            })

if __name__ == "__main__":
    # sections_size 15
    patch_data("./data/train/", train_data=True, sections_size=5, patches_per_image=600)
    patch_data("./data/val/", train_data=False, sections_size=5, patches_per_image=600)

    # sections_size 11
    patch_data("./data/train/", train_data=True, sections_size=11, patches_per_image=600)
    patch_data("./data/val/", train_data=False, sections_size=11, patches_per_image=600)

    # sections_size 5
    patch_data("./data/train/", train_data=True, sections_size=15, patches_per_image=600)
    patch_data("./data/val/", train_data=False, sections_size=15, patches_per_image=600)

    # convert_to_8bit("./data/train/")