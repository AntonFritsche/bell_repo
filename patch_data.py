import os
import numpy as np
import cv2 as cv

from tqdm import trange


def patch_data(
        dir: str,
        train_data: bool,
        sections_size: int,
        patches_per_image: int
    ):
    if train_data:
        image_dir = os.path.join(dir, "train/")
        dst = os.path.join("data/", f"train_patches_{sections_size}")
    else:
        image_dir = os.path.join(dir, "val")
        dst = os.path.join("data/", f"val_patches_{sections_size}")
    os.makedirs(dst, exist_ok=True)

    # Schleife, die über alle Bilder iteriert
    for img_name in trange(len(os.listdir(image_dir))):
        img_name = os.listdir(image_dir)[img_name]
        img = cv.imread(os.path.join(image_dir, img_name))
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        if lab.shape[-1] == 4:
            lab = img[..., :3]

        # Separate Kanäle
        L = lab[:, :, 0] / 255
        a = lab[:, :, 1] / 255
        b = lab[:, :, 2] / 255

        scaled_image = np.stack([L, a, b], axis=-1)
        h, w, _ = scaled_image.shape

        # Generierung der zufälligen Bildausschnitte mit den zugehörigen Labels
        for i in range(patches_per_image):
            r = np.random.randint(0, h - sections_size)
            c = np.random.randint(0, w - sections_size)

            center_x = r + sections_size // 2
            center_y = c + sections_size // 2

            patch = scaled_image[r:r+sections_size, c:c+sections_size, 0]
            a = scaled_image[center_x, center_y, 1]
            b = scaled_image[center_x, center_y, 2]

            # Speichern der Bildausschnitte und Labels im npy-Format
            np.save(os.path.join(dst, f"{img_name}_{i}.npy"), {
                "patch": patch,
                "label": np.array([a, b])
            })

if __name__ == "__main__":
    # Sektionsgröße 5
    # patch_data("D:/projekte/bell_repo/data/", train_data=True, sections_size=5, patches_per_image=600)
    # patch_data("D:/projekte/bell_repo/data/", train_data=False, sections_size=5, patches_per_image=600)

    # Sektionsgröße 13
    # patch_data("D:/projekte/bell_repo/data/", train_data=True, sections_size=13, patches_per_image=600)
    # patch_data("D:/pcrojekte/bell_repo/data/", train_data=False, sections_size=13, patches_per_image=600)

    # Sektionsgröße 15
    # patch_data("D:/projekte/bell_repo/data/", train_data=True, sections_size=15, patches_per_image=600)
    # patch_data("D:/projekte/bell_repo/data/", train_data=False, sections_size=15, patches_per_image=600)

    # Sektionsgröße 50
    patch_data("D:/projekte/bell_repo/data/", train_data=True, sections_size=50, patches_per_image=100)
    patch_data("D:/projekte/bell_repo/data/", train_data=False, sections_size=50, patches_per_image=100)