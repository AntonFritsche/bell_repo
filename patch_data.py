import os
import numpy as np

from skimage.util import img_as_float
from skimage import io, color
from tqdm import trange


def patch_data(dir: str, train_data: bool, sections_size: int, patches_per_image: int):
    if train_data:
        dst = f"./data/train_patches_{sections_size}"
    else:
        dst = f"./data/val_patches_{sections_size}"
    os.makedirs(dst, exist_ok=True)

    for img_name in trange(len(os.listdir(dir))):
        img_name = os.listdir(dir)[img_name]
        img = io.imread(os.path.join(dir, img_name))
        if img.shape[-1] == 4:
            img = img[..., :3]

        # convert and normalize
        lab = color.rgb2lab(img_as_float(img))
        lab[..., 0] /= 100
        lab[..., 1:] /= 128

        h, w, _ = lab.shape

        for i in range(patches_per_image):
            r = np.random.randint(0, h - sections_size)
            c = np.random.randint(0, w - sections_size)

            center_x = r + sections_size // 2
            center_y = c + sections_size // 2

            patch = lab[r:r+sections_size, c:c+sections_size, 0]
            a = lab[center_x, center_y, 1]
            b = lab[center_x, center_y, 2]

            np.save(os.path.join(dst, f"{img_name}_{i}.npy"), {
                "patch": patch,
                "label": np.array([a, b])
            })

if __name__ == "__main__":
    # sections_size 15
    patch_data("./data/train/", train_data=True, sections_size=5, patches_per_image=500)
    patch_data("./data/val/", train_data=False, sections_size=5, patches_per_image=500)

    # sections_size 11
    patch_data("./data/train/", train_data=True, sections_size=11, patches_per_image=500)
    patch_data("./data/val/", train_data=False, sections_size=11, patches_per_image=500)

    # sections_size 5
    patch_data("./data/train/", train_data=True, sections_size=15, patches_per_image=500)
    patch_data("./data/val/", train_data=False, sections_size=15, patches_per_image=500)