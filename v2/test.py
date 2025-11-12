import os.path
import torch
import numpy as np
import cv2

from tqdm import trange
from skimage import io, color
from skimage.transform import resize


def break_up_image(image_name: str,section_size: int):
    image = io.imread(os.path.join("test", image_name))
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    image = resize(image, (500, 500), anti_aliasing=False)
    original_image = image.copy()
    image = color.rgb2lab(image).astype(np.float32)
    image[:, :, 0] = image[:, :, 0] / 100.0
    image[:, :, 1] = image[:, :, 1] / 128.0
    image[:, :, 2] = image[:, :, 2] / 128.0

    print(f"# L: max {np.max(image[:, :, 0])} min {np.min(image[:, :, 0])}")
    print(f"# A: max {np.max(image[:, :, 1])} min {np.min(image[:, :, 1])}")
    print(f"# B: max {np.max(image[:, :, 2])} min {np.min(image[:, :, 2])}")

    pred_image = np.zeros((500, 500, 3), dtype=np.float32)

    # model
    model = torch.load("conv_model_15.pth", weights_only=False)
    model.eval()

    # height
    for i in trange(0, image.shape[0]-section_size+1):
        # width
        for j in range(0, image.shape[1]-section_size+1):
            center_x = section_size // 2
            center_y = section_size // 2
            center_i = i + section_size // 2
            center_j = j + section_size // 2

            grayscale = cv2.extractChannel(image, coi=0)
            section = grayscale[i:i+section_size, j:j+section_size]
            section_tensor = torch.from_numpy(section).float().unsqueeze(0).unsqueeze(0)
            pred_section = model(section_tensor)

            L = cv2.extractChannel(section, coi=0)[center_x, center_y] * 100.0
            B = (pred_section[:, 1].detach().numpy() * 128.0) * 128.0
            A = (pred_section[:, 0].detach().numpy() * 128.0) * 128.0

            pred_image[center_i, center_j, 0] = np.uint8(L) # L channel
            pred_image[center_i, center_j, 1] = np.uint8(A) # A channel
            pred_image[center_i, center_j, 2] = np.uint8(B) # B channel

    print(f"# shape {pred_image.shape}")
    print(f"# L: max {np.max(pred_image[:, :, 0])} min {np.min(pred_image[:, :, 0])}")
    print(f"# A: max {np.max(pred_image[:, :, 1])} min {np.min(pred_image[:, :, 1])}")
    print(f"# B: max {np.max(pred_image[:, :, 2])} min {np.min(pred_image[:, :, 2])}")

    rgb_image = color.lab2rgb(pred_image)
    io.imsave(f"test/pred_{image_name}", rgb_image)
    io.imsave(f"test/{image_name}", original_image)

if __name__ == "__main__":
    # break_up_image("cat.png", 15)
    # break_up_image("bike.png", 15)
    break_up_image("landscape.png", 15)
