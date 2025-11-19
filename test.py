import os.path
import torch
import numpy as np

from tqdm import trange
from skimage.util import img_as_float
from skimage import io, color
from skimage.transform import resize
from model import ConvModel


def break_up_image(image_name: str,section_size: int):
    image = io.imread(os.path.join("test", image_name))
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    image = resize(image, (500, 500), anti_aliasing=False)
    original_image = image.copy()
    image = color.rgb2lab(img_as_float(image))

    print(f"# L: max {np.max(image[:, :, 0])} min {np.min(image[:, :, 0])}")
    print(f"# A: max {np.max(image[:, :, 1])} min {np.min(image[:, :, 1])}")
    print(f"# B: max {np.max(image[:, :, 2])} min {np.min(image[:, :, 2])}")

    pred_image = np.zeros((500, 500, 3), dtype=np.float32)

    # model
    model = ConvModel(section_size=section_size)
    model.load_state_dict(
    torch.load(f"result/model_sectionsize_{section_size}/conv_model_{section_size}.pth", map_location="cpu"))
    model.eval()

    # height
    for i in trange(0, image.shape[0]-section_size+1): # 0, 500 - sections_size + 1
        # width
        for j in range(0, image.shape[1]-section_size+1): # 0, 500 - sections_size + 1
            center_x = section_size // 2
            center_y = section_size // 2

            grayscale = image[:, :, 0]
            section = grayscale[i:i+section_size, j:j+section_size]
            section_tensor = torch.from_numpy(section).float().unsqueeze(0).unsqueeze(0)
            pred_section = model(section_tensor)

            L = section[center_x, center_y]
            B = pred_section[:, 1].detach().numpy()
            A = pred_section[:, 0].detach().numpy()

            pred_image[i, j, 0] = np.int16(L) # L channel
            pred_image[i, j, 1] = np.int16(A) # A channel
            pred_image[i, j, 2] = np.int16(B) # B channel

    print(f"# shape {pred_image.shape}")
    print(f"# L: max {np.max(pred_image[:, :, 0])} min {np.min(pred_image[:, :, 0])}")
    print(f"# A: max {np.max(pred_image[:, :, 1])} min {np.min(pred_image[:, :, 1])}")
    print(f"# B: max {np.max(pred_image[:, :, 2])} min {np.min(pred_image[:, :, 2])}")

    rgb_image = color.lab2rgb(pred_image)
    io.imsave(f"test/pred_{image_name}", rgb_image)
    io.imsave(f"test/{image_name}", original_image)

if __name__ == "__main__":
    break_up_image("cat.png", 5)
    break_up_image("bike.png", 5)
    break_up_image("landscape.png", 5)
