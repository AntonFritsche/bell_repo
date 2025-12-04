import os.path
import torch
import numpy as np

from tqdm import trange
import cv2 as cv
from model import ConvModel


def break_up_image(image_name:str, section_size:int):
    image = cv.imread(os.path.join("test", image_name))
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    # image = cv.resize(image, (500, 500))
    original_image = image.copy()
    image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    image /= 255

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
            grayscale = image[:, :, 0]
            section = grayscale[i:i+section_size, j:j+section_size]
            section_tensor = torch.from_numpy(section).float().unsqueeze(0).unsqueeze(0)
            pred_section = model(section_tensor)

            center_x = i + section_size // 2
            center_y = j + section_size // 2

            L = grayscale[center_x, center_y]
            A = float(pred_section[0, 0].detach().cpu().numpy())
            B = float(pred_section[0, 1].detach().cpu().numpy())

            pred_image[center_x, center_y, 0] = L # L channel
            pred_image[center_x, center_y, 1] = A # A channel
            pred_image[center_x, center_y, 2] = B # B channel

    print(f"# shape {pred_image.shape}")
    print(f"# L: max {np.max(pred_image[:, :, 0])} min {np.min(pred_image[:, :, 0])}")
    print(f"# A: max {np.max(pred_image[:, :, 1])} min {np.min(pred_image[:, :, 1])}")
    print(f"# B: max {np.max(pred_image[:, :, 2])} min {np.min(pred_image[:, :, 2])}\n")

    pred_image_uint8 = (pred_image * 255).astype(np.uint8)
    rgb_image = cv.cvtColor(pred_image_uint8, cv.COLOR_LAB2RGB)

    print(f"# R: max {np.max(rgb_image[:, :, 0])} min {np.min(rgb_image[:, :, 0])}")
    print(f"# G: max {np.max(rgb_image[:, :, 1])} min {np.min(rgb_image[:, :, 1])}")
    print(f"# B: max {np.max(rgb_image[:, :, 2])} min {np.min(rgb_image[:, :, 2])}")

    cv.imwrite(f"test/pred_{image_name}", rgb_image)
    
def test_integrity(image_name:str, section_size:int):
    image = cv.imread(os.path.join("test", image_name))
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    print(f"# R: max {np.max(image[:, :, 0])} min {np.min(image[:, :, 0])}")
    print(f"# G: max {np.max(image[:, :, 1])} min {np.min(image[:, :, 1])}")
    print(f"# B: max {np.max(image[:, :, 2])} min {np.min(image[:, :, 2])}\n")

    image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    image /= 255

    print(f"# L: max {np.max(image[:, :, 0])} min {np.min(image[:, :, 0])}")
    print(f"# A: max {np.max(image[:, :, 1])} min {np.min(image[:, :, 1])}")
    print(f"# B: max {np.max(image[:, :, 2])} min {np.min(image[:, :, 2])}\n")

    image = image[:section_size, :section_size]
    print(f"# shape of image: {image.shape}")
    print("# image L channel:", image[:, :, 0], image[:, :, 0].shape)
    print("# image A channel:", image[:, :, 1], image[:, :, 1].shape)
    print("# image B channel:", image[:, :, 2], image[:, :, 2].shape)
    image = image[:, :, 0]
    image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)

    model = ConvModel(section_size=section_size)
    model.load_state_dict(torch.load(
        f"result/model_sectionsize_{section_size}/conv_model_{section_size}.pth",
        map_location="cpu")
    )
    model.eval()
    pred = model(image)
    print("# model prediction A: ", pred[:, 0].item() * 255)
    print("# model prediction B: ", pred[:, 1].item() * 255)

    pred_image = np.zeros((section_size, section_size, 3), dtype=np.float32)
    L = image[section_size//2, section_size//2].numpy()
    A = pred[:, 0].item()
    B = pred[:, 1].item()

    pred_image[:, :, 0] = L
    pred_image[:, :, 1] = A
    pred_image[:, :, 2] = B

    pred_image = cv.cvtColor(pred_image, cv.COLOR_LAB2RGB)
    pred_image_uint8 = (pred_image * 255).astype(np.uint8)

    print(f"\n# R: max {np.max(pred_image_uint8[:, :, 0])} min {np.min(pred_image_uint8[:, :, 0])}")
    print(f"# G: max {np.max(pred_image_uint8[:, :, 1])} min {np.min(pred_image_uint8[:, :, 1])}")
    print(f"# B: max {np.max(pred_image_uint8[:, :, 2])} min {np.min(pred_image_uint8[:, :, 2])}")

if __name__ == "__main__":
    break_up_image("cat.png", 15)
    # break_up_image("bike.png", 5)
    # break_up_image("landscape.png", 5)
    # test_integrity("landscape.png", 15)
