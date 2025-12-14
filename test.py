import os.path
import cv2 as cv
import torch
import numpy as np

from tqdm import trange
from model import ConvModel_v1, ConvModel_v2
import matplotlib.pyplot as plt


def predict_image(
        dir:str,
        image:str,
        section_size:int,
        model_version:str
    ):
    image = cv.imread(os.path.join("test", image))
    image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    image = image / 255

    pred_image = np.zeros((500, 500, 3), dtype=np.float32)

    # laden des Modells
    model = ConvModel_v1(section_size=section_size) if model_version == "v1" else ConvModel_v2(section_size=section_size)
    model.load_state_dict(
    torch.load(f"result/model_{model_version}_sectionsize_{section_size}/conv_model_{section_size}.pth", map_location="cpu"))
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

            A = float(pred_section[0, 0].detach().cpu().numpy())
            B = float(pred_section[0, 1].detach().cpu().numpy())

            pred_image[:, :, 0] = grayscale # L channel
            pred_image[center_x, center_y, 1] = A # A channel
            pred_image[center_x, center_y, 2] = B # B channel

    # Umwandlung des vorhergesaten Bildes in den RGB-Farbraum
    pred_image_uint8 = (pred_image * 255).astype(np.uint8)
    pred_image_uint8 = cv.cvtColor(pred_image_uint8, cv.COLOR_LAB2RGB)
    return pred_image_uint8

def test_integrity(
        image_name:str,
        section_size:int
    ):
    image = cv.imread(os.path.join("test", image_name))
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    image /= 255

    image = image[:section_size, :section_size]
    image = image[:, :, 0]
    image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)

    model = ConvModel_v1(section_size=section_size)
    # model = ConvModel_v2(section_size=section_size)
    model.load_state_dict(torch.load(
        f"result/model_sectionsize_{section_size}/conv_model_{section_size}.pth",
        map_location="cpu")
    )
    model.eval()
    pred = model(image)
    # print("# model prediction A: ", pred[:, 0].item() * 255)
    # print("# model prediction B: ", pred[:, 1].item() * 255)

    pred_image = np.zeros((section_size, section_size, 3), dtype=np.float32)
    L = image[section_size//2, section_size//2].numpy()
    A = pred[:, 0].item()
    B = pred[:, 1].item()

    pred_image[:, :, 0] = L
    pred_image[:, :, 1] = A
    pred_image[:, :, 2] = B

    pred_image = cv.cvtColor(pred_image, cv.COLOR_LAB2RGB)
    pred_image_uint8 = (pred_image * 255).astype(np.uint8)

    # print(f"\n# R: max {np.max(pred_image_uint8[:, :, 0])} min {np.min(pred_image_uint8[:, :, 0])}")
    # print(f"# G: max {np.max(pred_image_uint8[:, :, 1])} min {np.min(pred_image_uint8[:, :, 1])}")
    # print(f"# B: max {np.max(pred_image_uint8[:, :, 2])} min {np.min(pred_image_uint8[:, :, 2])}")

def image_grid(
        images:list[np.ndarray],
        model:str
    ):
    nrows, ncols = 3, 2
    fig_size = [6, 8]

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size)

    for i, axi in enumerate(ax.flat):
        axi.imshow(images[i])
        if i == 0:
            axi.set_title("Original")
        elif i == 1:
            axi.set_title("Vorhersage")
        axi.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join("./result/image_grid/", f"image_grid_{model}.png"))

if __name__ == "__main__":
    img_1 = cv.imread("./test/bike.jpg")
    img_2 = cv.imread("./test/cat.jpg")
    img_3 = cv.imread("./test/landscape.jpg")

    img_1_v1_pred = predict_image("./test/", "bike.jpg", 13, "v1")
    img_1_v2_pred = predict_image("./test/", "bike.jpg", 13, "v2")

    img_2_v1_pred = predict_image("./test/", "cat.jpg", 13, "v1")
    img_2_v2_pred = predict_image("./test/", "cat.jpg", 13, "v2")

    img_3_v1_pred = predict_image("./test/", "landscape.jpg", 13, "v1")
    img_3_v2_pred = predict_image("./test/", "landscape.jpg", 13, "v2")

    image_list_v1 = [img_1_v1_pred, img_2_v1_pred, img_3_v1_pred]
    image_list_v2 = [img_1_v2_pred, img_2_v2_pred, img_3_v2_pred]

    image_list_v1.insert(0, cv.cvtColor(cv.imread(os.path.join("./test/", "bike.jpg")), cv.COLOR_BGR2RGB))
    image_list_v1.insert(2, cv.cvtColor(cv.imread(os.path.join("./test/", "cat.jpg")), cv.COLOR_BGR2RGB))
    image_list_v1.insert(4, cv.cvtColor(cv.imread(os.path.join("./test/", "landscape.jpg")), cv.COLOR_BGR2RGB))

    image_list_v2.insert(0, cv.cvtColor(cv.imread(os.path.join("./test/", "bike.jpg")), cv.COLOR_BGR2RGB))
    image_list_v2.insert(2, cv.cvtColor(cv.imread(os.path.join("./test/", "cat.jpg")), cv.COLOR_BGR2RGB))
    image_list_v2.insert(4, cv.cvtColor(cv.imread(os.path.join("./test/", "landscape.jpg")), cv.COLOR_BGR2RGB))

    fig, axes = plt.subplots(nrows=3, ncols=2, sharey=True)

    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]
    ax5, ax6 = axes[2]

    ax1.hist(img_1_v1_pred.ravel(), bins=256)
    ax2.hist(img_1.ravel(), bins=256)
    ax3.hist(img_2_v1_pred.ravel(), bins=256)
    ax4.hist(img_2.ravel(), bins=256)
    ax5.hist(img_3_v1_pred.ravel(), bins=256)
    ax6.hist(img_3.ravel(), bins=256)

    ax1.set_title("Histogramm Original")
    ax2.set_title("Histogramm Vorhersage")
    ax3.set_title("Histogramm Original")
    ax4.set_title("Histogramm Vorhersage")
    ax5.set_title("Histogramm Original")
    ax6.set_title("Histogramm Vorhersage")

    plt.tight_layout()
    plt.savefig(os.path.join("./result/histogram/", "histogram_v1.png"))

    fig, axes = plt.subplots(nrows=3, ncols=2, sharey=True)

    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]
    ax5, ax6 = axes[2]

    ax1.hist(img_1_v2_pred.ravel(), bins=256)
    ax2.hist(img_1.ravel(), bins=256)
    ax3.hist(img_2_v2_pred.ravel(), bins=256)
    ax4.hist(img_2.ravel(), bins=256)
    ax5.hist(img_3_v2_pred.ravel(), bins=256)
    ax6.hist(img_3.ravel(), bins=256)

    ax1.set_title("Histogramm Original")
    ax2.set_title("Histogramm Vorhersage")
    ax3.set_title("Histogramm Original")
    ax4.set_title("Histogramm Vorhersage")
    ax5.set_title("Histogramm Original")
    ax6.set_title("Histogramm Vorhersage")

    plt.tight_layout()
    plt.savefig(os.path.join("./result/histogram/", "histogram_v2.png"))

    image_grid(image_list_v1, "v1")
    image_grid(image_list_v2, "v2")

    # test_integrity("landscape.png", 15)
