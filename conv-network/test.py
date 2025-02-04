import torch
import cv2
from test_color import show_image
from PIL import Image
from matplotlib import pyplot as plt

def test_cuda():
    print(f"cuda 0 : {torch.device('cuda:0')}")
    print(f"cuda 1 : {torch.device('cuda:1')}")
    print(f"cuda 2 : {torch.device('cuda:2')}")

    print(torch.cuda.is_available())
    print(torch.version.hip)


# noinspection PyRedeclaration
def show_image(input_image):
    image = Image.open(input_image)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Reconstructed Image")
    plt.show()

def test_cv2(row_path):
    img = cv2.imread("cat.png")
    res_img = cv2.resize(img, (300, 500), interpolation=cv2.INTER_AREA)
    cv2.imwrite("cat.png", res_img)

    row_image = cv2.imread(row_path)

    image_1 = row_image[499, :]
    cv2.imwrite("test_image.png", image_1)

    image_2 = row_image[:, 499]
    cv2.imwrite("test_image_2.png", image_2)

    # show_image("test_image.png")
    # show_image("test_image_2.png")
    cv2.imshow("test_image", image_1)

test_cv2(r"E:\Programmierung\Datein\Python\bell_repo\conv-network\cat.png")
