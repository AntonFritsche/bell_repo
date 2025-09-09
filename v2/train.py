import os
import torch
import sys
import cv2
import matplotlib.pyplot as plt

from model import ConvModel
from torch.nn import MSELoss
from dataset import RuntimeABSectionDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    training_loss = []
    validation_loss = []

    epochs = 10
    train_directory = "./data/train"
    val_directory = "./data/val"
    result_directory = "./result"
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    section_size = 17
    lr = 1e-4
    batch_size_train = 128
    batch_size_val = 256

    conv_model = ConvModel(section_size=section_size)
    conv_model = conv_model.to(device)

    print("# Parameters: ", sum(p.numel() for p in conv_model.parameters() if p.requires_grad))

    loss_fn = MSELoss()
    optimizer = torch.optim.AdamW(conv_model.parameters(), lr=lr)

    train_dataset = RuntimeABSectionDataset(train_directory, section_size, return_centers=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=16)

    val_dataset = RuntimeABSectionDataset(val_directory, section_size, return_centers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, num_workers=16)

    print(len(train_dataset), len(val_dataset))

    for epoch in range(1, epochs + 1):
        loss_acc = 0.0
        samples_seen = 0

        conv_model.train()
        training_progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            colour='green',
            file=sys.stdout,
            desc=f'Epoch {epoch} / {epochs} - Training  '
        )

        for batch_idx, (data, label) in training_progress:
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()
            prediction = conv_model(data)
            loss = loss_fn(prediction, label)

            training_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            loss_acc += loss.item() * data.size(0)
            samples_seen += data.size(0)

            training_progress.set_postfix(
                {
                    "Loss": loss_acc / samples_seen
                }
            )

        conv_model.eval()
        validation_progress = tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            colour='green',
            file=sys.stdout,
            desc=f'Epoch {epoch} / {epochs} - Validation  '
        )

        with torch.no_grad():
            reconstructed_image = torch.zeros(size=(500, 500, 3), dtype=torch.uint8).to(device)

            for batch_idx, (data, label, center_x, center_y) in validation_progress:
                data, label = data.to(device), label.to(device)

                prediction = conv_model(data)
                loss = loss_fn(prediction, label)

                validation_loss.append(loss.item())

                l = data[:, 0, section_size // 2, section_size // 2] * 255.0
                a = prediction[:, 0] * 255.0
                b = prediction[:, 1] * 255.0

                reconstructed_image[center_x, center_y, 0] = l.type(torch.uint8)
                reconstructed_image[center_x, center_y, 1] = a.type(torch.uint8)
                reconstructed_image[center_x, center_y, 2] = b.type(torch.uint8)

            print("L range: ", reconstructed_image[:, :, 0].min().item(), reconstructed_image[:, :, 0].max().item())
            print("A range: ", reconstructed_image[:, :, 1].min().item(), reconstructed_image[:, :, 1].max().item())
            print("B range: ", reconstructed_image[:, :, 2].min().item(), reconstructed_image[:, :, 2].max().item())

            reconstruction_filename = f"reconstruction_{epoch}.png"
            reconstruction_path = os.path.join(result_directory, reconstruction_filename)

            reconstructed_image = reconstructed_image.cpu().numpy()
            reconstructed_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_LAB2BGR)

            cv2.imwrite(reconstruction_path, reconstructed_image)

    fig_train, ax = plt.subplots()
    ax.plot(training_loss, [i for i in range(epochs)])
    fig_train.suptitle("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.savefig(os.path.join(result_directory, "training_loss.png"))
    plt.show()

    fig_val, ax = plt.subplots()
    ax.plot(training_loss, [i for i in range(epochs)])
    fig_train.suptitle("validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.savefig(os.path.join(result_directory, "validation_loss.png"))
    plt.show()

if __name__ == '__main__':
    main()