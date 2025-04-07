import os
from model import ConvModel
import torch
from torch.nn import MSELoss
from dataset import RuntimeABSectionDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import cv2

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs = 20
    train_directory = "./data"
    result_directory = "./result"
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    section_size = 13
    lr = 1e-4

    conv_model = ConvModel(1, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 32, 32, 32, 32, 2)
    conv_model = conv_model.to(device)

    loss_fn = MSELoss()
    optimizer = torch.optim.AdamW(conv_model.parameters(), lr=lr)

    train_dataset = RuntimeABSectionDataset(train_directory, section_size, False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16)

    # todo: validate on other dataset / image
    val_dataset = RuntimeABSectionDataset(train_directory, section_size, True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True, num_workers=16)

    for epoch in range(1, epochs + 1):
        loss_acc = 0.0
        samples_seen = 0

        progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            colour='green',
            file=sys.stdout,
            desc=f'Epoch {epoch} / {epochs} - Training  '
        )

        conv_model.train()

        for batch_idx, (data, label) in progress:
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()
            prediction = conv_model(data)
            loss = loss_fn(prediction, label)
            loss.backward()
            optimizer.step()

            loss_acc += loss.item() * data.size(0)
            samples_seen += data.size(0)

            progress.set_postfix(
                {
                    "Loss": loss_acc / samples_seen
                }
            )

        conv_model.eval()

        with torch.no_grad():
            reconstructed_image = torch.zeros(size=(500, 500, 3), dtype=torch.uint8).to(device)

            for batch_idx, (data, label, center_x, center_y) in tqdm(
                    enumerate(val_loader),
                    total=len(val_loader),
                    desc=f"Reconstruction (Epoch {epoch} / {epochs})"
            ):
                data, label = data.to(device), label.to(device)

                prediction = conv_model(data)

                l = data[:, 0, section_size // 2, section_size // 2]
                a = prediction[:, 0]
                b = prediction[:, 1]

                reconstructed_image[center_x, center_y, 0] = l.type(torch.uint8)
                reconstructed_image[center_x, center_y, 1] = a.type(torch.uint8)
                reconstructed_image[center_x, center_y, 2] = b.type(torch.uint8)

            print("L range: ", reconstructed_image[:, :, 0].min().item(), reconstructed_image[:, :, 0].max().item())
            print("A range: ", reconstructed_image[:, :, 1].min().item(), reconstructed_image[:, :, 1].max().item())
            print("B range: ", reconstructed_image[:, :, 2].min().item(), reconstructed_image[:, :, 2].max().item())

            reconstruction_filename = f"reconstruction_{epoch}.png"
            reconstruction_path = os.path.join(result_directory, reconstruction_filename)

            reconstructed_image = reconstructed_image.cpu().numpy()
            reconstructed_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_LAB2BGR)*255.0

            cv2.imwrite(reconstruction_path, reconstructed_image)




if __name__ == '__main__':
    main()