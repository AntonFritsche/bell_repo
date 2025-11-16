import os
import torch
import sys
import matplotlib.pyplot as plt

from model import ConvModel
from torch.nn import MSELoss
from dataset import Section_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsummary import summary


def test_section_sizes():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # exponentielle Abhängigkeit der section_size und der Parameter
    section_sizes = [7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    parameters = []
    for size in section_sizes:
        conv_model = ConvModel(section_size=size)
        conv_model = conv_model.to(device)

        print("# Parameters: ", sum(p.numel() for p in conv_model.parameters() if p.requires_grad))
        parameters.append(sum(p.numel() for p in conv_model.parameters() if p.requires_grad))
        summary(conv_model, (1, size, size))
    fig, ax = plt.subplots()
    plt.xticks(section_sizes)
    ax = fig.gca()
    ax.plot(section_sizes, parameters)
    ax.set_xlabel("Sectionsgröße")
    ax.set_ylabel("Parameter")
    plt.show()

def main(section_size: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_flush_denormal(True)

    training_losses = []
    validation_losses = []

    epochs = 5
    lr = 1e-6
    batch_size_train = 128
    batch_size_val = 256
    train_directory = "./data/train"
    val_directory = "./data/val"
    result_directory = "./result/" + "model_sectionsize_" + str(section_size)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    # model
    conv_model = ConvModel(section_size=section_size)
    conv_model = conv_model.to(device)
    # test_section_sizes()

    summary(conv_model, (1, section_size, section_size))

    loss_fn = MSELoss()
    optimizer = torch.optim.AdamW(conv_model.parameters(), lr=lr)

    train_dataset = Section_Dataset(train_directory, section_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=16)
    val_dataset = Section_Dataset(val_directory, section_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, num_workers=16)

    print("# len train_loader: ", len(train_loader))
    print("# len val_loader: ", len(val_loader))

    for epoch in range(1, epochs + 1):
        # training
        loss_acc = 0.0
        samples_seen = 0

        # validation
        val_loss_acc = 0.0
        val_samples_seen = 0

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

            loss.backward()
            optimizer.step()

            loss_acc += loss.item() * data.size(0)
            samples_seen += data.size(0)

            training_progress.set_postfix(
                {
                    "train loss": loss_acc / samples_seen,
                }
            )

        training_losses.append(loss_acc / samples_seen)

        conv_model.eval()
        validation_progress = tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            colour='green',
            file=sys.stdout,
            desc=f'Epoch {epoch} / {epochs} - Validation  '
        )

        with torch.no_grad():
            for batch_idx, (data, label) in validation_progress:
                data, label = data.to(device), label.to(device)

                prediction = conv_model(data)
                loss = loss_fn(prediction, label)

                val_loss_acc += loss.item() * data.size(0)
                val_samples_seen += data.size(0)

            validation_losses.append(val_loss_acc / val_samples_seen)
        print(f"Epoch {epoch} / {epochs} - Validation Loss: {round(val_loss_acc / val_samples_seen, 5)}")

    torch.save(conv_model, f"result/model_sectionsize_{section_size}/conv_model_15.pth")

    fig_train, ax = plt.subplots()
    ax.plot([i for i in range(epochs)], training_losses)
    # plt.semilogy()
    plt.xticks(range(epochs))
    fig_train.suptitle(f"Training Loss section_size {section_size}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.savefig(os.path.join(result_directory, "training_loss.png"))
    # plt.show()

    fig_val, ax = plt.subplots()
    ax.plot([i for i in range(epochs)], validation_losses)
    # plt.semilogy()
    plt.xticks(range(epochs))
    fig_val.suptitle(f"Validation Loss section_size {section_size}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.savefig(os.path.join(result_directory, "validation_loss.png"))
    # plt.show()

if __name__ == '__main__':
    # unterschiedliche section_sizes
    # main(5)
    # main(11)
    main(15)