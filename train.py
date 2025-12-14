import os
import torch
import sys
import matplotlib.pyplot as plt
import torch.nn as nn

from model import ConvModel_v1, ConvModel_v2, ConvModel_v3
from torch.nn import MSELoss
from dataset import Section_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsummary import summary


def test_section_sizes():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # exponentielle Abhängigkeit der section_size und der Parameter
    section_sizes = [7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    parameters_v1 = []
    parameters_v2 = []
    for size in section_sizes:
        conv_model_v1 = ConvModel_v1(section_size=size)
        conv_model_v2 = ConvModel_v2(section_size=size)
        conv_model_v1 = conv_model_v1.to(device)
        conv_model_v2 = conv_model_v2.to(device)

        print("# Parameteranzahl: ", sum(p.numel() for p in conv_model_v1.parameters() if p.requires_grad))
        print("# Parameteranzahl: ", sum(p.numel() for p in conv_model_v2.parameters() if p.requires_grad))

        parameters_v1.append(sum(p.numel() for p in conv_model_v1.parameters() if p.requires_grad))
        parameters_v2.append(sum(p.numel() for p in conv_model_v2.parameters() if p.requires_grad))

        summary(conv_model_v1, (1, size, size))
        summary(conv_model_v2, (1, size, size))

    fig, ax = plt.subplots(2, 1)

    ax[0].set_xticks(section_sizes)
    ax[1].set_xticks(section_sizes)

    ax[0].plot(section_sizes, parameters_v1)
    ax[1].plot(section_sizes, parameters_v2)

    ax[0].set_xlabel("Sektionsgröße")
    ax[1].set_xlabel("Sektionsgröße")

    ax[0].set_ylabel("Parameter")
    ax[1].set_ylabel("Parameter")
    plt.subplots_adjust(hspace=0.4)
    plt.show()

def train(
        section_size: int,
        data: str
    ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_flush_denormal(True)

    training_losses = []
    validation_losses = []

    epochs = 15
    lr = 1e-4
    batch_size_train = 256
    batch_size_val = 256
    train_directory = os.path.join(data, f"train_patches_{section_size}")
    val_directory = os.path.join(data, f"val_patches_{section_size}")
    result_directory = f"./result/model_sectionsize_{section_size}"
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    # model
    # conv_model = ConvModel_v1(section_size=section_size)
    # conv_model = ConvModel_v2(section_size=section_size)
    conv_model = ConvModel_v3(section_size=section_size)
    conv_model = conv_model.to(device)
    # test_section_sizes()

    summary(conv_model, (1, section_size, section_size))

    loss_fn = MSELoss()
    optimizer = torch.optim.AdamW(conv_model.parameters(), lr=lr)

    train_dataset = Section_Dataset(train_directory)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=4)
    val_dataset = Section_Dataset(val_directory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, num_workers=4)

    print("# len train_loader: ", len(train_loader))
    print("# len val_loader: ", len(val_loader))
    best_val_loss = float("inf")
    best_model_path = "result/model_checkpoint.pth"

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

        total_loss = loss_acc / samples_seen
        training_losses.append(total_loss)

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

            total_loss = val_loss_acc / val_samples_seen
            validation_losses.append(val_loss_acc / val_samples_seen)
            print(f"Epoch {epoch}: Validation Loss: {total_loss:.4f}")

        if total_loss < best_val_loss:
            best_val_loss = total_loss

            torch.save({
                'epoch': epoch,
                'model_state_dict': conv_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, best_model_path)

    checkpoint = torch.load(best_model_path)
    torch.save(checkpoint, f"result/model_sectionsize_{section_size}/conv_model_{section_size}.pth")

    fig_train, ax = plt.subplots()
    ax.plot([i for i in range(1, epochs+1)], training_losses)
    # plt.semilogy()
    plt.xticks(range(epochs))
    fig_train.suptitle(f"Training Loss section_size {section_size}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.savefig(os.path.join(result_directory, "training_loss.png"))
    # plt.show()

    fig_val, ax = plt.subplots()
    ax.plot([i for i in range(1, epochs+1)], validation_losses)
    # plt.semilogy()
    plt.xticks(range(epochs))
    fig_val.suptitle(f"Validation Loss section_size {section_size}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.savefig(os.path.join(result_directory, "validation_loss.png"))
    # plt.show()

if __name__ == '__main__':
    # unterschiedliche section_sizes
    # train(
    #     section_size=5,
    #     data="data/"
    # )
    # train(
    #     section_size=50,
    #     data="data/"
    # )
    train(
        section_size=13,
        data="data/"
    )

    # test_section_sizes()
