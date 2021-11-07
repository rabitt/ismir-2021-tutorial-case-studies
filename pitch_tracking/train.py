import os
import glob
import librosa
import numpy as np
from constants import compute_hcqt, CQT_FREQUENCIES
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter


class PitchSalience(nn.Module):
    def __init__(self):
        super(PitchSalience, self).__init__()
        self.conv1 = nn.Conv2d(5, 16, (3, 3), padding="same")
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, (3, 3), padding="same")
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 1, (3, 3), padding="same")
        self.flatten = nn.Flatten(1, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input is (batch, channels, time, freq)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)  # (batch, 1, time, freq)

        x = torch.transpose(x, 1, 2)  # (batch, time, 1, freq)
        x = torch.transpose(x, 2, 3)  # (batch, time, freq, 1)
        return x


class PitchData(Dataset):
    def __init__(self, data_path):
        self.example_files = glob.glob(os.path.join(data_path, "*.npz"))

    def __len__(self):
        return len(self.example_files)

    def __getitem__(self, idx):
        example_data = np.load(self.example_files[idx])
        return (
            torch.tensor(example_data["hcqt"]).float(),
            torch.tensor(example_data["target_salience"]).float(),
        )


def plot_predictions(model, cqt, salience):
    predicted_salience = model(cqt).detach()
    fig = plt.figure(figsize=(10, 10))
    n_examples = np.min([cqt.shape[0], 3])
    for i in range(n_examples):
        plt.subplot(n_examples, 3, 1 + (n_examples * i))
        plt.imshow(cqt[i, 1, :, :].T, origin="lower", cmap="magma")
        plt.clim(0, 1)
        plt.colorbar()
        plt.title(f"CQT")
        plt.axis("tight")

        plt.subplot(n_examples, 3, 2 + (n_examples * i))
        plt.imshow(salience[i, :, :, 0].T, origin="lower", cmap="magma")
        plt.clim(0, 1)
        plt.colorbar()
        plt.title("Target")
        plt.axis("tight")

        plt.subplot(n_examples, 3, 3 + (n_examples * i))
        plt.imshow(predicted_salience[i, :, :, 0].T, origin="lower", cmap="magma")
        plt.clim(0, 1)
        plt.colorbar()
        plt.title("Prediction")
        plt.axis("tight")

    plt.tight_layout()
    return fig


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    batch_size = 4

    model = PitchSalience().to(device)
    model = model
    print(model)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.ones([1]) * 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    training_dataset = PitchData("data/train")
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    validation_dataset = PitchData("data/validation")
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    writer = SummaryWriter("tensorboard")

    epochs = 1
    n_validation_batches = len(validation_dataloader)
    print_frequency = 100
    loss_counter = 0
    plot_counter = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        ## train
        model.train()
        running_loss = 0.0
        for batch, (X, y) in enumerate(train_dataloader):
            X = X.to(device)
            y = y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch % print_frequency == 0:
                print(f"batch: {batch}")
                print(f"loss: {running_loss/print_frequency}")
                writer.add_scalar("training loss", running_loss / print_frequency, loss_counter)
                loss_counter += 1
                running_loss = 0.0

            if batch % 1000 == 0:
                writer.add_figure(
                    "(Train) Predictions", plot_predictions(model, X, y), global_step=plot_counter
                )
                plot_counter += 1

                # validate
                model.eval()
                validation_loss = 0
                with torch.no_grad():
                    for X, y in validation_dataloader:
                        X, y = X.to(device), y.to(device)
                        pred = model(X)
                        validation_loss += loss_fn(pred, y).item()

                    writer.add_figure(
                        "(Validation) Predictions",
                        plot_predictions(model, X, y),
                        global_step=plot_counter,
                    )
                    validation_loss /= n_validation_batches
                    print(f"Avg validation loss: {validation_loss} \n")
                    writer.add_scalar("validation loss", validation_loss, loss_counter)
                model.train()

    # save the model
    torch.save(model.state_dict(), "pitch_salience.pt")

    print("Done!")


if __name__ == "__main__":
    main()