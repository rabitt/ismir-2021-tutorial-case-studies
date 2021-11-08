import argparse
import os
import glob

import numpy as np
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
        # no sigmoid at the end here, because we are using BCEWithLogitsLoss
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


def visualize(model, hcqt, salience):
    """Visualize the model inputs, predictions and targets

    Args:
        model (nn.Module): pytorch model
        hcqt (np.ndarray): hcqt matrix
        salience (np.ndarray): target salience matrix

    Returns:
        plt.Figure: matplotlib figure handle
    """
    predicted_salience = model(hcqt).detach()
    fig = plt.figure(figsize=(10, 10))
    n_examples = 3
    for i in range(n_examples):

        # plot the input
        plt.subplot(n_examples, 3, 1 + (n_examples * i))
        # use channel 1 of the hcqt, which corresponds to h=1
        plt.imshow(hcqt[i, 1, :, :].T, origin="lower", cmap="magma")
        plt.clim(0, 1)
        plt.colorbar()
        plt.title(f"HCQT")
        plt.axis("tight")

        # plot the target salience
        plt.subplot(n_examples, 3, 2 + (n_examples * i))
        plt.imshow(salience[i, :, :, 0].T, origin="lower", cmap="magma")
        plt.clim(0, 1)
        plt.colorbar()
        plt.title("Target")
        plt.axis("tight")

        # plot the predicted salience
        plt.subplot(n_examples, 3, 3 + (n_examples * i))
        plt.imshow(torch.sigmoid(predicted_salience[i, :, :, 0].T), origin="lower", cmap="magma")
        plt.clim(0, 1)
        plt.colorbar()
        plt.title("Prediction")
        plt.axis("tight")

    plt.tight_layout()
    return fig


def main(args):

    # Use the GPU if it's available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # initialize the model
    model = PitchSalience().to(device)
    print(model)

    # define the loss function & optimizer
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.ones([1]) * 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # initialize the training dataset/loader
    training_dataset = PitchData(os.path.join(args.data_dir, "train"))
    train_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)

    # initialize the validation dataset/loader
    validation_dataset = PitchData(os.path.join(args.data_dir, "validation"))
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False
    )

    # initialize the tensorboard writer
    writer = SummaryWriter(args.tensorboard_dir)

    # training & validation loop
    n_validation_batches = len(validation_dataloader)
    loss_counter = 0
    plot_counter = 0
    for epoch in range(args.n_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        ## train
        model.train()
        running_loss = 0.0
        for batch, (X, y) in enumerate(train_dataloader):

            # move the data to the GPU, ifusing
            X = X.to(device)
            y = y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log the loss every 100 steps
            running_loss += loss.item()
            if batch % 100 == 0:
                print(f"[{batch}] loss: {running_loss/100}")
                writer.add_scalar("training loss", running_loss / 100, loss_counter)
                loss_counter += 1
                running_loss = 0.0

            # visualize and validate every 1000 steps
            if batch % 1000 == 0:
                writer.add_figure(
                    "(Train) Predictions", visualize(model, X, y), global_step=plot_counter
                )
                plot_counter += 1

                # run validation
                model.eval()
                validation_loss = 0
                with torch.no_grad():
                    for X, y in validation_dataloader:
                        X, y = X.to(device), y.to(device)
                        pred = model(X)
                        validation_loss += loss_fn(pred, y).item()

                    # visualize the last validation batch
                    writer.add_figure(
                        "(Validation) Predictions",
                        visualize(model, X, y),
                        global_step=plot_counter,
                    )

                    # compute validation loss over full validation set
                    validation_loss /= n_validation_batches
                    print(f"Avg validation loss: {validation_loss} \n")
                    writer.add_scalar("validation loss", validation_loss, loss_counter)

                # put the model back in training mode
                model.train()

    # save the model
    torch.save(model.state_dict(), args.model_save_dir)
    print(f"Done! Saved model to {args.model_save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a pitch tracking model")
    parser.add_argument("data_dir", type=str, help="directory where the data lives")
    parser.add_argument("model_save_dir", type=str, help="Path to save the model")
    parser.add_argument(
        "tensorboard_dir", type=str, help="Path to save the tensorboard visualizations"
    )
    parser.add_argument("--n_epochs", type=int, help="Number of epochs to train for", default=10)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=4)
    args = parser.parse_args()
    main(args)
