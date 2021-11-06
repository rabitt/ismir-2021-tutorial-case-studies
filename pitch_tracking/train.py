import librosa
import numpy as np
from pitch_tracking.constants import COMPUTE_CQT, FREQUENCIES
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import mirdata
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter


class PitchSalience(nn.Module):
    def __init__(self):
        super(PitchSalience, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 3), padding="same")
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
    def __init__(self, split, examples_per_track=1):
        assert split in ["train", "validation"]
        self.split = split
        self.examples_per_track = examples_per_track

        self.medleydb_pitch = mirdata.initialize("medleydb_pitch")

        rng = np.random.default_rng(seed=42)
        track_ids_permuted = rng.permutation(self.medleydb_pitch.track_ids)
        split_idx = int(np.round(0.9 * len(track_ids_permuted)))
        if split == "train":
            self.track_ids = track_ids_permuted[:split_idx]
        else:
            self.track_ids = track_ids_permuted[split_idx:]

        # time-frequency parameters
        self.target_sr = 22050
        self.bins_per_semitone = 3
        self.n_octaves = 6
        self.fmin = 32.7
        self.bins_per_octave = 12 * self.bins_per_semitone
        self.n_bins = self.n_octaves * self.bins_per_octave
        self.hop_length = 512  # 23 ms hop
        self.n_time_frames = 50  # 1.16 seconds
        self.n_audio_samples = self.hop_length * self.n_time_frames

        self.frequencies = librosa.cqt_frequencies(self.n_bins, self.fmin, self.bins_per_octave)
        self.get_times = lambda n_bins: librosa.frames_to_time(
            np.arange(n_bins), sr=self.target_sr, hop_length=self.hop_length
        )

    def __len__(self):
        return len(self.track_ids) * self.examples_per_track

    def __getitem__(self, idx):
        track_id_idx = int(np.floor(idx / self.examples_per_track))
        print(self.track_ids[track_id_idx])
        track = self.medleydb_pitch.track(self.track_ids[track_id_idx])

        audio, sr = track.audio
        audio = librosa.resample(audio, sr, self.target_sr)
        cqt = librosa.cqt(
            audio,
            sr=self.target_sr,
            hop_length=self.hop_length,
            fmin=self.fmin,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
        )
        n_times = cqt.shape[1]
        times = self.get_times(n_times)
        freqs = self.frequencies

        pitch_data = track.pitch
        target_salience = pitch_data.to_matrix(times, "s", freqs, "hz")
        random_time_idx = np.random.randint(0, n_times - self.n_time_frames)

        sample = (1.0 / 80.0) * librosa.amplitude_to_db(
            np.abs(cqt[:, random_time_idx : random_time_idx + self.n_time_frames].T), ref=1.0
        ) + 1.0
        label = target_salience[random_time_idx : random_time_idx + self.n_time_frames, :]
        audio_start_idx = int(times[random_time_idx] * self.target_sr)
        return (
            torch.tensor(sample[np.newaxis, :, :]).float(),
            torch.tensor(label[:, :, np.newaxis]).float(),
            {
                "track_id": self.track_ids[track_id_idx],
                "audio": audio[audio_start_idx : audio_start_idx + self.n_audio_samples],
                "instrument": track.instrument,
            },
        )


def plot_predictions(model, cqt, salience, metadata):
    predicted_salience = model(cqt).detach()
    fig = plt.figure(figsize=(15, 7))
    n_examples = np.min([cqt.shape[0], 3])
    for i in range(n_examples):
        plt.subplot(n_examples, 3, 1 + (n_examples * i))
        plt.imshow(cqt[i, 0, :, :].T, origin="lower", cmap="magma")
        plt.clim(0, 1)
        plt.title(f'CQT: {metadata["track_id"][i]}, {metadata["instrument"][i]}')
        plt.axis("tight")

        plt.subplot(n_examples, 3, 2 + (n_examples * i))
        plt.imshow(salience[i, :, :, 0].T, origin="lower", cmap="magma")
        plt.clim(0, 1)
        plt.title("Target")
        plt.axis("tight")

        plt.subplot(n_examples, 3, 3 + (n_examples * i))
        plt.imshow(predicted_salience[i, :, :, 0].T, origin="lower", cmap="magma")
        plt.clim(0, 1)
        plt.title("Prediction")
        plt.axis("tight")

    plt.suptitle(f'{metadata["track_id"][i]}, {metadata["instrument"][i]}')
    plt.tight_layout()
    return fig


def run_inference(model, audio):
    cqt = COMPUTE_CQT(audio)
    framed_cqt = librosa.util.frame(...)
    predicted_salience = model(framed_cqt)
    unwrapped_prediction = overlap_add(predicted_salience)
    predicted_pitch_idx = np.argmax(unwrapped_prediction)
    predicted_pitch = [FREQUENCIES[f] for f in predicted_pitch_idx]
    predicted_salience = [unwrapped_prediction[f] for f in predicted_pitch_idx]
    return predicted_pitch, predicted_salience


def train(dataloader, model, loss_fn, optimizer, device, writer, epoch):
    model.train()
    running_loss = 0.0
    for batch, (X, y, m) in enumerate(dataloader):
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
        if batch % 100 == 0:
            n_examples_used = batch * len(X)
            print(f"loss: {running_loss/1000}  [{n_examples_used:>5d}/{total_examples:>5d}]")
            writer.add_scalar("training loss", running_loss / 1000, epoch)
            running_loss = 0.0

            writer.add_figure(
                "(Train) Predictions", plot_predictions(model, X, y, m), global_step=epoch
            )
            for i in range(3):
                writer.add_audio(
                    f'{m["track_id"][i]}: {m["instrument"][i]}',
                    m["audio"][i],
                    global_step=epoch,
                    sample_rate=22050,
                )

            break


def validate(dataloader, model, loss_fn, device, writer, epoch):
    num_batches = len(dataloader)
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for X, y, m in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            validation_loss += loss_fn(pred, y).item()
            break

    validation_loss /= num_batches
    print(f"Avg validation loss: {validation_loss:>8f} \n")
    writer.add_scalar("validation loss", validation_loss / 1000, epoch)


def evaluate(model):
    pass


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    batch_size = 4

    model = PitchSalience().to(device)
    model = model
    print(model)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.ones([1]) * 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    training_dataset = PitchData("train", 1)
    validation_dataset = PitchData("validation", 1)

    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    writer = SummaryWriter("tensorboard")
    writer.add_graph(model, training_dataset[0][0][np.newaxis, :, :, :])
    epochs = 5
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device, writer, epoch)
        validate(validation_dataloader, model, loss_fn, device, writer, epoch)

    evaluate(model)
    print("Done!")


if __name__ == "__main__":
    main()