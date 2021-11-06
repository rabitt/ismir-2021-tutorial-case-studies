import argparse
import librosa
from librosa.core import pitch
import mirdata
import numpy as np


from constants import (
    TARGET_SR,
    BINS_PER_SEMITONE,
    N_OCTAVES,
    FMIN,
    BINS_PER_OCTAVE,
    N_BINS,
    HOP_LENGTH,
    N_TIME_FRAMES,
    N_AUDIO_SAMPLES,
    FREQUENCIES,
    GET_TIMES,
)
from pitch_tracking.constants import COMPUTE_CQT


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


def pitch_data_generator():
    medleydb_pitch = mirdata.initialize("medleydb_pitch")
    for track in medleydb_pitch.load_tracks():
        y, sr = track.audio
        yield (y, sr, track.pitch)

    # vocadito = mirdata.initialize("vocadito")
    # for track in vocadito.load_tracks():
    #     y, sr = track.audio
    #     yield(y, sr, track.f0)

    tonas = mirdata.initialize("tonas")
    for track in tonas.load_tracks():
        y, sr = track.audio
        yield (y, sr, track.f0_corrected)


def apply_augmentations_to_example(audio, pitch_data):
    return audio, pitch_data


def generate_examples(split):
    medleydb_pitch = mirdata.initialize("medleydb_pitch")
    split_track_ids = medleydb_pitch.get_track_splits([0.9, 0.1])
    if split == "train":
        track_ids = split_track_ids[0]
    else:
        track_ids = split_track_ids[1]

    for track_id in track_ids:
        track = medleydb_pitch.track(track_id)
        y, sr = track.audio
        yield (y, sr, track.pitch)


def generate_samples_from_example(audio_in, sr_in, pitch_data):
    audio = librosa.resample(audio_in, sr_in, TARGET_SR)

    audio, pitch_data = apply_augmentations_to_example(audio, pitch_data)

    cqt = COMPUTE_CQT(audio)
    n_times = cqt.shape[1]
    times = GET_TIMES(n_times)

    target_salience = pitch_data.to_matrix(times, "s", FREQUENCIES, "hz")

    time_indexes = np.arange(0, n_times, step=N_TIME_FRAMES / 2)

    for t_idx in time_indexes:
        sample = cqt[np.newaxis, :, t_idx : t_idx + N_TIME_FRAMES]
        label = target_salience[t_idx : t_idx + N_TIME_FRAMES, :, np.newaxis]
        yield (sample, label, t_idx)


def main(args):

    train_example_generator = generate_examples("train")
    validation_example_generator = generate_examples("validation")

    for generator, savedir in zip(
        [train_example_generator, validation_example_generator], ["train", "validation"]
    ):
        for audio, sr, pitch_data in generator:
            fname = os.path.join(savedir, "asdf.npz")
            for sample, label, t_idx in generate_samples_from_example(audio, sr, pitch_data):
                np.savez(fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tf record files to train pitch tracker")
    parser.add_argument(
        "asdf",
        type=str,
        help="",
    )
    parser.add_argument(
        "--vocals-only",
        action="store_true",
        help="Only generate dataset for vocals",
    )
    args = parser.parse_args()
    main(args)