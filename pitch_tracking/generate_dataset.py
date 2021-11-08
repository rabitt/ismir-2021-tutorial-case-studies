import argparse
from re import L
import mirdata
import numpy as np
import os

from constants import (
    N_EXAMPLES_PER_TRACK,
    N_TIME_FRAMES,
    CQT_FREQUENCIES,
    compute_hcqt,
    get_cqt_times,
    load_audio,
)


def generate_examples(split):
    """Generate full track audio - annotation examples

    Args:
        split (str): data split. One of 'train' or 'validation'

    Yields:
        (np.ndarray, mirdata.annotations.F0Data): audio, annotation tuple
    """
    medleydb_pitch = mirdata.initialize("medleydb_pitch")
    split_track_ids = medleydb_pitch.get_random_track_splits([0.9, 0.1])
    if split == "train":
        track_ids = split_track_ids[0]
    else:
        track_ids = split_track_ids[1]

    total_track_ids = len(track_ids)
    for i, track_id in enumerate(track_ids):
        print(f"Track {i+1}/{total_track_ids}: {track_id}")
        track = medleydb_pitch.track(track_id)
        y = load_audio(track.audio_path)
        yield (y, track.pitch)


def generate_samples_from_example(audio, pitch_data):
    """Generate training samples given an audio - annotation tuple

    Args:
        audio (np.ndarray): full length audio signal (sr=22050)
        pitch_data (mirdata.annotations.F0Data): f0 data object

    Yields:
        (np.ndarray, np.ndarray, int): harmonic CQT, target salience and slice index
    """
    hcqt = compute_hcqt(audio)
    n_times = hcqt.shape[1]
    times = get_cqt_times(n_times)

    target_salience = pitch_data.to_matrix(times, "s", CQT_FREQUENCIES, "hz")

    all_time_indexes = np.arange(0, n_times - N_TIME_FRAMES)
    time_indexes = np.random.choice(all_time_indexes, size=(N_EXAMPLES_PER_TRACK), replace=False)

    # split example into time slices
    for t_idx in time_indexes:
        sample = hcqt[:, t_idx : t_idx + N_TIME_FRAMES]
        label = target_salience[t_idx : t_idx + N_TIME_FRAMES, :, np.newaxis]
        yield (sample, label, t_idx)


def main(args):

    # create data save directories if they don't exist
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    splits = ["train", "validation"]
    for split in splits:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            os.mkdir(split_path)

    # initialize split-wise generators
    generators = [generate_examples(split) for split in splits]

    # generate training and validation data
    for generator, split in zip(generators, splits):
        print(f"Generating {split} data...")
        for i, (audio, pitch_data) in enumerate(generator):
            for hcqt, target_salience, t_idx in generate_samples_from_example(audio, pitch_data):
                fname = os.path.join(data_dir, split, f"{i}-{t_idx}.npz")
                np.savez(fname, hcqt=hcqt, target_salience=target_salience)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate npz files for training/validation")
    parser.add_argument("data_dir", type=str, help="Directory to save the data in")
    args = parser.parse_args()
    main(args)
