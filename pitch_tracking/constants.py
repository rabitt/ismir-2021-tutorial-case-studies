import librosa
import numpy as np

TARGET_SR = 22050
BINS_PER_SEMITONE = 3
N_OCTAVES = 6
FMIN = 32.7
BINS_PER_OCTAVE = 12 * BINS_PER_SEMITONE
N_BINS = N_OCTAVES * BINS_PER_OCTAVE
HOP_LENGTH = 512  # 23 ms hop
N_TIME_FRAMES = 50  # 1.16 seconds
N_AUDIO_SAMPLES = HOP_LENGTH * N_TIME_FRAMES
N_EXAMPLES_PER_TRACK = 100

CQT_FREQUENCIES = librosa.cqt_frequencies(N_BINS, FMIN, BINS_PER_OCTAVE)


def load_audio(audio_path):
    y, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    return y


def compute_hcqt(audio):
    cqt = librosa.cqt(
        audio,
        sr=TARGET_SR,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
    )
    cqt = (1.0 / 80.0) * librosa.amplitude_to_db(np.abs(cqt), ref=1.0) + 1.0

    hcqt = librosa.interp_harmonics(cqt, CQT_FREQUENCIES, [0.5, 1, 2, 3, 4])
    return hcqt.transpose([0, 2, 1])


def get_cqt_times(n_bins):
    return librosa.frames_to_time(np.arange(n_bins), sr=TARGET_SR, hop_length=HOP_LENGTH)
