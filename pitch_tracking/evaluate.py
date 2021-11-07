import librosa
import mir_eval
import numpy as np
import torch
from train import PitchSalience
from constants import CQT_FREQUENCIES, compute_hcqt, load_audio, get_cqt_times
import mirdata


def run_inference(model, audio_path):
    model.eval()
    y = load_audio(audio_path)
    hcqt = compute_hcqt(y)
    hcqt_tensor = torch.tensor(hcqt[np.newaxis, :, :, :]).float()

    unwrapped_prediction = model(hcqt_tensor).detach()

    n_times = hcqt.shape[1]
    times = get_cqt_times(n_times)
    # output_salience = np.zeros(hcqt[0, :, :].shape)
    # framed_cqt = librosa.util.frame(...)
    # predicted_salience = model(framed_cqt)
    # unwrapped_prediction = overlap_add(predicted_salience)
    predicted_pitch_idx = np.argmax(unwrapped_prediction, axis=0)
    predicted_pitch = np.array([CQT_FREQUENCIES[f] for f in predicted_pitch_idx])
    predicted_salience = np.array([unwrapped_prediction[f] for f in predicted_pitch_idx])
    return mirdata.annotations.F0Data(
        times, "s", predicted_pitch, "hz", predicted_salience, "likelihood"
    )


def evaluate(model):
    scores = {}
    vocadito = mirdata.initialize("vocadito")
    for track_id in vocadito.track_ids:
        track = vocadito.track(track_id)
        ref_times, ref_freqs, _ = track.f0.to_mir_eval()

        estimated_f0 = run_inference(model, track.audio_path)
        est_times, est_freqs, est_voicing = estimated_f0.to_mir_eval()
        scores[track_id] = mir_eval.melody.evaluate(
            ref_times, ref_freqs, est_times, est_freqs, est_voicing=est_voicing
        )
        print(scores)
        break


def main():

    model = PitchSalience()
    model.load_state_dict(torch.load("pitch_salience.pt"))
    model.eval()

    evaluate(model)


if __name__ == "__main__":
    main()
