import argparse
import os

import librosa
import soundfile
import mir_eval
import numpy as np
import torch
from torch import nn
from train import PitchSalience
from constants import CQT_FREQUENCIES, compute_hcqt, load_audio, get_cqt_times
import mirdata


def run_inference(model, audio_path):
    """Run model inference on a full-length audio file

    Args:
        model (nn.Module): pytorch model
        audio_path (str): path to audio file to run inference on

    Returns:
        mirdata.annotations.F0Data: f0 data object, containing predicted f0 data
    """
    model.eval()  # put the model in evaluation mode

    # load audio (at the fixed sample rate)
    y = load_audio(audio_path)

    # compute input features
    hcqt = compute_hcqt(y)
    n_times = hcqt.shape[1]

    # the number of frames to run inference on at a time
    slice_size = 200
    outputs = []

    with torch.no_grad():
        # loop over the full time range
        for i in np.arange(0, n_times, step=slice_size):
            hcqt_tensor = torch.tensor(hcqt[np.newaxis, :, i : i + slice_size, :]).float()
            predicted_salience = model(hcqt_tensor)
            predicted_salience = nn.Sigmoid()(predicted_salience).detach()
            outputs.append(predicted_salience)

    # concatenate the outputs
    # NOTE: this is not the best approach! This will have boundary effects
    # every slice_size frames. To improve this, use e.g. overlap add
    unwrapped_prediction = np.hstack(outputs)[0, :n_times, :, 0].astype(float)

    # decode the output predictions into a single time series using viterbi decoding
    transition_matrix = librosa.sequence.transition_local(len(CQT_FREQUENCIES), 5)
    predicted_pitch_idx = librosa.sequence.viterbi(unwrapped_prediction.T, transition_matrix)

    # compute f0 and amplitudes using predicted indexes
    predicted_pitch = np.array([CQT_FREQUENCIES[f] for f in predicted_pitch_idx])
    predicted_salience = np.array(
        [unwrapped_prediction[i, f] for i, f in enumerate(predicted_pitch_idx)]
    )
    times = get_cqt_times(n_times)
    return mirdata.annotations.F0Data(
        times, "s", predicted_pitch, "hz", predicted_salience, "likelihood"
    )


def sonify_outputs(save_path, times, freqs, voicing):
    """Sonify the model outputs

    Args:
        save_path (str): path to save the audio file
        times (np.ndarray): time stamps (in seconds)
        freqs (np.ndarray): f0 values (in Hz)
        voicing (np.ndarray): voicing (between 0 and 1)
    """
    y = mir_eval.sonify.pitch_contour(times, freqs, 8000, amplitudes=voicing)
    soundfile.write(save_path, y, 8000)


def evaluate(model, sonification_dir):
    """Run evaluation on vocadito

    Args:
        model (nn.Module): pytorch model
        sonification_dir (str): path to save sonifications

    """
    scores = {}
    vocadito = mirdata.initialize("vocadito")
    # loop over the tracks in vocadito
    for track_id in vocadito.track_ids:
        track = vocadito.track(track_id)

        # get the reference f0 in mir_eval format
        ref_times, ref_freqs, _ = track.f0.to_mir_eval()

        # run model inference on this track
        estimated_f0 = run_inference(model, track.audio_path)

        # get the estimated f0 in mir_eval format
        est_times, est_freqs, est_voicing = estimated_f0.to_mir_eval()

        # sonify the estimates
        sonify_outputs(
            os.path.join(sonification_dir, f"{track_id}_f0est.wav"),
            est_times,
            est_freqs,
            est_voicing,
        )

        # compute metrics
        scores[track_id] = mir_eval.melody.evaluate(
            ref_times, ref_freqs, est_times, est_freqs, est_voicing=est_voicing
        )

        # print the scores for this track
        print(f"{track_id}: {scores[track_id]}")

    # metric averages
    print("===Average Metrics===")
    for metric in [
        "Overall Accuracy",
        "Raw Pitch Accuracy",
        "Raw Chroma Accuracy",
        "Voicing Recall",
        "Voicing False Alarm",
    ]:
        print(metric)
        print(np.mean([s[metric] for s in scores.values()]))


def main(args):

    # load the model
    model = PitchSalience()
    model.load_state_dict(torch.load(args.model_save_dir))

    # create sonification dir if it doesn't exist
    if not os.path.exists(args.sonification_dir):
        os.mkdir(args.sonification_dir)

    # run evaluation
    evaluate(model, args.sonification_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a pitch tracking model")
    parser.add_argument("model_save_dir", type=str, help="Path to the saved model")
    parser.add_argument("sonification_dir", type=str, help="Path to save sonifications")
    args = parser.parse_args()
    main(args)
