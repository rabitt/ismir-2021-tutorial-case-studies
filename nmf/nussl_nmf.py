import librosa
import librosa.display
import numpy as np
import nussl
import matplotlib.pyplot as plt
import pretty_midi


def plot_nmf(components, activations, freqs, times, yaxis_note=False):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 7.5))
    n_components = components.shape[0]

    for n in range(n_components):
        ax1.plot(times, activations[n, :] + n)
        ax2.plot(components[n, :] + n, freqs)

    ax1.set_ylabel('Activation')
    ax1.set_xlabel('Time (sec)')
    ax1.xaxis.set_major_formatter(librosa.display.TimeFormatter())

    ax2.set_xlabel('Component')
    ax2.semilogy()
    if yaxis_note:
        ax2.yaxis.set_major_formatter(librosa.display.NoteFormatter())
        ax2.set_ylabel('Note')
    else:
        ax2.yaxis.set_major_formatter(librosa.display.LogHzFormatter())
        ax2.set_ylabel('Freq (Hz)')

    plt.show()


def load_signal(path):
    signal = nussl.AudioSignal(path)
    signal.to_mono()
    return signal


def do_nmf(signal, n_components, mode='frob'):
    nmf = nussl.separation.base.nmf_mixin.NMFMixin()

    solver = 'mu' if mode != 'frob' else 'cd'

    if mode == 'frob':
        b = 'frobenius'
    elif mode == 'kl':
        b = 'kullback-leibler'
    else:
        b = 'itakura-saito'

    nmf_model, components, activations = nmf.fit(signal, n_components,
                                                 solver=solver, beta_loss=b)

    activations = np.squeeze(activations)
    return nmf_model, components, activations


def windowed_peak_picker(arr, win_size, thresh_pct, do_diff=True):

    # Scale between [0.0, 1.0]
    arr -= np.min(arr)
    arr /= np.max(arr)

    # Zero everything below threshold %
    arr = np.multiply(arr, (arr >= thresh_pct))

    if do_diff:
        # Do peak picking on the discrete difference, i.e. arr[1] - arr[0], etc
        # don't care about offsets --> take max(0.0, diff)
        arr = np.maximum(0.0, np.diff(arr))

    peaks = []
    while np.sum(np.abs(arr)) > 0.0:
        peak_idx = np.argmax(arr)
        peaks.append(peak_idx)
        lower = int(np.maximum(0, peak_idx - win_size))
        upper = int(np.minimum(len(arr)-1, peak_idx + win_size))
        arr[lower:upper] = 0
    return peaks


def find_pitch(component, freqs):
    peak_idx = np.argmax(component)
    peak_freq = freqs[peak_idx]
    return int(np.round(librosa.hz_to_midi(peak_freq)))


def convert_to_pretty_midi(components, activations, win_size, note_len,
                           time_vector, freq_vector, pgm, is_drum, name='',
                           velocity=80, thresh_pct=0.5, pitch_dict=None):
    n_components = components.shape[0]
    inst = pretty_midi.Instrument(pgm, is_drum=is_drum, name=name)

    # Convert win_size from seconds to time frame units
    fr = time_vector[1] - time_vector[0]
    win_size = int(win_size / fr)

    for n in range(n_components):
        peak_idx = windowed_peak_picker(activations[n, :], win_size, thresh_pct)
        note_times = [time_vector[p] for p in peak_idx]

        if pitch_dict is None:
            pitch = find_pitch(components[n, :], freq_vector)
        else:
            pitch = pitch_dict[n]

        for start in note_times:
            note = pretty_midi.Note(velocity, pitch, start, start+note_len)
            inst.notes.append(note)

    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(inst)
    return pm


def drums():

    note_dist = 0.125  # seconds
    note_len = 0.125  # seconds

    # Do drum example
    drum_path = 'input/808_drums.wav'
    n_components = 5  # num notes, known ahead of time
    drums = load_signal(drum_path)

    nmf_model, components, activations = do_nmf(drums, n_components)
    plot_nmf(components, activations, drums.freq_vector, drums.time_bins_vector)

    # Convert it into a PrettyMIDI object
    drum_pgm = 0
    is_drum = True
    name = 'drums'
    pitch_dict = {
        0: librosa.note_to_midi('C1'),
        1: librosa.note_to_midi('C1'),
        2: librosa.note_to_midi('C1'),
        3: librosa.note_to_midi('C1'),
        4: librosa.note_to_midi('C1'),
    }
    drums_pm = convert_to_pretty_midi(components, activations, note_dist, note_len,
                                      drums.time_bins_vector, drums.freq_vector,
                                      drum_pgm, is_drum, name, pitch_dict=pitch_dict)


def piano():
    note_dist = 0.125  # seconds
    note_len = 0.125  # seconds
    do_plot = False

    piano_path = 'input/piano.wav'
    piano = load_signal(piano_path)
    n_components = 8  # num notes, known ahead of time

    nmf_model, components, activations = do_nmf(piano, n_components)

    if do_plot:
        plot_nmf(components, activations,
                 piano.freq_vector, piano.time_bins_vector, yaxis_note=True)

    # Convert it into a PrettyMIDI object
    piano_pgm = 0  # MIDI piano --> program number 0
    is_drum = False
    name = 'piano'
    piano_pm = convert_to_pretty_midi(components, activations, note_dist, note_len,
                                      piano.time_bins_vector, piano.freq_vector,
                                      piano_pgm, is_drum, name, pitch_dict=None)

    # Save audio file
    out_path = 'estimates/piano_sines.wav'
    sines = piano_pm.synthesize()
    piano_sines = nussl.AudioSignal(audio_data_array=sines, sample_rate=44100)
    piano_sines.write_audio_to_file(out_path)



if __name__ == '__main__':
    piano()