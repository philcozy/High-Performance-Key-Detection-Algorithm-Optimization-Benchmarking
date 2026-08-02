import numpy as np
from scipy.io import wavfile
from scipy.signal import windows, resample_poly, stft

# ---------- constants ----------

TARGET_SR = 4410                        # sample rate audio is resampled to before analysis
FRAME_SIZE = 16384                      # STFT window length, in samples
HOP_SIZE = FRAME_SIZE // 4              # STFT hop length, in samples
OVERLAP_SIZE = FRAME_SIZE - HOP_SIZE    #
NUM_BANDS = 72                          # number of CQT bins (6 octaves x 12 semitones)
NUM_CHROMA = 12                         # number of pitch classes after folding
Q_STRETCH = 1.2                         # multiplier applied to the CQT quality factor
REFERENCE_FREQ = 32.70                  # Hz, frequency of CQT band 0 (C1)
OCTAVE_WEIGHTS = np.array([0.6, 1, 1, 1, 0.8, 0.6])

STATES = (
    "C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
    "c", "c#", "d", "eb", "e", "f", "f#", "g", "ab", "a", "bb", "b",
)

# Key profiles: (major_weights, minor_weights), each 12 values indexed C..B
KRUMHANSL_SCHMUCKLER = (
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
)
BELLMAN_BUDGE = (
    [16.80, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 20.28, 1.80, 8.04, 0.62, 10.57],
    [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 21.07, 7.49, 1.53, 0.92, 10.21],
)
SHAATH = (
    [7.239, 3.504, 3.584, 2.845, 5.819, 4.559, 2.448, 6.995, 3.391, 4.556, 4.074, 4.459],
    [7.003, 3.144, 4.359, 5.404, 3.672, 4.089, 3.907, 6.200, 3.634, 2.872, 5.355, 3.832],
)
TEMPERLEY = (
    [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
    [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0],
)

PROFILES = {
    "temperley": TEMPERLEY,
    "bellman_budge": BELLMAN_BUDGE,
    "shaath": SHAATH,
    "krumhansl_schmuckler": KRUMHANSL_SCHMUCKLER,
}


# ---------- stage 1: load + preprocess audio ----------

def preprocess(audio_path):
    """Load a WAV file and return (mono float64 audio in [-1, 1], TARGET_SR)."""
    sr, audio = wavfile.read(audio_path)
    audio = audio.astype(np.float64) / 32768.0

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != TARGET_SR:
        audio = resample_poly(audio, TARGET_SR, sr)

    return audio, TARGET_SR


# ---------- stage 2: STFT magnitude spectrum ----------

def spectrum(audio, sr):
    """Compute the STFT of audio and sum magnitudes across time into one spectrum."""
    _, _, stft_matrix = stft(
        audio, fs=sr, window="hamming", nperseg=FRAME_SIZE, noverlap=OVERLAP_SIZE
    )
    return np.abs(stft_matrix).sum(axis=1)


# ---------- stage 3: approximate CQT ----------

def _band_center_freq(band_index, reference_freq=REFERENCE_FREQ):
    """Return the center frequency, in Hz, of the given semitone band index."""
    return reference_freq * (2 ** (band_index / 12))


def build_cqt_kernel(num_bands=NUM_BANDS, frame_size=FRAME_SIZE, sample_rate=TARGET_SR):
    """Build per-band (start_bin, hann_window) kernels approximating a CQT."""
    quality_factor = Q_STRETCH / (2 ** (1.0 / 12) - 1)

    offsets = []
    kernels = []
    for band_index in range(num_bands):
        center_freq = _band_center_freq(band_index)
        center_bin = center_freq * (frame_size / sample_rate)

        bin_width_hz = center_freq / quality_factor
        bins_needed = bin_width_hz / (sample_rate / frame_size)

        begin = int(np.ceil(center_bin - bins_needed / 2))
        end = int(np.floor(center_bin + bins_needed / 2))

        offsets.append(begin)
        kernels.append(windows.hann(end - begin + 1))

    return offsets, kernels


def apply_cqt_kernel(magnitude, offsets, kernels):
    """Project an FFT magnitude spectrum onto the CQT kernels, one value per band."""
    cqt_values = np.zeros(len(kernels))
    for i, (offset, window) in enumerate(zip(offsets, kernels)):
        bins = magnitude[offset : offset + len(window)]
        cqt_values[i] = np.dot(bins, window)
    return cqt_values


def cqt(magnitude):
    """Map an FFT magnitude spectrum to a NUM_BANDS-bin approximate CQT."""
    offsets, kernels = build_cqt_kernel()
    return apply_cqt_kernel(magnitude, offsets, kernels)


# ---------- stage 4: fold CQT into chroma ----------

def fold(cqt_bins, num_chroma=NUM_CHROMA, weights=OCTAVE_WEIGHTS):
    """Fold a multi-octave CQT into a single NUM_CHROMA-bin chroma vector."""
    num_octaves = len(cqt_bins) // num_chroma

    reshaped_cqt = cqt_bins.reshape(num_octaves, num_chroma)
    return ( reshaped_cqt * weights[:, np.newaxis] ).sum(axis=0)


# ---------- stage 5: key classification ----------

def get_profile(key_index, major, minor):
    """Return the rotated major/minor profile for key_index (0-11 major, 12-23 minor)."""
    root = key_index % 12
    profile = major if key_index < 12 else minor
    return np.roll(profile, root)

def get_all_scores(chroma, major, minor):
    """Return the Pearson correlation between chroma and each of the 24 key profiles."""
    return [
        float(np.corrcoef(chroma, get_profile(i, major, minor))[0, 1])
        for i in range(24)
    ]


def classify(chroma, major, minor):
    """Return (tonic, mode) for the best-matching key profile."""
    scores = get_all_scores(chroma, major, minor)
    best_index = int(np.argmax(scores))
    tonic = STATES[best_index]
    mode = "major" if best_index < 12 else "minor"
    return tonic, mode


# ---------- pipeline entry point ----------

def detect_key(audio_path, major=SHAATH[0], minor=SHAATH[1]):
    """Run the full pipeline on a WAV file and return its (tonic, mode)."""
    audio, sr = preprocess(audio_path)
    mag = spectrum(audio, sr)
    cqt_bins = cqt(mag)
    chroma = fold(cqt_bins)
    return classify(chroma, major, minor)