import numpy as np
import soxr
from scipy.io import wavfile
from scipy.signal import windows, resample_poly, stft

# ---------- constants ----------

TARGET_SR = 4410                        # sample rate audio is resampled to before analysis
FRAME_SIZE = 16384                      # STFT window length, in samples
HOP_SIZE = FRAME_SIZE // 4              # STFT hop length, in samples
OVERLAP_SIZE = FRAME_SIZE - HOP_SIZE    #
NUM_BANDS = 72                          # number of CQT bins (6 octaves x 12 semitones)
NUM_CHROMA = 12                         # number of pitch classes after folding
Q_STRETCH = 0.9                         # scaling factor for CQT filter quality factor (Q)
REFERENCE_FREQ = 32.70                  # Hz, frequency of CQT band 0 (C1)
LOG_GAMMA = 100.0                       # compression strength in log(1 + gamma * |X|)
OCTAVE_WEIGHTS = np.array([
    0.39997267549999998559, 0.55634425248300645173, 0.52496636345143543600,
    0.60847548384277727607, 0.59898115679999996974, 0.49072435317960994006,
])

STATES = (
    "C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
    "c", "c#", "d", "eb", "e", "f", "f#", "g", "ab", "a", "bb", "b",
)

# Key profiles: (major_weights, minor_weights), each 12 values indexed C..B
SHAATH = (
    [7.23900502618145225142, 3.50351166725158691406, 3.58445177536649417505, 2.84511816478676315967,
     5.81898892118549859731, 4.55865057415321039969, 2.44778850545506543313, 6.99473192146829525484,
     3.39106613673504853068, 4.55614256655143456953, 4.07392666663523606019, 4.45932757378886890365],
    [7.00255045060284420089, 3.14360279015996679775, 4.35904319714962529275, 5.40418120718934069657,
     3.67234420879306133756, 4.08971184917797891956, 3.90791435991553992579, 6.19960288562316463867,
     3.63424625625277419871, 2.87241191079875557435, 5.35467999794542670600, 3.83242038595048351013],
)

# Functions detect_key() calls, in order; the benchmark times each one
STAGES = ("preprocess", "spectrum", "cqt", "fold", "classify")

PROFILES = {
    "shaath": SHAATH,
}


# ---------- stage 1: load + preprocess audio ----------

def preprocess(audio_path):
    """Load a WAV file and return (mono float64 audio in [-1, 1], TARGET_SR)."""
    sr, audio = wavfile.read(audio_path)
    audio = audio.astype(np.float64) / 32768.0

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != TARGET_SR:
        audio = soxr.resample(audio, sr, TARGET_SR, quality='HQ')

    return audio, TARGET_SR


# ---------- stage 2: STFT magnitude spectrum ----------

def spectrum(audio, sr):
    """Compute the STFT of audio and sum magnitudes across time into one spectrum."""
    _, _, stft_matrix = stft(
        audio, fs=sr, window="hamming", nperseg=FRAME_SIZE, noverlap=OVERLAP_SIZE
    )
    return np.abs(stft_matrix)


# ---------- stage 3: approximate CQT ----------

def _band_center_freq(band_index, reference_freq=REFERENCE_FREQ):
    """Return the center frequency, in Hz, of the given semitone band index."""
    return reference_freq * (2 ** (band_index / 12))


def build_cqt_kernel_matrix(num_bands=NUM_BANDS, frame_size=FRAME_SIZE, sample_rate=TARGET_SR):
    Q = Q_STRETCH * (2 ** (1.0 / 12) - 1)
    num_fft_bins = frame_size // 2 + 1

    kernel_matrix = np.zeros((num_bands, num_fft_bins))

    for band_index in range(num_bands):
        center_freq = _band_center_freq(band_index)
        center_bin = center_freq * (frame_size / sample_rate)

        bin_width_hz = center_freq * Q
        bins_needed = bin_width_hz / (sample_rate / frame_size)

        begin = int(np.ceil(center_bin - bins_needed / 2))
        end = int(np.floor(center_bin + bins_needed / 2))

        raw_window = windows.hann(end - begin + 1)
        kernel = raw_window / raw_window.sum()
        kernel_matrix[band_index, begin : end + 1] = kernel

    return kernel_matrix


def apply_cqt_kernel(magnitude, kernel_matrix):
    """Project each STFT frame onto the CQT kernels -> shape (NUM_BANDS, frames)."""
    return np.matmul(kernel_matrix, magnitude)


def cqt(magnitude):
    """Map an FFT magnitude spectrum to a NUM_BANDS-bin approximate CQT."""
    kernel_matrix = build_cqt_kernel_matrix()
    return apply_cqt_kernel(magnitude, kernel_matrix)


# ---------- stage 4: log compress each frames --------
def compress(cqt_mag):
    """Return log(1 + LOG_GAMMA * |X|) per frame, summed across time into one spectrum."""
    cqt_mag = cqt_mag / cqt_mag.max()
    return np.log1p(LOG_GAMMA * cqt_mag).sum(axis=1)

# ---------- stage 5: fold CQT into chroma ----------

def fold(cqt_bins, num_chroma=NUM_CHROMA, weights=OCTAVE_WEIGHTS):
    """Fold a multi-octave CQT into a single NUM_CHROMA-bin chroma vector."""
    num_octaves = len(cqt_bins) // num_chroma

    reshaped_cqt = cqt_bins.reshape(num_octaves, num_chroma)
    return ( reshaped_cqt * weights[:, np.newaxis] ).sum(axis=0)


# ---------- stage 6: key classification ----------

def get_profile(key_index, major, minor):
    """Return the rotated major/minor profile for key_index (0-11 major, 12-23 minor)."""
    root = key_index % 12
    profile = major if key_index < 12 else minor
    return np.roll(profile, root)

def cosine_similarity(x, y):
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

def get_all_scores(chroma, major, minor):
    """Calculate cosine similarity scores between the input chroma vector and all 24 major/minor key profiles."""
    scores = []
    for i in range(24):
        profile = get_profile(i, major, minor)
        score = cosine_similarity(chroma, profile)
        scores.append(score)
    return scores

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
    cqt_mag = cqt(mag)
    cqt_log = compress(cqt_mag)
    chroma = fold(cqt_log)
    return classify(chroma, major, minor)