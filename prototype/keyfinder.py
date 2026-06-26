from scipy.io import wavfile
from scipy.signal import windows, resample_poly
from scipy import fft
import numpy as np

TARGET_SR = 11025
FRAMESIZE = 16384
HOPSIZE = FRAMESIZE // 4
BANDS = 72
DIRECTSKSTRETCH = 1.0

# ---------- approx CQT ----------

def _band_freq(i, fref=32.70):
    return fref * (2 ** (i / 12))

def build_cqt_kernel(bands=BANDS, framesize=FRAMESIZE, sample_rate=TARGET_SR):
    Q = DIRECTSKSTRETCH * 1.0 / (2 ** (1.0 / 12) - 1)

    offsets, kernels = [], []
    for i in range(bands):
        center_freq = _band_freq(i)
        center_bin = center_freq * (framesize / sample_rate)

        bin_width_hz = center_freq / Q
        fft_bins_needed = bin_width_hz / (sample_rate / framesize)

        begin = int(np.ceil(center_bin - fft_bins_needed / 2))
        end = int(np.floor(center_bin + fft_bins_needed / 2))

        coeffs = windows.hann(end - begin + 1)
        offsets.append(begin)
        kernels.append(coeffs)
    return offsets, kernels

def apply_cqt_kernel(mag, offsets, kernels):
    cqt = np.zeros(len(kernels))
    for i, (offset, coeffs) in enumerate(zip(offsets, kernels)):
        bins = mag[offset: offset + len(coeffs)]
        cqt[i] = np.dot(bins, coeffs)
    return cqt

# ---------- key profiles ----------

_bb_major    = [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 20.28, 1.8, 8.04, 0.62, 10.57]
_bb_minor    = [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 21.07, 7.49, 1.53, 0.92, 10.21]
_shaath_major = [7.239, 3.504, 3.584, 2.845, 5.819, 4.559, 2.448, 6.995, 3.391, 4.556, 4.074, 4.459]
_shaath_minor = [7.003, 3.144, 4.359, 5.404, 3.672, 4.089, 3.907, 6.200, 3.634, 2.872, 5.355, 3.832]
_temperley_major = [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]
_temperley_minor = [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]

PROFILES = {
    'temperley': (_temperley_major, _temperley_minor),
    'bellman_budge': (_bb_major, _bb_minor),
    'shaath': (_shaath_major, _shaath_minor),
}

STATES = ('C','Db','D','Eb','E','F','F#','G','Ab','A','Bb','B',
          'c','c#','d','eb','e','f','f#','g','ab','a','bb','b')

def _get_profile(idx, major, minor):
    r = -(idx % 12)
    return (major[r:] + major[:r]) if idx < 12 else (minor[r:] + minor[:r])

def get_all_scores(chroma, major=_temperley_major, minor=_temperley_minor):
    """Return correlation scores for all 24 keys."""
    return [np.dot(chroma, _get_profile(i, major, minor)) for i in range(24)]

def get_key(chroma, major, minor):
    """Return (tonic_str, mode_str) for best-matching key"""
    scores = get_all_scores(chroma, major, minor)
    idx = int(np.argmax(scores))
    return STATES[idx], ('major' if idx < 12 else 'minor')

# ---------- pipeline steps ----------

def load_audio(audio_path):
    """Load and pre-process audio: mono, float64, downsampled to TARGET_SR.
    Returns (audio, sr)."""
    sr, audio = wavfile.read(audio_path)
    audio = audio.astype(np.float64)

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if sr != TARGET_SR:
        audio = resample_poly(audio, TARGET_SR, sr)
    return audio

def compute_avg_spectrum(audio):
    """Average normalised magnitude spectrum across Hamming-windowed frames.
    Returns sum_mag array of shape (FRAMESIZE//2 + 1,)."""
    hamming = windows.hamming(FRAMESIZE)

    num_frames = ((len(audio) - FRAMESIZE) // HOPSIZE) + 1
    if num_frames < 1:
        return None
    
    sum_mag = np.zeros((FRAMESIZE // 2) + 1)
    for i in range(num_frames):
        start = i * HOPSIZE
        frame = audio[start: start + FRAMESIZE] * hamming

        mag = np.abs(fft.rfft(frame))
        total = mag.sum()
        if total > 0:
            sum_mag += mag / total
    return sum_mag

def compute_cqt(sum_mag):
    """Map average spectrum to 72-bin approx CQT."""
    offsets, kernels = build_cqt_kernel()
    return apply_cqt_kernel(sum_mag, offsets, kernels)

def compute_chroma(cqt_72):
    """Fold 72-bin CQT to 12-bin chroma"""
    return cqt_72.reshape(6, 12).sum(axis=0)

# ---------- convenience ----------

def detect_key(audio_path, major=_shaath_major, minor=_shaath_minor):
    audio = load_audio(audio_path)
    sum_mag = compute_avg_spectrum(audio)
    if sum_mag is None:
        return None
    cqt_72 = compute_cqt(sum_mag)
    chroma = compute_chroma(cqt_72)
    return get_key(chroma, major, minor)
