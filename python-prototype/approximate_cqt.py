import numpy as np

def band_freq(band_index):
    A1_hz = 55.0
    semitones = 12
    return A1_hz * (2 ** (band_index / semitones)) 

def build_kernal(bands, framesize, sample_rate):
    # Q = center / bandwidth ≥ 1 / 0.0595
    Q =  1.2 * 1 / (2 ** (1.0 / 12) - 1) 
    
    """
    band 0  (A1, 55 Hz):
    offsets[0] = 5          ← "start at FFT bin 5"
    kernels[0] = [0.3, 0.8, 1.2, 0.8, 0.3]   ← 5 weights, one per bin
    """

    # init lookup table
    offsets = []
    kernels = []

    for i in range(bands):
        center_freq = band_freq(i)
        center_bin = center_freq * framesize / sample_rate
        bin_width = center_bin / Q

        begin = np.ceil(center_bin - bin_width/2)
        end = np.floor(center_bin + bin_width/2)

        if end - begin < 2:
            center = int(np.round(center_bin))
            begin, end = center - 1, center + 1
        
        bins_in_window = np.arange(begin, end+1)
        n = bins_in_window - begin

        coeffs = 1.0 - np.cos(2 * np.pi * n / bins_in_window)
        coeffs = coeffs / coeffs.sum()

        offsets.append(int(begin))
        kernels.append(coeffs)

    return offsets, kernels

def apply_kernel(mag, offsets, kernels):

    cqt = np.zeros(len(kernels)) # 72 bands

    for i, (offset, coeffs) in enumerate(zip(offsets, kernels)):
        bins = mag[offset : offset + len(coeffs)]
        cqt[i] = np.dot(bins, coeffs)

    return cqt