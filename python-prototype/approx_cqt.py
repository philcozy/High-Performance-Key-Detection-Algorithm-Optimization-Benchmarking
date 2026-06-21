import numpy as np
import scipy.signal.windows as windows

def band_freq(i, fref):
    return fref * (2 ** (i / 12)) 

def build_kernal(bands, framesize, sample_rate):
    # Q = 20.4 > 16.8 (the gap)
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
        center_freq = band_freq(i, 55.0)
        center_bin  = center_freq * framesize / sample_rate # bin k = f × N / sample_rate

        bin_width_hz    = center_freq / Q
        fft_bins_needed = bin_width_hz / (sample_rate / framesize)

        begin = int( np.ceil (center_bin - fft_bins_needed / 2) )
        end   = int( np.floor(center_bin + fft_bins_needed / 2) )
        
        # indexs of fft bins
        bins_in_window = np.arange(begin, end+1)

        # coeffs is a hann window by the size of fft bins needed
        coeffs = windows.hann( len(bins_in_window) )

        offsets.append(begin)
        kernels.append(coeffs)

    return offsets, kernels

def apply_kernel(mag, offsets, kernels):

    cqt = np.zeros( len(kernels) ) # 72 bands

    for i, (offset, coeffs) in enumerate(zip(offsets, kernels)):
        bins = mag[offset : offset + len(coeffs)]
        cqt[i] = np.dot(bins, coeffs)

    return cqt