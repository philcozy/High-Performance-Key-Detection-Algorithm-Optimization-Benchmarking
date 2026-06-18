from scipy.io import wavfile
from scipy import signal
from scipy import fft
import numpy as np
import approximate_cqt as cqt
import matplotlib.pyplot as plt

# param
# fft
sample_rate = 11025
framesize = 4096
hopsize = framesize / 4

# cqt
bands = 72
offsets, kernels = cqt.build_kernal(bands, framesize, sample_rate)

# A440 sin wave
t = np.linspace(0, 2.0, int(sample_rate * 2.0), endpoint=False)
audio_data = np.sin(2 * np.pi * 440 * t)

# pick a frame
start = 3 * framesize
end = start + framesize
hamming = signal.windows.hamming(framesize)
frame = audio_data[start:end] * hamming

# fft
mag = np.abs( fft.rfft(frame) )

# approximate cqt
cqt_72 = cqt.apply_kernel(mag, offsets, kernels)

# fold to chroma
chroma = cqt_72.reshape(6, 12).sum(axis=0)
print(chroma)


