from scipy.io import wavfile
from scipy import signal
from scipy import fft
import numpy as np
import approximate_cqt as cqt
import matplotlib.pyplot as plt

# audio_io
sample_rate, audio_data = wavfile.read("audio_io.wav")

if(len(audio_data.shape) > 1):
    audio_data = audio_data[:, 0]

# param
framesize = 4096
hopsize = framesize / 4

# pick a frame
start = 3 * framesize
end = start + framesize
hamming = signal.windows.hamming(framesize)
frame = audio_data[start:end] * hamming

# fft
mag = np.abs( fft.rfft(frame) )

# param
bands = 72

# lookup table
offsets, kernels = cqt.build_kernal(bands, framesize, sample_rate)

# approximate cqt
cqt_72 = cqt.apply_kernel(mag, offsets, kernels)

# fold to chroma
cqt_72.reshape(6, 12).sum(axis=0)

