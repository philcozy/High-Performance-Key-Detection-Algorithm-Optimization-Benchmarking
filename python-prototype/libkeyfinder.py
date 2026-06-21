# python files
import approx_cqt as cqt
import keyprofile as kp

# lib
from scipy.io import wavfile
from scipy.signal import windows, butter, filtfilt
from scipy import fft
import numpy as np
import matplotlib.pyplot as plt

# audio io
sample_rate, audio_data = wavfile.read("audio_b_major.wav")

# mono reduction
if(len(audio_data.shape) > 1):
    audio_data = np.mean(audio_data, axis=1)

# down sample : increase frequecy resolution -> 11025 / 16384 = 0.673, sufficient for lower octave notes (A1 -> A#1 = 3.27hz), around 7 bins
# param
framesize = 16384
hopsize = framesize // 4
hamming = windows.hamming(framesize)

#fft
num_frames = ( (len(audio_data) - framesize) // hopsize ) + 1
sum_mag = np.zeros( (framesize // 2) + 1 )

for i in range(num_frames):
    start = i * hopsize     
    end = start + framesize

    frame = audio_data[start:end] * hamming
    mag = np.abs( fft.rfft(frame) )
    sum_mag += mag / sum(mag)

# approximate cqt -> mapping fft to 72 bins with different bin width
bands = 72
offsets, kernels = cqt.build_kernal(bands, framesize, sample_rate)
cqt_72 = cqt.apply_kernel(sum_mag, offsets, kernels)

# fold to chroma
chroma = cqt_72.reshape(6, 12).sum(axis=0)

# find best score from key profile
key, tone = kp.get_key(chroma)
print(key, tone)
