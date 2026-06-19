from scipy import signal
from scipy import fft
import numpy as np
import matplotlib.pyplot as plt

# param
sr = 11025
fs = 4096
hp = fs / 4

# A major chord tones
t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
freqs = [440.00, 554.37, 659.25]   # A, C#, E
audio = sum(np.sin(2 * np.pi * f * t) for f in freqs)
audio /= np.max(np.abs(audio))

# pick a frame
start = 3 * fs
hamming = signal.windows.hamming(fs)
frame = audio[start:start+fs] * hamming

# fft
mag = np.abs( fft.rfft(frame) )
freq = fft.rfftfreq(fs, d=1/sr)

plt.plot(freq, mag, color='b')
plt.grid(True)
plt.show()