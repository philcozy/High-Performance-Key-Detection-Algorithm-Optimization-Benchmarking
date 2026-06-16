from scipy.io import wavfile
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

sample_rate, audio_data = wavfile.read("audio_io.wav")

if(len(audio_data.shape) > 1):
    audio_data = audio_data[:, 0]

freq, time, Sxx = scipy.signal.spectrogram(audio_data, fs=sample_rate)

Sxx_db = 10 * np.log10(Sxx + 1e-10)

plt.pcolormesh(time, freq, Sxx_db, cmap="viridis")

plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")

plt.show()