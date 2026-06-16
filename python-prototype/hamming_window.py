import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt

size = 64

# own version
arr = np.zeros(size, dtype=np.float32)
for i in range(len(arr)):
    arr[i] = 0.54 - 0.46 * math.cos(2 * math.pi * i / (size - 1))

plt.plot(arr)

plt.ylabel("Amplitude")
plt.xlabel("Sample")

plt.show()


# scipy version
window = signal.windows.hamming(size)
plt.plot(window)

plt.ylabel("Amplitude")
plt.xlabel("Sample")

plt.show()