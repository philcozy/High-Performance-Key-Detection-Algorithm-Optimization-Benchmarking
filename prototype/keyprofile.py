import numpy as np

# Bellman-Budge starts with A
nbb_major = [ 0.0804, 0.0062, 0.1057, 0.1680, 0.0086, 0.1295, 0.0141, 0.1349, 0.1193, 0.0125, 0.2028, 0.0180 ]
nbb_minor = [ 0.0153, 0.0092, 0.1021, 0.1816, 0.0069, 0.1299, 0.1334, 0.0107, 0.1115, 0.0138, 0.2107, 0.0749 ]

# Sha'ath starts with A
shaath_major = [0.08747, 0.05473, 0.06883, 0.15177, 0.05330, 0.08317, 0.05569, 0.10468, 0.09775, 0.06023, 0.12404, 0.05712 ]
shaath_minor = [0.09147, 0.07676, 0.07286, 0.14548, 0.06159, 0.08090, 0.12365, 0.05976, 0.08113, 0.05838, 0.10917, 0.09147 ]

# Temperley starts with A
temperley_major = [0.08537, 0.03659, 0.09756, 0.12195, 0.04878, 0.08537, 0.04878, 0.10976, 0.09756, 0.04878, 0.10976, 0.04878 ]
temperley_minor = [0.04878, 0.03659, 0.09756, 0.12195, 0.04878, 0.08537, 0.10976, 0.04878, 0.09756, 0.04878, 0.10976, 0.08537 ]

states = ('C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B',
          'c', 'c#', 'd', 'eb', 'e', 'f', 'f#', 'g', 'ab', 'a', 'bb', 'b')

tone = ("major", "minor")

def get_profile_for_key(idx, major=temperley_major, minor=temperley_minor):
    rotation = -(idx % 12)
    if idx < 12: 
        return (major[rotation:] + major[:rotation])
    else: 
        return (minor[rotation:] + minor[:rotation])

def get_key(chroma):
    best_score = -1
    idx = 0
    for i in range(24):
        score = np.dot(chroma, get_profile_for_key(i))
        if score > best_score:
            best_score = score
            idx = i
    return states[idx], tone[int(idx / 12)]
    