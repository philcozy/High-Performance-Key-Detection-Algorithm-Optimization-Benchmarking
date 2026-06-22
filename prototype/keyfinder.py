# lib
from scipy.io import wavfile
from scipy.signal import windows, resample_poly
from scipy import fft
import numpy as np
# python files
import approx_cpt as cqt
import keyprofile as kp

TARGET_SR = 11025
FRAMESIZE = 16384
HOPSIZE = FRAMESIZE // 4
BANDS = 72

def detect_key(audio_path):
    # audio io
    sr, audio = wavfile.read(audio_path)

    # handle input format
    if(len(audio.shape) > 1):
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float64)

    # down sample
    # increase frequecy resolution -> 11025 / 16384 = 0.673, sufficient for lower octave notes (A1 -> A#1 = 3.27hz), around 7 bins
    if sr != TARGET_SR:
        audio = resample_poly(audio, TARGET_SR, sr)
    sr = TARGET_SR

    #fft
    hamming = windows.hamming(FRAMESIZE)
    num_frames = ( (len(audio) - FRAMESIZE) // HOPSIZE ) + 1
    if num_frames < 1:
        return None # track too short
    
    sum_mag = np.zeros( (FRAMESIZE // 2) + 1 )

    for i in range(num_frames):
        start = i * HOPSIZE    
        end = start + FRAMESIZE

        frame = audio[start:end] * hamming
        mag = np.abs( fft.rfft(frame) )
        total = sum(mag)

        if total > 0:
            sum_mag += mag / sum(mag)

    # approximate cqt -> mapping fft to 72 bins with different bin width
    offsets, kernels = cqt.build_kernal(BANDS, FRAMESIZE, sr)
    cqt_72 = cqt.apply_kernel(sum_mag, offsets, kernels)

    # fold to chroma
    chroma = cqt_72.reshape(6, 12).sum(axis=0)

    # find best score from key profile
    return kp.get_key(chroma)
