"""
Debug pipeline: side-by-side comparison of librosa vs prototype at each step.
Usage: python debug_pipeline.py <audio.wav>

Output figures are saved to debug/output/.
Each figure has two subplots per row:  left = librosa,  right = prototype.
"""

import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import librosa

# ---- import prototype ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'prototype'))
from keyfinder import (
    load_audio, compute_avg_spectrum, compute_cqt, compute_chroma,
    get_all_scores, get_key, STATES, TARGET_SR, FRAMESIZE,
    _temperley_major, _temperley_minor,
)

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

PITCH_C_START = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
KEY_LABELS = list(STATES)


def _save(fig, name, stem):
    path = OUTPUT_DIR / f'{stem}_{name}.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def run_debug(audio_path):
    audio_path = Path(audio_path)
    stem = audio_path.stem
    print(f'\n=== debug pipeline: {audio_path.name} ===\n')

    # ---- prototype pipeline ----
    proto_audio   = load_audio(str(audio_path))
    proto_sum_mag = compute_avg_spectrum(proto_audio)
    proto_cqt72   = compute_cqt(proto_sum_mag)
    proto_chroma  = compute_chroma(proto_cqt72)
    proto_scores  = get_all_scores(proto_chroma)
    proto_key     = get_key(proto_chroma, _temperley_major, _temperley_minor)

    # ---- librosa pipeline: audio -> chroma_cqt -> keyprofile ----
    lib_audio, lib_sr = librosa.load(str(audio_path), sr=None, mono=True)
    lib_chroma_cqt  = librosa.feature.chroma_cqt(y=lib_audio, sr=lib_sr)  # (12, T) C-starting
    lib_chroma_sum  = lib_chroma_cqt.sum(axis=1)                           # (12,) C-starting
    lib_scores      = get_all_scores(lib_chroma_sum)
    lib_key         = get_key(lib_chroma_sum, _temperley_major, _temperley_minor)

    print(f'  librosa  key: {lib_key[0]} {lib_key[1]}')
    print(f'  prototype key: {proto_key[0]} {proto_key[1]}')

    # ======================================================
    # Figure 1: Audio waveform
    # ======================================================
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 3))
    fig.suptitle('Step 1 — Audio waveform', fontsize=13)

    t_lib   = np.arange(len(lib_audio))   / lib_sr
    t_proto = np.arange(len(proto_audio)) / TARGET_SR

    ax_l.plot(t_lib,   lib_audio,   linewidth=0.4)
    ax_l.set_title('librosa (native sr)')
    ax_l.set_xlabel('Time (s)')
    ax_l.set_ylabel('Amplitude')

    ax_r.plot(t_proto, proto_audio, linewidth=0.4, color='tab:orange')
    ax_r.set_title(f'prototype (resampled to {TARGET_SR} Hz)')
    ax_r.set_xlabel('Time (s)')
    ax_r.set_ylabel('Amplitude')

    plt.tight_layout()
    _save(fig, '1_waveform', stem)

    # ======================================================
    # Figure 2: Intermediate frequency representation
    #   left  = librosa chroma_cqt heatmap (C-starting, over time)
    #   right = prototype 72-bin CQT (A-starting, summed)
    # ======================================================
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle('Step 2 — Frequency representation', fontsize=13)

    img = ax_l.imshow(lib_chroma_cqt, aspect='auto', origin='lower',
                      interpolation='nearest')
    ax_l.set_title('librosa chroma_cqt (C-starting)')
    ax_l.set_xlabel('Frame')
    ax_l.set_ylabel('Pitch class')
    ax_l.set_yticks(range(12))
    ax_l.set_yticklabels(PITCH_C_START)
    plt.colorbar(img, ax=ax_l, label='Energy')

    band_labels = [f'{PITCH_C_START[i%12]}{i//12+1}' for i in range(72)]
    ax_r.bar(range(72), proto_cqt72, color='tab:orange', width=0.8)
    ax_r.set_title('prototype approx CQT (C1 -> B6, 72 bins)')
    ax_r.set_xlabel('CQT band')
    ax_r.set_ylabel('Energy')
    ax_r.set_xticks(range(0, 72, 12))
    ax_r.set_xticklabels([band_labels[i] for i in range(0, 72, 12)], rotation=30)

    plt.tight_layout()
    _save(fig, '2_cqt', stem)

    # ======================================================
    # Figure 3: Chroma (12-bin, both C-starting)
    # ======================================================
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Step 3 — Chroma (C-starting, normalised)', fontsize=13)

    lib_norm   = lib_chroma_sum   / (lib_chroma_sum.max()   or 1)
    proto_norm = proto_chroma   / (proto_chroma.max()   or 1)

    ax_l.bar(PITCH_C_START, lib_norm)
    ax_l.set_title('librosa chroma_cqt (summed, C-starting)')
    ax_l.set_xlabel('Pitch class')
    ax_l.set_ylabel('Normalised energy')
    ax_l.set_ylim(0, 1.1)

    ax_r.bar(PITCH_C_START, proto_norm, color='tab:orange')
    ax_r.set_title('prototype chroma (folded CQT, C-starting)')
    ax_r.set_xlabel('Pitch class')
    ax_r.set_ylabel('Normalised energy')
    ax_r.set_ylim(0, 1.1)

    plt.tight_layout()
    _save(fig, '3_chroma', stem)

    # ======================================================
    # Figure 4: Key profile scores (24 keys)
    # ======================================================
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle('Step 4 — Key profile scores (24 keys)', fontsize=13)

    colors_l = ['tab:red' if s == max(lib_scores)   else 'tab:blue'   for s in lib_scores]
    colors_r = ['tab:red' if s == max(proto_scores) else 'tab:orange' for s in proto_scores]

    ax_l.bar(range(24), lib_scores,   color=colors_l)
    ax_l.set_title(f'librosa  →  {lib_key[0]} {lib_key[1]}')
    ax_l.set_xlabel('Key')
    ax_l.set_ylabel('Correlation score')
    ax_l.set_xticks(range(24))
    ax_l.set_xticklabels(KEY_LABELS, rotation=90, fontsize=7)

    ax_r.bar(range(24), proto_scores, color=colors_r)
    ax_r.set_title(f'prototype  →  {proto_key[0]} {proto_key[1]}')
    ax_r.set_xlabel('Key')
    ax_r.set_ylabel('Correlation score')
    ax_r.set_xticks(range(24))
    ax_r.set_xticklabels(KEY_LABELS, rotation=90, fontsize=7)

    plt.tight_layout()
    _save(fig, '4_key_scores', stem)

    print(f'\nAll figures saved to {OUTPUT_DIR}\n')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python debug_pipeline.py <audio.wav>')
        sys.exit(1)
    run_debug(sys.argv[1])
