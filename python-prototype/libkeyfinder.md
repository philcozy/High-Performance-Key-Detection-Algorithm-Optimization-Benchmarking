- fft
    with sample rate and frame size N:
    - `bin_frequency[k] = k × (sample_rate / N)`
    - `k = f × N / sample_rate`
    So bin 0 = 0 Hz, bin 1 = sr/N Hz, etc. 
    The bin width (frequency resolution) is always `sr/N` Hz — uniform, fixed.

- musical
    Given a reference pitch (A4 = 440 Hz) and a semitone offset i from it:
    - `f(i) = f_ref × 2^(i/12)`
    band i across 72 bands starts from a fixed low reference frequency (around A1 = 55 Hz) and goes up chromatically.

- Q facator
    The quality factor of a bandpass filter:
    `Q = center_frequency / bandwidth`
    `bandwidth = center / Q` where Q is fixed

    Two adjacent semitones always have a frequency ratio of 2^(1/12) ≈ 1.0595. So the gap between them, expressed as a fraction of their center frequency, is always the same number (2^(1/12) - 1) ≈ 0.0595
    `bandwidth ≤ gap = center × 0.0595 → Q = center / bandwidth ≥ 1 / 0.0595 ≈ 17`

- Hann Window
    `w(n) = 0.5 × (1 - cos(2πn / N))`
    `w(n) = 1 - cos(2πn / N)`
    This is just a scaled Hann (same shape, twice the amplitude). Since everything gets normalized afterward it doesn't matter

- Normalization
    `coeffs_normalized[j] = coeffs[j] / sum(coeffs) × center_frequency`