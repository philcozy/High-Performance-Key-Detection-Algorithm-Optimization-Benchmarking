"""Parse key strings and score key-finding predictions using the MIREX scheme."""

PITCH_CLASS = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
}

# MIREX score awarded per prediction category
SCORE_CORRECT = 1.0
SCORE_FIFTH = 0.5
SCORE_RELATIVE = 0.3
SCORE_PARALLEL = 0.2
SCORE_WRONG = 0.0

FIFTH_INTERVALS = (5, 7)          # semitone offsets a perfect fifth away
RELATIVE_MAJOR_OFFSET = 9         # major tonic -> its relative minor, in semitones up
RELATIVE_MINOR_OFFSET = 3         # minor tonic -> its relative major, in semitones up


def parse_key(key_str):
    """
    Parse a key string into (tonic_idx, mode).

    Handles formats like:
      'C# major', 'a minor', 'F#min', 'Ebmaj', 'A:minor', 'b'
    """
    s = key_str.strip()

    # Normalise whitespace and separators
    for sep in ('\t', ':'):
        s = s.replace(sep, ' ')
    s = ' '.join(s.split())  # collapse multiple spaces

    # Split tonic from mode
    if ' ' in s:
        tonic, mode_str = s.split(' ', 1)
    else:
        # No separator — try to split after the tonic (1 or 2 chars)
        # e.g. 'Ebminor', 'C#maj', 'Cmajor', 'amin'
        for length in (2, 1):
            if len(s) > length:
                candidate = s[:length]
                rest = s[length:]
                # valid tonic candidates end in b, #, or letter only
                if candidate[0].isalpha() and (
                    len(candidate) == 1 or candidate[1] in ('#', 'b')
                ):
                    tonic, mode_str = candidate, rest
                    break
        else:
            # Single token with no mode — infer from case
            tonic = s
            mode_str = 'minor' if s[0].islower() else 'major'

    # Normalise tonic capitalisation: 'eb' -> 'Eb', 'EB' -> 'Eb'
    tonic = tonic[0].upper() + tonic[1:].lower() if len(tonic) > 1 else tonic.upper()

    if tonic not in PITCH_CLASS:
        raise ValueError(f'Unknown tonic: {tonic!r} (from {key_str!r})')

    # Normalise mode
    m = mode_str.strip().lower()
    if m in ('minor', 'min', 'm'):
        mode = 'minor'
    elif m in ('major', 'maj', 'M', ''):
        mode = 'major'
    elif m.startswith('min'):
        mode = 'minor'
    elif m.startswith('maj'):
        mode = 'major'
    else:
        raise ValueError(f'Unknown mode: {mode_str!r} (from {key_str!r})')

    return PITCH_CLASS[tonic], mode


def mirex_score(true_key, pred_key):
    """
    MIREX key-finding scoring scheme.
    Both keys as (tonic_idx, mode) tuples.
    Returns (score, category).

    Scoring:
      1.0  correct       — exact match
      0.5  fifth         — same mode, tonic a perfect fifth up or down
      0.3  relative      — relative major/minor (same key signature)
      0.2  parallel      — same tonic, different mode
      0.0  wrong         — everything else
    """
    t_true, m_true = true_key
    t_pred, m_pred = pred_key
    diff = (t_pred - t_true) % 12

    if t_true == t_pred and m_true == m_pred:
        return SCORE_CORRECT, 'correct'

    if m_true == m_pred and diff in FIFTH_INTERVALS:
        return SCORE_FIFTH, 'fifth'

    # Relative: major predicted as its relative minor
    if m_true == 'major' and m_pred == 'minor' and diff == RELATIVE_MAJOR_OFFSET:
        return SCORE_RELATIVE, 'relative'

    # Relative: minor predicted as its relative major
    if m_true == 'minor' and m_pred == 'major' and diff == RELATIVE_MINOR_OFFSET:
        return SCORE_RELATIVE, 'relative'

    if t_true == t_pred and m_true != m_pred:
        return SCORE_PARALLEL, 'parallel'

    return SCORE_WRONG, 'wrong'