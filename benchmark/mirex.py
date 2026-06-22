PITCH_CLASS = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
}

def parse_key(key_str):
    """'C# major', 'a minor', 'F#min' → (tonic_idx, 'major'|'minor')."""
    s = key_str.strip().replace('\t', ' ')
    # Tolerate 'Cmaj', 'a:min', etc.
    for sep in (' ', ':', '\t'):
        if sep in s:
            tonic, mode = s.split(sep, 1)
            break
    else:
        # No separator — assume lowercase = minor, uppercase = major
        tonic, mode = s, 'minor' if s[0].islower() else 'major'

    mode = mode.strip().lower()
    mode = 'minor' if mode.startswith('m') and 'a' not in mode[:3] else mode
    mode = 'minor' if 'min' in mode else 'major'

    tonic = tonic.strip()
    tonic = tonic[0].upper() + tonic[1:]
    return PITCH_CLASS[tonic], mode


def mirex_score(true_key, pred_key):
    """Both keys as (tonic_idx, mode) tuples. Returns (score, category)."""
    t_true, m_true = true_key
    t_pred, m_pred = pred_key
    diff = (t_pred - t_true) % 12

    if t_true == t_pred and m_true == m_pred:
        return 1.0, 'correct'
    if m_true == m_pred and diff in (5, 7):
        return 0.5, 'fifth'
    if m_true == 'major' and m_pred == 'minor' and diff == 9:
        return 0.3, 'relative'
    if m_true == 'minor' and m_pred == 'major' and diff == 3:
        return 0.3, 'relative'
    if t_true == t_pred and m_true != m_pred:
        return 0.2, 'parallel'
    return 0.0, 'wrong'