"""Parse key strings and score key-finding predictions using the MIREX scheme."""

import re

PITCH_CLASS = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
}

MAJOR = 'major'
MINOR = 'minor'

# A key string is a tonic, optional spaces, then a mode word. Anything after is ignored
KEY_PATTERN = re.compile(r'^([A-Ga-g][#b]?)\s*([A-Za-z]*)')

# Accepted mode words.
# Whole words are compared, never first letters: 'major' and 'minor' both start with 'm'.
MODE_BY_WORD = {
    'M': MAJOR, 'maj': MAJOR, 'major': MAJOR,
    'm': MINOR, 'min': MINOR, 'minor': MINOR,
}

# MIREX score awarded per prediction category
CATEGORY_SCORES = {
    'correct': 1.0,       # exact match
    'fifth': 0.5,         # same mode, tonic a perfect fifth up or down
    'relative': 0.3,      # relative major/minor (same key signature)
    'parallel': 0.2,      # same tonic, different mode
    'wrong': 0.0,         # everything else
}

FIFTH_INTERVALS = (5, 7)          # semitone offsets a perfect fifth away
RELATIVE_MAJOR_OFFSET = 9         # major tonic -> its relative minor, in semitones up
RELATIVE_MINOR_OFFSET = 3         # minor tonic -> its relative major, in semitones up


# ---------- parsing ----------

def parse_key(key_str):
    """Parse a key string into (tonic_idx, mode)."""

    s = key_str.replace('\t', ' ').replace(':', ' ').strip()

    match = KEY_PATTERN.match(s)
    if not match:
        raise ValueError(f'Invalid key format: {key_str!r}')

    raw_tonic, raw_mode = match.groups() # Eb Minor -> (Eb, Minor)

    tonic = raw_tonic.capitalize() # eb -> Eb
    if tonic not in PITCH_CLASS:
        raise ValueError(f'Unknown tonic: {tonic!r} (from {key_str!r})')

    return PITCH_CLASS[tonic], parse_mode(raw_mode, raw_tonic, key_str)


def parse_mode(raw_mode, raw_tonic, key_str):
    """Return the mode of a key string, given its mode word and its tonic as written."""

    if not raw_mode: # A -> major, c -> minor
        return MINOR if raw_tonic[0].islower() else MAJOR

    mode = MODE_BY_WORD.get(raw_mode) or MODE_BY_WORD.get(raw_mode.lower())
    if mode is None:
        raise ValueError(f'Unknown mode: {raw_mode!r} (from {key_str!r})')
    return mode


# ---------- scoring ----------

def categorize(true_key, pred_key):
    """
    Return the MIREX category of one prediction: 'correct', 'fifth',
    'relative', 'parallel' or 'wrong'. Both keys are (tonic_idx, mode) tuples.
    """
    true_tonic, true_mode = true_key
    pred_tonic, pred_mode = pred_key
    same_tonic = true_tonic == pred_tonic
    same_mode = true_mode == pred_mode
    diff = (pred_tonic - true_tonic) % 12 # keep in 0 ~ 11

    if same_tonic and same_mode:
        return 'correct'

    if same_mode and diff in FIFTH_INTERVALS:
        return 'fifth'

    # Relative: a major key predicted as its relative minor, or the other way round
    if true_mode == MAJOR and pred_mode == MINOR and diff == RELATIVE_MAJOR_OFFSET:
        return 'relative'
    if true_mode == MINOR and pred_mode == MAJOR and diff == RELATIVE_MINOR_OFFSET:
        return 'relative'

    if same_tonic and not same_mode:
        return 'parallel'

    return 'wrong'


def mirex_score(true_key, pred_key):
    """Return (score, category) for one prediction under the MIREX scheme."""
    category = categorize(true_key, pred_key)
    return CATEGORY_SCORES[category], category
