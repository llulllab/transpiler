"""
Music theory helpers: resolve Sonic Pi note names, chords and scales to
MIDI note numbers.

MIDI number convention (same as Sonic Pi / General MIDI):
  C4  = 60  (middle C)
  C-1 = 0
  G9  = 127
"""
from __future__ import annotations
import re

# ---------------------------------------------------------------------------
# Note name → semitone offset within octave
# ---------------------------------------------------------------------------

_LETTER_SEMITONE: dict[str, int] = {
    'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11,
    'c': 0, 'd': 2, 'e': 4, 'f': 5, 'g': 7, 'a': 9, 'b': 11,
}

# Accidental modifiers (semitones)
_ACCIDENTAL: dict[str, int] = {
    's':  1,   '#':  1,   'S':  1,
    'b': -1,   'f': -1,   'F': -1,   'B': -1,
    'ss': 2,   'x':  2,
    'bb': -2,  'ff': -2,
    'sb': 0,   'bs': 0,
}

# Pattern: letter, optional accidental, optional octave integer
_NOTE_RE = re.compile(
    r'^([A-Ga-g])(ss|bb|ff|sb|bs|[s#SbfFBx])?(-?\d+)?$'
)

# Sonic Pi default octave when not specified (e.g. :C → C4 = 60)
_DEFAULT_OCTAVE = 4


def note_to_midi(n: object) -> float | None:
    """
    Convert a Sonic Pi note representation to a MIDI number (float).

    Accepted inputs:
      - int / float      → returned as-is
      - ':C4'  ':Cs4'  ':Eb3'  ':C'  (with or without leading colon)
      - 'r'  ':r'  ':rest'           → returns None (rest)
    """
    if n is None:
        return None
    if isinstance(n, (int, float)):
        return float(n)
    if not isinstance(n, str):
        raise ValueError(f"Cannot resolve note: {n!r}")

    n = n.lstrip(':').strip()
    if n.lower() in ('r', 'rest'):
        return None

    m = _NOTE_RE.match(n)
    if not m:
        raise ValueError(f"Unrecognised note name: {n!r}")

    letter, acc, octave_str = m.groups()
    semitone = _LETTER_SEMITONE[letter]
    if acc:
        semitone += _ACCIDENTAL.get(acc, 0)
    octave = int(octave_str) if octave_str is not None else _DEFAULT_OCTAVE
    # MIDI = (octave + 1) * 12 + semitone  (C-1 = 0, C4 = 60)
    midi = (octave + 1) * 12 + semitone
    return float(midi)


# ---------------------------------------------------------------------------
# Chord types
# ---------------------------------------------------------------------------

CHORD_INTERVALS: dict[str, list[int]] = {
    # Triads
    'major':            [0, 4, 7],
    'minor':            [0, 3, 7],
    'major_triad':      [0, 4, 7],
    'minor_triad':      [0, 3, 7],
    'dim':              [0, 3, 6],
    'diminished':       [0, 3, 6],
    'aug':              [0, 4, 8],
    'augmented':        [0, 4, 8],
    'sus2':             [0, 2, 7],
    'sus4':             [0, 5, 7],
    # Seventh chords
    'major7':           [0, 4, 7, 11],
    'minor7':           [0, 3, 7, 10],
    'dom7':             [0, 4, 7, 10],
    '7':                [0, 4, 7, 10],
    'dominant7':        [0, 4, 7, 10],
    'dim7':             [0, 3, 6, 9],
    'diminished7':      [0, 3, 6, 9],
    'm7b5':             [0, 3, 6, 10],
    'half_diminished':  [0, 3, 6, 10],
    'aug7':             [0, 4, 8, 10],
    'augmented7':       [0, 4, 8, 10],
    'major7sharp5':     [0, 4, 8, 11],
    # Extended
    'major9':           [0, 4, 7, 11, 14],
    'minor9':           [0, 3, 7, 10, 14],
    '9':                [0, 4, 7, 10, 14],
    'add9':             [0, 4, 7, 14],
    'add11':            [0, 4, 7, 17],
    # Other common names
    '1':                [0],
    '5':                [0, 7],
    'power':            [0, 7],
    'major6':           [0, 4, 7, 9],
    'minor6':           [0, 3, 7, 9],
    'six':              [0, 4, 7, 9],
    'minor_major7':     [0, 3, 7, 11],
}


def chord(root: object, kind: str = 'major',
          invert: int = 0, num_octaves: int = 1) -> list[float]:
    """
    Return a list of MIDI note numbers for a chord.

    Args:
        root:        Root note (anything accepted by note_to_midi)
        kind:        Chord quality string, e.g. 'minor', 'major7'
        invert:      Inversion number (0 = root position)
        num_octaves: Repeat across N octaves
    """
    root_midi = note_to_midi(root)
    if root_midi is None:
        return []
    intervals = CHORD_INTERVALS.get(str(kind))
    if intervals is None:
        raise ValueError(f"Unknown chord type: {kind!r}")

    notes = []
    for oct_i in range(num_octaves):
        for iv in intervals:
            notes.append(root_midi + iv + oct_i * 12)

    # Apply inversion
    for _ in range(invert % len(intervals)):
        notes.append(notes.pop(0) + 12)

    return notes


# ---------------------------------------------------------------------------
# Scale types
# ---------------------------------------------------------------------------

SCALE_INTERVALS: dict[str, list[int]] = {
    # Western diatonic
    'major':                [0, 2, 4, 5, 7, 9, 11],
    'ionian':               [0, 2, 4, 5, 7, 9, 11],
    'natural_minor':        [0, 2, 3, 5, 7, 8, 10],
    'minor':                [0, 2, 3, 5, 7, 8, 10],
    'aeolian':              [0, 2, 3, 5, 7, 8, 10],
    'harmonic_minor':       [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor_asc':    [0, 2, 3, 5, 7, 9, 11],
    # Modes
    'dorian':               [0, 2, 3, 5, 7, 9, 10],
    'phrygian':             [0, 1, 3, 5, 7, 8, 10],
    'lydian':               [0, 2, 4, 6, 7, 9, 11],
    'mixolydian':           [0, 2, 4, 5, 7, 9, 10],
    'locrian':              [0, 1, 3, 5, 6, 8, 10],
    # Pentatonic
    'major_pentatonic':     [0, 2, 4, 7, 9],
    'minor_pentatonic':     [0, 3, 5, 7, 10],
    'pentatonic_major':     [0, 2, 4, 7, 9],
    'pentatonic_minor':     [0, 3, 5, 7, 10],
    # Blues
    'blues_major':          [0, 2, 3, 4, 7, 9],
    'blues_minor':          [0, 3, 5, 6, 7, 10],
    # Other
    'chromatic':            list(range(12)),
    'whole_tone':           [0, 2, 4, 6, 8, 10],
    'diminished':           [0, 2, 3, 5, 6, 8, 9, 11],
    'octatonic':            [0, 2, 3, 5, 6, 8, 9, 11],
    'prometheus':           [0, 2, 4, 6, 9, 10],
    'egyptian':             [0, 2, 5, 7, 10],
    'hirajoshi':            [0, 2, 3, 7, 8],
    'kumoi':                [0, 2, 3, 7, 9],
    'iwato':                [0, 1, 5, 6, 10],
    'in':                   [0, 1, 5, 7, 8],
    'insen':                [0, 1, 5, 7, 10],
    'scriabin':             [0, 1, 4, 7, 9],
    'enigmatic':            [0, 1, 4, 6, 8, 10, 11],
    'neapolitan_major':     [0, 1, 3, 5, 7, 9, 11],
    'neapolitan_minor':     [0, 1, 3, 5, 7, 8, 11],
    'double_harmonic':      [0, 1, 4, 5, 7, 8, 11],
    'hungarian_minor':      [0, 2, 3, 6, 7, 8, 11],
    'yo':                   [0, 3, 5, 7, 10],
    'ritusen':              [0, 2, 5, 7, 9],
}


def scale(root: object, kind: str = 'major', num_octaves: int = 1) -> list[float]:
    """
    Return a list of MIDI note numbers for a scale.

    Args:
        root:        Root note (anything accepted by note_to_midi)
        kind:        Scale quality string, e.g. 'minor_pentatonic'
        num_octaves: Number of octaves to generate
    """
    root_midi = note_to_midi(root)
    if root_midi is None:
        return []
    intervals = SCALE_INTERVALS.get(str(kind))
    if intervals is None:
        raise ValueError(f"Unknown scale: {kind!r}")

    notes = []
    for oct_i in range(num_octaves):
        for iv in intervals:
            notes.append(root_midi + iv + oct_i * 12)
    # Include the octave above the last if num_octaves > 1 (Sonic Pi does this)
    if num_octaves > 1:
        notes.append(root_midi + num_octaves * 12)
    return notes


# ---------------------------------------------------------------------------
# Extra theory helpers
# ---------------------------------------------------------------------------

def chord_invert(notes: list, shift: int) -> list[float]:
    """
    Invert a chord by moving notes up/down by octaves.

    shift > 0: move that many notes from the bottom to the top (+12 each).
    shift < 0: move that many notes from the top to the bottom (-12 each).
    """
    notes = list(notes)
    if not notes:
        return notes
    shift = shift % len(notes)
    for _ in range(shift):
        notes.append(notes.pop(0) + 12)
    return notes


def degree(degree_val: object, tonic: object, scale_name: str = 'major') -> float:
    """
    Return the MIDI note for the nth scale degree (1-based, or Roman numeral).

    degree_val: int (1-based) or string 'I','II','III'... or ':i',':ii'...
    """
    # Roman numeral → integer
    _roman = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6, 'vii': 7}
    if isinstance(degree_val, str):
        d = degree_val.lstrip(':').lower()
        degree_val = _roman.get(d, 1)
    degree_val = int(degree_val)
    notes = scale(tonic, scale_name, num_octaves=2)
    if not notes:
        return 60.0
    idx = (degree_val - 1) % len(notes)
    return notes[idx]


def chord_degree(
    degree_val: object,
    tonic: object,
    scale_name: str = 'major',
    num_notes: int = 4,
) -> list[float]:
    """
    Build a chord from a scale degree by stacking every-other scale note.

    degree_val: 1-based integer or Roman numeral string.
    """
    _roman = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6, 'vii': 7}
    if isinstance(degree_val, str):
        d = degree_val.lstrip(':').lower()
        degree_val = _roman.get(d, 1)
    degree_val = int(degree_val)
    # Generate enough scale notes to pick from
    notes = scale(tonic, scale_name, num_octaves=3)
    if not notes:
        return []
    # Start at degree (1-based), take every other note
    start = (degree_val - 1) % len(notes)
    result = []
    pos = start
    for _ in range(num_notes):
        if pos < len(notes):
            result.append(notes[pos])
        pos += 2
    return result


def note_range(start_note: object, end_note: object) -> list[float]:
    """Return all MIDI notes from start_note to end_note (inclusive)."""
    start = note_to_midi(start_note)
    end = note_to_midi(end_note)
    if start is None or end is None:
        return []
    lo, hi = (int(min(start, end)), int(max(start, end)))
    return [float(n) for n in range(lo, hi + 1)]


# ---------------------------------------------------------------------------
# Convenience: beats-per-second given BPM
# ---------------------------------------------------------------------------

def beat_duration(bpm: float) -> float:
    """Return the duration of one beat in seconds."""
    return 60.0 / bpm
