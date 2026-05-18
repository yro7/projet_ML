"""Dataset loading and signal trimming utilities."""

import os
import re
import tempfile
from pathlib import Path

import numpy as np

_NUMBA_CACHE_DIR = Path(tempfile.gettempdir()) / "ml_fil_rouge_numba_cache"
_NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_NUMBA_CACHE_DIR))

import librosa

try:
    from .config import DATA_DIR, DEFAULT_SAMPLE_RATE, GENRES, WORDS
except ImportError:  # Allows importing from a notebook whose cwd is projet_fil_rouge/
    from config import DATA_DIR, DEFAULT_SAMPLE_RATE, GENRES, WORDS


def list_audio_files(data_dir=DATA_DIR):
    """Return WAV files in deterministic order."""

    return sorted(Path(data_dir).glob("*.wav"))


def parse_label(file_path, words=WORDS):
    """Return the integer label encoded in a filename."""

    stem = Path(file_path).stem
    for label, word in enumerate(words):
        if word in stem:
            return label
    raise ValueError(f"Cannot infer word label from filename: {file_path}")


def parse_genre(file_path, genres=GENRES):
    """Return the speaker genre encoded at the start of a filename."""

    stem = Path(file_path).stem
    match = re.match(r"([A-Za-z])\d+", stem)
    if match and match.group(1) in genres:
        return match.group(1)
    raise ValueError(f"Cannot infer genre from filename: {file_path}")


def energy_trim(record, target_length):
    """Trim or pad a signal around its energy barycenter."""

    signal = np.asarray(record, dtype=float).reshape(-1)
    target_length = int(target_length)
    if target_length <= 0:
        raise ValueError("target_length must be positive")

    if signal.shape[0] < target_length:
        pad_total = target_length - signal.shape[0]
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(signal, (pad_left, pad_right), mode="constant")

    if signal.shape[0] == target_length:
        return signal.copy()

    energy = signal**2
    total_energy = energy.sum()
    if total_energy <= np.finfo(float).eps:
        center = signal.shape[0] // 2
    else:
        center = int(np.floor((energy * np.arange(signal.shape[0])).sum() / total_energy))

    start = center - target_length // 2
    start = max(0, min(start, signal.shape[0] - target_length))
    end = start + target_length
    return signal[start:end]


def load_dataset(
    data_dir=DATA_DIR,
    words=WORDS,
    genres=GENRES,
    sample_rate=DEFAULT_SAMPLE_RATE,
    target_length=None,
    return_paths=False,
):
    """Load WAV files and return X, y, speaker genres, and sampling frequency.

    The default sample rate mirrors librosa's default used in the original
    notebook, which keeps the existing dimensions reproducible.
    """

    paths = list_audio_files(data_dir)
    if not paths:
        raise FileNotFoundError(f"No .wav files found in {data_dir}")

    records = []
    labels = []
    speaker_genres = []
    sample_rates = []

    for path in paths:
        record, fs = librosa.load(path, sr=sample_rate, mono=True)
        records.append(record)
        labels.append(parse_label(path, words=words))
        speaker_genres.append(parse_genre(path, genres=genres))
        sample_rates.append(fs)

    if len(set(sample_rates)) != 1:
        raise ValueError(f"Multiple sampling frequencies found: {sorted(set(sample_rates))}")

    trim_length = min(len(record) for record in records) if target_length is None else target_length
    print(f"The smallest record contains {trim_length} samples")
    X = np.vstack([energy_trim(record, trim_length) for record in records])
    y = np.asarray(labels, dtype=int)
    speaker_genres = np.asarray(speaker_genres)
    fs = sample_rates[0]

    if return_paths:
        return X, y, speaker_genres, fs, paths
    return X, y, speaker_genres, fs
