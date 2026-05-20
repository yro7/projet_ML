"""Dataset loading and signal trimming utilities."""

from dataclasses import dataclass
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
    from .config import (
        DATA_DIR,
        DATASET_CURRENT,
        DATASET_GOOGLE_SPEECH_COMMANDS,
        DATASET_PERSONAL_AUGMENTED,
        DEFAULT_SAMPLE_RATE,
        GENRES,
        GOOGLE_DEFAULT_WORDS,
        GOOGLE_SPEECH_COMMANDS_DIR,
        PERSONAL_DATA_DIR,
        WORDS,
    )
except ImportError:  # Allows importing from a notebook whose cwd is projet_fil_rouge/
    from config import (
        DATA_DIR,
        DATASET_CURRENT,
        DATASET_GOOGLE_SPEECH_COMMANDS,
        DATASET_PERSONAL_AUGMENTED,
        DEFAULT_SAMPLE_RATE,
        GENRES,
        GOOGLE_DEFAULT_WORDS,
        GOOGLE_SPEECH_COMMANDS_DIR,
        PERSONAL_DATA_DIR,
        WORDS,
    )


@dataclass(frozen=True)
class DatasetBundle:
    """Loaded dataset plus metadata used by benchmarks and notebooks."""

    name: str
    X: np.ndarray
    y: np.ndarray
    genres: np.ndarray
    fs: int
    target_names: tuple
    paths: tuple
    metadata: dict


def list_audio_files(data_dir=DATA_DIR):
    """Return WAV files in deterministic order."""

    return sorted(Path(data_dir).glob("*.wav"))


def _has_wav_files(data_dir):
    return Path(data_dir).exists() and any(Path(data_dir).glob("*.wav"))


def _has_google_speech_commands(data_dir=GOOGLE_SPEECH_COMMANDS_DIR, words=GOOGLE_DEFAULT_WORDS):
    data_dir = Path(data_dir)
    return data_dir.exists() and all(
        any((data_dir / word).glob("*.wav")) for word in words
    )


def list_datasets(include_missing=False):
    """Return available registered dataset names.

    With include_missing=True, return every known dataset name even if its local
    files are not present yet.
    """

    datasets = [
        DATASET_CURRENT,
        DATASET_PERSONAL_AUGMENTED,
        DATASET_GOOGLE_SPEECH_COMMANDS,
    ]
    if include_missing:
        return datasets

    available = []
    if _has_wav_files(DATA_DIR):
        available.append(DATASET_CURRENT)
    if _has_wav_files(DATA_DIR) and _has_wav_files(PERSONAL_DATA_DIR):
        available.append(DATASET_PERSONAL_AUGMENTED)
    if _has_google_speech_commands():
        available.append(DATASET_GOOGLE_SPEECH_COMMANDS)
    return available


def parse_label(file_path, words=WORDS):
    """Return the integer label encoded in a filename."""

    stem = Path(file_path).stem
    for label, word in enumerate(words):
        if word in stem:
            return label
    raise ValueError(f"Cannot infer word label from filename: {file_path}")


def _path_matches_words(file_path, words):
    stem = Path(file_path).stem
    return any(word in stem for word in words)


def parse_genre(file_path, genres=GENRES):
    """Return the speaker genre encoded at the start of a filename."""

    stem = Path(file_path).stem
    match = re.match(r"([A-Za-z])\d+", stem)
    if match and match.group(1) in genres:
        return match.group(1)
    raise ValueError(f"Cannot infer genre from filename: {file_path}")


def parse_speaker_prefix(file_path):
    """Return the alphabetic speaker prefix encoded at the start of a filename."""

    stem = Path(file_path).stem
    match = re.match(r"([A-Za-z]+)\d+", stem)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot infer speaker prefix from filename: {file_path}")


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


def _stack_records(records, target_length=None, verbose=True):
    trim_length = min(len(record) for record in records) if target_length is None else target_length
    if verbose:
        print(f"The smallest record contains {trim_length} samples")
    return np.vstack([energy_trim(record, trim_length) for record in records])


def _bundle_from_paths(
    name,
    paths,
    words,
    label_parser,
    genre_parser,
    sample_rate=DEFAULT_SAMPLE_RATE,
    target_length=None,
    metadata=None,
    strict=True,
):
    paths = tuple(sorted(Path(path) for path in paths))
    if not paths:
        raise FileNotFoundError(f"No .wav files found for dataset '{name}'")

    records = []
    labels = []
    speaker_genres = []
    sample_rates = []
    used_paths = []
    skipped_paths = []

    for path in paths:
        try:
            label = label_parser(path, words)
            speaker_genre = genre_parser(path)
        except ValueError:
            if strict:
                raise
            skipped_paths.append(path)
            continue

        record, fs = librosa.load(path, sr=sample_rate, mono=True)
        records.append(record)
        labels.append(label)
        speaker_genres.append(speaker_genre)
        sample_rates.append(fs)
        used_paths.append(path)

    if not records:
        raise FileNotFoundError(f"No usable .wav files found for dataset '{name}'")

    if len(set(sample_rates)) != 1:
        raise ValueError(f"Multiple sampling frequencies found: {sorted(set(sample_rates))}")

    X = _stack_records(records, target_length=target_length)
    y = np.asarray(labels, dtype=int)
    speaker_genres = np.asarray(speaker_genres)

    metadata = dict(metadata or {})
    metadata.update(
        {
            "n_skipped_files": len(skipped_paths),
            "skipped_paths": tuple(skipped_paths),
        }
    )
    return DatasetBundle(
        name=name,
        X=X,
        y=y,
        genres=speaker_genres,
        fs=sample_rates[0],
        target_names=tuple(words),
        paths=tuple(used_paths),
        metadata=metadata,
    )


def _filename_label_parser(path, words):
    return parse_label(path, words=words)


def _google_label_parser(path, words):
    label_name = Path(path).parent.name
    try:
        return tuple(words).index(label_name)
    except ValueError as exc:
        raise ValueError(f"Cannot infer Google label from folder: {path}") from exc


def _unknown_genre_parser(path):
    return "unknown"


def _current_bundle(words, sample_rate, target_length, strict):
    words = tuple(WORDS if words is None else words)
    paths = [
        path for path in list_audio_files(DATA_DIR)
        if strict or _path_matches_words(path, words)
    ]
    return _bundle_from_paths(
        name=DATASET_CURRENT,
        paths=paths,
        words=words,
        label_parser=_filename_label_parser,
        genre_parser=lambda path: parse_genre(path, genres=GENRES),
        sample_rate=sample_rate,
        target_length=target_length,
        metadata={"data_dirs": (DATA_DIR,)},
        strict=strict,
    )


def _personal_augmented_bundle(words, sample_rate, target_length, strict):
    words = tuple(WORDS if words is None else words)
    current_paths = list_audio_files(DATA_DIR)
    personal_paths = list_audio_files(PERSONAL_DATA_DIR)
    if not personal_paths:
        raise FileNotFoundError(
            f"No personal .wav files found in {PERSONAL_DATA_DIR}. "
            "Expected files like P01_avance.wav."
        )

    paths = current_paths + personal_paths
    if not strict:
        paths = [path for path in paths if _path_matches_words(path, words)]

    return _bundle_from_paths(
        name=DATASET_PERSONAL_AUGMENTED,
        paths=paths,
        words=words,
        label_parser=_filename_label_parser,
        genre_parser=parse_speaker_prefix,
        sample_rate=sample_rate,
        target_length=target_length,
        metadata={"data_dirs": (DATA_DIR, PERSONAL_DATA_DIR)},
        strict=strict,
    )


def _google_speech_commands_bundle(words, sample_rate, target_length, strict):
    words = tuple(GOOGLE_DEFAULT_WORDS if words is None else words)
    data_dir = Path(GOOGLE_SPEECH_COMMANDS_DIR)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Google Speech Commands directory not found: {data_dir}. "
            "Download/extract it locally first."
        )

    paths = []
    missing_words = []
    for word in words:
        word_paths = sorted((data_dir / word).glob("*.wav"))
        if not word_paths:
            missing_words.append(word)
        paths.extend(word_paths)

    if missing_words and strict:
        raise FileNotFoundError(
            "Missing Google Speech Commands wav files for: "
            + ", ".join(missing_words)
        )

    return _bundle_from_paths(
        name=DATASET_GOOGLE_SPEECH_COMMANDS,
        paths=paths,
        words=words,
        label_parser=_google_label_parser,
        genre_parser=_unknown_genre_parser,
        sample_rate=sample_rate,
        target_length=target_length,
        metadata={"data_dirs": (data_dir,), "missing_words": tuple(missing_words)},
        strict=strict,
    )


def load_named_dataset(
    name=DATASET_CURRENT,
    words=None,
    sample_rate=DEFAULT_SAMPLE_RATE,
    target_length=None,
    strict=True,
):
    """Load a registered dataset by name and return a DatasetBundle."""

    if name == DATASET_CURRENT:
        return _current_bundle(words, sample_rate, target_length, strict)
    if name == DATASET_PERSONAL_AUGMENTED:
        return _personal_augmented_bundle(words, sample_rate, target_length, strict)
    if name == DATASET_GOOGLE_SPEECH_COMMANDS:
        return _google_speech_commands_bundle(words, sample_rate, target_length, strict)

    valid_names = ", ".join(list_datasets(include_missing=True))
    raise ValueError(f"Unknown dataset '{name}'. Expected one of: {valid_names}")


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

    X = _stack_records(records, target_length=target_length)
    y = np.asarray(labels, dtype=int)
    speaker_genres = np.asarray(speaker_genres)
    fs = sample_rates[0]

    if return_paths:
        return X, y, speaker_genres, fs, paths
    return X, y, speaker_genres, fs
