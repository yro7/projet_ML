"""Pure preprocessing functions and sklearn-compatible transformers."""

import os
import tempfile
from pathlib import Path

_NUMBA_CACHE_DIR = Path(tempfile.gettempdir()) / "ml_fil_rouge_numba_cache"
_NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_NUMBA_CACHE_DIR))

import librosa
import numpy as np
from scipy import signal
from scipy.fft import fft
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from ..config import (
        DEFAULT_N_COMPONENTS,
        DEFAULT_N_MFCC,
        DEFAULT_SAMPLE_RATE,
        DEFAULT_STFT_NPERSEG,
        RANDOM_SEED,
    )
except ImportError:  # Allows importing from a notebook whose cwd is projet_fil_rouge/
    from config import (
        DEFAULT_N_COMPONENTS,
        DEFAULT_N_MFCC,
        DEFAULT_SAMPLE_RATE,
        DEFAULT_STFT_NPERSEG,
        RANDOM_SEED,
    )


def _ensure_2d(X):
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array, got shape {X.shape}")
    return X


def _validate_positive_int(value, name):
    if value is not None and int(value) <= 0:
        raise ValueError(f"{name} must be positive or None")


def _aggregate_time(values, stat, axis=-1):
    if stat == "mean":
        return np.mean(values, axis=axis)
    if stat == "max":
        return np.max(values, axis=axis)
    raise ValueError("stat must be 'mean' or 'max'")


def fft_magnitude(X, idx_frequence_max=None):
    """Return the magnitude of the FFT, optionally truncated on frequency axis."""

    _validate_positive_int(idx_frequence_max, "idx_frequence_max")
    X = _ensure_2d(X)
    features = np.abs(fft(X, axis=1))
    if idx_frequence_max is not None:
        features = features[:, : int(idx_frequence_max)]
    return features


def stft_magnitude(
    X,
    fs=DEFAULT_SAMPLE_RATE,
    nperseg=DEFAULT_STFT_NPERSEG,
    idx_frequence_max=None,
):
    """Return STFT magnitudes with shape (n_samples, n_freqs, n_times)."""

    _validate_positive_int(idx_frequence_max, "idx_frequence_max")
    X = _ensure_2d(X)
    _, _, zxx = signal.stft(X, fs=fs, nperseg=nperseg, axis=-1)
    magnitudes = np.abs(zxx)
    if idx_frequence_max is not None:
        magnitudes = magnitudes[:, : int(idx_frequence_max), :]
    return magnitudes


def stft_features(
    X,
    stat="mean",
    idx_frequence_max=None,
    fs=DEFAULT_SAMPLE_RATE,
    nperseg=DEFAULT_STFT_NPERSEG,
):
    """Aggregate STFT magnitudes over time into a 2D feature matrix."""

    magnitudes = stft_magnitude(
        X,
        fs=fs,
        nperseg=nperseg,
        idx_frequence_max=idx_frequence_max,
    )
    return _aggregate_time(magnitudes, stat=stat, axis=2)


def mfcc_coefficients(X, sr=DEFAULT_SAMPLE_RATE, n_mfcc=DEFAULT_N_MFCC):
    """Return MFCC coefficients with shape (n_samples, n_mfcc, n_times)."""

    _validate_positive_int(n_mfcc, "n_mfcc")
    X = _ensure_2d(X)
    coeffs = [
        librosa.feature.mfcc(y=signal_row.astype(float), sr=sr, n_mfcc=int(n_mfcc))
        for signal_row in X
    ]
    return np.asarray(coeffs)


def mfcc_features(X, stat="mean", sr=DEFAULT_SAMPLE_RATE, n_mfcc=DEFAULT_N_MFCC):
    """Aggregate MFCC coefficients over time into a 2D feature matrix."""

    coeffs = mfcc_coefficients(X, sr=sr, n_mfcc=n_mfcc)
    return _aggregate_time(coeffs, stat=stat, axis=2)


def effective_n_components(n_components, X):
    """Keep PCA valid inside small train folds."""

    if n_components is None:
        return min(DEFAULT_N_COMPONENTS, X.shape[0], X.shape[1])
    if isinstance(n_components, int):
        return min(n_components, X.shape[0], X.shape[1])
    return n_components


def fit_transform_pca(X_train, X_test=None, n_components=DEFAULT_N_COMPONENTS, random_state=RANDOM_SEED):
    """Fit PCA on train only, then transform train and optional test data."""

    X_train = _ensure_2d(X_train)
    pca = PCA(
        n_components=effective_n_components(n_components, X_train),
        random_state=random_state,
    )
    X_train_pca = pca.fit_transform(X_train)
    if X_test is None:
        return X_train_pca, None, pca
    return X_train_pca, pca.transform(_ensure_2d(X_test)), pca


class FFTTransformer(BaseEstimator, TransformerMixin):
    """sklearn transformer wrapping fft_magnitude."""

    def __init__(self, idx_frequence_max=None):
        self.idx_frequence_max = idx_frequence_max

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return fft_magnitude(X, idx_frequence_max=self.idx_frequence_max)


class FFTPCAFeatureExtractor(BaseEstimator, TransformerMixin):
    """FFT magnitude followed by PCA, with optional scaling after PCA."""

    def __init__(
        self,
        idx_frequence_max=None,
        n_components=DEFAULT_N_COMPONENTS,
        scale=False,
        random_state=RANDOM_SEED,
    ):
        self.idx_frequence_max = idx_frequence_max
        self.n_components = n_components
        self.scale = scale
        self.random_state = random_state

    def fit(self, X, y=None):
        fft_features = fft_magnitude(X, idx_frequence_max=self.idx_frequence_max)
        self.pca_ = PCA(
            n_components=effective_n_components(self.n_components, fft_features),
            random_state=self.random_state,
        )
        pca_features = self.pca_.fit_transform(fft_features)
        self.scaler_ = StandardScaler().fit(pca_features) if self.scale else None
        return self

    def transform(self, X, y=None):
        fft_features = fft_magnitude(X, idx_frequence_max=self.idx_frequence_max)
        pca_features = self.pca_.transform(fft_features)
        if self.scaler_ is not None:
            return self.scaler_.transform(pca_features)
        return pca_features


class STFTTransformer(BaseEstimator, TransformerMixin):
    """sklearn transformer returning aggregated STFT features."""

    def __init__(
        self,
        stat="mean",
        idx_frequence_max=None,
        fs=DEFAULT_SAMPLE_RATE,
        nperseg=DEFAULT_STFT_NPERSEG,
    ):
        self.stat = stat
        self.idx_frequence_max = idx_frequence_max
        self.fs = fs
        self.nperseg = nperseg

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return stft_features(
            X,
            stat=self.stat,
            idx_frequence_max=self.idx_frequence_max,
            fs=self.fs,
            nperseg=self.nperseg,
        )


class MFCCTransformer(BaseEstimator, TransformerMixin):
    """sklearn transformer returning aggregated MFCC features."""

    def __init__(self, stat="mean", n_mfcc=DEFAULT_N_MFCC, sr=DEFAULT_SAMPLE_RATE):
        self.stat = stat
        self.n_mfcc = n_mfcc
        self.sr = sr

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return mfcc_features(X, stat=self.stat, sr=self.sr, n_mfcc=self.n_mfcc)


def make_fft_pca_preprocessor(
    idx_frequence_max=None,
    n_components=DEFAULT_N_COMPONENTS,
    scale=False,
    random_state=RANDOM_SEED,
):
    """Factory used by benchmark/manual CV code."""

    return FFTPCAFeatureExtractor(
        idx_frequence_max=idx_frequence_max,
        n_components=n_components,
        scale=scale,
        random_state=random_state,
    )


def make_fft_pca_pipeline(
    idx_frequence_max=None,
    n_components=DEFAULT_N_COMPONENTS,
    scale=False,
    random_state=RANDOM_SEED,
):
    """Optional sklearn Pipeline equivalent for notebook-style experiments."""

    steps = [
        ("fft", FFTTransformer(idx_frequence_max=idx_frequence_max)),
        (
            "pca",
            PCA(
                n_components=n_components,
                random_state=random_state,
            ),
        ),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def default_fft_pca_param_grid():
    return {
        "idx_frequence_max": [1_000, 3_000, 9_261],
        "n_components": [5, 10, 20],
        "scale": [False, True],
    }
