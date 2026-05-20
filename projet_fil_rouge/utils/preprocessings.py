"""Pure preprocessing functions and sklearn-compatible transformers."""

import os
import tempfile
from pathlib import Path

_NUMBA_CACHE_DIR = Path(tempfile.gettempdir()) / "ml_fil_rouge_numba_cache"
_NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_NUMBA_CACHE_DIR))

import numpy as np
import librosa
from scipy import signal
from scipy.fft import fft
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import KernelPCA, NMF, PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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


def _summarize(values, stats, axis=-1):
    summaries = []
    for stat in stats:
        if stat == "mean":
            summaries.append(np.mean(values, axis=axis))
        elif stat == "std":
            summaries.append(np.std(values, axis=axis))
        elif stat == "max":
            summaries.append(np.max(values, axis=axis))
        elif stat == "min":
            summaries.append(np.min(values, axis=axis))
        elif stat == "max_abs":
            summaries.append(np.max(np.abs(values), axis=axis))
        elif stat == "energy":
            summaries.append(np.mean(values**2, axis=axis))
        else:
            raise ValueError(
                "stats entries must be one of: 'mean', 'std', 'max', 'min', "
                "'max_abs', 'energy'"
            )
    return np.concatenate([np.atleast_1d(summary) for summary in summaries])


def _require_pywt():
    try:
        import pywt
    except ImportError as exc:
        raise ImportError(
            "Wavelet preprocessing requires PyWavelets. "
            "Install the project requirements or run: pip install PyWavelets"
        ) from exc
    return pywt


def preprocess_fft(X, idx_frequence_max=None):
    """Return FFT magnitude features as a 2D matrix.

    This is the simple API to use in notebooks and manual exercises.
    """

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


def preprocess_stft(
    X,
    stat="mean",
    idx_frequence_max=None,
    fs=DEFAULT_SAMPLE_RATE,
    nperseg=DEFAULT_STFT_NPERSEG,
):
    """Return aggregated STFT features as a 2D matrix."""

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


def preprocess_mfcc(X, stat="mean", sr=DEFAULT_SAMPLE_RATE, n_mfcc=DEFAULT_N_MFCC):
    """Return aggregated MFCC features as a 2D matrix."""

    coeffs = mfcc_coefficients(X, sr=sr, n_mfcc=n_mfcc)
    return _aggregate_time(coeffs, stat=stat, axis=2)


def preprocess_mfcc_summary(
    X,
    sr=DEFAULT_SAMPLE_RATE,
    n_mfcc=DEFAULT_N_MFCC,
    stats=("mean", "std"),
    include_delta=False,
    include_delta2=False,
):
    """Return richer MFCC summaries as a 2D matrix.

    Deltas add first- and second-order temporal derivatives before the same
    summary statistics are computed.
    """

    coeffs = mfcc_coefficients(X, sr=sr, n_mfcc=n_mfcc)
    features = []
    for sample_coeffs in coeffs:
        matrices = [sample_coeffs]
        if include_delta:
            matrices.append(librosa.feature.delta(sample_coeffs, mode="nearest"))
        if include_delta2:
            matrices.append(
                librosa.feature.delta(sample_coeffs, order=2, mode="nearest")
            )
        features.append(
            np.concatenate(
                [_summarize(matrix, stats=stats, axis=1) for matrix in matrices]
            )
        )
    return np.asarray(features)


def _effective_lda_n_components(n_components, X, y):
    if y is None:
        raise ValueError("FFTLDAFeatureExtractor requires y during fit")

    max_components = min(len(np.unique(y)) - 1, X.shape[1])
    if max_components <= 0:
        raise ValueError("LDA requires at least two classes in y")
    if n_components is None:
        return max_components
    return min(int(n_components), max_components)


def effective_n_components(n_components, X):
    """Keep PCA valid inside small train folds."""

    if n_components is None:
        return min(DEFAULT_N_COMPONENTS, X.shape[0], X.shape[1])
    if isinstance(n_components, int):
        return min(n_components, X.shape[0], X.shape[1])
    return n_components


def preprocess_pca_train_test(
    X_train,
    X_test=None,
    n_components=DEFAULT_N_COMPONENTS,
    random_state=RANDOM_SEED,
):
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
    """sklearn transformer wrapping preprocess_fft."""

    def __init__(self, idx_frequence_max=None):
        self.idx_frequence_max = idx_frequence_max

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return preprocess_fft(X, idx_frequence_max=self.idx_frequence_max)


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
        fft_features = preprocess_fft(X, idx_frequence_max=self.idx_frequence_max)
        self.pca_ = PCA(
            n_components=effective_n_components(self.n_components, fft_features),
            random_state=self.random_state,
        )
        pca_features = self.pca_.fit_transform(fft_features)
        self.scaler_ = StandardScaler().fit(pca_features) if self.scale else None
        return self

    def transform(self, X, y=None):
        fft_features = preprocess_fft(X, idx_frequence_max=self.idx_frequence_max)
        pca_features = self.pca_.transform(fft_features)
        if self.scaler_ is not None:
            return self.scaler_.transform(pca_features)
        return pca_features


class FFTKernelPCAFeatureExtractor(BaseEstimator, TransformerMixin):
    """FFT magnitude followed by Kernel PCA."""

    def __init__(
        self,
        idx_frequence_max=None,
        n_components=DEFAULT_N_COMPONENTS,
        kernel="rbf",
        gamma=None,
        degree=3,
        coef0=1,
        scale=True,
        random_state=RANDOM_SEED,
        n_jobs=None,
    ):
        self.idx_frequence_max = idx_frequence_max
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.scale = scale
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        fft_features = preprocess_fft(X, idx_frequence_max=self.idx_frequence_max)
        self.scaler_ = StandardScaler().fit(fft_features) if self.scale else None
        reduced_input = (
            self.scaler_.transform(fft_features)
            if self.scaler_ is not None
            else fft_features
        )
        self.kernel_pca_ = KernelPCA(
            n_components=effective_n_components(self.n_components, reduced_input),
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.kernel_pca_.fit(reduced_input)
        return self

    def transform(self, X, y=None):
        fft_features = preprocess_fft(X, idx_frequence_max=self.idx_frequence_max)
        reduced_input = (
            self.scaler_.transform(fft_features)
            if self.scaler_ is not None
            else fft_features
        )
        return self.kernel_pca_.transform(reduced_input)


class FFTLDAFeatureExtractor(BaseEstimator, TransformerMixin):
    """FFT magnitude followed by supervised Linear Discriminant Analysis."""

    def __init__(
        self,
        idx_frequence_max=None,
        n_components=None,
        scale=True,
        solver="svd",
        shrinkage=None,
    ):
        self.idx_frequence_max = idx_frequence_max
        self.n_components = n_components
        self.scale = scale
        self.solver = solver
        self.shrinkage = shrinkage

    def fit(self, X, y=None):
        fft_features = preprocess_fft(X, idx_frequence_max=self.idx_frequence_max)
        self.scaler_ = StandardScaler().fit(fft_features) if self.scale else None
        reduced_input = (
            self.scaler_.transform(fft_features)
            if self.scaler_ is not None
            else fft_features
        )
        self.lda_ = LinearDiscriminantAnalysis(
            n_components=_effective_lda_n_components(
                self.n_components,
                reduced_input,
                y,
            ),
            solver=self.solver,
            shrinkage=self.shrinkage,
        )
        self.lda_.fit(reduced_input, y)
        return self

    def transform(self, X, y=None):
        fft_features = preprocess_fft(X, idx_frequence_max=self.idx_frequence_max)
        reduced_input = (
            self.scaler_.transform(fft_features)
            if self.scaler_ is not None
            else fft_features
        )
        return self.lda_.transform(reduced_input)


class FFTNMFFeatureExtractor(BaseEstimator, TransformerMixin):
    """FFT magnitude followed by Non-negative Matrix Factorization."""

    def __init__(
        self,
        idx_frequence_max=None,
        n_components=DEFAULT_N_COMPONENTS,
        init="nndsvda",
        max_iter=1_000,
        scale=False,
        random_state=RANDOM_SEED,
    ):
        self.idx_frequence_max = idx_frequence_max
        self.n_components = n_components
        self.init = init
        self.max_iter = max_iter
        self.scale = scale
        self.random_state = random_state

    def fit(self, X, y=None):
        fft_features = preprocess_fft(X, idx_frequence_max=self.idx_frequence_max)
        self.nmf_ = NMF(
            n_components=effective_n_components(self.n_components, fft_features),
            init=self.init,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        nmf_features = self.nmf_.fit_transform(fft_features)
        self.scaler_ = StandardScaler().fit(nmf_features) if self.scale else None
        return self

    def transform(self, X, y=None):
        fft_features = preprocess_fft(X, idx_frequence_max=self.idx_frequence_max)
        nmf_features = self.nmf_.transform(fft_features)
        if self.scaler_ is not None:
            return self.scaler_.transform(nmf_features)
        return nmf_features


class FFTSVDFeatureExtractor(BaseEstimator, TransformerMixin):
    """FFT magnitude followed by Truncated SVD."""

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
        fft_features = preprocess_fft(X, idx_frequence_max=self.idx_frequence_max)
        self.svd_ = TruncatedSVD(
            n_components=effective_n_components(self.n_components, fft_features),
            random_state=self.random_state,
        )
        svd_features = self.svd_.fit_transform(fft_features)
        self.scaler_ = StandardScaler().fit(svd_features) if self.scale else None
        return self

    def transform(self, X, y=None):
        fft_features = preprocess_fft(X, idx_frequence_max=self.idx_frequence_max)
        svd_features = self.svd_.transform(fft_features)
        if self.scaler_ is not None:
            return self.scaler_.transform(svd_features)
        return svd_features


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
        return preprocess_stft(
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
        return preprocess_mfcc(X, stat=self.stat, sr=self.sr, n_mfcc=self.n_mfcc)


class MFCCSummaryTransformer(BaseEstimator, TransformerMixin):
    """sklearn transformer returning richer MFCC summary features."""

    def __init__(
        self,
        n_mfcc=DEFAULT_N_MFCC,
        stats=("mean", "std"),
        include_delta=False,
        include_delta2=False,
        sr=DEFAULT_SAMPLE_RATE,
        scale=False,
    ):
        self.n_mfcc = n_mfcc
        self.stats = stats
        self.include_delta = include_delta
        self.include_delta2 = include_delta2
        self.sr = sr
        self.scale = scale

    def fit(self, X, y=None):
        features = preprocess_mfcc_summary(
            X,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            stats=self.stats,
            include_delta=self.include_delta,
            include_delta2=self.include_delta2,
        )
        self.scaler_ = StandardScaler().fit(features) if self.scale else None
        return self

    def transform(self, X, y=None):
        features = preprocess_mfcc_summary(
            X,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            stats=self.stats,
            include_delta=self.include_delta,
            include_delta2=self.include_delta2,
        )
        if self.scaler_ is not None:
            return self.scaler_.transform(features)
        return features


class WaveletTransformer(BaseEstimator, TransformerMixin):
    """Discrete wavelet feature extractor with optional PCA reduction."""

    def __init__(
        self,
        wavelet="db4",
        level=4,
        representation="packet_energy",
        stats=("energy", "mean", "std", "max_abs"),
        n_components=None,
        scale=False,
        random_state=RANDOM_SEED,
    ):
        self.wavelet = wavelet
        self.level = level
        self.representation = representation
        self.stats = stats
        self.n_components = n_components
        self.scale = scale
        self.random_state = random_state

    def fit(self, X, y=None):
        features = self._wavelet_features(X)
        if self.n_components is None:
            self.pca_ = None
            reduced_features = features
        else:
            self.pca_ = PCA(
                n_components=effective_n_components(self.n_components, features),
                random_state=self.random_state,
            )
            reduced_features = self.pca_.fit_transform(features)
        self.scaler_ = StandardScaler().fit(reduced_features) if self.scale else None
        return self

    def transform(self, X, y=None):
        features = self._wavelet_features(X)
        if self.pca_ is not None:
            features = self.pca_.transform(features)
        if self.scaler_ is not None:
            return self.scaler_.transform(features)
        return features

    def _wavelet_features(self, X):
        pywt = _require_pywt()
        X = _ensure_2d(X)
        features = []
        for signal_row in X:
            effective_level = self._effective_level(signal_row.shape[0], pywt)
            if self.representation == "packet_energy":
                packet = pywt.WaveletPacket(
                    data=signal_row,
                    wavelet=self.wavelet,
                    mode="symmetric",
                    maxlevel=effective_level,
                )
                nodes = packet.get_level(effective_level, order="freq")
                features.append([np.mean(node.data**2) for node in nodes])
            else:
                coeffs = pywt.wavedec(
                    signal_row,
                    self.wavelet,
                    level=effective_level,
                    mode="symmetric",
                )
                if self.representation == "approximation":
                    features.append(coeffs[0])
                elif self.representation == "coefficients":
                    features.append(np.concatenate(coeffs))
                elif self.representation == "stats":
                    features.append(
                        np.concatenate(
                            [
                                _summarize(coeff, self.stats, axis=0)
                                for coeff in coeffs
                            ]
                        )
                    )
                else:
                    raise ValueError(
                        "representation must be one of: 'packet_energy', "
                        "'approximation', 'coefficients', 'stats'"
                    )
        return np.asarray(features, dtype=float)

    def _effective_level(self, signal_length, pywt):
        _validate_positive_int(self.level, "level")
        wavelet = pywt.Wavelet(self.wavelet)
        max_level = pywt.dwt_max_level(signal_length, wavelet.dec_len)
        return min(int(self.level), max_level)


PREPROCESSORS = {
    "fft": FFTTransformer,
    "fft_pca": FFTPCAFeatureExtractor,
    "fft_kernel_pca": FFTKernelPCAFeatureExtractor,
    "fft_lda": FFTLDAFeatureExtractor,
    "fft_nmf": FFTNMFFeatureExtractor,
    "fft_svd": FFTSVDFeatureExtractor,
    "stft": STFTTransformer,
    "mfcc": MFCCTransformer,
    "mfcc_summary": MFCCSummaryTransformer,
    "wavelet": WaveletTransformer,
}


def make_preprocessor(preprocessor=None, **params):
    """Create a fresh preprocessor from a registry name, class, factory, or estimator."""

    if preprocessor is None:
        return None

    if isinstance(preprocessor, str):
        if preprocessor not in PREPROCESSORS:
            valid_names = ", ".join(sorted(PREPROCESSORS))
            raise ValueError(f"Unknown preprocessor '{preprocessor}'. Expected one of: {valid_names}")
        return PREPROCESSORS[preprocessor](**params)

    if isinstance(preprocessor, type):
        return preprocessor(**params)

    if callable(preprocessor) and not hasattr(preprocessor, "fit"):
        return preprocessor(**params)

    estimator = clone(preprocessor)
    if params:
        estimator.set_params(**params)
    return estimator


def default_fft_pca_param_grid():
    return {
        "idx_frequence_max": [1_000, 3_000, 9_261],
        "n_components": [5, 10, 20],
        "scale": [False, True],
    }


def default_stft_param_grid():
    return {
        "stat": ["mean", "max"],
        "idx_frequence_max": [500, 1_000, None],
        "nperseg": [200, DEFAULT_STFT_NPERSEG],
    }


def default_mfcc_param_grid():
    return {
        "stat": ["mean", "max"],
        "n_mfcc": [8, DEFAULT_N_MFCC, 20],
    }


def default_mfcc_summary_param_grid():
    return {
        "n_mfcc": [DEFAULT_N_MFCC, 20],
        "stats": [("mean", "std")],
        "include_delta": [False, True],
        "include_delta2": [False],
        "scale": [True],
    }


def default_fft_kernel_pca_param_grid():
    return {
        "idx_frequence_max": [500, 1_000, 3_000],
        "n_components": [5, 10, 20],
        "kernel": ["rbf", "cosine"],
        "gamma": [None, 0.001],
        "scale": [True],
    }


def default_fft_lda_param_grid():
    return {
        "idx_frequence_max": [500, 1_000, 3_000],
        "n_components": [None],
        "scale": [True],
    }


def default_fft_nmf_param_grid():
    return {
        "idx_frequence_max": [500, 1_000, 3_000],
        "n_components": [5, 10, 20],
        "scale": [False, True],
    }


def default_fft_svd_param_grid():
    return {
        "idx_frequence_max": [500, 1_000, 3_000],
        "n_components": [5, 10, 20],
        "scale": [False, True],
    }


def default_wavelet_param_grid():
    return {
        "wavelet": ["db4", "sym5", "coif1"],
        "level": [3, 4, 5],
        "representation": ["packet_energy", "stats", "approximation"],
        "n_components": [None, 10],
        "scale": [True],
    }


PREPROCESSOR_PARAM_GRIDS = {
    "fft_pca": default_fft_pca_param_grid,
    "fft_kernel_pca": default_fft_kernel_pca_param_grid,
    "fft_lda": default_fft_lda_param_grid,
    "fft_nmf": default_fft_nmf_param_grid,
    "fft_svd": default_fft_svd_param_grid,
    "stft": default_stft_param_grid,
    "mfcc": default_mfcc_param_grid,
    "mfcc_summary": default_mfcc_summary_param_grid,
    "wavelet": default_wavelet_param_grid,
}


def get_preprocessor_param_grid(name):
    """Return the default unprefixed parameter grid for a registered preprocessor."""

    if name not in PREPROCESSOR_PARAM_GRIDS:
        valid_names = ", ".join(sorted(PREPROCESSOR_PARAM_GRIDS))
        raise ValueError(f"Unknown preprocessor grid '{name}'. Expected one of: {valid_names}")
    return {param: list(values) for param, values in PREPROCESSOR_PARAM_GRIDS[name]().items()}
