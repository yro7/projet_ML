"""Plotting helpers only"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

try:
    from ..config import DEFAULT_SAMPLE_RATE, GENRES, WORDS
except ImportError:  # Allows importing from a notebook whose cwd is projet_fil_rouge/
    from config import DEFAULT_SAMPLE_RATE, GENRES, WORDS


def plot_pca_explained_variance(pca, ax=None):
    """Plot cumulative explained variance for a fitted PCA."""

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(np.cumsum(pca.explained_variance_ratio_))
    ax.set_xlabel("Nombre de composantes")
    ax.set_ylabel("Variance expliquée cumulée")
    return ax


def plot_confusion_matrix(y_true, y_pred, labels=WORDS, title=None, ax=None, cmap="Blues"):
    """Plot a confusion matrix from predictions."""

    if ax is None:
        _, ax = plt.subplots()
    cm = confusion_matrix(y_true, y_pred)
    display_labels = labels if labels is not None and len(labels) == cm.shape[0] else None
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    display.plot(cmap=cmap, ax=ax, colorbar=False)
    if title:
        ax.set_title(title)
    return ax


def _result_confusion_matrix(result, split):
    metrics = result.get(f"{split}_metrics", {})
    if "confusion_matrix" in metrics:
        return np.asarray(metrics["confusion_matrix"])

    y_true = result.get(f"y_{split}")
    y_pred = result.get(f"y_{split}_pred")
    if y_true is not None and y_pred is not None:
        return confusion_matrix(y_true, y_pred)
    return None


def plot_benchmark_result(result, labels=None, title_prefix=""):
    """Plot train/test confusion matrices already present in a benchmark result."""

    matrices = []
    for split, label in (("train", "Train"), ("test", "Test")):
        matrix = _result_confusion_matrix(result, split)
        if matrix is not None:
            matrices.append((label, matrix))

    if not matrices:
        raise ValueError("No train/test confusion matrix found in result")

    fig, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 4))
    axes = np.atleast_1d(axes)
    default_labels = WORDS if labels is None else labels

    for ax, (split_label, matrix) in zip(axes, matrices):
        display_labels = default_labels if default_labels is not None and len(default_labels) == matrix.shape[0] else None
        display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=display_labels)
        display.plot(cmap="Blues", ax=ax, colorbar=False)
        title = f"Matrice de confusion {split_label.lower()}"
        ax.set_title(f"{title_prefix} - {title}" if title_prefix else title)

    fig.tight_layout()
    return fig, axes


def show_subplots_for_transformed_data(
    transformed_data,
    y,
    genres,
    method="stft",
    words=WORDS,
    list_genres=GENRES,
    fs=DEFAULT_SAMPLE_RATE,
    n_instances=3,
):
    """Show transformed signals by word and genre.

    Expected shapes:
    - STFT: (n_samples, n_freqs, n_times)
    - MFCC: (n_samples, n_coeffs, n_times)
    """

    transformed_data = np.asarray(transformed_data)
    y = np.asarray(y)
    genres = np.asarray(genres)
    figures = []

    for genre in list_genres:
        fig, axes = plt.subplots(len(words), n_instances, figsize=(5 * n_instances, 10))
        axes = np.asarray(axes).reshape(len(words), n_instances)
        fig.suptitle(f"Spectrogrammes pour le genre : {genre}")

        for word_label, word in enumerate(words):
            indices = np.where((y == word_label) & (genres == genre))[0][:n_instances]
            for col in range(n_instances):
                ax = axes[word_label, col]
                if col >= len(indices):
                    ax.axis("off")
                    continue

                sample = transformed_data[indices[col]]
                if method == "stft":
                    ax.imshow(
                        20 * np.log10(np.abs(sample) + 1e-10),
                        origin="lower",
                        aspect="auto",
                        cmap="jet",
                    )
                elif method == "mfcc":
                    import librosa.display

                    librosa.display.specshow(sample, sr=fs, ax=ax, x_axis="time")
                else:
                    raise ValueError("method must be 'stft' or 'mfcc'")

                if col == 0:
                    ax.set_ylabel(word)
                if word_label == 0:
                    ax.set_title(f"Instance {col + 1}")

        fig.tight_layout()
        figures.append(fig)

    return figures
