"""Project-wide constants.

Keep this module small: it is shared by data loading, preprocessing, models,
and evaluation without owning any business logic.
"""

from pathlib import Path

RANDOM_SEED = 51

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "FichierTest"

WORDS = ("avance", "recule", "tournegauche")
GENRES = ("M", "F")

DEFAULT_SAMPLE_RATE = 22_050
DEFAULT_N_COMPONENTS = 20
DEFAULT_N_MFCC = 13
DEFAULT_STFT_NPERSEG = 400


def seed_everything(seed=RANDOM_SEED):
    """Seed numpy and torch when available."""

    import numpy as np

    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
