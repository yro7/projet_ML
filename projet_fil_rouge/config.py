"""Config générale du projet"""

from pathlib import Path

RANDOM_SEED = 51

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "FichierTest"
PERSONAL_DATA_DIR = PROJECT_DIR / "FichierPerso"
GOOGLE_SPEECH_COMMANDS_DIR = PROJECT_DIR / "SpeechCommands"

DATASET_CURRENT = "current"
DATASET_PERSONAL_AUGMENTED = "personal_augmented"
DATASET_GOOGLE_SPEECH_COMMANDS = "google_speech_commands"

WORDS = ("avance", "recule", "tournegauche")
GOOGLE_DEFAULT_WORDS = ("forward", "backward", "left")
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
