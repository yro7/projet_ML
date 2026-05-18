"""Gradient boosting classifier definition."""

from sklearn.ensemble import GradientBoostingClassifier

try:
    from ...config import RANDOM_SEED
except ImportError:  # Allows importing from a notebook whose cwd is projet_fil_rouge/
    from config import RANDOM_SEED


def make_classifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=2,
    random_state=RANDOM_SEED,
    **params,
):
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        **params,
    )


def default_param_grid():
    return {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.5],
    }
