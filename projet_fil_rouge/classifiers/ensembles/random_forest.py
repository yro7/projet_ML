"""Random forest classifier definition."""

from sklearn.ensemble import RandomForestClassifier

try:
    from ...config import RANDOM_SEED
except ImportError:  # Allows importing from a notebook whose cwd is projet_fil_rouge/
    from config import RANDOM_SEED


def make_classifier(
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=RANDOM_SEED,
    n_jobs=-1,
    **params,
):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
        **params,
    )


def default_param_grid():
    return {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_leaf": [1, 2],
    }
