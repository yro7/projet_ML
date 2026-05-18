"""Neural network classifier definition.

This uses sklearn's MLPClassifier to keep the common fit/predict contract.
Torch-specific exercises can still live in notebooks or a later dedicated
module without changing evaluation code.
"""

from sklearn.neural_network import MLPClassifier

try:
    from ..config import RANDOM_SEED
except ImportError:  # Allows importing from a notebook whose cwd is projet_fil_rouge/
    from config import RANDOM_SEED


def make_classifier(
    hidden_layer_sizes=(32,),
    activation="relu",
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=500,
    random_state=RANDOM_SEED,
    **params,
):
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state,
        **params,
    )


def default_param_grid():
    return {
        "hidden_layer_sizes": [(16,), (32,), (32, 16)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
    }
