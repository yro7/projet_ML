# Guide notebook consumer

Ce guide montre comment utiliser le notebook comme consumer leger de la librairie.
Les registries exposent les preprocessors, classifiers et grilles par defaut par nom.

## Grid search SVM

```python
from projet_fil_rouge.classifiers import get_classifier_param_grid
from projet_fil_rouge.evaluation.benchmark import run_grid_search
from projet_fil_rouge.utils.preprocessings import get_preprocessor_param_grid

result = run_grid_search(
    X_train=X,
    y_train=y,
    preprocessor="fft_pca",
    preprocessor_param_grid=get_preprocessor_param_grid("fft_pca"),
    classifier="svm",
    classifier_param_grid=get_classifier_param_grid("svm"),
    cv=3,
    n_jobs=1,
)
```

## Benchmark bagging_tree

```python
from projet_fil_rouge.evaluation.benchmark import train_test_benchmark

result = train_test_benchmark(
    X_raw=X,
    y=y,
    preprocessor="fft_pca",
    preprocessor_params={"idx_frequence_max": 1_000, "n_components": 10, "scale": True},
    classifier="bagging_tree",
    classifier_params={"n_estimators": 50, "max_depth": 2},
)
```

## Plot du resultat

```python
from projet_fil_rouge.config import WORDS
from projet_fil_rouge.utils.plots import plot_benchmark_result

plot_benchmark_result(result, labels=WORDS, title_prefix="Bagging tree")
```
