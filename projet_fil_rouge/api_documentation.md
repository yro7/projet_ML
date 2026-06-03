# Documentation API Simple

Cette documentation décrit l'organisation et l'interface des trois dossiers principaux du projet : **`classifiers`**, **`evaluation`** et **`utils`** (situés dans [projet_fil_rouge](file:///Users/arkanyota/FilRouge/projet_ML/projet_fil_rouge)).

---

## 1. Classifiers (`classifiers/`)

Ce dossier gère l'enregistrement et la configuration des algorithmes de classification. Il fournit des wrappers pour scikit-learn ainsi que des implémentations personnalisées.

### 📂 Fichiers et Rôles

*   **[`__init__.py`](file:///Users/arkanyota/FilRouge/projet_ML/projet_fil_rouge/classifiers/__init__.py)**
    *   **Régistre central** : Regroupe tous les classifieurs disponibles (`CLASSIFIERS`) et leurs grilles d'hyperparamètres par défaut (`CLASSIFIER_PARAM_GRIDS`).
    *   **`make_classifier(classifier, **params)`** : Instancie un classifieur à partir de sa clé dans le registre (ex: `"logistic_regression"`), de son type ou d'une instance existante.
    *   **`get_classifier_param_grid(name)`** : Renvoie une copie de la grille de paramètres associée à un classifieur.
*   **[`specs.py`](file:///Users/arkanyota/FilRouge/projet_ML/projet_fil_rouge/classifiers/specs.py)**
    *   Définit la structure **`ClassifierSpec`** (dataclass) utilisée pour orchestrer et labelliser les benchmarks.
    *   Fournit les fonctions de création de specs :
        *   `create_base_classifier_specs(fast)` : Régression logistique, SVM et Réseau de neurones de base.
        *   `create_bagging_classifier_specs(fast)` : Ensembles bagging combinés avec chacun des classifieurs de base.
        *   `create_preset_classifier_specs(fast)` : Modèles d'ensembles prédéfinis (Bagging Tree, Random Forest, Gradient Boosting, AdaBoost).
        *   `create_all_classifier_specs(...)` et `classifier_specs_by_key(...)`.
*   **[`logistic_regression.py`](file:///Users/arkanyota/FilRouge/projet_ML/projet_fil_rouge/classifiers/logistic_regression.py)**
    *   Wrapper pour `sklearn.linear_model.LogisticRegression`.
*   **[`neural_network.py`](file:///Users/arkanyota/FilRouge/projet_ML/projet_fil_rouge/classifiers/neural_network.py)**
    *   Wrapper pour `sklearn.neural_network.MLPClassifier`.
*   **[`svm.py`](file:///Users/arkanyota/FilRouge/projet_ML/projet_fil_rouge/classifiers/svm.py)**
    *   Wrapper pour `sklearn.svm.SVC`.

---

### 👥 Ensembles (`classifiers/ensembles/`)

Ce sous-dossier contient les modèles d'ensembles (bagging, boosting) personnalisés ou importés :

*   **[`adaboost.py`](file:///Users/arkanyota/FilRouge/projet_ML/projet_fil_rouge/classifiers/ensembles/adaboost.py)**
    *   **`ScratchAdaBoostClassifier`** : Une classe personnalisée implémentant AdaBoost binaire de manière pédagogique avec gestion de poids.
    *   `make_classifier(...)` : Factory instanciant l'AdaBoost de scikit-learn combiné avec un arbre de décision peu profond.
*   **[`bagging.py`](file:///Users/arkanyota/FilRouge/projet_ML/projet_fil_rouge/classifiers/ensembles/bagging.py)**
    *   **`BaggingClassifier`** : Une implémentation maison de bootstrap aggregation pouvant envelopper n'importe quel classifieur de base scikit-learn.
    *   **`TreeBaggingClassifier`** : Version spécialisée utilisant des arbres de décision par défaut.
*   **[`gradient_boosting.py`](file:///Users/arkanyota/FilRouge/projet_ML/projet_fil_rouge/classifiers/ensembles/gradient_boosting.py)**
    *   Wrapper pour `sklearn.ensemble.GradientBoostingClassifier`.
*   **[`random_forest.py`](file:///Users/arkanyota/FilRouge/projet_ML/projet_fil_rouge/classifiers/ensembles/random_forest.py)**
    *   Wrapper pour `sklearn.ensemble.RandomForestClassifier`.

---

## 2. Evaluation (`evaluation/`)

Ce dossier orchestre les entraînements, les benchmarks, la validation croisée et le calcul/sauvegarde des scores.

### 📂 Fichiers et Rôles

*   **[`__init__.py`](file:///Users/arkanyota/FilRouge/projet_ML/projet_fil_rouge/evaluation/__init__.py)**
    *   Expose les fonctions d'analyse, d'orchestration de tâches de benchmark et d'exécution des méthodes d'évaluation.
*   **[`benchmark.py`](file:///Users/arkanyota/FilRouge/projet_ML/projet_fil_rouge/evaluation/benchmark.py)**
    *   **`build_estimator_pipeline(classifier, preprocessor)`** : Assemble un `Pipeline` scikit-learn.
    *   **`run_grid_search(...)`** : Exécute `GridSearchCV` sur l'ensemble du pipeline (prétraitement + classifieur).
    *   **`train_test_benchmark(...)`** : Réalise un découpage train/test simple en veillant à ajuster le prétraitement uniquement sur les données d'entraînement.
*   **[`manual_cv.py`](file:///Users/arkanyota/FilRouge/projet_ML/projet_fil_rouge/evaluation/manual_cv.py)**
    *   **`manual_loo_score(...)`** : Validation croisée *Leave-One-Out* explicite avec prétraitement ajusté à chaque pli pour éviter la fuite de données (*data leakage*).
    *   **`manual_grid_search(...)`** : Grille de recherche personnalisée combinant les paramètres de prétraitement et de modèle en utilisant le LOO.
*   **[`fit_methods.py`](file:///Users/arkanyota/FilRouge/projet_ML/projet_fil_rouge/evaluation/fit_methods.py)**
    *   Définit la structure **`FitMethodSpec`** qui enveloppe les runners d'évaluation :
        *   `run_grid_search_fit_method` : Exécute GridSearchCV.
        *   `run_train_test_fit_method` : Évalue sur un split train/test.
        *   `run_manual_loo_fit_method` : Évalue via le Leave-One-Out manuel.
*   **[`benchmark_matrix.py`](file:///Users/arkanyota/FilRouge/projet_ML/projet_fil_rouge/evaluation/benchmark_matrix.py)**
    *   **Orchestrateur matriciel** : Combine les ensembles de données, les méthodes d'évaluation, les prétraitements et les classifieurs en une file de tâches.
    *   Gère la reprise après interruption en identifiant les tâches déjà exécutées.
*   **[`benchmark_analysis.py`](file:///Users/arkanyota/FilRouge/projet_ML/projet_fil_rouge/evaluation/benchmark_analysis.py)**
    *   Charge et traite les fichiers CSV générés par les benchmarks.
    *   Génère des statistiques descriptives : top des modèles, agrégations par classifieur/prétraitement, et journal d'erreurs.
*   **[`metrics.py`](file:///Users/arkanyota/FilRouge/projet_ML/projet_fil_rouge/evaluation/metrics.py)**
    *   Fonctions utilitaires de calcul d'exactitude (`accuracy`), matrice de confusion (`confusion`) et d'un dictionnaire récapitulatif (`classification_summary`).
*   **[`results_io.py`](file:///Users/arkanyota/FilRouge/projet_ML/projet_fil_rouge/evaluation/results_io.py)**
    *   Gère les lectures et écritures de résultats dans les fichiers CSV de benchmark.

---

## 3. Utils (`utils/`)

Regroupe les étapes de transformation de données audio et les outils de visualisation graphique.

### 📂 Fichiers et Rôles

*   **[`plots.py`](file:///Users/arkanyota/FilRouge/projet_ML/projet_fil_rouge/utils/plots.py)**
    *   **`plot_pca_explained_variance(pca, ax)`** : Graphique de la variance cumulée d'une PCA.
    *   **`plot_confusion_matrix(...)`** : Affiche une matrice de confusion normalisée.
    *   **`plot_benchmark_result(...)`** : Affiche les matrices de confusion train et test d'un résultat.
    *   **`show_subplots_for_transformed_data(...)`** : Trace des grilles de spectrogrammes (STFT) ou de coefficients Cepstraux (MFCC) pour chaque mot et genre musical.
*   **[`preprocessings.py`](file:///Users/arkanyota/FilRouge/projet_ML/projet_fil_rouge/utils/preprocessings.py)**
    *   **Fonctions d'extraction brutes** :
        *   `preprocess_fft(...)` : Transformée de Fourier rapide.
        *   `stft_magnitude(...)` & `preprocess_stft(...)` : Spectrogramme STFT moyen/max.
        *   `mfcc_coefficients(...)` & `preprocess_mfcc(...)` : Coefficients MFCC moyens/max.
        *   `preprocess_mfcc_summary(...)` : Résumé MFCC enrichi avec dérivées temporelles (Deltas et Delta-Deltas).
    *   **Transformeurs compatibles scikit-learn (`TransformerMixin`)** :
        *   `FFTTransformer`
        *   `FFTPCAFeatureExtractor` : FFT suivie d'une réduction PCA et mise à l'échelle.
        *   `FFTKernelPCAFeatureExtractor` : FFT suivie d'une Kernel PCA.
        *   `FFTLDAFeatureExtractor` : FFT suivie d'une LDA supervisée.
        *   `FFTNMFFeatureExtractor` : FFT suivie d'une factorisation NMF.
        *   `FFTSVDFeatureExtractor` : FFT suivie d'une troncature SVD.
        *   `STFTTransformer`
        *   `MFCCTransformer`
        *   `MFCCSummaryTransformer`
        *   `WaveletTransformer` : Décomposition en ondelettes discrètes (DWT) ou paquets d'ondelettes.
    *   **`make_preprocessor(preprocessor, **params)`** : Factory instanciant le transformeur demandé à partir de son nom dans le registre `PREPROCESSORS`.
