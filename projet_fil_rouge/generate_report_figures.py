"""Generate all figures for the LaTeX report.

Run from project root:
    python3 projet_fil_rouge/generate_report_figures.py
"""
import sys
from pathlib import Path
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GridSearchCV, LeaveOneOut, train_test_split, cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
)
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from projet_fil_rouge.config import RANDOM_SEED, WORDS, seed_everything
from projet_fil_rouge.data import load_dataset
from projet_fil_rouge.utils.preprocessings import (
    preprocess_fft, stft_magnitude, mfcc_coefficients,
    preprocess_stft, preprocess_mfcc, FFTTransformer,
)
from projet_fil_rouge.classifiers.ensembles.bagging import BaggingClassifier
from projet_fil_rouge.classifiers.ensembles.adaboost import ScratchAdaBoostClassifier

seed_everything(RANDOM_SEED)

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

WORDS_LIST = list(WORDS)

def savefig(fig, name):
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Load data
# ============================================================
print("Loading dataset...")
X, y, genres, fs = load_dataset()
print(f"  X={X.shape}, y={y.shape}, fs={fs}")

X_fft = preprocess_fft(X)

# ============================================================
# Fig 1: PCA explained variance
# ============================================================
print("\n[Fig 1] PCA explained variance...")
pca_full = PCA(n_components=20, random_state=RANDOM_SEED).fit(X_fft)
fig, ax = plt.subplots(figsize=(8, 4))
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
ax.bar(range(1, 21), pca_full.explained_variance_ratio_, alpha=0.6, label="Individuelle")
ax.plot(range(1, 21), cumvar, "ro-", label="Cumulée")
ax.set_xlabel("Composante principale")
ax.set_ylabel("Variance expliquée")
ax.set_title("Variance expliquée par l'ACP sur $|\\hat{X}|$")
ax.legend()
ax.set_xticks(range(1, 21))
ax.grid(axis="y", alpha=0.3)
savefig(fig, "pca_explained_variance.pdf")

# ============================================================
# Fig 2: STFT spectrograms (3x3 male, 3x3 female)
# ============================================================
print("\n[Fig 2] STFT spectrograms...")
X_stft = stft_magnitude(X, fs=fs, nperseg=400)

def plot_spectrograms(X_transformed, y, genres, method, gender, gender_label):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle(f"{method.upper()} — Locuteurs {gender_label}", fontsize=14)
    mask = np.array([g == gender for g in genres])
    for word_idx, word in enumerate(WORDS_LIST):
        word_mask = (y == word_idx) & mask
        indices = np.where(word_mask)[0][:3]
        for j, idx in enumerate(indices):
            ax = axes[word_idx][j]
            if X_transformed.ndim == 3:
                ax.imshow(X_transformed[idx], aspect="auto", origin="lower", cmap="viridis")
            ax.set_title(f"{word} (#{j+1})")
            if j == 0:
                ax.set_ylabel("Fréquence")
            ax.set_xlabel("Temps")
    plt.tight_layout()
    return fig

fig_stft_m = plot_spectrograms(X_stft, y, genres, "stft", "M", "Masculins")
savefig(fig_stft_m, "stft_spectrograms_male.pdf")

fig_stft_f = plot_spectrograms(X_stft, y, genres, "stft", "F", "Féminins")
savefig(fig_stft_f, "stft_spectrograms_female.pdf")

# ============================================================
# Fig 3: MFCC coefficients (3x3 male, 3x3 female)
# ============================================================
print("\n[Fig 3] MFCC spectrograms...")
X_mfcc = mfcc_coefficients(X, sr=fs, n_mfcc=13)

fig_mfcc_m = plot_spectrograms(X_mfcc, y, genres, "mfcc", "M", "Masculins")
savefig(fig_mfcc_m, "mfcc_spectrograms_male.pdf")

fig_mfcc_f = plot_spectrograms(X_mfcc, y, genres, "mfcc", "F", "Féminins")
savefig(fig_mfcc_f, "mfcc_spectrograms_female.pdf")

# ============================================================
# Partie I setup: train/test split
# ============================================================
print("\n[Partie I] Preparing train/test split...")
X_abs = preprocess_fft(X)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_abs, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y,
)

# Helper: FFT transformer for pipelines
class FFT:
    def __init__(self, idx_frequence_max=None):
        self.idx_frequence_max = idx_frequence_max
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return preprocess_fft(X, idx_frequence_max=self.idx_frequence_max)
    def get_params(self, deep=True):
        return {"idx_frequence_max": self.idx_frequence_max}
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

# ============================================================
# Fig 4: Confusion matrices — LR sans scaler
# ============================================================
print("\n[Fig 4] LR Pipeline + GridSearchCV (sans scaler)...")
pipeline = Pipeline([
    ('fft', FFT()),
    ('pca', PCA(random_state=RANDOM_SEED)),
    ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
])
param_grid = {
    'fft__idx_frequence_max': [1000, 3000],
    'pca__n_components': [5, 20],
    'clf__C': [0.1, 10.0],
}
grid_search = GridSearchCV(pipeline, param_grid, cv=LeaveOneOut(), scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_raw, y_train)

print(f"  Best params: {grid_search.best_params_}")
print(f"  Best LOO score: {grid_search.best_score_:.4f}")
print(f"  Test score: {grid_search.score(X_test_raw, y_test):.4f}")

y_train_oof = cross_val_predict(grid_search.best_estimator_, X_train_raw, y_train, cv=LeaveOneOut())
y_test_pred = grid_search.predict(X_test_raw)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_true, y_pred, title in [
    (axes[0], y_train, y_train_oof, f"Train (LOO) — Acc={accuracy_score(y_train, y_train_oof):.2%}"),
    (axes[1], y_test, y_test_pred, f"Test — Acc={accuracy_score(y_test, y_test_pred):.2%}"),
]:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=WORDS_LIST)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
plt.suptitle("Régression logistique (FFT + PCA)", fontsize=13)
plt.tight_layout()
savefig(fig, "cm_logistic_regression.pdf")

# ============================================================
# Fig 5: Confusion matrices — LR avec StandardScaler
# ============================================================
print("\n[Fig 5] LR Pipeline + StandardScaler...")
pipeline_scaled = Pipeline([
    ('fft', FFT()),
    ('pca', PCA(random_state=RANDOM_SEED)),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
])
grid_search_scaled = GridSearchCV(pipeline_scaled, param_grid, cv=LeaveOneOut(), scoring='accuracy', n_jobs=-1)
grid_search_scaled.fit(X_train_raw, y_train)

print(f"  Best LOO (scaled): {grid_search_scaled.best_score_:.4f} vs {grid_search.best_score_:.4f}")
print(f"  Test (scaled): {grid_search_scaled.score(X_test_raw, y_test):.4f}")

y_train_oof_s = cross_val_predict(grid_search_scaled.best_estimator_, X_train_raw, y_train, cv=LeaveOneOut())
y_test_pred_s = grid_search_scaled.predict(X_test_raw)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_true, y_pred, title in [
    (axes[0], y_train, y_train_oof_s, f"Train (LOO) — Acc={accuracy_score(y_train, y_train_oof_s):.2%}"),
    (axes[1], y_test, y_test_pred_s, f"Test — Acc={accuracy_score(y_test, y_test_pred_s):.2%}"),
]:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=WORDS_LIST)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
plt.suptitle("Régression logistique + StandardScaler (FFT + PCA)", fontsize=13)
plt.tight_layout()
savefig(fig, "cm_logistic_regression_scaled.pdf")

# ============================================================
# Fig 6: Confusion matrices — SVM RBF
# ============================================================
print("\n[Fig 6] SVM RBF Pipeline...")
pipeline_svm = Pipeline([
    ('fft', FFT()),
    ('pca', PCA(random_state=RANDOM_SEED)),
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', random_state=RANDOM_SEED)),
])
param_grid_svm = {
    'fft__idx_frequence_max': [1000, 3000],
    'pca__n_components': [10, 20],
    'clf__C': [0.1, 10.0],
    'clf__gamma': [0.001, 'scale'],
}
grid_search_svm = GridSearchCV(pipeline_svm, param_grid_svm, cv=LeaveOneOut(), scoring='accuracy', n_jobs=-1)
grid_search_svm.fit(X_train_raw, y_train)

print(f"  Best params: {grid_search_svm.best_params_}")
print(f"  Best LOO: {grid_search_svm.best_score_:.4f}")
print(f"  Test: {grid_search_svm.score(X_test_raw, y_test):.4f}")

y_train_oof_svm = cross_val_predict(grid_search_svm.best_estimator_, X_train_raw, y_train, cv=LeaveOneOut())
y_test_pred_svm = grid_search_svm.predict(X_test_raw)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_true, y_pred, title in [
    (axes[0], y_train, y_train_oof_svm, f"Train (LOO) — Acc={accuracy_score(y_train, y_train_oof_svm):.2%}"),
    (axes[1], y_test, y_test_pred_svm, f"Test — Acc={accuracy_score(y_test, y_test_pred_svm):.2%}"),
]:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=WORDS_LIST)
    disp.plot(ax=ax, cmap="Oranges", colorbar=False)
    ax.set_title(title)
plt.suptitle("SVM RBF (FFT + PCA + StandardScaler)", fontsize=13)
plt.tight_layout()
savefig(fig, "cm_svm_rbf.pdf")

# ============================================================
# Fig 7-8: Bagging + Random Forest (binary)
# ============================================================
print("\n[Fig 7] Bagging & Random Forest (binary)...")
y1 = np.where(y == 1, 1, 0)
pca_bag = PCA(n_components=20, random_state=RANDOM_SEED)
X_pca_bag = pca_bag.fit_transform(X_abs)

X_train_bag, X_test_bag, y_train_bag, y_test_bag = train_test_split(
    X_pca_bag, y1, test_size=0.2, random_state=RANDOM_SEED, stratify=y1,
)

# Bagging from scratch
bag_scratch = BaggingClassifier(
    base_classifier=DecisionTreeClassifier(max_depth=2, random_state=RANDOM_SEED),
    n_estimators=100, random_state=RANDOM_SEED,
)
bag_scratch.fit(X_train_bag, y_train_bag)
y_pred_bag = bag_scratch.predict(X_test_bag)
bag_acc = accuracy_score(y_test_bag, y_pred_bag)

indiv_accs = [accuracy_score(y_test_bag, est.predict(X_test_bag)) for est in bag_scratch.estimators_]
mean_indiv = np.mean(indiv_accs)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=RANDOM_SEED)
rf.fit(X_train_bag, y_train_bag)
rf_acc = rf.score(X_test_bag, y_test_bag)

print(f"  Bagging scratch acc: {bag_acc:.4f}, mean indiv: {mean_indiv:.4f}, RF: {rf_acc:.4f}")

# Bar chart comparison
fig, ax = plt.subplots(figsize=(8, 5))
methods = ["Arbre individuel\n(moyenne)", "Bagging\n(from scratch)", "Random Forest\n(sklearn)"]
accs = [mean_indiv, bag_acc, rf_acc]
colors = ["#e74c3c", "#3498db", "#2ecc71"]
bars = ax.bar(methods, accs, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{acc:.2%}",
            ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("Accuracy")
ax.set_title("Comparaison Bagging vs Random Forest (Classification binaire)")
ax.set_ylim(0, 1.15)
ax.grid(axis="y", alpha=0.3)
savefig(fig, "bagging_rf_comparison.pdf")

# ============================================================
# Fig 8: AdaBoost from scratch + sklearn comparison
# ============================================================
print("\n[Fig 8] AdaBoost comparison...")
ada_scratch = ScratchAdaBoostClassifier(
    base_classifier=DecisionTreeClassifier(max_depth=2, random_state=RANDOM_SEED),
    n_estimators=100, random_state=RANDOM_SEED,
)
ada_scratch.fit(X_train_bag, y_train_bag)
ada_acc = accuracy_score(y_test_bag, ada_scratch.predict(X_test_bag))

# AdaBoost sklearn
ada_sk = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2, random_state=RANDOM_SEED),
    random_state=RANDOM_SEED,
)
grid_ada = GridSearchCV(ada_sk, {"n_estimators": [50, 100], "learning_rate": [0.1, 1.0]},
                         cv=LeaveOneOut(), scoring="accuracy", n_jobs=-1)
grid_ada.fit(X_train_bag, y_train_bag)
ada_sk_acc = grid_ada.score(X_test_bag, y_test_bag)

# Gradient Boosting
gb_clf = GradientBoostingClassifier(max_depth=2, random_state=RANDOM_SEED)
grid_gb = GridSearchCV(gb_clf, {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
                        cv=LeaveOneOut(), scoring="accuracy", n_jobs=-1)
grid_gb.fit(X_train_bag, y_train_bag)
gb_acc = grid_gb.score(X_test_bag, y_test_bag)

print(f"  AdaBoost scratch: {ada_acc:.4f}, sklearn: {ada_sk_acc:.4f}, GB: {gb_acc:.4f}")
print(f"  AdaBoost best params: {grid_ada.best_params_}")
print(f"  GB best params: {grid_gb.best_params_}")

fig, ax = plt.subplots(figsize=(10, 5))
methods = ["Bagging\n(scratch)", "AdaBoost\n(scratch)", "AdaBoost\n(sklearn)", "Gradient\nBoosting"]
accs = [bag_acc, ada_acc, ada_sk_acc, gb_acc]
colors = ["#3498db", "#e67e22", "#e74c3c", "#9b59b6"]
bars = ax.bar(methods, accs, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{acc:.2%}",
            ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("Accuracy")
ax.set_title("Comparaison des méthodes d'ensemble (Classification binaire)")
ax.set_ylim(0, 1.15)
ax.grid(axis="y", alpha=0.3)
savefig(fig, "ensemble_comparison.pdf")

# ============================================================
# Fig 9-10: Neural Network (PyTorch)
# ============================================================
print("\n[Fig 9-10] Neural Network PyTorch...")
import torch
from torch import nn

X_preprocessed = PCA(n_components=20, random_state=RANDOM_SEED).fit_transform(X_abs)
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
    X_preprocessed, y, test_size=0.5, random_state=RANDOM_SEED, stratify=y,
)

X_tr = torch.tensor(X_train_nn).reshape((X_train_nn.shape[0], 1, -1)).float()
X_te = torch.tensor(X_test_nn).reshape((X_test_nn.shape[0], 1, -1)).float()
y_tr = torch.nn.functional.one_hot(torch.tensor(y_train_nn), num_classes=3).reshape((X_train_nn.shape[0], 1, -1)).float()
y_te = torch.nn.functional.one_hot(torch.tensor(y_test_nn), num_classes=3).reshape((X_test_nn.shape[0], 1, -1)).float()

# Simple model (logistic regression)
class NNClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(20, 3))
    def forward(self, xb):
        return self.network(xb)

seed_everything(RANDOM_SEED)
model = NNClassification()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
num_epochs = 15
train_losses, test_losses = [], []

for epoch in range(num_epochs):
    model.train()
    ep_train = []
    for i in range(X_tr.shape[0]):
        optimizer.zero_grad()
        pred = model(X_tr[i])
        target = torch.argmax(y_tr[i], dim=-1)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        ep_train.append(loss.detach())
    model.eval()
    ep_test = []
    with torch.no_grad():
        for i in range(X_te.shape[0]):
            pred = model(X_te[i])
            target = torch.argmax(y_te[i], dim=-1)
            loss = loss_fn(pred, target)
            ep_test.append(loss.detach())
    train_losses.append(torch.stack(ep_train).mean().item())
    test_losses.append(torch.stack(ep_test).mean().item())

print(f"  Final train loss: {train_losses[-1]:.4f}, test loss: {test_losses[-1]:.4f}")

# Fig 9: Loss curves
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, num_epochs+1), train_losses, "b-o", label="Train Loss", markersize=4)
ax.plot(range(1, num_epochs+1), test_losses, "r-s", label="Test Loss", markersize=4)
ax.set_xlabel("Epoch")
ax.set_ylabel("CrossEntropyLoss")
ax.set_title("Courbes de perte — Régression logistique (PyTorch)")
ax.legend()
ax.grid(alpha=0.3)
savefig(fig, "nn_loss_curves.pdf")

# Fig 10: NN confusion matrix
model.eval()
y_true_nn, y_pred_nn = [], []
with torch.no_grad():
    for i in range(X_te.shape[0]):
        pred = model(X_te[i])
        y_pred_nn.append(torch.argmax(pred, dim=-1).item())
        y_true_nn.append(torch.argmax(y_te[i], dim=-1).item())

nn_acc = accuracy_score(y_true_nn, y_pred_nn)
print(f"  NN accuracy: {nn_acc:.4f}")

fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_true_nn, y_pred_nn)
disp = ConfusionMatrixDisplay(cm, display_labels=WORDS_LIST)
disp.plot(ax=ax, cmap="Greens", colorbar=False)
ax.set_title(f"PyTorch — Régression logistique (Acc={nn_acc:.2%})")
savefig(fig, "cm_nn_logistic.pdf")

# ============================================================
# Fig 11: Regularized NN
# ============================================================
print("\n[Fig 11] Regularized NN...")

class RegularizedNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(20, 64), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(64, 3),
        )
    def forward(self, xb):
        return self.network(xb)

seed_everything(RANDOM_SEED)
model_reg = RegularizedNN(dropout_rate=0.3)
optimizer_reg = torch.optim.Adam(model_reg.parameters(), lr=0.005, weight_decay=1e-3)
num_epochs_reg = 25
reg_train, reg_test = [], []

for epoch in range(num_epochs_reg):
    model_reg.train()
    ep_train = []
    for i in range(X_tr.shape[0]):
        optimizer_reg.zero_grad()
        pred = model_reg(X_tr[i])
        target = torch.argmax(y_tr[i], dim=-1)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer_reg.step()
        ep_train.append(loss.detach())
    model_reg.eval()
    ep_test = []
    with torch.no_grad():
        for i in range(X_te.shape[0]):
            pred = model_reg(X_te[i])
            target = torch.argmax(y_te[i], dim=-1)
            loss = loss_fn(pred, target)
            ep_test.append(loss.detach())
    reg_train.append(torch.stack(ep_train).mean().item())
    reg_test.append(torch.stack(ep_test).mean().item())

model_reg.eval()
y_true_reg, y_pred_reg = [], []
with torch.no_grad():
    for i in range(X_te.shape[0]):
        pred = model_reg(X_te[i])
        y_true_reg.append(torch.argmax(y_te[i], dim=-1).item())
        y_pred_reg.append(torch.argmax(pred, dim=-1).item())
reg_acc = accuracy_score(y_true_reg, y_pred_reg)
print(f"  Regularized NN acc: {reg_acc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(range(1, num_epochs_reg+1), reg_train, "b-o", label="Train", markersize=3)
axes[0].plot(range(1, num_epochs_reg+1), reg_test, "r-s", label="Test", markersize=3)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("Courbes de perte — FNN Régularisé")
axes[0].legend(); axes[0].grid(alpha=0.3)

cm = confusion_matrix(y_true_reg, y_pred_reg)
disp = ConfusionMatrixDisplay(cm, display_labels=WORDS_LIST)
disp.plot(ax=axes[1], cmap="Greens", colorbar=False)
axes[1].set_title(f"FNN Régularisé (Acc={reg_acc:.2%})")
plt.tight_layout()
savefig(fig, "nn_regularized.pdf")

# ============================================================
# Fig 12: CNN 1D on raw signals
# ============================================================
print("\n[Fig 12] CNN 1D...")
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    X, y, test_size=0.5, random_state=RANDOM_SEED, stratify=y,
)
X_tr_cnn = torch.tensor(X_train_cnn).reshape((X_train_cnn.shape[0], 1, -1)).float()
X_te_cnn = torch.tensor(X_test_cnn).reshape((X_test_cnn.shape[0], 1, -1)).float()
y_tr_cnn = torch.tensor(y_train_cnn).long()
y_te_cnn = torch.tensor(y_test_cnn).long()

class AudioCNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=15, stride=4), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(8, 16, kernel_size=7, stride=2), nn.ReLU(), nn.MaxPool1d(4),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_size = self.conv(torch.zeros(1, 1, X.shape[1])).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat_size, 64), nn.ReLU(), nn.Linear(64, 3))
    def forward(self, x):
        return self.fc(self.conv(x))

seed_everything(RANDOM_SEED)
cnn_model = AudioCNN1D()
optimizer_cnn = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
cnn_train_losses, cnn_test_losses = [], []

for epoch in range(15):
    cnn_model.train()
    ep_train = []
    for i in range(X_tr_cnn.shape[0]):
        optimizer_cnn.zero_grad()
        pred = cnn_model(X_tr_cnn[i:i+1])
        loss = loss_fn(pred, y_tr_cnn[i:i+1])
        loss.backward()
        optimizer_cnn.step()
        ep_train.append(loss.detach())
    cnn_model.eval()
    ep_test = []
    with torch.no_grad():
        for i in range(X_te_cnn.shape[0]):
            pred = cnn_model(X_te_cnn[i:i+1])
            loss = loss_fn(pred, y_te_cnn[i:i+1])
            ep_test.append(loss.detach())
    cnn_train_losses.append(torch.stack(ep_train).mean().item())
    cnn_test_losses.append(torch.stack(ep_test).mean().item())

cnn_model.eval()
y_pred_cnn = []
with torch.no_grad():
    for i in range(X_te_cnn.shape[0]):
        pred = cnn_model(X_te_cnn[i:i+1])
        y_pred_cnn.append(torch.argmax(pred, dim=-1).item())
cnn_acc = accuracy_score(y_test_cnn, y_pred_cnn)
print(f"  CNN acc: {cnn_acc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(range(1, 16), cnn_train_losses, "b-o", label="Train", markersize=3)
axes[0].plot(range(1, 16), cnn_test_losses, "r-s", label="Test", markersize=3)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("Courbes de perte — CNN 1D (signaux bruts)")
axes[0].legend(); axes[0].grid(alpha=0.3)

cm = confusion_matrix(y_test_cnn, y_pred_cnn)
disp = ConfusionMatrixDisplay(cm, display_labels=WORDS_LIST)
disp.plot(ax=axes[1], cmap="Purples", colorbar=False)
axes[1].set_title(f"CNN 1D — Acc={cnn_acc:.2%}")
plt.tight_layout()
savefig(fig, "cnn_1d_results.pdf")

# ============================================================
# Fig 13-16: Benchmark analysis (Part V)
# ============================================================
print("\n[Fig 13-16] Benchmark analysis from CSV...")
CSV_PATH = ROOT / "projet_fil_rouge" / "results" / "part5_benchmark_aggressive.csv"
rows = []
with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

scored_rows = []
for r in rows:
    if r["status"] == "ok":
        try:
            r["_score"] = float(r["score"])
            scored_rows.append(r)
        except (ValueError, TypeError):
            pass

# Fig 13: Bar chart by preprocessor
by_prep = defaultdict(list)
for r in scored_rows:
    by_prep[r["preprocessor"]].append(r["_score"])

prep_names = sorted(by_prep, key=lambda k: -np.mean(by_prep[k]))
prep_means = [np.mean(by_prep[k]) for k in prep_names]
prep_labels = {
    "mfcc_summary": "MFCC\nSummary", "mfcc": "MFCC", "fft_lda": "FFT+LDA",
    "fft_pca": "FFT+PCA", "fft_svd": "FFT+SVD", "fft_nmf": "FFT+NMF",
    "wavelet": "Wavelet", "fft_kernel_pca": "FFT+\nKernel PCA", "stft": "STFT",
}

fig, ax = plt.subplots(figsize=(12, 5))
colors_prep = plt.cm.viridis(np.linspace(0.2, 0.9, len(prep_names)))
bars = ax.bar([prep_labels.get(n, n) for n in prep_names], prep_means, color=colors_prep,
              edgecolor="black", linewidth=0.5)
for bar, mean in zip(bars, prep_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{mean:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Accuracy moyenne")
ax.set_title("Performance moyenne par méthode de preprocessing (35 669 expériences)")
ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.3)
savefig(fig, "benchmark_by_preprocessor.pdf")

# Fig 14: Bar chart by classifier
by_clf = defaultdict(list)
for r in scored_rows:
    by_clf[r["classifier"]].append(r["_score"])

clf_names = sorted(by_clf, key=lambda k: -np.mean(by_clf[k]))
clf_means = [np.mean(by_clf[k]) for k in clf_names]
clf_labels = {
    "random_forest": "Random\nForest", "bagging_neural_network": "Bagging\nNN",
    "neural_network": "Neural\nNetwork", "adaboost": "AdaBoost",
    "gradient_boosting": "Gradient\nBoosting", "bagging_tree": "Bagging\nTree",
    "bagging_logistic_regression": "Bagging\nLR", "logistic_regression": "Logistic\nRegression",
    "bagging_svm": "Bagging\nSVM", "svm": "SVM",
}

fig, ax = plt.subplots(figsize=(14, 5))
colors_clf = plt.cm.plasma(np.linspace(0.2, 0.9, len(clf_names)))
bars = ax.bar([clf_labels.get(n, n) for n in clf_names], clf_means, color=colors_clf,
              edgecolor="black", linewidth=0.5)
for bar, mean in zip(bars, clf_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{mean:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Accuracy moyenne")
ax.set_title("Performance moyenne par classifier (35 669 expériences)")
ax.set_ylim(0, 0.85)
ax.grid(axis="y", alpha=0.3)
savefig(fig, "benchmark_by_classifier.pdf")

# Fig 15: Heatmap preprocessor x classifier
print("  Generating heatmap...")
by_pc = defaultdict(list)
for r in scored_rows:
    by_pc[(r["preprocessor"], r["classifier"])].append(r["_score"])

heatmap_data = np.zeros((len(prep_names), len(clf_names)))
for i, p in enumerate(prep_names):
    for j, c in enumerate(clf_names):
        scores = by_pc.get((p, c), [])
        heatmap_data[i, j] = np.mean(scores) if scores else np.nan

fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto", vmin=0.3, vmax=1.0)
ax.set_xticks(range(len(clf_names)))
ax.set_xticklabels([clf_labels.get(n, n) for n in clf_names], rotation=45, ha="right")
ax.set_yticks(range(len(prep_names)))
ax.set_yticklabels([prep_labels.get(n, n) for n in prep_names])
for i in range(len(prep_names)):
    for j in range(len(clf_names)):
        val = heatmap_data[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if val > 0.65 else "black")
plt.colorbar(im, ax=ax, label="Accuracy moyenne")
ax.set_title("Matrice Preprocessing × Classifier (accuracy moyenne)")
plt.tight_layout()
savefig(fig, "benchmark_heatmap.pdf")

# Fig 16: Bar chart by fit method
by_fm = defaultdict(list)
for r in scored_rows:
    by_fm[r["fit_method"]].append(r["_score"])

fm_names = sorted(by_fm, key=lambda k: -np.mean(by_fm[k]))
fm_means = [np.mean(by_fm[k]) for k in fm_names]
fm_labels = {"grid_search_cv": "GridSearchCV", "train_test": "Train/Test", "manual_loo": "Manual LOO"}

fig, ax = plt.subplots(figsize=(8, 5))
colors_fm = ["#2ecc71", "#3498db", "#e74c3c"]
bars = ax.bar([fm_labels.get(n, n) for n in fm_names], fm_means, color=colors_fm,
              width=0.4, edgecolor="black", linewidth=0.5)
for bar, mean in zip(bars, fm_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{mean:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Accuracy moyenne")
ax.set_title("Performance moyenne par méthode d'évaluation")
ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.3)
savefig(fig, "benchmark_by_fit_method.pdf")

# ============================================================
# Fig 17: Noise robustness benchmark (with 10 realizations)
# ============================================================
print("\n[Fig 17] Noise robustness benchmark...")
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

def add_noise_to_dataset(X, snr_db, random_state=RANDOM_SEED):
    if snr_db is None or np.isinf(snr_db):
        return X.copy()
    np.random.seed(random_state)
    X_noisy = np.zeros_like(X)
    for i in range(X.shape[0]):
        x = X[i, :]
        power = np.mean(x ** 2)
        if power <= 0:
            X_noisy[i, :] = x
            continue
        noise_power = power / (10 ** (snr_db / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power), size=x.shape)
        X_noisy[i, :] = x + noise
    return X_noisy

from projet_fil_rouge.utils.preprocessings import preprocess_mfcc_summary

snr_levels_dict = {
    "Clean": None,
    "25dB": 25.0,
    "20dB": 20.0,
    "15dB": 15.0,
    "10dB": 10.0,
    "5dB": 5.0,
    "0dB": 0.0
}

noise_classifiers = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED),
    "SVM (RBF)": SVC(C=10.0, kernel="rbf", gamma="scale", random_state=RANDOM_SEED),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
    "Bagging Tree": BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED),
        n_estimators=50,
        random_state=RANDOM_SEED
    ),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, random_state=RANDOM_SEED),
        n_estimators=100,
        learning_rate=0.1,
        random_state=RANDOM_SEED
    ),
    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(32,),
        alpha=0.001,
        max_iter=1000,
        random_state=RANDOM_SEED
    )
}

n_reals = 10
mean_results = {clf_name: [] for clf_name in noise_classifiers}
std_results = {clf_name: [] for clf_name in noise_classifiers}

for lvl_name, snr_v in snr_levels_dict.items():
    print(f"  Evaluating SNR: {lvl_name}...")
    if snr_v is None:
        X_feats = preprocess_mfcc_summary(X, sr=fs, n_mfcc=13)
        for clf_name, clf in noise_classifiers.items():
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            scs = cross_val_score(pipe, X_feats, y, cv=LeaveOneOut(), n_jobs=-1)
            mean_results[clf_name].append(np.mean(scs))
            std_results[clf_name].append(0.0)
    else:
        all_scs = {clf_name: [] for clf_name in noise_classifiers}
        for r in range(n_reals):
            X_n = add_noise_to_dataset(X, snr_v, random_state=RANDOM_SEED + r)
            X_feats = preprocess_mfcc_summary(X_n, sr=fs, n_mfcc=13)
            for clf_name, clf in noise_classifiers.items():
                pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
                scs = cross_val_score(pipe, X_feats, y, cv=LeaveOneOut(), n_jobs=-1)
                all_scs[clf_name].append(np.mean(scs))
        for clf_name in noise_classifiers:
            mean_results[clf_name].append(np.mean(all_scs[clf_name]))
            std_results[clf_name].append(np.std(all_scs[clf_name]))

fig, ax = plt.subplots(figsize=(11, 7))
colors_map = {
    "Logistic Regression": "#34495e", "SVM (RBF)": "#e74c3c", "Random Forest": "#2ecc71",
    "Bagging Tree": "#3498db", "AdaBoost": "#e67e22", "Neural Network": "#9b59b6"
}
markers_map = {
    "Logistic Regression": "o", "SVM (RBF)": "s", "Random Forest": "^",
    "Bagging Tree": "D", "AdaBoost": "v", "Neural Network": "p"
}
x_lbls = list(snr_levels_dict.keys())
x_idxs = np.arange(len(x_lbls))

for clf_name in noise_classifiers:
    mns = np.array(mean_results[clf_name])
    sds = np.array(std_results[clf_name])
    ax.plot(x_idxs, mns, marker=markers_map[clf_name], color=colors_map[clf_name], label=clf_name, linewidth=2, markersize=7)
    ax.fill_between(x_idxs, mns - sds, mns + sds, color=colors_map[clf_name], alpha=0.15)
    ax.text(x_idxs[-1] + 0.08, mns[-1], f"{mns[-1]:.1%} ± {sds[-1]:.1%}" if sds[-1] > 0 else f"{mns[-1]:.1%}", color=colors_map[clf_name], va="center", fontweight="bold", fontsize=8)

ax.set_xticks(x_idxs)
ax.set_xticklabels(x_lbls, fontsize=11)
ax.set_xlabel("Niveau de Bruit (SNR)", fontsize=12, fontweight="bold")
ax.set_ylabel("Accuracy (LOO CV)", fontsize=12, fontweight="bold")
ax.set_title("Résilience au Bruit des Classifieurs avec Intervalles de Confiance (MFCC Summary)", fontsize=13, fontweight="bold", pad=15)
ax.set_ylim(0.25, 1.05)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
ax.set_xlim(-0.25, len(x_lbls) - 0.5)
plt.tight_layout()
savefig(fig, "noise_robustness.pdf")
fig_png = plt.figure(fig.number)
fig_png.savefig(FIGURES_DIR / "noise_robustness.png", dpi=150, bbox_inches="tight")
plt.close(fig_png)

# ============================================================
# Store all numerical results for LaTeX
# ============================================================

results = {
    "lr_best_params": str(grid_search.best_params_),
    "lr_loo_score": f"{grid_search.best_score_:.4f}",
    "lr_test_score": f"{grid_search.score(X_test_raw, y_test):.4f}",
    "lr_scaled_loo_score": f"{grid_search_scaled.best_score_:.4f}",
    "lr_scaled_test_score": f"{grid_search_scaled.score(X_test_raw, y_test):.4f}",
    "svm_best_params": str(grid_search_svm.best_params_),
    "svm_loo_score": f"{grid_search_svm.best_score_:.4f}",
    "svm_test_score": f"{grid_search_svm.score(X_test_raw, y_test):.4f}",
    "bag_acc": f"{bag_acc:.4f}",
    "bag_mean_indiv": f"{mean_indiv:.4f}",
    "rf_acc": f"{rf_acc:.4f}",
    "ada_scratch_acc": f"{ada_acc:.4f}",
    "ada_sklearn_acc": f"{ada_sk_acc:.4f}",
    "ada_best_params": str(grid_ada.best_params_),
    "gb_acc": f"{gb_acc:.4f}",
    "gb_best_params": str(grid_gb.best_params_),
    "nn_acc": f"{nn_acc:.4f}",
    "nn_reg_acc": f"{reg_acc:.4f}",
    "cnn_acc": f"{cnn_acc:.4f}",
    "benchmark_total": str(len(rows)),
    "benchmark_ok": str(len(scored_rows)),
}

import json
results_path = FIGURES_DIR / "numerical_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nNumerical results saved to {results_path}")

print("\n✅ All figures generated successfully!")
print(f"   Directory: {FIGURES_DIR}")
import os
for fn in sorted(os.listdir(FIGURES_DIR)):
    print(f"   - {fn}")
