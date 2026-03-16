import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FIGURES_DIR = Path(__file__).parent / "figures"

rng = np.random.default_rng(7)


def simulate_population(n_neurons: int, n_trials_per_class: int, modulation: float = 5.0):
    """Simulate Poisson spike counts for two conditions.

    Each neuron has a baseline rate. Half the neurons are tuned, increasing their
    rate for class 1 and decreasing for class 0 (and vice versa for the other half).
    """
    baseline = rng.uniform(5.0, 15.0, size=n_neurons)  # spikes/s

    # tuning vectors
    sign = np.ones(n_neurons)
    sign[n_neurons // 2 :] = -1.0

    rates_class0 = baseline + sign * modulation
    rates_class1 = baseline - sign * modulation

    X0 = rng.poisson(rates_class0, size=(n_trials_per_class, n_neurons)).astype(float)
    X1 = rng.poisson(rates_class1, size=(n_trials_per_class, n_neurons)).astype(float)

    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(len(X0)), np.ones(len(X1))]).astype(int)

    return X, y


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    n_neurons = 64
    n_trials_per_class = 200

    X, y = simulate_population(n_neurons, n_trials_per_class)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test, pca_train, pca_test = train_test_split(
        X_scaled, y, X_pca, test_size=0.25, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Decoder accuracy: {acc:.3f}")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    ax.set_title("Confusion matrix")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Decision boundary in PCA space
    x_min, x_max = pca_test[:, 0].min() - 1.0, pca_test[:, 0].max() + 1.0
    y_min, y_max = pca_test[:, 1].min() - 1.0, pca_test[:, 1].max() + 1.0

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_std = pca.inverse_transform(grid)
    probs = clf.predict_proba(grid_std)[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(xx, yy, probs, levels=np.linspace(0, 1, 25), alpha=0.7)
    sc = ax.scatter(pca_test[:, 0], pca_test[:, 1], c=y_test,
                    edgecolor="k", linewidth=0.5, cmap="bwr")
    ax.set_title("PCA projection with logistic decision surface")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(cf, ax=ax, label="P(y=1)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "decoder_pca.png", dpi=150)
    plt.close(fig)

    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
