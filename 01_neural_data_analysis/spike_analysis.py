"""Spike train analysis.

Produces three outputs:
1. Spike raster for a simulated Poisson population
2. Per-neuron firing rate bar chart
3. PCA projection of population activity (binned counts)

Modes:
- demo (default): simulate Poisson spike trains
- real: load spike times from CSV (columns: neuron_id, spike_time_s)

Example:
    python spike_analysis.py
    python spike_analysis.py --real --data ../data/example_spikes.csv
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent.parent / "data" / "example_spikes.csv"
FIGURES_DIR = Path(__file__).parent / "figures"


def poisson_spike_train(rate_hz: float, duration_s: float, dt: float = 0.001, seed=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_steps = int(duration_s / dt)
    spikes = rng.random(n_steps) < rate_hz * dt
    return np.where(spikes)[0] * dt


def simulate_population(
    n_neurons: int = 20,
    baseline_rate_hz: float = 10.0,
    duration_s: float = 2.0,
    dt: float = 0.001,
    seed: int = 0,
) -> tuple[list[np.ndarray], float]:
    rng = np.random.default_rng(seed)
    rates = np.clip(rng.normal(baseline_rate_hz, 2.0, size=n_neurons), 0.1, None)
    trains = [
        poisson_spike_train(r, duration_s, dt=dt, seed=rng.integers(0, 2**32))
        for r in rates
    ]
    return trains, duration_s


def load_spikes_csv(path: Path, max_neurons: int | None = 50) -> tuple[list[np.ndarray], float]:
    df = pd.read_csv(path)
    df = df.sort_values(["neuron_id", "spike_time_s"])
    trains = [g["spike_time_s"].to_numpy() for _, g in df.groupby("neuron_id")]
    if max_neurons is not None:
        trains = trains[:max_neurons]
    duration_s = float(df["spike_time_s"].max())
    return trains, duration_s


def bin_spike_trains(trains: list[np.ndarray], duration_s: float, bin_size_s: float = 0.05) -> np.ndarray:
    """Return a (n_neurons, n_bins) array of binned spike counts."""
    edges = np.arange(0, duration_s + bin_size_s, bin_size_s)
    return np.array([np.histogram(t, bins=edges)[0] for t in trains], dtype=float)


def plot_raster(trains: list[np.ndarray], title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, times in enumerate(trains):
        ax.vlines(times, i + 0.5, i + 1.5, lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron")
    ax.set_title(title)
    ax.set_ylim(0.5, len(trains) + 0.5)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_firing_rates(firing_rates: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(np.arange(len(firing_rates)) + 1, firing_rates, color="#0066cc", alpha=0.8, width=0.7)
    ax.axhline(firing_rates.mean(), color="#cc0000", ls="--", lw=1, label=f"mean {firing_rates.mean():.1f} Hz")
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("Per-neuron firing rates")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_pca(counts: np.ndarray, save_path: Path) -> None:
    """PCA on the (n_bins, n_neurons) activity matrix — each bin is an observation."""
    X = counts.T  # shape: (n_bins, n_neurons)
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(6, X_scaled.shape[1]))
    pca.fit(X_scaled)
    proj = pca.transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Scree plot
    var_ratio = pca.explained_variance_ratio_
    axes[0].bar(np.arange(1, len(var_ratio) + 1), var_ratio * 100, color="#0066cc", alpha=0.8)
    axes[0].plot(np.arange(1, len(var_ratio) + 1), np.cumsum(var_ratio) * 100,
                 "o--", color="#cc0000", ms=4, label="cumulative")
    axes[0].set_xlabel("PC")
    axes[0].set_ylabel("Variance explained (%)")
    axes[0].set_title("PCA scree plot")
    axes[0].legend(frameon=False, fontsize=8)

    # PC1 vs PC2 trajectory
    sc = axes[1].scatter(proj[:, 0], proj[:, 1],
                         c=np.arange(len(proj)), cmap="viridis", s=10, alpha=0.7)
    axes[1].set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    axes[1].set_title("Population activity trajectory (PC space)")
    fig.colorbar(sc, ax=axes[1], label="Time bin")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true",
                        help="Load spikes from CSV instead of simulating.")
    parser.add_argument("--data", type=str, default=str(DATA_PATH),
                        help="Path to spikes CSV (columns: neuron_id, spike_time_s).")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.real:
        path = Path(args.data)
        if not path.exists():
            raise SystemExit(f"Missing data file: {path}\n"
                             "Expected CSV with columns: neuron_id, spike_time_s\n"
                             "See data/example_spikes.csv for format reference.")
        trains, duration_s = load_spikes_csv(path)
        label = "real"
    else:
        trains, duration_s = simulate_population()
        label = "simulated Poisson"

    firing_rates = np.array([len(t) / duration_s for t in trains])
    print(f"Neurons: {len(trains)}  |  Duration: {duration_s:.2f} s")
    print(f"Mean firing rate: {np.mean(firing_rates):.2f} Hz  "
          f"(range {firing_rates.min():.1f}–{firing_rates.max():.1f} Hz)")

    plot_raster(trains, title=f"Spike raster ({label})",
                save_path=FIGURES_DIR / "spike_raster.png")
    plot_firing_rates(firing_rates, save_path=FIGURES_DIR / "firing_rates.png")

    counts = bin_spike_trains(trains, duration_s)
    plot_pca(counts, save_path=FIGURES_DIR / "population_pca.png")

    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
