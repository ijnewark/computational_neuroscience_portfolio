"""Neural information theory: mutual information and Fisher information.

Asks: how much does a neural population tell us about the stimulus?

Two complementary measures:
1. Mutual information I(S; R) — model-free, estimated from spike count histograms.
2. Fisher information J(s) — local discriminability; for a Gaussian tuning curve
   J(s) ∝ (f'(s))² / σ² and sets the Cramér–Rao bound on stimulus estimation error.

Both are central to van Rossum-style theoretical analyses of neural coding efficiency.

Reference concepts:
  Dayan & Abbott (2001) *Theoretical Neuroscience*, Ch. 4.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import xlogy

FIGURES_DIR = Path(__file__).parent / "figures"

# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def entropy_from_counts(counts: np.ndarray) -> float:
    """Shannon entropy (bits) from a 1-D array of non-negative counts."""
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def mutual_information(joint: np.ndarray) -> float:
    """Mutual information (bits) from a joint probability matrix P(s, r).

    joint[i, j] = P(S=s_i, R=r_j)
    """
    joint = joint / joint.sum()
    p_s = joint.sum(axis=1, keepdims=True)   # marginal over responses
    p_r = joint.sum(axis=0, keepdims=True)   # marginal over stimuli
    outer = p_s * p_r
    # I(S;R) = sum p(s,r) log2(p(s,r) / p(s)p(r))
    log_ratio = np.where(joint > 0, np.log2(np.where(joint > 0, joint / (outer + 1e-300), 1.0)), 0.0)
    return float(np.sum(joint * log_ratio))


# ──────────────────────────────────────────────────────────────────────────────
# Tuning curves
# ──────────────────────────────────────────────────────────────────────────────

def gaussian_tuning(s: np.ndarray, preferred: float, width: float, r_max: float) -> np.ndarray:
    """Gaussian tuning curve: f(s) = r_max * exp(-(s - preferred)^2 / (2 * width^2))."""
    return r_max * np.exp(-0.5 * ((s - preferred) / width) ** 2)


def simulate_responses(
    stimuli: np.ndarray,
    preferred: np.ndarray,
    width: float = 20.0,
    r_max: float = 40.0,
    n_trials: int = 200,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample Poisson spike counts for each (stimulus, neuron) pair.

    Returns shape (n_stimuli, n_neurons, n_trials).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n_stim = len(stimuli)
    n_neurons = len(preferred)
    # mean rates: shape (n_stim, n_neurons)
    rates = np.array([[gaussian_tuning(s, p, width, r_max) for p in preferred]
                      for s in stimuli])
    counts = rng.poisson(rates[:, :, np.newaxis] * np.ones((1, 1, n_trials)))
    return counts.astype(float)


# ──────────────────────────────────────────────────────────────────────────────
# Fisher information for a Gaussian tuning curve population
# ──────────────────────────────────────────────────────────────────────────────

def population_fisher(
    s: np.ndarray,
    preferred: np.ndarray,
    width: float,
    r_max: float,
) -> np.ndarray:
    """Summed Fisher information over a population at each stimulus value s.

    For Poisson noise: J_i(s) = [f_i'(s)]^2 / f_i(s)  (assuming f_i > 0)
    """
    J = np.zeros(len(s))
    for p in preferred:
        f = gaussian_tuning(s, p, width, r_max)
        # derivative w.r.t. s
        df = f * (-(s - p) / width ** 2)
        safe = f > 1e-6
        J[safe] += (df[safe] ** 2) / f[safe]
    return J


# ──────────────────────────────────────────────────────────────────────────────
# MI vs population size
# ──────────────────────────────────────────────────────────────────────────────

def mi_vs_population_size(
    stimuli: np.ndarray,
    max_neurons: int = 40,
    width: float = 20.0,
    r_max: float = 40.0,
    n_trials: int = 300,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute I(S;R) as population size grows (neurons added in order of preferred)."""
    if rng is None:
        rng = np.random.default_rng(0)

    # Space preferred stimuli evenly
    preferred_all = np.linspace(stimuli.min(), stimuli.max(), max_neurons)
    sizes = np.arange(1, max_neurons + 1)
    mi_vals = np.zeros(len(sizes))

    for k, n in enumerate(sizes):
        prefs = preferred_all[:n]
        counts = simulate_responses(stimuli, prefs, width=width, r_max=r_max,
                                    n_trials=n_trials, rng=rng)
        # Summarise population response as total spike count (simple scalar)
        total = counts.sum(axis=1).mean(axis=-1)  # (n_stim,) mean total count
        # Discretise into bins and build joint histogram
        bins = np.linspace(0, total.max() + 1, 30)
        joint = np.zeros((len(stimuli), len(bins) - 1))
        for si in range(len(stimuli)):
            c = counts[si].sum(axis=0)  # (n_trials,)
            joint[si], _ = np.histogram(c, bins=bins)
        joint = joint / joint.sum()
        mi_vals[k] = mutual_information(joint)

    return sizes, mi_vals


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_tuning_curves(stimuli: np.ndarray, preferred: np.ndarray,
                       width: float, r_max: float, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(preferred)))
    for p, c in zip(preferred, cmap):
        ax.plot(stimuli, gaussian_tuning(stimuli, p, width, r_max), color=c, lw=1.5)
    ax.set_xlabel("Stimulus (deg)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("Gaussian tuning curves (orientation population)")
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(preferred.min(), preferred.max()))
    fig.colorbar(sm, ax=ax, label="Preferred orientation (deg)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_fisher(stimuli: np.ndarray, J: np.ndarray, crb: np.ndarray, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(stimuli, J, color="#0066cc", lw=2)
    axes[0].set_xlabel("Stimulus (deg)")
    axes[0].set_ylabel("Fisher information (Hz/deg²)")
    axes[0].set_title("Population Fisher information $J(s)$")

    axes[1].plot(stimuli, crb, color="#cc0000", lw=2)
    axes[1].set_xlabel("Stimulus (deg)")
    axes[1].set_ylabel("Cramér–Rao bound (deg²)")
    axes[1].set_title("Estimation variance lower bound $1/J(s)$")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_mi_scaling(sizes: np.ndarray, mi_vals: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, mi_vals, "o-", color="#0066cc", ms=4, lw=1.5)
    ax.set_xlabel("Population size (neurons)")
    ax.set_ylabel("Mutual information $I(S; R)$ (bits)")
    ax.set_title("Information scaling with population size")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # Orientation stimuli: 0–180 degrees
    stimuli = np.linspace(0, 180, 13)
    preferred = np.linspace(0, 180, 8)
    width = 25.0   # tuning width (deg)
    r_max = 40.0   # peak rate (Hz)

    # Tuning curves
    plot_tuning_curves(stimuli, preferred, width, r_max,
                       save_path=FIGURES_DIR / "tuning_curves.png")

    # Fisher information
    s_fine = np.linspace(0, 180, 500)
    J = population_fisher(s_fine, preferred, width, r_max)
    crb = np.where(J > 1e-6, 1.0 / J, np.nan)
    plot_fisher(s_fine, J, crb, save_path=FIGURES_DIR / "fisher_information.png")

    print(f"Mean Fisher information: {np.nanmean(J):.3f} Hz/deg²")
    print(f"Mean CRB: {np.nanmean(crb):.4f} deg²  "
          f"(≈ {np.sqrt(np.nanmean(crb)):.3f} deg RMS error)")

    # Mutual information vs population size
    sizes, mi_vals = mi_vs_population_size(stimuli, max_neurons=30, width=width,
                                           r_max=r_max, n_trials=300, rng=rng)
    plot_mi_scaling(sizes, mi_vals, save_path=FIGURES_DIR / "mi_scaling.png")
    print(f"I(S;R) with 1 neuron: {mi_vals[0]:.3f} bits")
    print(f"I(S;R) with 30 neurons: {mi_vals[-1]:.3f} bits")
    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
