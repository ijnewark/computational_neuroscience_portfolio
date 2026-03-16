"""Leaky integrate-and-fire neuron demo.

Produces three plots:
1. Membrane voltage trace with spike events highlighted
2. Spike raster across a small parameter sweep
3. f-I curve (firing rate vs injected current)
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path(__file__).parent / "figures"


class LIFNeuron:
    def __init__(
        self,
        v_rest: float = -65.0,
        v_reset: float = -65.0,
        v_thresh: float = -50.0,
        r_m: float = 10.0,   # MOhm
        tau_m: float = 10.0,  # ms
    ):
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.r_m = r_m
        self.tau_m = tau_m

    def step(self, v: float, i_inj: float, dt_ms: float) -> tuple[float, bool]:
        dv = (-(v - self.v_rest) + self.r_m * i_inj) / self.tau_m
        v_next = v + dt_ms * dv
        spiked = v_next >= self.v_thresh
        if spiked:
            v_next = self.v_reset
        return v_next, spiked


def simulate(
    neuron: LIFNeuron,
    duration_ms: float = 500.0,
    dt_ms: float = 0.1,
    i_baseline: float = 3.0,
    noise_sd: float = 0.2,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_steps = int(duration_ms / dt_ms)
    v = neuron.v_rest
    v_trace = np.empty(n_steps)
    spikes = []

    for k in range(n_steps):
        i_inj = i_baseline + noise_sd * rng.standard_normal()
        v, spiked = neuron.step(v, i_inj, dt_ms)
        v_trace[k] = v
        if spiked:
            spikes.append(k * dt_ms)

    return np.arange(n_steps) * dt_ms, v_trace, np.array(spikes)


def fi_curve(
    neuron: LIFNeuron,
    i_range: np.ndarray,
    duration_ms: float = 2000.0,
    dt_ms: float = 0.1,
    noise_sd: float = 0.0,
) -> np.ndarray:
    """Compute firing rate (Hz) for each input current in i_range."""
    rates = np.empty(len(i_range))
    for idx, i in enumerate(i_range):
        _, _, spikes = simulate(neuron, duration_ms=duration_ms, dt_ms=dt_ms,
                                i_baseline=i, noise_sd=noise_sd, seed=idx)
        rates[idx] = len(spikes) / (duration_ms / 1000.0)
    return rates


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    lif = LIFNeuron()

    # --- Figure 1: voltage trace ---
    t, v_trace, spikes = simulate(lif, duration_ms=500.0, i_baseline=3.0)

    print(f"Total spikes: {len(spikes)}")
    print(f"Firing rate: {len(spikes) / (t[-1] / 1000):.2f} Hz")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t, v_trace, color="#333333", lw=0.8, label="V_m")
    ax.axhline(lif.v_thresh, color="#cc0000", lw=1, ls="--", label="threshold")
    for sp in spikes:
        ax.axvline(sp, color="#0066cc", lw=0.6, alpha=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane potential (mV)")
    ax.set_title("LIF membrane potential")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "lif_voltage_trace.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: f-I curve ---
    i_vals = np.linspace(0.0, 6.0, 30)
    rates = fi_curve(lif, i_vals, duration_ms=2000.0, noise_sd=0.0)

    # Analytical rheobase for reference: I_rh = (V_thresh - V_rest) / R_m
    i_rheobase = (lif.v_thresh - lif.v_rest) / lif.r_m
    print(f"Rheobase (analytical): {i_rheobase:.2f} nA")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(i_vals, rates, "o-", color="#0066cc", ms=4, lw=1.5)
    ax.axvline(i_rheobase, color="#cc0000", ls="--", lw=1, label=f"rheobase ≈ {i_rheobase:.2f} nA")
    ax.set_xlabel("Input current $I$ (nA)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("LIF f–I curve")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "lif_fi_curve.png", dpi=150)
    plt.close(fig)
    print("Figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
