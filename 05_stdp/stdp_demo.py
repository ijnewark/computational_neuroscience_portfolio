"""Spike-timing dependent plasticity (STDP) demo.

Implements the classic asymmetric Hebbian STDP learning rule (Bi & Poo 1998):
  ΔW = A+ * exp(-Δt / τ+)   if Δt > 0  (pre before post → potentiation)
  ΔW = -A- * exp( Δt / τ-)  if Δt < 0  (post before pre → depression)

where Δt = t_post - t_pre.

Produces three figures:
1. STDP learning window: ΔW vs Δt
2. Synaptic weight evolution during a Poisson-driven simulation
3. Weight distribution before and after learning

Reference:
  Bi, G., & Poo, M. (1998). Synaptic modifications in cultured hippocampal neurons.
  J. Neurosci., 18(24), 10464–10472.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field

FIGURES_DIR = Path(__file__).parent / "figures"


# ──────────────────────────────────────────────────────────────────────────────
# STDP rule
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class STDPParams:
    A_plus: float = 0.005      # potentiation amplitude
    A_minus: float = 0.00525   # depression amplitude (slightly > A+ → weight decay)
    tau_plus: float = 20.0     # potentiation time constant (ms)
    tau_minus: float = 20.0    # depression time constant (ms)
    w_min: float = 0.0         # hard lower bound on weight
    w_max: float = 1.0         # hard upper bound on weight


def stdp_window(dt: np.ndarray, params: STDPParams) -> np.ndarray:
    """STDP weight change as a function of spike timing difference Δt = t_post - t_pre."""
    dw = np.where(
        dt > 0,
        params.A_plus  * np.exp(-dt / params.tau_plus),
        -params.A_minus * np.exp( dt / params.tau_minus),
    )
    # dt == 0: undefined — set to 0
    dw = np.where(dt == 0, 0.0, dw)
    return dw


# ──────────────────────────────────────────────────────────────────────────────
# Synapse model with online trace-based STDP
# ──────────────────────────────────────────────────────────────────────────────

class STDPSynapse:
    """Single synapse with online STDP using eligibility traces.

    Traces allow efficient O(n_steps) simulation:
      x_pre  += 1 at each pre-synaptic spike, decays with τ+
      x_post += 1 at each post-synaptic spike, decays with τ-
    Weight updates:
      on pre spike:  ΔW = -A- * x_post  (depression)
      on post spike: ΔW = +A+ * x_pre   (potentiation)
    """

    def __init__(self, w_init: float, params: STDPParams):
        self.w = w_init
        self.params = params
        self._x_pre = 0.0
        self._x_post = 0.0

    def step(self, dt_ms: float, pre_spike: bool, post_spike: bool) -> float:
        p = self.params
        # Decay traces
        self._x_pre  *= np.exp(-dt_ms / p.tau_plus)
        self._x_post *= np.exp(-dt_ms / p.tau_minus)

        # Apply STDP updates and update traces
        if pre_spike:
            self.w -= p.A_minus * self._x_post
            self._x_pre += 1.0

        if post_spike:
            self.w += p.A_plus * self._x_pre
            self._x_post += 1.0

        self.w = np.clip(self.w, p.w_min, p.w_max)
        return self.w


# ──────────────────────────────────────────────────────────────────────────────
# Simulation
# ──────────────────────────────────────────────────────────────────────────────

def poisson_spike_train(rate_hz: float, duration_ms: float, dt_ms: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Boolean spike array of shape (n_steps,)."""
    n = int(duration_ms / dt_ms)
    return rng.random(n) < rate_hz * dt_ms / 1000.0


def simulate_weight_evolution(
    pre_rate_hz: float = 20.0,
    post_rate_hz: float = 20.0,
    duration_ms: float = 60_000.0,
    dt_ms: float = 1.0,
    w_init: float = 0.5,
    params: STDPParams | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a single synapse with independent Poisson pre/post firing."""
    if params is None:
        params = STDPParams()
    rng = np.random.default_rng(seed)
    synapse = STDPSynapse(w_init=w_init, params=params)

    pre_spikes  = poisson_spike_train(pre_rate_hz,  duration_ms, dt_ms, rng)
    post_spikes = poisson_spike_train(post_rate_hz, duration_ms, dt_ms, rng)

    n_steps = int(duration_ms / dt_ms)
    t = np.arange(n_steps) * dt_ms
    w_trace = np.empty(n_steps)

    for k in range(n_steps):
        w_trace[k] = synapse.step(dt_ms, bool(pre_spikes[k]), bool(post_spikes[k]))

    return t, w_trace


def simulate_population_weights(
    n_synapses: int = 200,
    pre_rate_hz: float = 20.0,
    post_rate_hz: float = 20.0,
    duration_ms: float = 60_000.0,
    dt_ms: float = 1.0,
    params: STDPParams | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate many independent synapses; return initial and final weight distributions."""
    if params is None:
        params = STDPParams()
    rng = np.random.default_rng(seed)
    w_init = rng.uniform(0.3, 0.7, size=n_synapses)
    w_final = np.empty(n_synapses)

    for i in range(n_synapses):
        _, w_trace = simulate_weight_evolution(
            pre_rate_hz=pre_rate_hz,
            post_rate_hz=post_rate_hz,
            duration_ms=duration_ms,
            dt_ms=dt_ms,
            w_init=w_init[i],
            params=params,
            seed=int(rng.integers(0, 2**31)),
        )
        w_final[i] = w_trace[-1]

    return w_init, w_final


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_stdp_window(params: STDPParams, save_path: Path) -> None:
    dt_vals = np.linspace(-100, 100, 400)
    dw = stdp_window(dt_vals, params)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dt_vals[dt_vals > 0], dw[dt_vals > 0], color="#0066cc", lw=2, label="LTP (potentiation)")
    ax.plot(dt_vals[dt_vals < 0], dw[dt_vals < 0], color="#cc0000", lw=2, label="LTD (depression)")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlabel(r"$\Delta t = t_{\rm post} - t_{\rm pre}$ (ms)")
    ax.set_ylabel(r"$\Delta W$")
    ax.set_title("STDP learning window (Bi & Poo 1998)")
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_weight_evolution(t_ms: np.ndarray, w_trace: np.ndarray, save_path: Path) -> None:
    t_s = t_ms / 1000.0
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t_s, w_trace, lw=0.6, color="#333333", alpha=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Synaptic weight $W$")
    ax.set_title("Synaptic weight evolution under independent Poisson activity")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_weight_distributions(w_init: np.ndarray, w_final: np.ndarray, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    bins = np.linspace(0, 1, 25)

    axes[0].hist(w_init,  bins=bins, color="#888888", edgecolor="white", lw=0.5)
    axes[0].set_title("Initial weight distribution")
    axes[0].set_xlabel("Weight")
    axes[0].set_ylabel("Count")

    axes[1].hist(w_final, bins=bins, color="#0066cc", edgecolor="white", lw=0.5)
    axes[1].set_title("Final weight distribution (after STDP)")
    axes[1].set_xlabel("Weight")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    params = STDPParams()

    # 1. STDP window
    plot_stdp_window(params, FIGURES_DIR / "stdp_window.png")
    print("STDP window saved.")

    # 2. Single synapse weight trace
    t, w_trace = simulate_weight_evolution(
        pre_rate_hz=20.0, post_rate_hz=20.0,
        duration_ms=60_000.0, params=params, seed=1,
    )
    plot_weight_evolution(t, w_trace, FIGURES_DIR / "weight_evolution.png")
    print(f"Weight evolution: {w_trace[0]:.3f} → {w_trace[-1]:.3f}")

    # 3. Population weight distributions
    print("Simulating population (this takes ~30 s)...")
    w_init, w_final = simulate_population_weights(
        n_synapses=200, pre_rate_hz=20.0, post_rate_hz=20.0,
        duration_ms=60_000.0, params=params, seed=0,
    )
    plot_weight_distributions(w_init, w_final, FIGURES_DIR / "weight_distributions.png")
    print(f"Mean initial weight: {w_init.mean():.3f}")
    print(f"Mean final weight:   {w_final.mean():.3f}")
    print(f"Bimodality index: {((w_final < 0.3).sum() + (w_final > 0.7).sum()) / len(w_final):.2f}")
    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
