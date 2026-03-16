# Module 05 — Spike-Timing-Dependent Plasticity (STDP)

Demonstrates the classic asymmetric Hebbian STDP learning rule (Bi & Poo 1998). Simulates a pair of pre- and post-synaptic neurons driven by Poisson spike trains and tracks how synaptic weights evolve under the STDP rule.

## Files

| File | Description |
|------|-------------|
| `stdp_demo.py` | Main STDP simulation script |
| `figures/stdp_window.png` | The STDP learning window (ΔW vs Δt) |
| `figures/weight_evolution.png` | Synaptic weight trajectory over the simulation |
| `figures/weight_distributions.png` | Weight distribution before and after learning |

## Usage

```bash
python stdp_demo.py
```

## STDP Rule

The weight change ΔW depends on the relative timing of pre- and post-synaptic spikes (Δt = t_post − t_pre):

```
ΔW = +A_plus  * exp(-Δt / tau_plus)   if Δt > 0  (pre before post → potentiation)
ΔW = -A_minus * exp(+Δt / tau_minus)  if Δt < 0  (post before pre → depression)
```

**Default parameters:** `A_plus = 0.005`, `A_minus = 0.00525`, `tau_plus = tau_minus = 20 ms`

## Key Concepts

- **Hebbian plasticity** — synapses that fire together wire together; STDP is the temporally precise version
- **Causal potentiation** — pre → post ordering strengthens the synapse (pre–post)
- **Anti-causal depression** — post → pre ordering weakens the synapse
- **Asymmetry** — slight dominance of depression (A− > A+) leads to competition and sparse coding

## Reference

Bi, G. & Poo, M. (1998). Synaptic modifications in cultured hippocampal neurons. *J. Neurosci.*, 18(24), 10464–10472.
