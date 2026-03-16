# Module 02 — Neuron Models

Implements a leaky integrate-and-fire (LIF) neuron with Euler integration. Sweeps injected current to produce a frequency–current (F–I) curve, and plots the membrane voltage trace for a single trial.

## Files

| File | Description |
|------|-------------|
| `lif_model.py` | Main LIF simulation script |
| `lif_notebook.ipynb` | Interactive notebook with parameter exploration |
| `figures/lif_voltage_trace.png` | Membrane potential over time |
| `figures/lif_fi_curve.png` | Firing rate vs injected current |

## Usage

```bash
python lif_model.py
```

## Model Description

The LIF model is governed by:

```
tau_m * dV/dt = -(V - V_rest) + R * I_ext
```

When `V` reaches the threshold `V_th`, a spike is recorded and `V` is reset to `V_reset`.

**Default parameters:** `tau_m = 20 ms`, `V_rest = -70 mV`, `V_th = -55 mV`, `R = 10 MOhm`

## Key Concepts

- **Leaky integration** — the membrane acts as a RC circuit; charge leaks away between inputs
- **F–I curve** — the relationship between input current and output spike rate; a core descriptor of neural gain
- **Refractory period** — a brief interval after each spike during which the neuron cannot fire again
