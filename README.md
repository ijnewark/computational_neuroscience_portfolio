# Computational Neuroscience Portfolio

A collection of self-contained Python modules covering core computational neuroscience topics, built as part of a BSc Neuroscience programme. Each module includes a standalone script, a Jupyter notebook walkthrough, and output figures.

## Repository Structure

```
computational_neuroscience_portfolio/
├── 01_neural_data_analysis/   # Spike sorting, rasters, firing rates, PCA
│   ├── spike_analysis.py
│   ├── neural_analysis.ipynb
│   └── figures/
├── 02_neuron_models/          # Leaky integrate-and-fire (LIF) neuron
│   ├── lif_model.py
│   ├── lif_notebook.ipynb
│   └── figures/
├── 03_neural_information/     # Mutual information & Fisher information
│   ├── info_theory.py
│   └── figures/
├── 04_machine_learning/       # Neural population decoder (logistic regression + PCA)
│   ├── neural_decoder.py
│   ├── decoder_notebook.ipynb
│   └── figures/
├── 05_stdp/                   # Spike-timing-dependent plasticity
│   ├── stdp_demo.py
│   └── figures/
├── data/
│   └── 01_neural_data_analysis/
│       ├── spikes.csv         # Minimal example spike data
│       └── example_spikes.csv # Larger simulated recording
├── requirements.txt
└── README.md
```

## Modules

### 01 — Neural Data Analysis
Loads spike-time data, computes per-neuron firing rates, plots raster diagrams, and runs PCA on population activity vectors.

**Run:** `python 01_neural_data_analysis/spike_analysis.py`

**Key outputs:** `spike_raster.png`, `firing_rates.png`, `population_pca.png`

---

### 02 — Neuron Models
Implements a leaky integrate-and-fire (LIF) neuron with Euler integration. Sweeps input current to produce an F–I curve.

**Run:** `python 02_neuron_models/lif_model.py`

**Key outputs:** `lif_voltage_trace.png`, `lif_fi_curve.png`

---

### 03 — Neural Information Theory
Estimates mutual information between stimulus and spike counts via plugin entropy, and computes Fisher information for a Gaussian tuning curve population.

**Run:** `python 03_neural_information/info_theory.py`

**Key outputs:** `tuning_curves.png`, `mi_scaling.png`, `fisher_information.png`

---

### 04 — Machine Learning Decoder
Simulates a two-class neural population, applies PCA for dimensionality reduction, and trains a logistic regression decoder. Reports accuracy and plots a confusion matrix.

**Run:** `python 04_machine_learning/neural_decoder.py`

**Key outputs:** `decoder_pca.png`, `confusion_matrix.png`

---

### 05 — Spike-Timing-Dependent Plasticity (STDP)
Demonstrates the classic asymmetric Hebbian STDP rule (Bi & Poo 1998). Runs a Poisson-driven simulation and tracks synaptic weight evolution.

**Run:** `python 05_stdp/stdp_demo.py`

**Key outputs:** `stdp_window.png`, `weight_evolution.png`, `weight_distributions.png`

---

## Data

| File | Description |
|------|-------------|
| `data/01_neural_data_analysis/spikes.csv` | Minimal toy dataset (3 neurons, 9 spikes) |
| `data/01_neural_data_analysis/example_spikes.csv` | Larger simulated recording (8 neurons, 10 s, Gamma ISIs) |

Both files use the format `neuron_id, spike_time_s`. Compatible with any NWB/spike-sorted recording exported to CSV.

## Installation

```bash
git clone https://github.com/ijnewark/computational_neuroscience_portfolio.git
cd computational_neuroscience_portfolio
pip install -r requirements.txt
```

## Requirements

See `requirements.txt`. Core dependencies: `numpy`, `matplotlib`, `scipy`, `scikit-learn`.

## License

MIT — see `LICENSE`.
