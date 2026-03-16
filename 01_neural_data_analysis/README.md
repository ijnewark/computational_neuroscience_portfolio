# Module 01 — Neural Data Analysis

Loads spike-time data (CSV format), computes per-neuron firing rates, produces raster diagrams, and applies PCA to the neural population activity matrix.

## Files

| File | Description |
|------|-------------|
| `spike_analysis.py` | Main script |
| `neural_analysis.ipynb` | Step-by-step notebook walkthrough |
| `figures/spike_raster.png` | Spike raster across neurons and time |
| `figures/firing_rates.png` | Mean firing rates per neuron |
| `figures/population_pca.png` | PCA of population activity (trials x neurons) |

## Usage

```bash
# Simulated data
python spike_analysis.py

# Real spike-sorted data
python spike_analysis.py --real --data ../../data/01_neural_data_analysis/example_spikes.csv
```

## Key Concepts

- **Raster plot** — each row is a neuron; tick marks show spike times
- **Firing rate** — spikes per second, computed over the full recording window
- **Population PCA** — reduces the high-dimensional neural state to 2D for visualisation

## Data Format

Expects a CSV with columns `neuron_id` (int) and `spike_time_s` (float).
Sample files are in `../../data/01_neural_data_analysis/`.
