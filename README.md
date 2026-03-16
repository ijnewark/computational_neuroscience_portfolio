# Data

## example_spikes.csv

A small example spike dataset for use with `01_neural_data_analysis/spike_analysis.py --real`.

**Format:** CSV with two columns — `neuron_id` (integer) and `spike_time_s` (float, seconds).

**Contents:** 8 neurons, 10 s simulated recording. Inter-spike intervals follow a Gamma distribution (shape k=2, mean rate range 5–35 Hz), giving mild regularity (CV ≈ 0.7) typical of cortical neurons in vivo.

**Usage:**
```bash
python 01_neural_data_analysis/spike_analysis.py --real --data data/example_spikes.csv
```

## Using real recordings

This CSV format is compatible with any spike-sorted recording. Public sources:

| Source | Format | Notes |
|--------|--------|-------|
| [CRCNS](https://crcns.org) | various | Many freely available datasets, including cortex and hippocampus |
| [Allen Brain Atlas](https://portal.brain-map.org) | NWB | Neuropixels recordings from mouse cortex |
| [OpenNeuro](https://openneuro.org) | BIDS/NWB | Human and animal datasets |

To use NWB files, install `pynwb` and export spike times as CSV with the expected column names.
