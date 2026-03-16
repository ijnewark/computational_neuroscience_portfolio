# Module 04 — Machine Learning Decoder

Demonstrates neural population decoding using classical machine learning. Simulates Poisson spike counts for two stimulus conditions, reduces dimensionality with PCA, trains a logistic regression classifier, and evaluates performance.

## Files

| File | Description |
|------|-------------|
| `neural_decoder.py` | Main decoding script |
| `decoder_notebook.ipynb` | Notebook walkthrough |
| `figures/decoder_pca.png` | 2D PCA scatter coloured by class |
| `figures/confusion_matrix.png` | Confusion matrix showing classification accuracy |

## Usage

```bash
python neural_decoder.py
```

## Pipeline

1. **Simulate** — Poisson spike counts for `n_neurons` neurons under two conditions
2. **Split** — 80/20 train–test split with `sklearn`
3. **Scale** — standardise features with `StandardScaler`
4. **Reduce** — PCA to 2 components for visualisation
5. **Decode** — `LogisticRegression` classifier
6. **Evaluate** — accuracy score and confusion matrix

## Key Concepts

- **Population vector** — the joint activity of many neurons carries more information than any single cell
- **PCA** — finds axes of maximum variance; useful for visualising class separability
- **Logistic regression** — linear decoder; a classic choice for neural decoding benchmarks
