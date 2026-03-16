# Module 03 — Neural Information Theory

Estimates the information content of neural spike count responses using two complementary measures: mutual information (MI) and Fisher information (FI). Based on theoretical frameworks in Dayan & Abbott (2001), Ch. 4.

## Files

| File | Description |
|------|-------------|
| `info_theory.py` | Main script |
| `figures/tuning_curves.png` | Gaussian tuning curves for the neural population |
| `figures/mi_scaling.png` | Mutual information as a function of population size |
| `figures/fisher_information.png` | Fisher information and the Cramér–Rao bound |

## Usage

```bash
python info_theory.py
```

## Key Concepts

- **Mutual information** `I(S; R)` — model-free measure of how much the stimulus S can be inferred from population response R; estimated via plugin entropy from spike count histograms
- **Fisher information** `J(s)` — local discriminability; for a Gaussian tuning curve, `J(s) ∝ (f'(s))^2 / σ^2`; sets the Cramér–Rao lower bound on decoding error
- **Efficient coding** — both measures capture the theoretical limit of how much information a neural code can carry

## Reference

Dayan, P. & Abbott, L.F. (2001). *Theoretical Neuroscience*. MIT Press.
