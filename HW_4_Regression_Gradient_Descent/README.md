# HW 4 — Regression & Gradient Descent

Linear regression with gradient descent implemented from scratch to predict housing prices from area and number of bedrooms.

## Task
Given a housing dataset (square footage, bedrooms, price), learn weights `w` that minimize mean squared error using gradient descent.

## Approach
1. **Normalization** — standardized features to zero mean and unit variance (implemented from scratch)
2. **Loss function** — mean squared error: `L(w) = (1/2m) Σ (ŷ - y)²`
3. **Batch gradient descent** — full-dataset gradient updates over 80 cycles; logged loss every 10 cycles
4. **Stochastic gradient descent** — per-example updates with random shuffling after each pass
5. **Learning rate comparison** — ran batch GD with α ∈ {0.01, 0.03, 0.05, 0.1, 0.2, 0.5} and plotted loss curves for each
6. **Prediction** — used the best-performing model (α = 0.5) to predict the price of a house with 2,650 sq ft and 4 bedrooms

## Key Observations
- Higher learning rates (0.5) converged faster without diverging on this normalized dataset
- Batch GD and SGD converged to similar solutions; SGD was noisier but faster per cycle
- Loss curves clearly show the effect of learning rate on convergence speed

## Files
| File | Description |
|---|---|
| `HW_4_Gradient_Descent_ebc5802.py` | Full implementation |
| `housing.txt` | Raw housing dataset (area, bedrooms, price) |
| `normalized.txt` | Normalized version generated at runtime |
| `RegressionGradientDescent.pdf` | Assignment specification |
| `Edison Chen HW4 Report.pdf` | Written report with plots and analysis |
