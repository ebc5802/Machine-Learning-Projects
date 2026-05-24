# HW 1 — Perceptron Algorithm

Spam email classifier built from scratch using the perceptron learning algorithm — no ML libraries, pure Python.

## Task
Binary classification: given an email, predict **spam (1)** or **not spam (-1)**.

## Approach
1. **Preprocessing** — split raw email data into train (4,000) and validation (1,000) sets
2. **Feature extraction** — bag-of-words vectors; words appearing fewer than `thresh` times are filtered out
3. **Training** — perceptron update rule: if `y · (w · x) ≤ 0`, update `w ← w + y·x`
4. **Hyperparameter search** — grid search over vocabulary threshold and max epochs to minimize validation error
5. **Analysis** — extracted top-12 highest and lowest weighted words from the learned classifier

## Key Result
~2.5% test error rate on held-out spam data

## Files
| File | Description |
|---|---|
| `HW_1_ebc5802.py` | Full implementation |
| `split_data.py` | Train/validation split utility |
| `simpleTestsPoo.py` | Simple sanity check tests |
| `hw1_data/` | Spam train, spam test, train, validation splits |
| `Perceptron_HW.pdf` | Assignment specification |
| `Edison Chen HW#1 Report.pdf` | Written report with results and analysis |
