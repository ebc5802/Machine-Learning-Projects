# HW 6 — NLP with RNNs in PyTorch

Built a character-level RNN language model in PyTorch to generate new dinosaur names, trained on a dataset of 1,500+ real dinosaur names.

## Task
Train a recurrent neural network to learn the statistical patterns of dinosaur name characters, then sample from it to generate novel, plausible-sounding names.

## Approach
1. **Dataset** — `dinos.txt`: 1,500+ real dinosaur names, lowercased, 27 unique characters (a–z + newline as `<EOS>`)
2. **Tokenization** — each name split into individual characters; `<EOS>` token appended to mark sequence end
3. **Model** — character-level RNN: at each timestep, the hidden state is updated using the current character embedding and previous hidden state; output layer predicts a distribution over the next character
4. **Training** — cross-entropy loss on next-character predictions; gradients clipped to prevent exploding gradients
5. **Sampling** — at inference time, sample characters one at a time from the output distribution until `<EOS>` is generated

## Key Concept
Unlike feedforward networks, RNNs maintain a **hidden state** that carries information across timesteps — essential for modeling sequential data where context from earlier characters influences later ones.

## Files
| File | Description |
|---|---|
| `Edison_Chen_12_3_2023_Assignment_6.ipynb` | Full notebook with model, training loop, and name generation |
| `dinos.txt` | Training dataset of dinosaur names |
