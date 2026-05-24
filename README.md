# Machine Learning Projects

Projects guided by NYUAD Fall '23 Machine Learning class with Professor [Keith Ross](https://engineering.nyu.edu/faculty/keith-ross).

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)

This course was my first real deep dive into the math and mechanics behind machine learning — not just using tools, but understanding what's actually happening under the hood. Many of the assignments were intentionally done with only math libraries (NumPy, pure Python): deriving weight updates by hand, implementing eigendecomposition without sklearn, building gradient descent from scratch before ever touching an optimizer. That foundation made the later PyTorch assignments click in a way they wouldn't have otherwise. This repo marks my first stepping stone into training my own models — and the point where ML stopped feeling like a black box.

---

## Assignments

| # | Topic | Key Concepts | Format |
|---|---|---|---|
| [1](#hw1) | Perceptron Algorithm | Linear classification, bag-of-words, hyperparameter search | Python |
| [2](#hw2) | Support Vector Machines | Margin maximization, kernel methods | Written |
| [3](#hw3) | Principal Component Analysis | Eigendecomposition, dimensionality reduction, eigenfaces | Python |
| [4](#hw4) | Regression & Gradient Descent | Batch GD, stochastic GD, feature normalization | Python |
| [5](#hw5) | Computer Vision with CNNs | Conv/pool layers, CIFAR-10 classification, early stopping | Jupyter (PyTorch) |
| [6](#hw6) | NLP with RNNs | Character-level language modeling, sequence generation | Jupyter (PyTorch) |

---

<a name="hw1"></a>
### HW 1 — Perceptron Algorithm [`HW_1_Perceptron_Algorithm/`](./HW_1_Perceptron_Algorithm/)

Built a **spam email classifier** from scratch using the perceptron learning algorithm — no ML libraries, pure Python.

- Parsed raw email text into bag-of-words feature vectors with vocabulary filtering
- Implemented the perceptron update rule, error calculation, and epoch loop by hand
- Tuned two hyperparameters (vocabulary frequency threshold + max epochs) via grid search on a validation set
- Identified the highest- and lowest-weighted words in the learned classifier
- **Result:** ~2.5% test error rate on the spam dataset

---

<a name="hw2"></a>
### HW 2 — Support Vector Machines [`HW_2_Support_Vector_Machines/`](./HW_2_Support_Vector_Machines/)

Theoretical and mathematical deep-dive into SVMs — written assignment covering:

- Hard-margin and soft-margin SVM formulations
- Margin maximization and the dual optimization problem
- Kernel trick for non-linearly separable data

---

<a name="hw3"></a>
### HW 3 — Principal Component Analysis [`HW_3_Principal_Component_Analysis/`](./HW_3_Principal_Component_Analysis/)

Applied PCA to a dataset of **400 grayscale face images** (64×64 px) to compute eigenfaces and explore dimensionality reduction.

- Computed the mean face and centered the data matrix
- Derived eigenvectors via the covariance matrix trick (computing L = AᵀA to avoid a 4096×4096 matrix)
- Visualized the first 16 principal components (eigenfaces)
- Reconstructed faces using 5, 10, 25, 50, 100, 200, 300, and 399 PCs — demonstrating how few components capture most visual information
- Plotted proportion of variance explained per component

---

<a name="hw4"></a>
### HW 4 — Regression & Gradient Descent [`HW_4_Regression_Gradient_Descent/`](./HW_4_Regression_Gradient_Descent/)

Implemented **linear regression with gradient descent** from scratch to predict housing prices (area + bedrooms → price).

- Normalized features (zero mean, unit variance) from scratch
- Implemented batch gradient descent and stochastic gradient descent
- Compared 6 learning rates (α = 0.01, 0.03, 0.05, 0.1, 0.2, 0.5) — plotted loss curves per α
- Made a real prediction: house with 2,650 sq ft and 4 bedrooms

---

<a name="hw5"></a>
### HW 5 — Computer Vision with CNNs [`HW_5_Computer_Vision_Using_Convolutional_NNs_In_PyTorch/`](./HW_5_Computer_Vision_Using_Convolutional_NNs_In_PyTorch/)

Built and trained a **6-layer CNN on CIFAR-10** (60,000 images, 10 classes) using PyTorch.

- Implemented the convolution operation manually, then built with `nn.Conv2d` + `nn.MaxPool2d`
- Architecture: 3 conv blocks (Conv→ReLU→Conv→ReLU→MaxPool) → 3 FC layers
- Trained with Adam optimizer (lr = 0.001, 20 epochs, batch size 128)
- Implemented early stopping with best-weights checkpointing
- **Result:** ~76.7% test accuracy

---

<a name="hw6"></a>
### HW 6 — NLP with RNNs [`HW_6_NLP_Using_RNNs_In_PyTorch/`](./HW_6_NLP_Using_RNNs_In_PyTorch/)

Built a **character-level RNN language model** in PyTorch to generate new dinosaur names, trained on a dataset of 1,500+ real dinosaur names.

- Tokenized names at the character level with `<EOS>` tokens
- Implemented a recurrent network with hidden state across timesteps
- Trained to predict the next character given previous characters
- Sampled the model to generate novel, plausible-sounding dinosaur names

---

## Final Project — Rock Paper Scissors with Computer Vision CNNs

Applied CNNs to real-time **hand gesture recognition** — classifying Rock, Paper, and Scissors from images using a convolutional neural network.

📄 Full writeup, results, and demo: [Notion](https://edison-chen.notion.site/Rock-Paper-Scissors-w-Computer-Vision-CNNs-434b4dd7571e477fb7c53060f788508b?pvs=74)

---

## Practice HW (Redo)

Re-implementations of HW 1 and HW 2 done for additional practice. See [`Practice_HW_(Redo)/`](./Practice_HW_(Redo)/).
