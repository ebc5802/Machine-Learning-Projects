# HW 5 — Computer Vision with CNNs in PyTorch

Built and trained a convolutional neural network on CIFAR-10 to classify images across 10 object categories.

## Dataset
**CIFAR-10** — 60,000 color images (32×32 px), 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Split into 45,000 train / 5,000 validation / 10,000 test.

## Model Architecture
```
Input: 3 × 32 × 32
→ Conv2d(3→32) + ReLU + Conv2d(32→64) + ReLU + MaxPool2d → 64 × 16 × 16
→ Conv2d(64→128) + ReLU + Conv2d(128→128) + ReLU + MaxPool2d → 128 × 8 × 8
→ Conv2d(128→256) + ReLU + Conv2d(256→256) + ReLU + MaxPool2d → 256 × 4 × 4
→ Flatten → Linear(4096→1024) + ReLU → Linear(1024→512) + ReLU → Linear(512→10)
```

## Training
- **Optimizer:** Adam (lr = 0.001)
- **Loss:** Cross-entropy
- **Epochs:** 20, batch size 128
- **Hardware:** Google Colab (T4 GPU)

## Key Implementations
- Manually implemented the 2D convolution operation (`apply_filter`) before using `nn.Conv2d`
- Built GPU-aware training with `DeviceDataLoader`
- Implemented **early stopping** with patience parameter and best-weights checkpointing (`best_model.pth`)

## Result
**~76.7% test accuracy** (up from 10% random baseline)

## Files
| File | Description |
|---|---|
| `Edison Chen Assignment 5.ipynb` | Full notebook with training, evaluation, and plots |
| `Edison Chen HW5 Report.pdf` | Written report |
