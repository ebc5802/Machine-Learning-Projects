# HW 3 — Principal Component Analysis

Applied PCA from scratch to a dataset of 400 grayscale face images (64×64 px) to compute eigenfaces and explore dimensionality reduction.

## Task
Decompose a face image dataset into its principal components and reconstruct faces using progressively fewer dimensions.

## Approach
1. **Mean face** — computed the average across all 400 images and visualized it
2. **Centering** — subtracted the mean face from each image to get the centered matrix A
3. **Covariance trick** — instead of computing the 4096×4096 covariance matrix, computed L = AAᵀ (400×400) and recovered the eigenvectors of the full covariance via the relation `U = Aᵀv / ||Aᵀv||`
4. **Eigenfaces** — visualized the first 16 principal components as face images
5. **Reconstruction** — reconstructed a random face using 5, 10, 25, 50, 100, 200, 300, and 399 PCs to show quality vs. compression tradeoff
6. **Variance plot** — plotted the proportion of variance explained by each component

## Key Insight
~50 principal components are sufficient to produce a recognizable reconstruction of any face in the dataset — the first few PCs capture global lighting and facial structure, while later ones add fine detail.

## Files
| File | Description |
|---|---|
| `Edison_Chen_PCA_HW.py` | Full implementation (NumPy + matplotlib) |
| `faces.csv` | 400 face images as flattened pixel rows |
| `PCA_assignment.pdf` | Assignment specification |
| `Edison Chen HW#3 Report.pdf` | Written report with visualizations and analysis |
| `Assignment_3.zip` | Original assignment archive |
