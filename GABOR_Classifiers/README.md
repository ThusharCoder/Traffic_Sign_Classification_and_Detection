# Gabor Features + Classifiers

Experiments pairing **Gabor filter-bank features** with four classical classifiers, evaluated with 5-fold cross-validation across six image resolutions. See the [main README](../README.md) for the shared protocol, dataset layout, and requirements.

## Feature Descriptor

`extract_gabor_features(image, target_size)` converts the image to grayscale, resizes it, and convolves it with a bank of Gabor kernels (7×7, ψ = 0):

| Parameter | Values |
|---|---|
| Orientation (θ) | 0°, 30°, 45°, 60°, 90°, 120°, 135°, 150° |
| Scale (σ) | 1, 2, 3, 5, 7 |
| Wavelength (λ) | 3, 5, 7 |
| Aspect ratio (γ) | 0.5, 0.75, 1.0 |

All **360** filter responses are flattened and concatenated into a single feature vector.

## Notebooks

| Notebook | Classifier | Output CSVs |
|---|---|---|
| `Gabor + SVM.ipynb` | Linear SVM | `Gabor_SVM_5Fold_{All,Average}_Results.csv` |
| `Gabor + Gradient Boosting.ipynb` | Gradient Boosting | `Gabor_GradientBoosting_5Fold_{All,Average}_Results.csv` |
| `Gabor + Random Forest.ipynb` | Random Forest (100 trees) | `Gabor_RandomForest_5Fold_{All,Average}_Results.csv` |
| `Gabor + Ensemble.ipynb` | Soft-voting ensemble (SVM + GB + RF) | `Gabor_Ensemble_5Fold_{All,Average}_Results.csv` |

Each notebook is self-contained: open it, adjust `baseDir` if needed, and **Run All**.

> ⚠ Gabor is the most expensive descriptor in this project (360 convolutions per image); expect long runtimes at higher resolutions.
