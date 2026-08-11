# LTP Features + Classifiers

Experiments pairing **Local Ternary Pattern (LTP) features** with four classical classifiers, evaluated with 5-fold cross-validation across six image resolutions. See the [main README](../README.md) for the shared protocol, dataset layout, and requirements.

## Feature Descriptor

Two functions implement the descriptor:

- **`compute_ltp(image, threshold=5)`** — for every interior pixel, each of its 8 neighbors is compared against the center with a ternary rule: differences above `+threshold` set a bit in a *positive* code, differences below `-threshold` set a bit in a *negative* code. The two code maps are summarized as 256-bin normalized histograms and concatenated into a **512-dimensional feature vector**.
- **`extract_ltp_features(image, target_size, threshold=5)`** — converts the image to 8-bit grayscale, resizes it, and computes the LTP histogram.

The ternary threshold (±5) makes LTP more robust to noise than plain LBP.

## Notebooks

| Notebook | Classifier | Output CSVs |
|---|---|---|
| `LTP + SVM.ipynb` | Linear SVM | `LTP_SVM_5Fold_{All,Average}_Results.csv` |
| `LTP + Gradient Boosting.ipynb` | Gradient Boosting | `LTP_GradientBoosting_5Fold_{All,Average}_Results.csv` |
| `LTP + Random Forest.ipynb` | Random Forest (100 trees) | `LTP_RandomForest_5Fold_{All,Average}_Results.csv` |
| `LTP + Ensemble.ipynb` | Soft-voting ensemble (SVM + GB + RF) | `LTP_Ensemble_SVM_RF_GB_5Fold_{All,Average}_Results.csv` |

Each notebook is self-contained: open it, adjust `baseDir` if needed, and **Run All**.

> ⚠ The LTP computation is a pure-Python per-pixel loop, so extraction time grows quickly with image resolution.
