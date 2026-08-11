# HOG Features + Classifiers

Experiments pairing **Histogram of Oriented Gradients (HOG) features** with four classical classifiers, evaluated with 5-fold cross-validation across six image resolutions. See the [main README](../README.md) for the shared protocol, dataset layout, and requirements.

## Feature Descriptor

`extract_hog_features(image, target_size)` converts the image to grayscale, resizes it, and computes the HOG descriptor (`skimage.feature.hog`):

| Parameter | Value |
|---|---|
| Orientation bins | 9 |
| Pixels per cell | 8 × 8 |
| Cells per block | 2 × 2 |
| Block normalization | L2-Hys |

The block-normalized gradient histograms are returned as a single flattened feature vector (dimensionality depends on image size).

## Notebooks

| Notebook | Classifier | Output CSVs |
|---|---|---|
| `HOG + SVM.ipynb` | Linear SVM | `HOG_SVM_5Fold_{All,Average}_Results.csv` |
| `HOG + Gradient Boosting.ipynb` | Gradient Boosting | `HOG_GradientBoosting_5Fold_{All,Average}_Results.csv` |
| `HOG + Random Forest.ipynb` | Random Forest (100 trees) | `HOG_RandomForest_5Fold_{All,Average}_Results.csv` |
| `HOG + Ensemble.ipynb` | Soft-voting ensemble (SVM + GB + RF) | `HOG_Ensemble_SVM_RF_GB_5Fold_{All,Average}_Results.csv` |

Each notebook is self-contained: open it, adjust `baseDir` if needed, and **Run All**.
