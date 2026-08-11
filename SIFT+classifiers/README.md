# SIFT Features + Classifiers

Experiments pairing **SIFT descriptors encoded as a Bag of Visual Words (BoVW)** with four classical classifiers, evaluated with 5-fold cross-validation across six image resolutions. See the [main README](../README.md) for the shared protocol, dataset layout, and requirements.

## Feature Pipeline

Unlike the other descriptors in this project, SIFT produces a *variable* number of keypoint descriptors per image, so a vocabulary step turns them into fixed-length vectors:

1. **`extract_sift_descriptors(image, target_size)`** — grayscale + resize, then OpenCV `SIFT_create().detectAndCompute()` returns the image's 128-dimensional keypoint descriptors.
2. **`load_descriptors(directory, target_size)`** — walks the class sub-directories and collects per-image descriptor sets and labels (images with no detectable keypoints are skipped).
3. **Visual vocabulary** — per fold, KMeans (`NUM_VISUAL_WORDS = 100`) is fitted on all stacked *training* descriptors.
4. **`build_bovw_features(descriptor_list, kmeans)`** — each image becomes a 100-bin, L1-normalized histogram of visual-word occurrences.

## Notebooks

| Notebook | Classifier | Output CSVs |
|---|---|---|
| `SIFT + SVM.ipynb` | Linear SVM | `SIFT_SVM_5Fold_{All,Average}_Results.csv` |
| `SIFT + Gradient Boosting.ipynb` | Gradient Boosting | `SIFT_GradientBoosting_5Fold_{All,Average}_Results.csv` |
| `SIFT + Random Forest.ipynb` | Random Forest (100 trees) | `SIFT_RandomForest_5Fold_{All,Average}_Results.csv` |
| `SIFT + Ensemble.ipynb` | Soft-voting ensemble (SVM + GB + RF) | `SIFT_Ensemble_SVM_RF_GB_5Fold_{All,Average}_Results.csv` |

Each notebook is self-contained: open it, adjust `baseDir` if needed, and **Run All**.

> Note: very small image sizes (e.g. 8×8) may yield few or no SIFT keypoints, which can reduce the number of usable images at those resolutions.
