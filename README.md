# Image Classification with Handcrafted Features & Classifiers

Traffic sign recognition is an important component of intelligent transportation systems and road safety applications. However, many existing datasets are collected under controlled conditions or from regions with different road environments and traffic sign characteristics. This creates a gap in developing models that perform reliably under local and real-world conditions. To address this, a dedicated traffic sign dataset was collected from Mysore, covering diverse road environments and varying conditions such as daylight, nighttime, and rainfall. The dataset captures the appearance, placement, and environmental variations of traffic signs encountered on Mysore roads. Working with a locally collected dataset enables the study of classification performance under realistic regional conditions. This also provides a foundation for developing and evaluating robust traffic sign classification methods specifically suited to real-world Indian road scenarios.



This project focuses on **four handcrafted feature extractors** against **four classical machine-learning classifiers** for image classification, evaluated with **5-fold cross-validation** across **six image resolutions**.

---

## Directory Structure

```
Traffic_Sign_Classification/
├── README.md                        ← this file
├── Gabor+classiifers/               ← Gabor filter-bank features
│   ├── Gabor + SVM.ipynb
│   ├── Gabor + Gradient Boosting.ipynb
│   ├── Gabor + Random Forest.ipynb
│   ├── Gabor + Ensemble.ipynb
│   └── README.md
├── HOG+classifiers/                 ← Histogram of Oriented Gradients features
│   ├── HOG + SVM.ipynb
│   ├── HOG + Gradient Boosting.ipynb
│   ├── HOG + Random Forest.ipynb
│   ├── HOG + Ensemble.ipynb
│   └── README.md
├── LTP+classifiers/                 ← Local Ternary Pattern features
│   ├── LTP + SVM.ipynb
│   ├── LTP + Gradient Boosting.ipynb
│   ├── LTP + Random Forest.ipynb
│   ├── LTP + Ensemble.ipynb
│   └── README.md
└── SIFT+classifiers/                ← SIFT + Bag of Visual Words features
    ├── SIFT + SVM.ipynb
    ├── SIFT + Gradient Boosting.ipynb
    ├── SIFT + Random Forest.ipynb
    ├── SIFT + Ensemble.ipynb
    └── README.md
```

Each subfolder holds one notebook per feature-classifier combination (16 experiments in total). Every notebook is **fully self-contained** — it includes its own feature extraction, data loading, training, evaluation, and results export — so any notebook can be run independently.

---

## Feature Descriptors

| Feature | Description | Feature Vector |
|---|---|---|
| **Gabor** | Filter bank: 8 orientations × 5 scales (σ) × 3 wavelengths (λ) × 3 aspect ratios (γ) = 360 filters | Concatenated flattened filter responses |
| **HOG** | Histogram of Oriented Gradients: 9 orientation bins, 8×8-pixel cells, 2×2-cell blocks, L2-Hys normalization | Flattened block-normalized histograms |
| **LTP** | Local Ternary Pattern: ternary threshold ±5, 8-neighborhood, positive + negative code maps | 2 × 256-bin histograms = 512-dim |
| **SIFT** | SIFT keypoint descriptors encoded as a Bag of Visual Words (KMeans vocabulary, 100 visual words) | 100-dim normalized word histogram |

## Classifiers

| Classifier | Configuration |
|---|---|
| **SVM** | `SVC(kernel="linear", probability=True, random_state=42)` |
| **Gradient Boosting** | `GradientBoostingClassifier(random_state=42)` |
| **Random Forest** | `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)` |
| **Ensemble** | `VotingClassifier(voting="soft")` combining the three classifiers above |

---

## Experimental Protocol (common to all notebooks)

- **Dataset**: expected at `E:\THUSHAR\DATASET\Croped_5Fold` (set by `baseDir` inside each notebook) with the layout:

  ```
  Croped_5Fold/
  ├── fold_1/
  │   ├── train/<class_name>/<images>
  │   └── test/<class_name>/<images>
  ├── fold_2/ ...
  └── fold_5/ ...
  ```

- **Validation**: 5-fold cross-validation (`fold_1` … `fold_5`).
- **Image resolutions tested**: 8×8, 16×16, 32×32, 64×64, 128×128, 196×210.
- **Metrics**: accuracy, precision, recall, and F1 — both **weighted** and **macro** averages.
- **Visualization**: a confusion-matrix heatmap per (image size, fold) run.

## Outputs

Each notebook writes two CSV files to its working directory:

| File pattern | Contents |
|---|---|
| `<Feature>_<Classifier>_5Fold_All_Results.csv` | Metrics for every fold × image size |
| `<Feature>_<Classifier>_5Fold_Average_Results.csv` | Metrics averaged over the 5 folds, per image size |

> Note: the HOG, LTP, and SIFT ensemble notebooks use the pattern `<Feature>_Ensemble_SVM_RF_GB_5Fold_...csv`, while the Gabor ensemble uses `Gabor_Ensemble_5Fold_...csv`.

---

## Requirements

```
numpy  pandas  opencv-python  scikit-image  scikit-learn  matplotlib  seaborn  tqdm  joblib
```

Install with:

```
pip install numpy pandas opencv-python scikit-image scikit-learn matplotlib seaborn tqdm joblib
```

## How to Run

1. Ensure the dataset is available at the path configured in `baseDir` (see the *Experiment Configuration* cell of each notebook), or edit `baseDir` to point to your copy.
2. Open the desired notebook and **Run All** cells.
3. Collect the exported CSVs from the notebook's working directory.

> ⚠ Full runs are compute-intensive: each notebook processes 5 folds × 6 image sizes, and feature extraction (especially the 360-filter Gabor bank and per-pixel LTP) can take a long time on large datasets.
