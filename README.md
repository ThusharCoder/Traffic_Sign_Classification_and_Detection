# Image Classification with Handcrafted Features & Classical Machine Learning

Traffic sign recognition is an important component of intelligent transportation systems and road safety applications. However, many existing datasets are collected under controlled conditions or from regions with different road environments and traffic sign characteristics. This creates a gap in developing models that perform reliably under local and real-world conditions.

To address this, a dedicated traffic sign dataset was collected from **Mysuru, Karnataka, India**, covering diverse road environments and varying environmental conditions such as daylight, nighttime, and rainfall. The dataset captures the appearance, placement, and environmental variations of traffic signs encountered on roads in Mysuru.

Working with a locally collected dataset enables the study of traffic sign classification performance under realistic regional conditions. It also provides a foundation for developing and evaluating robust traffic sign classification methods specifically suited to real-world Indian road scenarios.

This project focuses on **four handcrafted feature extraction techniques** combined with **four classical machine-learning classifiers**. The experiments are conducted using **5-fold validation** across **six different image resolutions**, allowing the effect of feature representation, classifier selection, and image resolution on traffic sign classification performance to be investigated.

---

## Dataset

### Dataset Collection

The dataset consists of **4,595 traffic sign images** collected from different locations across the city of **Mysuru**.

The images were collected under diverse real-world environmental conditions, including:

- Daylight conditions
- Nighttime conditions
- Rainy conditions
- Different road environments
- Different viewing angles and sign placements
- Variations in background and surrounding objects

Unlike datasets consisting only of isolated traffic signs, an individual image in this dataset **may contain one or more traffic signs/symbols**. This allows the dataset to represent more realistic traffic scenarios encountered on roads.

The original images can have **different image dimensions**, as they were collected from real-world sources rather than being captured at a fixed resolution.

### Annotation

The 4,595 original images were manually annotated using **LabelImg**.

For every annotated image, a corresponding **XML annotation file** was generated containing the bounding-box information and class labels of the traffic signs present in the image.

The original images and their corresponding XML annotation files can be accessed from the following locations:

**Original Images:**  
`[LINK TO ORIGINAL IMAGES]`

**XML Annotations:**  
`[LINK TO XML ANNOTATIONS]`

### Image Cropping

Since an image may contain one or more traffic signs, the annotated bounding boxes were used to extract individual traffic-sign regions from the original images.

A dedicated cropping script was developed to:

1. Read each original image.
2. Locate its corresponding XML annotation file.
3. Extract the bounding-box coordinates of each annotated traffic sign.
4. Crop each traffic sign from the original image.
5. Organize the cropped images according to their respective traffic sign classes.

The cropping code used to generate the individual traffic-sign images is provided at:

**Cropping Code:**  
`[LINK TO CROPPING CODE]`

The cropping process resulted in approximately **11,166 individual traffic-sign images** from the original 4,595 images.

The resulting dataset contains **16 traffic sign classes**, but other classes are added to it and a total 
of **14,562** images were created which spread over 24 different classses and the class wise distribution is 
shown below.

### Dataset Class Distribution

The class-wise distribution of the resulting **14,562 cropped traffic-sign images** is shown below.


| Sl. No. | Class | Total |
|--------:|-------------------------------|------:|
| 1 | BikeParking | 60 |
| 2 | CarParking | 81 |
| 3 | EatingPlace | 24 |
| 4 | GoSlow | 599 |
| 5 | GoodsProhibited | 185 |
| 6 | ImportantRoadAhead | 166 |
| 7 | MajorRoadAhead | 2179 |
| 8 | MedianRoadAhead | 1748 |
| 9 | NarrowBridge | 156 |
| 10 | NoLeftTurn | 255 |
| 11 | NoParking | 245 |
| 12 | NoParkingBothDirections | 180 |
| 13 | NoRightTurn | 60 |
| 14 | NoStraightAhead | 443 |
| 15 | OneWay | 437 |
| 16 | PedestrianCrossing | 2883 |
| 17 | PedestrianCrossingGoSlow | 182 |
| 18 | RightHandCurve | 504 |
| 19 | RoadHumpAhead | 170 |
| 20 | SchoolAhead | 158 |
| 21 | SideRoadLeft | 1939 |
| 22 | SpeedLimitForty | 616 |
| 23 | UncontrolledIntersection | 1131 |
| 24 | Uturn | 161 |
| **Total** | | **14,562** |

The class distribution demonstrates that the dataset is naturally imbalanced, with some traffic sign classes having substantially more samples than others. This class imbalance is retained during the experiments to reflect the distribution of traffic signs encountered during the data collection process.

### Five-Fold Dataset

The resulting **14,562 cropped traffic-sign images** were organized into a **5-fold dataset**.

Each fold contains separate training and testing sets:

```text
Croped_5Fold/
├── fold_1/
│   ├── train/
│   │   ├── BikeParking/
│   │   ├── CarParking/
│   │   ├── ...
│   │   └── Uturn/
│   └── test/
│       ├── BikeParking/
│       ├── CarParking/
│       ├── ...
│       └── Uturn/
│
├── fold_2/
│   ├── train/
│   └── test/
│
├── fold_3/
│   ├── train/
│   └── test/
│
├── fold_4/
│   ├── train/
│   └── test/
│
└── fold_5/
    ├── train/
    └── test/


```
# Machine Learning feature Extraction algorithms and classifiers
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

## Experimental Protocol

- **Dataset**: The notebooks use the `Croped_5Fold` dataset. The dataset path is configured using the `baseDir` variable inside each notebook. Users should update `baseDir` to the location of the dataset on their own system.

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
