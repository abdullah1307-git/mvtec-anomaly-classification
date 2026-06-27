# Explainable Multi-Product Industrial Anomaly Classification
### Using CNN Transfer Learning and Cross-Category Evaluation

**Course:** Machine Learning & Smart Systems
**Student:** Abdullah Rashid
**Institution:** University of Europe for Applied Sciences
**Semester:** 3rd Semester, Bachelor of Software Engineering

---

## Project Overview

This project develops an explainable CNN-based framework for automatically classifying industrial product images as **Normal** or **Anomalous (defective)** using the MVTec AD dataset. Three CNN models are trained, compared, and explained using Grad-CAM and SHAP explainability methods.

The project follows a complete machine learning research workflow including data preprocessing, model training, evaluation, explainability analysis, robustness testing, and deployment.

---

## Dataset

| Property | Details |
|---|---|
| Name | MVTec Anomaly Detection (MVTec AD) |
| Source | https://www.kaggle.com/datasets/ipythonx/mvtec-ad |
| Categories | 15 industrial product and texture categories |
| Task | Binary classification: Normal (0) vs Anomalous (1) |
| Total Images | 5000+ |
| Split | 60% Train, 20% Validation, 20% Test (stratified) |

### The 15 Categories
Bottle, Cable, Capsule, Carpet, Grid, Hazelnut, Leather, Metal Nut, Pill, Screw, Tile, Toothbrush, Transistor, Wood, Zipper

---

## Models Compared

| Model | Type | Description |
|---|---|---|
| Custom CNN | Baseline | Built from scratch with 4 convolutional blocks |
| EfficientNetV2S | Transfer Learning | Pretrained on ImageNet, fine-tuned on MVTec AD |
| ConvNeXtTiny | Transfer Learning | Pretrained on ImageNet, fine-tuned on MVTec AD |

Transfer learning models use a two-stage training strategy:
- **Stage 1 (Epochs 1-10):** Backbone frozen, only classifier head trained
- **Stage 2 (Epochs 11+):** Last 2 backbone blocks unfrozen for fine-tuning

---

## Results

| Model | Accuracy | Macro F1 | ROC-AUC | PR-AUC | Inference (ms) | Parameters | Size (MB) |
|---|---|---|---|---|---|---|---|
| Custom CNN | 0.6377 | 0.5499 | 0.5775 | 0.2821 | 0.790 | 422,530 | 1.63 |
| EfficientNetV2S | 0.8693 | 0.8142 | 0.9080 | 0.8159 | 3.154 | 15,221,034 | 79.10 |
| **ConvNeXtTiny** | **0.9066** | **0.8706** | **0.9544** | **0.8916** | 4.401 | 15,670,018 | 106.95 |

**Best Model: ConvNeXtTiny** with 90.66% Accuracy and 0.9544 ROC-AUC

### Key Findings
- ConvNeXtTiny outperformed all models across every metric
- Transfer learning models significantly outperformed the custom CNN baseline
- Best category: Carpet (100% accuracy)
- Most challenging category: Toothbrush (70% accuracy)
- Custom CNN was the fastest at 0.790 ms per image (best efficiency)

---

## Research Questions

| RQ | Question | Answer |
|---|---|---|
| RQ1 | Which architecture achieves strongest performance? | ConvNeXtTiny (90.66% acc, 0.8706 F1) |
| RQ2 | How much does performance vary across categories? | Range of 0.30 (70% to 100%) |
| RQ3 | Do Grad-CAM and SHAP overlap with anomalous regions? | See Figure 13 and Figure 14 |
| RQ4 | Which categories have highest false-negative rates? | Toothbrush (lowest recall) |
| RQ5 | Can a compact model give practical trade-off? | Custom CNN at 0.790 ms (fastest) |

---

## Project Features

- Stratified 60/20/20 train/validation/test split
- Class imbalance handling using weighted CrossEntropyLoss
- Data augmentation (random flips, rotation, color jitter)
- Two-stage transfer learning with early stopping
- Grad-CAM heatmaps showing which regions the model focuses on
- SHAP analysis showing positive and negative prediction evidence
- Robustness testing under 5 types of image degradation
- Per-category performance analysis across all 15 product types
- 17 result figures generated

---

## Repository Structure

    mvtec-anomaly-classification/
    |
    |-- mvtec_anomaly_classification.ipynb   (Main executed Kaggle notebook)
    |
    |-- figures/
    |   |-- figure_01.png   (Total image count per category)
    |   |-- figure_02.png   (Overall class distribution)
    |   |-- figure_03.png   (Sample images per category)
    |   |-- figure_04.png   (Class distribution across splits)
    |   |-- figure_05.png   (Training vs validation accuracy curves)
    |   |-- figure_06.png   (Training vs validation loss curves)
    |   |-- figure_07.png   (Confusion matrices for all 3 models)
    |   |-- figure_08.png   (ROC curves for all 3 models)
    |   |-- figure_09.png   (Precision-Recall curves)
    |   |-- figure_10.png   (Per-class F1 score for best model)
    |   |-- figure_11.png   (Per-category accuracy for best model)
    |   |-- figure_12.png   (Anomalous recall heatmap across models)
    |   |-- figure_13.png   (Grad-CAM heatmap visualizations)
    |   |-- figure_14.png   (SHAP explanation visualizations)
    |   |-- figure_15.png   (Robustness under image degradation)
    |   |-- figure_16.png   (Accuracy vs inference time trade-off)
    |   |-- figure_17.png   (Parameter count comparison)
    |
    |-- results_summary.csv              (Full model comparison table)
    |-- classification_report.txt        (Detailed per-class report)
    |-- README.md                        (This file)

---

## Technologies Used

| Category | Tools |
|---|---|
| Deep Learning | PyTorch, Torchvision |
| Pretrained Models | EfficientNetV2S, ConvNeXtTiny |
| Explainability | Grad-CAM (manual implementation), SHAP |
| Evaluation | Scikit-learn |
| Visualization | Matplotlib, Seaborn, OpenCV |
| Platform | Kaggle (GPU T4 x2) |
| Language | Python 3.10 |

---

## How to Run

1. Go to the Kaggle Notebook link below
2. The MVTec AD dataset is already attached
3. Make sure GPU is enabled in Settings (Accelerator: GPU T4 x2)
4. Click **Run All**
5. All 17 figures and model files will be saved to /kaggle/working/

---

## Links

| Resource | Link |
|---|---|
| Kaggle Notebook | https://www.kaggle.com/code/abdullah130704/notebook8b312b2661 |
| Dataset | https://www.kaggle.com/datasets/ipythonx/mvtec-ad |
| Live Demo | https://mvtec-anomaly-classification.streamlit.app/ |
