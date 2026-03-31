# Comprehensive Graph Documentation

## Breast Cancer Detection Using Vision Transformers - Complete Graph Catalog

**Total Graphs Generated: 42**

---

# SECTION A: Paper Figures (1-22)

Core figures following the academic paper structure for breast cancer detection research.

---

## Architecture & Methodology Figures (1-9)

### Figure 1: Architecture Diagram
- **File:** `paper_figures/Figure_01_Architecture_Diagram.png`
- **Type:** Block Diagram / System Architecture
- **Description:** End-to-end breast cancer identification pipeline
- **Components:** Dataset Collection → Swin ResUnet3+ Segmentation → MAD-ELM Detection → PIWCO Optimization → Cancer Prediction

### Figure 2: Sample Images
- **File:** `paper_figures/Figure_02_Sample_Images.png`
- **Type:** 2×2 Image Grid
- **Description:** Sample normal and abnormal mammogram images from both datasets
- **Panels:**
  - (a) Dataset 1 Normal (Benign)
  - (b) Dataset 1 Abnormal (Malignant)
  - (c) Dataset 2 Normal (Benign)
  - (d) Dataset 2 Abnormal (Malignant)

### Figure 3: ResUnet3+ Architecture
- **File:** `paper_figures/Figure_03_ResUnet3_Architecture.png`
- **Type:** Neural Network Architecture Diagram
- **Description:** Schematic of ResUnet3+ segmentation architecture
- **Components:** Input → Encoder 1-3 → ASPP → Decoder ResUnet3+ → Mask Output
- **Features:** Skip connections between encoder and decoder

### Figure 4: Swin ResUnet3+ Framework
- **File:** `paper_figures/Figure_04_Swin_ResUnet3_Framework.png`
- **Type:** Neural Network Architecture Diagram
- **Description:** Swin Transformer integrated with ResUnet3+ for segmentation
- **Components:** Input Mammogram → Patch Embedding → Swin Transformer → Feature Fusion → ResUnet3+ Decoder → Segmentation

### Figure 5: PIWCO Algorithm Flowchart
- **File:** `paper_figures/Figure_05_PIWCO_Flowchart.png`
- **Type:** Flowchart
- **Description:** Position-based Improved Invasive Weed and Crisscross Optimization
- **Steps:** Initialize Population → IIWO Exploration → COA Exploitation → Convergence Check → Update Positions → Evaluate Fitness → Return Best Parameters

### Figure 6: MAD Architecture
- **File:** `paper_figures/Figure_06_MAD_Architecture.png`
- **Type:** Neural Network Architecture Diagram
- **Description:** Multi-scale Attention-based DenseNet for feature extraction
- **Components:** Input → Conv Stem → Dense Blocks → Multi-scale Attention → Classifier

### Figure 7: ELM Architecture
- **File:** `paper_figures/Figure_07_ELM_Architecture.png`
- **Type:** Neural Network Diagram
- **Description:** Extreme Learning Machine structure
- **Components:** Input Layer (6 neurons) → Hidden Layer (6 neurons) → Output Layer (2 neurons)
- **Features:** Fully connected with random weights

### Figure 8: MAD-ELM Architecture
- **File:** `paper_figures/Figure_08_MAD_ELM_Architecture.png`
- **Type:** Combined Architecture Diagram
- **Description:** Integration of MAD feature extractor with ELM classifier
- **Components:** Input → MAD Feature Extractor → Feature Vector → ELM → Class Output

### Figure 9: Segmentation Results
- **File:** `paper_figures/Figure_09_Segmentation_Results.png`
- **Type:** 2×2 Image Grid
- **Description:** Visualization of segmentation results
- **Panels:**
  - (a) Dataset 1 Input
  - (b) Dataset 1 Segmented
  - (c) Dataset 2 Input
  - (d) Dataset 2 Segmented

---

## Performance Evaluation Graphs (10-21)

### Figure 10: Batch Size Evaluation - Dataset 1 (Algorithms)
- **File:** `paper_figures/Figure_10_Batch_Dataset1_Algorithms.png`
- **Type:** Line Graph (2×2 subplots)
- **X-axis:** Batch Size [4, 8, 15, 32, 50]
- **Subplots:** (a) Accuracy, (b) FDR, (c) FNR, (d) FPR
- **Compared Methods:** AVOA-AMAD-ELM, LO-AMAD-ELM, IIWO-AMAD-ELM, COA-AMAD-ELM, PIWCO-AMAD-ELM

### Figure 11: Batch Size Evaluation - Dataset 1 (Classifiers)
- **File:** `paper_figures/Figure_11_Batch_Dataset1_Classifiers.png`
- **Type:** Line Graph (2×2 subplots)
- **X-axis:** Batch Size [4, 8, 15, 32, 50]
- **Subplots:** (a) Accuracy, (b) FDR, (c) FNR, (d) FPR
- **Compared Methods:** DNN, CNN, DenseNet, MAD-ELM, PIWCO-AMAD-ELM

### Figure 12: K-Fold Evaluation - Dataset 1 (Algorithms)
- **File:** `paper_figures/Figure_12_KFold_Dataset1_Algorithms.png`
- **Type:** Line Graph (2×2 subplots)
- **X-axis:** K-Fold [2, 3, 4, 5]
- **Subplots:** (a) Accuracy, (b) FDR, (c) FNR, (d) FPR
- **Compared Methods:** AVOA-AMAD-ELM, LO-AMAD-ELM, IIWO-AMAD-ELM, COA-AMAD-ELM, PIWCO-AMAD-ELM

### Figure 13: K-Fold Evaluation - Dataset 1 (Classifiers)
- **File:** `paper_figures/Figure_13_KFold_Dataset1_Classifiers.png`
- **Type:** Line Graph (2×2 subplots)
- **X-axis:** K-Fold [2, 3, 4, 5]
- **Subplots:** (a) Accuracy, (b) FDR, (c) FNR, (d) FPR
- **Compared Methods:** DNN, CNN, DenseNet, MAD-ELM, PIWCO-AMAD-ELM

### Figure 14: Learning Percentage Evaluation - Dataset 1 (Algorithms)
- **File:** `paper_figures/Figure_14_LearnPct_Dataset1_Algorithms.png`
- **Type:** Line Graph (2×2 subplots)
- **X-axis:** Learning Percentage [35, 45, 55, 65, 75]
- **Subplots:** (a) Accuracy, (b) FDR, (c) FNR, (d) FPR
- **Compared Methods:** AVOA-AMAD-ELM, LO-AMAD-ELM, IIWO-AMAD-ELM, COA-AMAD-ELM, PIWCO-AMAD-ELM

### Figure 15: Learning Percentage Evaluation - Dataset 1 (Classifiers)
- **File:** `paper_figures/Figure_15_LearnPct_Dataset1_Classifiers.png`
- **Type:** Line Graph (2×2 subplots)
- **X-axis:** Learning Percentage [35, 45, 55, 65, 75]
- **Subplots:** (a) Accuracy, (b) FDR, (c) FNR, (d) FPR
- **Compared Methods:** DNN, CNN, DenseNet, MAD-ELM, PIWCO-AMAD-ELM

### Figure 16: Batch Size Evaluation - Dataset 2 (Algorithms)
- **File:** `paper_figures/Figure_16_Batch_Dataset2_Algorithms.png`
- **Type:** Line Graph (2×2 subplots)
- **X-axis:** Batch Size [4, 8, 15, 32, 50]
- **Subplots:** (a) Accuracy, (b) FDR, (c) FNR, (d) FPR
- **Compared Methods:** AVOA-AMAD-ELM, LO-AMAD-ELM, IIWO-AMAD-ELM, COA-AMAD-ELM, PIWCO-AMAD-ELM

### Figure 17: Batch Size Evaluation - Dataset 2 (Classifiers)
- **File:** `paper_figures/Figure_17_Batch_Dataset2_Classifiers.png`
- **Type:** Line Graph (2×2 subplots)
- **X-axis:** Batch Size [4, 8, 15, 32, 50]
- **Subplots:** (a) Accuracy, (b) FDR, (c) FNR, (d) FPR
- **Compared Methods:** DNN, CNN, DenseNet, MAD-ELM, PIWCO-AMAD-ELM

### Figure 18: K-Fold Evaluation - Dataset 2 (Algorithms)
- **File:** `paper_figures/Figure_18_KFold_Dataset2_Algorithms.png`
- **Type:** Line Graph (2×2 subplots)
- **X-axis:** K-Fold [2, 3, 4, 5]
- **Subplots:** (a) Accuracy, (b) FDR, (c) FNR, (d) FPR
- **Compared Methods:** AVOA-AMAD-ELM, LO-AMAD-ELM, IIWO-AMAD-ELM, COA-AMAD-ELM, PIWCO-AMAD-ELM

### Figure 19: K-Fold Evaluation - Dataset 2 (Classifiers)
- **File:** `paper_figures/Figure_19_KFold_Dataset2_Classifiers.png`
- **Type:** Line Graph (2×2 subplots)
- **X-axis:** K-Fold [2, 3, 4, 5]
- **Subplots:** (a) Accuracy, (b) FDR, (c) FNR, (d) FPR
- **Compared Methods:** DNN, CNN, DenseNet, MAD-ELM, PIWCO-AMAD-ELM

### Figure 20: Learning Percentage Evaluation - Dataset 2 (Algorithms)
- **File:** `paper_figures/Figure_20_LearnPct_Dataset2_Algorithms.png`
- **Type:** Line Graph (2×2 subplots)
- **X-axis:** Learning Percentage [35, 45, 55, 65, 75]
- **Subplots:** (a) Accuracy, (b) FDR, (c) FNR, (d) FPR
- **Compared Methods:** AVOA-AMAD-ELM, LO-AMAD-ELM, IIWO-AMAD-ELM, COA-AMAD-ELM, PIWCO-AMAD-ELM

### Figure 21: Learning Percentage Evaluation - Dataset 2 (Classifiers)
- **File:** `paper_figures/Figure_21_LearnPct_Dataset2_Classifiers.png`
- **Type:** Line Graph (2×2 subplots)
- **X-axis:** Learning Percentage [35, 45, 55, 65, 75]
- **Subplots:** (a) Accuracy, (b) FDR, (c) FNR, (d) FPR
- **Compared Methods:** DNN, CNN, DenseNet, MAD-ELM, PIWCO-AMAD-ELM

### Figure 22: Training and Validation Curves
- **File:** `paper_figures/Figure_22_Training_Validation_Curves.png`
- **Type:** Line Graph (1×2 subplots)
- **Subplots:**
  - (a) Model Accuracy vs Epochs (Training + Validation)
  - (b) Model Loss vs Epochs (Training + Validation)
- **X-axis:** Epochs
- **Y-axis:** Accuracy (%) / Loss

---

# SECTION B: Supplementary Tables (1-5)

Visual table outputs rendered as publication-quality images.

### Table 1: Features and Challenges
- **File:** `paper_tables/Table_01_Features_Challenges.png`
- **Columns:** Author, Methodology, Features, Challenges
- **Content:** Comparison of deep learning approaches for breast cancer detection

### Table 2: Hyperparameters
- **File:** `paper_tables/Table_02_Hyperparameters.png`
- **Columns:** Module, Parameter, Value
- **Content:** Training configuration, regularization settings, split ratios

### Table 3: Optimization Comparison
- **File:** `paper_tables/Table_03_Optimization_Comparison.png`
- **Columns:** Method, Dataset, Accuracy, Sensitivity, Specificity, Precision, F1, MCC
- **Content:** Performance of different optimization algorithms

### Table 4: Classifier Comparison
- **File:** `paper_tables/Table_04_Classifier_Comparison.png`
- **Columns:** Classifier, Dataset, Accuracy, Sensitivity, Specificity, Precision, F1, MCC
- **Content:** Performance of different classification methods

### Table 5: Deep Architecture Comparison
- **File:** `paper_tables/Table_05_Deep_Architecture_Comparison.png`
- **Columns:** Method, Dataset, Accuracy, Sensitivity, Specificity, Precision, F1, MCC
- **Content:** Project model results (MobileViT, EfficientViT, HybridViT)

---

# SECTION C: Extended Analysis Graphs (Extra 1-15)

Additional research insights and comprehensive analysis visualizations.

### Extra 1: Case vs Image Accuracy
- **File:** `extra_graphs/Extra_01_Case_vs_Image_Accuracy.png`
- **Type:** Grouped Bar Chart
- **Description:** Comparison of image-level vs case-level accuracy metrics
- **Insight:** Shows the gap between per-image and per-patient accuracy

### Extra 2: Accuracy Confidence Interval Forest Plot
- **File:** `extra_graphs/Extra_02_Accuracy_CI_Forest.png`
- **Type:** Forest Plot
- **Description:** Point estimates with 95% confidence intervals
- **Insight:** Statistical uncertainty visualization for each model

### Extra 3: Dataset Class Distribution
- **File:** `extra_graphs/Extra_03_Dataset_Class_Distribution.png`
- **Type:** Grouped Bar Chart
- **Description:** Benign vs Malignant image counts per dataset
- **Datasets:** DDSM, BreakHis, BUS_UC
- **Insight:** Class imbalance analysis

### Extra 4: Error Profile Heatmap
- **File:** `extra_graphs/Extra_04_Error_Profile_Heatmap.png`
- **Type:** Heatmap
- **Description:** FDR, FNR, FPR rates by model/dataset
- **Colormap:** Rocket (reversed - lower is better)
- **Insight:** Error pattern identification across runs

### Extra 5: Training Time vs Accuracy
- **File:** `extra_graphs/Extra_05_Training_Time_vs_Accuracy.png`
- **Type:** Scatter Plot
- **Description:** Trade-off between training duration and model accuracy
- **X-axis:** Training Time (minutes)
- **Y-axis:** Accuracy (%)
- **Insight:** Efficiency analysis

### Extra 6: ROC Curve Comparison
- **File:** `extra_graphs/Extra_06_ROC_Comparison.png`
- **Type:** Line Plot (Multiple curves)
- **Description:** Receiver Operating Characteristic curves for all models
- **X-axis:** False Positive Rate
- **Y-axis:** True Positive Rate
- **Features:** AUC values in legend, diagonal reference line
- **Insight:** Discrimination ability comparison

### Extra 7: Precision-Recall Curve Comparison
- **File:** `extra_graphs/Extra_07_Precision_Recall_Comparison.png`
- **Type:** Line Plot (Multiple curves)
- **Description:** Precision-Recall curves for all models
- **X-axis:** Recall
- **Y-axis:** Precision
- **Features:** Average Precision (AP) in legend
- **Insight:** Performance on imbalanced datasets

### Extra 8: Confusion Matrix Grid
- **File:** `extra_graphs/Extra_08_Confusion_Matrix_Grid.png`
- **Type:** Grid of Heatmaps
- **Description:** Confusion matrices for all model/dataset combinations
- **Layout:** Dynamic grid (up to 3 columns)
- **Labels:** Benign, Malignant (Actual vs Predicted)
- **Insight:** Detailed classification breakdown

### Extra 9: Metrics Radar Chart
- **File:** `extra_graphs/Extra_09_Metrics_Radar_Chart.png`
- **Type:** Polar/Radar Chart
- **Description:** Multi-metric comparison across models
- **Metrics:** Accuracy, Precision, Recall, Specificity, F1-Score
- **Features:** Filled polygons with transparency
- **Insight:** Holistic performance visualization

### Extra 10: Model Ranking Heatmap
- **File:** `extra_graphs/Extra_10_Model_Ranking_Heatmap.png`
- **Type:** Annotated Heatmap
- **Description:** Performance scores across all metrics
- **Metrics:** Accuracy, Precision, Recall, Specificity, F1, AUC, MCC
- **Colormap:** Red-Yellow-Green (higher is better)
- **Insight:** Comprehensive model ranking

### Extra 11: Epoch Loss Comparison
- **File:** `extra_graphs/Extra_11_Epoch_Loss_Comparison.png`
- **Type:** Line Plot (1×2 subplots)
- **Description:** Training dynamics across all models
- **Subplots:** (a) Training Loss, (b) Validation Loss
- **X-axis:** Epoch
- **Insight:** Convergence behavior comparison

### Extra 12: Accuracy by Dataset
- **File:** `extra_graphs/Extra_12_Accuracy_By_Dataset.png`
- **Type:** Grouped Bar Chart
- **Description:** Model accuracy breakdown by dataset
- **Models:** MobileViT, EfficientViT, HybridViT
- **Datasets:** DDSM, BreakHis, BUS_UC
- **Insight:** Dataset-specific performance

### Extra 13: Sensitivity vs Specificity Scatter
- **File:** `extra_graphs/Extra_13_Sensitivity_Specificity_Scatter.png`
- **Type:** Scatter Plot
- **Description:** Trade-off between sensitivity and specificity
- **X-axis:** Specificity (%)
- **Y-axis:** Sensitivity (%)
- **Features:** Color by dataset, shape by model
- **Insight:** Clinical threshold analysis

### Extra 14: F1 vs MCC Comparison
- **File:** `extra_graphs/Extra_14_F1_MCC_Comparison.png`
- **Type:** Grouped Bar Chart
- **Description:** Side-by-side F1-Score and Matthews Correlation Coefficient
- **Insight:** Balanced metric comparison

### Extra 15: Comprehensive Summary Table
- **File:** `extra_graphs/Extra_15_Comprehensive_Summary.png`
- **Type:** Table Image
- **Columns:** Model, Dataset, Acc, Sens, Spec, Prec, F1, AUC, MCC, Time(m)
- **Insight:** Complete results at a glance

---

# Summary Statistics

| Category | Count | Subplots |
|----------|-------|----------|
| Architecture Diagrams | 6 | - |
| Image Visualizations | 2 | 8 panels |
| Flowcharts | 1 | - |
| Performance Line Graphs | 12 | 48 subplots |
| Training Curves | 1 | 2 subplots |
| **Paper Figures Total** | **22** | **58** |
| Supplementary Tables | 5 | - |
| Extended Analysis Graphs | 15 | ~25 subplots |
| **GRAND TOTAL** | **42** | **~83** |

---

# Metrics Reference

## Primary Metrics
| Metric | Formula | Range |
|--------|---------|-------|
| Accuracy | (TP + TN) / Total | 0-100% |
| Precision | TP / (TP + FP) | 0-100% |
| Sensitivity/Recall | TP / (TP + FN) | 0-100% |
| Specificity | TN / (TN + FP) | 0-100% |
| F1-Score | 2 × (Prec × Recall) / (Prec + Recall) | 0-100% |

## Error Metrics
| Metric | Formula | Range |
|--------|---------|-------|
| FDR (False Discovery Rate) | FP / (FP + TP) | 0-100% |
| FNR (False Negative Rate) | FN / (FN + TP) | 0-100% |
| FPR (False Positive Rate) | FP / (FP + TN) | 0-100% |

## Advanced Metrics
| Metric | Description | Range |
|--------|-------------|-------|
| ROC-AUC | Area Under ROC Curve | 0-1 |
| MCC | Matthews Correlation Coefficient | -1 to 1 |
| Cohen's Kappa | Inter-rater Agreement | -1 to 1 |
| Average Precision | Area Under PR Curve | 0-1 |

---

# Evaluation Parameters

## Batch Size Sweep
- Values: [4, 8, 15, 32, 50]
- Purpose: Memory vs accuracy trade-off analysis

## K-Fold Cross-Validation
- Values: [2, 3, 4, 5]
- Purpose: Model stability and generalization assessment

## Learning Percentage (Train Split)
- Values: [35%, 45%, 55%, 65%, 75%]
- Purpose: Data efficiency analysis

---

# Datasets

## Dataset 1: BreakHis (Proxy)
- **Full Name:** Breast Cancer Histopathological Image Classification
- **Magnifications:** 40X, 100X, 200X, 400X
- **Classes:** Benign, Malignant

## Dataset 2: DDSM (Proxy)
- **Full Name:** Digital Database for Screening Mammography
- **Image Type:** Mammogram masses
- **Classes:** Benign Masses, Malignant Masses

## Dataset 3: BUS_UC
- **Full Name:** Breast Ultrasound Dataset
- **Image Type:** Ultrasound images
- **Classes:** Benign, Malignant

---

# File Structure

```
results/paper_style_graphs/
├── paper_figures/           # 22 paper figures
│   ├── Figure_01_Architecture_Diagram.png
│   ├── Figure_02_Sample_Images.png
│   ├── ...
│   └── Figure_22_Training_Validation_Curves.png
├── paper_tables/            # 5 supplementary tables
│   ├── Table_01_Features_Challenges.png
│   ├── Table_02_Hyperparameters.png
│   ├── Table_03_Optimization_Comparison.png
│   ├── Table_04_Classifier_Comparison.png
│   └── Table_05_Deep_Architecture_Comparison.png
├── extra_graphs/            # 15 extended analysis graphs
│   ├── Extra_01_Case_vs_Image_Accuracy.png
│   ├── Extra_02_Accuracy_CI_Forest.png
│   ├── ...
│   └── Extra_15_Comprehensive_Summary.png
└── graph_manifest.json      # Complete file listing
```

---

# How to Generate

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Generate all graphs
python generate_paper_graphs.py

# Or run full pipeline (trains + generates)
python run_per_dataset.py
```

---

# Source Data

Performance sweep data (Figures 10-21) is read from:
- `results/paper_style_source_data.json`

If missing, auto-generated from project training results.
