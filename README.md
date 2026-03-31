# Breast Cancer Classification using Vision Transformers

A comprehensive deep learning framework for breast cancer classification using state-of-the-art Vision Transformer architectures. This project compares **MaxViT**, **MobileViT**, and **EfficientViT** models across multiple breast cancer imaging datasets.

## 🎯 Overview

This project implements binary classification (Benign vs Malignant) for breast cancer detection using three modern Vision Transformer architectures:

| Model | Description | Key Features |
|-------|-------------|--------------|
| **MaxViT** | CNN + Transformer Hybrid | Strong local & global feature modeling |
| **MobileViT** | Lightweight Hybrid | Efficient & deployable on edge devices |
| **EfficientViT** | Optimized Transformer | Fast inference, low memory footprint |

## 📊 Datasets

The models are trained and evaluated on three benchmark datasets:

- **DDSM** - Digital Database for Screening Mammography
- **BUS_UC** - Breast Ultrasound Images (UC Dataset)
- **BreakHis** - Breast Cancer Histopathological Images (40X, 100X, 200X, 400X magnifications)

## 🏗️ Project Structure

```
├── config.py                 # Configuration and hyperparameters
├── dataset.py                # Dataset loading and preprocessing
├── utils.py                  # Utility functions and metrics
├── train_maxvit.py           # MaxViT training script
├── train_mobilevit.py        # MobileViT training script
├── train_efficientvit.py     # EfficientViT training script
├── train_hybrid.py           # Hybrid training approach
├── run_all.py                # Run all experiments
├── run_per_dataset.py        # Per-dataset training
├── generate_paper_graphs.py  # Generate publication-ready figures
├── BreastCancer_Training_Notebook.ipynb  # Jupyter notebook
└── results/                  # Training results and visualizations
    ├── EfficientViT_*/       # EfficientViT results per dataset
    ├── MobileViT_*/          # MobileViT results per dataset
    ├── HybridViT_*/          # HybridViT results per dataset
    └── paper_style_graphs/   # Publication-ready figures
```

## ⚙️ Configuration

Key training parameters (from `config.py`):

```python
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
```

### Anti-Overfitting Strategies

- Label smoothing (0.1)
- Dropout in classifier head (0.3)
- Gradient clipping (max_norm=1.0)
- CosineAnnealingWarmRestarts LR schedule with warmup
- Progressive unfreezing (freeze backbone initially)
- Mixup augmentation (alpha=0.2)
- Data augmentation (flips, rotation, color jitter, erasing)
- Weighted sampling for class imbalance
- Early stopping with patience

## 🚀 Usage

### Training Individual Models

```bash
# Train MaxViT
python train_maxvit.py

# Train MobileViT
python train_mobilevit.py

# Train EfficientViT
python train_efficientvit.py
```

### Run All Experiments

```bash
python run_all.py
```

### Per-Dataset Training

```bash
python run_per_dataset.py
```

### Generate Publication Figures

```bash
python generate_paper_graphs.py
```

## 📈 Results

Results are saved in the `results/` directory with:

- **Training curves** (loss and accuracy over epochs)
- **Confusion matrices**
- **ROC curves**
- **Precision-Recall curves**
- **Calibration plots**
- **Class-wise metrics**
- **JSON metrics files**

### Sample Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification accuracy |
| Precision | Positive predictive value |
| Recall/Sensitivity | True positive rate |
| Specificity | True negative rate |
| F1 Score | Harmonic mean of precision and recall |
| ROC-AUC | Area under ROC curve |
| MCC | Matthews Correlation Coefficient |

## 📋 Requirements

- Python 3.8+
- PyTorch
- timm (PyTorch Image Models)
- torchvision
- scikit-learn
- matplotlib
- seaborn
- numpy
- pandas
- Pillow

## 🔬 Evaluation Modes

The framework supports both **image-level** and **case-level** evaluation:

- **Image-level**: Each image is evaluated independently
- **Case-level**: Predictions are aggregated per patient/case for clinical relevance

## 📚 Citation

If you use this code in your research, please cite accordingly.

## 📄 License

This project is for research and educational purposes.

---

**Note**: Datasets are not included in this repository due to size constraints. Please download them from their respective sources.
