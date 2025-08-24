# ECG Anomaly Detection with Autoencoder & Hybrid Classifiers

## Overview

This project detects anomalies in ECG signals by combining self-supervised learning (autoencoder) with lightweight supervised classification. The approach addresses the limitations of reconstruction-error-only methods by leveraging learned features for classification.

## Methodology

### 1. **Autoencoder (Self-Supervised Pretraining)**

- A **1D Convolutional Autoencoder** is trained exclusively on **normal ECG beats** from the PTB Diagnostic ECG Database (PTBDB).
- The model learns to reconstruct normal physiological patterns.
- **Anomalies are detected** based on high reconstruction error (Mean Absolute Error).
- Two thresholding strategies:
  - **95th percentile** of normal reconstruction errors (baseline)
  - **Youden’s J statistic** on ROC curve (optimized threshold)

### 2. **Hybrid Classification (Supervised Refinement)**

To improve detection beyond thresholding, the **encoder’s latent features** are used as input to supervised classifiers:

| Classifier                       | Description                                                             |
| -------------------------------- | ----------------------------------------------------------------------- |
| **Random Forest (RF)**           | Tree-based ensemble; robust to overfitting, provides feature importance |
| **Multi-Layer Perceptron (MLP)** | Feedforward neural network for non-linear pattern detection             |
| **Logistic Regression (LR)**     | Linear baseline for evaluating separability in latent space             |

### 3. **Hyperparameter Tuning**

- **Tuned Random Forest** selected as final model.
- **GridSearchCV** with **5-fold StratifiedKFold** cross-validation.
- Parameter grid:
  ```python
  {
      'n_estimators': [100, 200],
      'max_depth': [10, 15],
      'min_samples_split': [2, 5],
      'min_samples_leaf': [1, 2],
      'max_features': ['sqrt']
  }
  ```

## Conclusion

The hybrid method significantly outperforms reconstruction-error-only anomaly detection, offering a robust and accurate approach for ECG anomaly detection.
