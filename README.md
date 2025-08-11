# ECG Anomaly Detection with Autoencoder & Hybrid Classifiers

## Overview

This project detects anomalies in ECG signals by combining self-supervised learning (autoencoder) with lightweight supervised classification. The approach addresses the limitations of reconstruction-error-only methods by leveraging learned features for classification.

## Methodology

1. Autoencoder

   - Learns normal ECG beat structure.
   - Anomalies detected using reconstruction error thresholds (basic & Youden’s J).

2. Hybrid Approach

   - Encoder outputs fed into classifiers:
   - Random Forest (RF)
   - Tuned RF with Stratified K-Fold cross-validation
   - MLP
   - Logistic Regression

3. Hyperparameter Tuning

   - RF tuned using GridSearchCV with 5-fold StratifiedKFold.
   - Reduced overfitting while preserving performance.

## Requirements

- Python 3.8+
- TensorFlow / Keras
- scikit-learn
- NumPy, Pandas, Matplotlib, Seaborn

## Conclusion

The hybrid method significantly outperforms reconstruction-error-only anomaly detection, offering a robust and accurate approach for ECG anomaly detection.
