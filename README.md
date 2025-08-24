# Model Comparison for Credit Card Fraud Detection

A machine learning and graph neural network (GNN) based project for detecting fraudulent credit card transactions using the **Kaggle Credit Card Fraud Detection dataset**.

---

## Overview

Detecting credit card fraud is challenging due to the highly **imbalanced dataset** and the complex, non-linear relationships in transaction data.
This project explores both **traditional machine learning models** and **Graph Neural Networks** (Spektral, PyTorch Geometric, TensorFlow) to evaluate performance in fraud detection tasks.

The project applies **GPU acceleration** with RAPIDS libraries (`cudf`, `cuml`, `cupy`) to significantly reduce training and inference time.

---

## Features

* Preprocessing and handling of highly imbalanced dataset
* GPU-accelerated data manipulation with **cuDF** and **cuML**
* Implementation of multiple GNN architectures (GCN, GIN, Spektral, PyG, TensorFlow GNN layers)
* Baseline comparisons with traditional ML models
* Evaluation with standard metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Confusion Matrix
* Visualization of results for better interpretability

---

## Tech Stack

* Python
* TensorFlow, PyTorch, PyTorch Geometric, Spektral
* RAPIDS libraries: cuDF, cuML, cuPy
* scikit-learn
* NumPy, Pandas, Matplotlib
* SciPy

---

## Results

* Models were evaluated on the **Credit Card Fraud Detection dataset**
* RAPIDS acceleration provided significant speed-ups in data preprocessing and training

* Accuracy, Precision, Recall, F1-score across models
<img width="1975" height="1264" alt="image" src="https://github.com/user-attachments/assets/486aaf3f-7676-4ae1-8085-7b588dc17250" />


* Confusion matrices for fraud vs non-fraud prediction
<img width="2090" height="3590" alt="image" src="https://github.com/user-attachments/assets/84909eb3-df63-4c8d-9fc8-2b14d122c863" />


# Comprehensive Performance Summary (Ranked by Average Rank)

| Model                        | Accuracy | Precision | Recall |   TNR  |   F1   | Time (s) | IsEnsemble | Accuracy_Rank | Precision_Rank | Recall_Rank | TNR_Rank | F1_Rank | Avg_Rank |
|------------------------------|----------|-----------|--------|--------|--------|----------|------------|---------------|----------------|-------------|----------|---------|----------|
| Random Forest (GPU)          | 0.9996   | 0.9773    | 0.8190 | 1.0000 | 0.8912 |  2.6374  | True       | 1             | 1              | 11          | 3        | 1       | 3.4000   |
| XGBoost (GPU)                | 0.9996   | 0.9263    | 0.8381 | 0.9999 | 0.8800 |  2.9965  | True       | 2             | 2              | 9           | 4        | 2       | 3.8000   |
| Non Linear SVM (RBF Kernel)  | 0.9989   | 0.6519    | 0.8381 | 0.9992 | 0.7333 |  4.1233  | False      | 3             | 4              | 9           | 6        | 3       | 5.0000   |
| 1D Conv (Tanh)               | 0.9884   | 0.1240    | 0.8762 | 0.9886 | 0.2172 | 13.3512  | False      | 7             | 5              | 7           | 7        | 5       | 6.2000   |
| Boosting-GNN Ensemble        | 0.9987   | 0.6744    | 0.5524 | 0.9995 | 0.6073 |  9.2159  | True       | 4             | 3              | 16          | 5        | 4       | 6.4000   |
| MLP (ELU)                    | 0.9878   | 0.1187    | 0.8762 | 0.9880 | 0.2091 | 18.3215  | False      | 8             | 6              | 7           | 8        | 6       | 7.0000   |
| LSTM (ReLU)                  | 0.9821   | 0.0884    | 0.9333 | 0.9822 | 0.1614 | 23.1496  | False      | 11            | 8              | 3           | 11       | 7       | 8.0000   |
| ANN (Mish)                   | 0.9811   | 0.0839    | 0.9333 | 0.9812 | 0.1540 | 15.7701  | False      | 12            | 10             | 3           | 12       | 9       | 9.2000   |
| Jump-Attentive GNN           | 0.9852   | 0.0891    | 0.7619 | 0.9856 | 0.1595 | 70.6747  | False      | 10            | 7              | 12          | 10       | 8       | 9.4000   |
| Graph Neural Network (Swish) | 0.9872   | 0.0864    | 0.6190 | 0.9879 | 0.1517 | 36.2708  | False      | 9             | 9              | 15          | 9        | 10      | 10.4000  |
| Logistic Regression (GPU)    | 0.9982   | 0.0000    | 0.0000 | 1.0000 | 0.0000 |  0.0592  | False      | 5             | 17             | 17          | 1        | 17      | 11.4000  |
| Linear SVM (GPU)             | 0.9982   | 0.0000    | 0.0000 | 1.0000 | 0.0000 |  0.0524  | False      | 5             | 17             | 17          | 1        | 17      | 11.4000  |
| MLP (Mish)                   | 0.9370   | 0.0266    | 0.9333 | 0.9370 | 0.0518 | 22.3506  | False      | 15            | 13             | 3           | 15       | 13      | 11.8000  |
| ANN (SELU)                   | 0.9277   | 0.0240    | 0.9619 | 0.9276 | 0.0467 | 16.2276  | False      | 16            | 14             | 1           | 16       | 14      | 12.2000  |
| PC-GNN                       | 0.9657   | 0.0366    | 0.6952 | 0.9662 | 0.0696 | 34.7876  | False      | 13            | 11             | 13          | 13       | 11      | 12.2000  |
| Graph Neural Network (Tanh)  | 0.9577   | 0.0282    | 0.6571 | 0.9582 | 0.0542 | 36.2043  | False      | 14            | 12             | 14          | 14       | 12      | 13.2000  |
| 1D Conv (ReLU)               | 0.9189   | 0.0206    | 0.9238 | 0.9189 | 0.0403 | 17.4886  | False      | 17            | 15             | 6           | 17       | 15      | 14.0000  |
| LSTM (Mish)                  | 0.8433   | 0.0111    | 0.9524 | 0.8431 | 0.0219 | 21.6792  | False      | 18            | 16             | 2           | 18       | 16      | 14.0000  |


# Best Performers by Category and Metric

SINGLE MODELS:
*Overall Best: Non Linear SVM (RBF Kernel, GPU) (Avg Rank: 5.00)
*Best Precision: Non Linear SVM (RBF Kernel, GPU) (Precision: 0.6519)
*Best Recall: ANN (SELU) (Recall: 0.9619)
*Best F1: Non Linear SVM (RBF Kernel, GPU) (F1: 0.7333)

ENSEMBLE MODELS:
*Overall Best: Random Forest (GPU) (Avg Rank: 3.40)
*Best Precision: Random Forest (GPU) (Precision: 0.9773)
*Best Recall: XGBoost (GPU) (Recall: 0.8381)
*Best F1: Random Forest (GPU) (F1: 0.8912)


<img width="986" height="819" alt="image" src="https://github.com/user-attachments/assets/b40c3420-e894-4e6a-a77b-06261be7d650" />


---

## Future Work

* Extend to real-time fraud detection pipelines
* Investigate graph transformer architectures for transaction graphs
* Explore anomaly detection methods combined with GNNs
* Apply explainability methods (e.g., GNNExplainer, SHAP) to interpret fraud predictions

---

## Acknowledgements

The dataset used in this project is the **Credit Card Fraud Detection Dataset** made available on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).

Special thanks to the open-source libraries and research communities that support graph learning and GPU-accelerated data science.

---

## Contact

**Stuti Srivastava**
Email: [stutisrivastava0923@gmail.com](mailto:stutisrivastava0923@gmail.com)
LinkedIn: [linkedin.com/in/stutisrivastava23](https://linkedin.com/in/stutisrivastava23)

---
