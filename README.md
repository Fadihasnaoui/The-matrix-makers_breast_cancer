# Breast Cancer Diagnosis — Machine Learning Project (CRISP-DM)

## 📌 Project Overview
This project applies machine learning techniques to support **breast cancer diagnosis**, with a strong focus on **early detection**, **clinical reliability**, and **interpretability**.  
The work follows the **CRISP-DM methodology**, ensuring a structured, transparent, and medically aligned data science process.

The objective is to assist clinicians in distinguishing between **benign** and **malignant** tumors using quantitative features extracted from breast cancer datasets.

---

## 🏥 Business Objectives (BO)

- **BO1** — Detect breast cancer as early as possible  
- **BO2** — Improve diagnostic decision-making and reduce errors  
- **BO3** — Reduce unnecessary tests, biopsies, and medical procedures  
- **BO4** — Prioritize patients requiring urgent medical attention  

---

## 🎯 Data Science Objectives (DSO)

- **DSO1** — Maximize Recall (True Positive Rate) to avoid missing malignant cases  
- **DSO2** — Minimize False Positives to reduce unnecessary medical interventions  
- **DSO3** — Ensure fast, interpretable, and clinically actionable models  
- **DSO4** — Achieve robust generalization across diverse tumor characteristics  

---

## 🧪 Datasets

This project uses **two complementary breast cancer datasets**, which were **merged to improve robustness and generalization**.

### 1️⃣ Wisconsin Diagnostic Breast Cancer (WDBC)
- Features extracted from digitized images of breast mass fine needle aspirates
- Includes **mean**, **standard error**, and **worst-case** measurements
- Binary target variable:  
  - `0` → Benign  
  - `1` → Malignant  

### 2️⃣ Wisconsin Prognostic Breast Cancer (WPBC)
- Contains tumor-related features associated with disease progression
- Introduces additional variability in tumor morphology
- Used to enrich the dataset and improve generalization

### 🔗 Dataset Integration
- Common and compatible features between **WDBC** and **WPBC** were identified
- Datasets were **cleaned, aligned, and merged**
- The merged dataset improves:
  - tumor heterogeneity representation
  - robustness across patient profiles (**DSO4**)

---

## 🔍 Methodology — CRISP-DM Phases

### Phase 1 — Business Understanding
Clinical and diagnostic objectives were defined to ensure alignment between machine learning performance and real medical needs.

---

### Phase 2 — Data Understanding
- Exploratory Data Analysis (EDA)
- Analysis of class distribution (benign vs malignant)
- Box plots for mean, standard error, and worst features
- Detection of variability, scale differences, and outliers
- Clinical interpretation of extreme values

---

### Phase 3 — Data Preparation
- Harmonization of WDBC and WPBC features
- Handling of missing and inconsistent values
- Feature standardization
- **Principal Component Analysis (PCA)**:
  - Dimensionality reduction
  - Analysis of feature representation (cos²)
  - Identification of dominant tumor characteristics
- Train/test split
- Retention of clinically meaningful outliers

---

### Phase 4 — Modeling
The following models were trained and compared:

- Linear Regression (baseline)
- ElasticNet Logistic Regression
- Linear SVM (C tuned)
- RBF SVM (GRU-SVM Proxy)
- L1NN / L2NN (Manhattan / Euclidean k-NN)
- Random Forest
- MLP (500-500-500 with Early Stopping)

---

### Phase 5 — Evaluation
- Accuracy, Recall (TPR), Selectivity (TNR), FPR, FNR
- Confusion matrices
- ROC curves and AUC comparison
- Model-to-DSO alignment analysis
- Clinical error analysis (false positives vs false negatives)

---

## 🏆 Model Recommendation (Summary)

- **Primary Screening Model:** MLP (Early Stopping) — optimized for high Recall  
- **Interpretable Clinical Model:** ElasticNet Logistic Regression  
- **Robust Validation Model:** Random Forest  

A **hybrid decision-support strategy** is recommended rather than relying on a single model.

---

## 📊 Key Visualizations
- Box plots (Mean / SE / Worst features)
- ROC curves for all evaluated models
- Confusion matrices
- PCA cos² heatmap (feature representation quality)
- LIME explanations for model interpretability

---

## 🚀 Deployment (Planned — Not Implemented Yet)

> This section is intentionally left for future work.

Planned deployment considerations include:
- Model serialization
- API-based inference service
- Clinical decision-support integration
- Threshold calibration based on hospital policy
- Monitoring model performance and data drift

---

## ⚠️ Disclaimer
This project is for **educational and research purposes only**.  
It is **not a medical device** and should not be used for clinical diagnosis without proper validation and regulatory approval.

---

## 👥 Team
**The Matrix-Makers**  
Data Science & Machine Learning Project Team

---

