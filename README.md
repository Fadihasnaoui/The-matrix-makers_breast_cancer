Breast Cancer Diagnosis — Machine Learning Project (CRISP-DM)
📌 Project Overview

This project applies machine learning techniques to support breast cancer diagnosis, with a strong focus on early detection, clinical reliability, and model interpretability.
The work follows the CRISP-DM methodology, ensuring a structured, transparent, and medically aligned data science process.

Beyond model development, the project includes a fully deployed web-based dashboard that allows users to interact with trained models, visualize predictions, explore explainability (SHAP & LIME), and inspect evaluation metrics.

The objective is to assist clinicians and students in distinguishing between benign and malignant tumors using quantitative features extracted from breast cancer datasets.

🏥 Business Objectives (BO)

BO1 — Detect breast cancer as early as possible

BO2 — Improve diagnostic decision-making and reduce errors

BO3 — Reduce unnecessary tests, biopsies, and medical procedures

BO4 — Prioritize patients requiring urgent medical attention

🎯 Data Science Objectives (DSO)

DSO1 — Maximize Recall (True Positive Rate) to avoid missing malignant cases

DSO2 — Minimize False Positives to reduce unnecessary medical interventions

DSO3 — Ensure fast, interpretable, and clinically actionable models

DSO4 — Achieve robust generalization across diverse tumor characteristics

🧪 Datasets

This project uses two complementary breast cancer datasets, which were merged to improve robustness and generalization.

1️⃣ Wisconsin Diagnostic Breast Cancer (WDBC)

Features extracted from digitized images of breast mass fine needle aspirates

Includes mean, standard error, and worst-case measurements

Binary target variable:

0 → Benign

1 → Malignant

2️⃣ Wisconsin Prognostic Breast Cancer (WPBC)

Contains tumor-related features associated with disease progression

Consists exclusively of malignant cases

Used to enrich the dataset and increase malignant class representation

🔗 Dataset Integration

Common and compatible features between WDBC and WPBC were identified

Datasets were cleaned, aligned, and merged

The enriched dataset improves:

class balance

tumor heterogeneity representation

robustness across patient profiles (DSO4)

🔍 Methodology — CRISP-DM Phases
Phase 1 — Business Understanding

Clinical and diagnostic objectives were defined to ensure alignment between machine learning performance and real medical needs.

Phase 2 — Data Understanding

Exploratory Data Analysis (EDA)

Class distribution analysis (benign vs malignant)

Box plots for mean, standard error, and worst features

Correlation analysis and clinical interpretation of feature behavior

Identification of clinically meaningful outliers

Phase 3 — Data Preparation

Harmonization of WDBC and WPBC features

Removal of non-informative identifiers

Feature standardization

Principal Component Analysis (PCA):

Dimensionality analysis

Feature representation quality (cos²)

Stratified train/test split

Retention of medically meaningful extreme values

Phase 4 — Modeling

The following models were trained and compared:

Linear Regression (baseline)

ElasticNet Logistic Regression

Linear SVM (C tuned)

RBF SVM (GRU-SVM Proxy)

L1NN / L2NN (Manhattan / Euclidean k-NN)

Random Forest

MLP (500–500–500 with Early Stopping)

Models were selected to cover different trade-offs between recall, specificity, interpretability, and generalization.

Phase 5 — Evaluation

Accuracy, Recall (TPR), Specificity (TNR)

False Positive Rate (FPR) and False Negative Rate (FNR)

Confusion matrices

ROC curves and AUC comparison

Model-to-DSO alignment analysis

Clinical risk analysis with emphasis on false negatives

🏆 Model Recommendation (Summary)

Rather than relying on a single model, the project adopts a decision-support perspective:

Primary high-capacity model: MLP (Early Stopping) — strong recall and generalization

Clinically interpretable model: ElasticNet Logistic Regression

Robust ensemble model: Random Forest

This hybrid approach balances performance, interpretability, and clinical safety.

📊 Explainability & Transparency

To support clinical trust:

SHAP is used for global and local feature impact analysis

LIME is used for instance-level explanations

Explanations are visualized directly in the deployed dashboard

Feature contributions are consistent with known tumor morphology indicators

🚀 Deployment (Implemented)

This project includes a fully functional deployment, demonstrating how machine learning models can be integrated into a real-world application.

🏗️ Deployment Architecture

The system follows a client–server architecture:

Frontend: React (Vite), hosted on Vercel

Backend: FastAPI (Python), hosted on Render

ML Layer: Pre-trained scikit-learn models with SHAP & LIME explainability

Security: JWT-based authentication and HTTPS communication

End User
   ↓
React Dashboard (Vercel)
   ↓ REST API
FastAPI Backend (Render)
   ↓
ML Models + SHAP/LIME

🧰 Technologies Used

Frontend: React (Vite), Recharts, Tailwind

Backend: FastAPI, Python

Machine Learning: scikit-learn

Explainability: SHAP, LIME

Authentication: JWT

Hosting: Vercel (frontend), Render (backend)

🔄 Application Workflow

User accesses the web dashboard

User logs in (JWT authentication)

User submits feature values for prediction

Backend:

Performs inference

Generates SHAP and LIME explanations

Returns evaluation metrics

Results are visualized interactively in the dashboard

📊 Monitoring & Reliability

API availability monitored via hosting platforms

Error handling implemented at backend level

Response time suitable for real-time interaction

Free-tier hosting may introduce cold-start latency

⚠️ Deployment Limitations

Hosted on free cloud tiers (non-guaranteed uptime)

No persistent clinical database

No regulatory certification

Intended for academic and educational use

⚠️ Disclaimer

This project is for educational and research purposes only.
It is not a medical device and must not be used for clinical diagnosis without proper validation and regulatory approval.

👥 Team

The Matrix-Makers
Machine Learning & Data Science Project Team