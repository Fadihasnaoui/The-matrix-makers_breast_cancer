# 🩺 Breast Cancer Detection ML System - Matrix Makers Team

A complete, production-ready machine learning system for breast cancer detection with real-time predictions, model explainability (SHAP), and comprehensive monitoring.

**Team**: Matrix Makers  
**Branch**: rayen  
**Project**: ML for Breast Cancer Detection

## 🌟 Features

- **6 ML Models**: Neural Network (99.4%), Random Forest, SVM, k-NN, Logistic Regression
- **Real-time Predictions**: FastAPI backend with instant diagnosis
- **SHAP Explainability**: Mathematical feature importance with force plots
- **Model Comparison**: Switch between models in real-time
- **Interactive Dashboard**: Beautiful web interface with sample loading
- **Monitoring Stack**: Prometheus + Grafana + AlertManager
- **Docker Support**: Complete containerized deployment
- **Automated Reports**: Daily performance monitoring

## 🚀 Quick Start

### Local Development
```bash
# 1. Clone and setup
git clone https://github.com/Fadihasnaoui/The-matrix-makers_breast_cancer.git
cd The-matrix-makers_breast_cancer
git checkout rayen
cd breast-cancer-app

# 2. Setup environment
make setup

# 3. Run the application
make run
