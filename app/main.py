"""
Breast Cancer Detection API - Enhanced Version 5.0.0
Complete ML System with Notebook Integrations: 8 models, ROC curves, Artifact system, etc.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import pickle
import json
import os
from pathlib import Path
import time
from datetime import datetime
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, generate_latest, REGISTRY
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# SHAP imports
import shap
import joblib

# Initialize FastAPI
app = FastAPI(
    title="Breast Cancer Detection API - Enhanced",
    description="Complete ML System with 8 models, ROC curves, Artifact system, Business Objectives tracking",
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Global variables
MODELS_DIR = Path("app/models")
ARTIFACTS_DIR = Path("artifacts")
current_model = None
scaler = None
pca = None
available_models = {}
model_info = {}
feature_names = []
sample_features = []
model_loaded = False
X_train = None
X_test = None
y_train = None
y_test = None
X_train_scaled = None
X_test_scaled = None
X_train_pca = None
start_time = time.time()
shap_explainer = None
feature_names_shap = []
X_train_shap = None

# Business and Data Science Objectives from notebook
BUSINESS_OBJECTIVES = {
    "bo1_early_detection": {"name": "Early Detection", "description": "Detect breast cancer as early as possible", "target": 95},
    "bo2_faster_diagnosis": {"name": "Faster Diagnosis", "description": "Accelerate and improve diagnostic decision-making", "target": 90},
    "bo3_reduce_costs": {"name": "Reduce Costs", "description": "Reduce unnecessary tests, biopsies, and medical procedures", "target": 85},
    "bo4_prioritize_urgent": {"name": "Prioritize Urgent", "description": "Prioritize patients who require urgent medical attention", "target": 95}
}

DATA_SCIENCE_OBJECTIVES = {
    "dso1_high_recall": {"name": "Maximize Early Detection", "target": "Recall ≥ 95%", "primary_model": "MLP", "metric": "recall"},
    "dso2_high_selectivity": {"name": "Minimize False Positives", "target": "Selectivity ≥ 90%", "primary_model": "Linear SVM", "metric": "selectivity"},
    "dso3_fast_interpretable": {"name": "Fast, Interpretable Results", "target": "Latency < 0.01 sec", "primary_model": "Logistic Regression", "metric": "speed"},
    "dso4_robust_generalization": {"name": "Robust Generalization", "target": "Accuracy ≥ 95%", "primary_model": "Random Forest", "metric": "accuracy"}
}

# Enhanced model configurations from notebook
MODEL_CONFIGS = {
    "mlp": {
        "name": "MLP Neural Network (500-500-500)",
        "class": MLPClassifier,
        "params": {"hidden_layer_sizes": (500, 500, 500), "max_iter": 2000, "random_state": 42}
    },
    "knn": {
        "name": "k-NN Classifier (k=5)",
        "class": KNeighborsClassifier,
        "params": {"n_neighbors": 5, "weights": "distance"}
    },
    "knn_optimized": {
        "name": "k-NN Optimized (k=15)",
        "class": KNeighborsClassifier,
        "params": {"n_neighbors": 15, "weights": "distance", "metric": "euclidean"}
    },
    "rf": {
        "name": "Random Forest",
        "class": RandomForestClassifier,
        "params": {"n_estimators": 200, "random_state": 42}
    },
    "svm_linear": {
        "name": "Linear SVM",
        "class": SVC,
        "params": {"kernel": "linear", "probability": True, "random_state": 42}
    },
    "svm_rbf": {
        "name": "SVM RBF",
        "class": SVC,
        "params": {"kernel": "rbf", "probability": True, "random_state": 42}
    },
    "logistic": {
        "name": "Logistic Regression",
        "class": LogisticRegression,
        "params": {"max_iter": 2000, "random_state": 42}
    },
    "linear": {
        "name": "Linear Regression",
        "class": LinearRegression,
        "params": {}
    }
}

# Metrics
model_accuracy_gauge = Gauge('api_model_accuracy', 'Current model accuracy')
model_recall_gauge = Gauge('api_model_recall', 'Current model recall')
model_precision_gauge = Gauge('api_model_precision', 'Current model precision')
model_f1_gauge = Gauge('api_model_f1', 'Current model F1 score')
model_auc_gauge = Gauge('api_model_auc', 'Current model AUC score')
predictions_counter = Counter('api_predictions_total', 'Total predictions made')

# Pydantic models
class PredictionRequest(BaseModel):
    features: List[float]
    model_id: str = "mlp"

class ModelSwitchRequest(BaseModel):
    model_id: str

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ============ CORE FUNCTIONS ============

def load_or_create_data():
    """Load data from CSV or create sample data"""
    global feature_names, sample_features

    print("📊 Loading data...")

    # Try to find data file
    data_paths = ["data.csv", "data/data.csv", "data_enriched.csv"]

    for path in data_paths:
        if os.path.exists(path):
            print(f"✅ Found data file: {path}")
            try:
                df = pd.read_csv(path)

                # Clean data
                if 'id' in df.columns:
                    df = df.drop('id', axis=1)
                if 'Unnamed: 32' in df.columns:
                    df = df.drop('Unnamed: 32', axis=1)

                # Map diagnosis
                if 'diagnosis' in df.columns:
                    # Handle both formats
                    if df['diagnosis'].dtype == object:
                        df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1, 'benign': 0, 'malignant': 1})
                    print(f"📈 Dataset balance: {df['diagnosis'].value_counts().to_dict()}")

                # Get feature names
                feature_names = [col for col in df.columns if col != 'diagnosis']
                if len(feature_names) > 30:
                    feature_names = feature_names[:30]

                # Get sample features
                if len(df) > 0:
                    # Try to get one benign sample
                    benign_samples = df[df['diagnosis'] == 0]
                    if len(benign_samples) > 0:
                        sample_features = benign_samples[feature_names].iloc[0].values.tolist()
                    else:
                        sample_features = df[feature_names].iloc[0].values.tolist()

                X = df[feature_names].values
                y = df['diagnosis'].values

                print(f"📈 Loaded {len(X)} samples, {len(feature_names)} features")
                print(f"   Malignant: {sum(y)}, Benign: {len(y)-sum(y)}")
                return X, y

            except Exception as e:
                print(f"❌ Error loading {path}: {e}")
                continue

    # Create sample data if no file found
    print("⚠️ Creating sample data...")
    return create_sample_data()

def create_sample_data():
    """Create sample breast cancer data"""
    global feature_names, sample_features

    feature_names = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean',
        'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se',
        'smoothness_se', 'compactness_se', 'concavity_se',
        'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
        'smoothness_worst', 'compactness_worst', 'concavity_worst',
        'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]

    np.random.seed(42)
    n_samples = 569
    X = []
    y = []

    for i in range(n_samples):
        if i < 357:  # Benign
            row = [np.random.uniform(6, 15), np.random.uniform(9, 25), np.random.uniform(40, 100),
                   np.random.uniform(150, 700), np.random.uniform(0.05, 0.1), np.random.uniform(0.02, 0.1),
                   np.random.uniform(0.0, 0.15), np.random.uniform(0.0, 0.05), np.random.uniform(0.15, 0.3),
                   np.random.uniform(0.05, 0.09), np.random.uniform(0.2, 1.5), np.random.uniform(0.3, 2.0),
                   np.random.uniform(1.5, 10), np.random.uniform(10, 100), np.random.uniform(0.002, 0.01),
                   np.random.uniform(0.002, 0.05), np.random.uniform(0.0, 0.04), np.random.uniform(0.0, 0.02),
                   np.random.uniform(0.005, 0.03), np.random.uniform(0.001, 0.005), np.random.uniform(7, 20),
                   np.random.uniform(12, 30), np.random.uniform(50, 120), np.random.uniform(200, 900),
                   np.random.uniform(0.07, 0.15), np.random.uniform(0.05, 0.25), np.random.uniform(0.0, 0.3),
                   np.random.uniform(0.0, 0.1), np.random.uniform(0.15, 0.4), np.random.uniform(0.05, 0.15)]
            y.append(0)
        else:  # Malignant
            row = [np.random.uniform(12, 28), np.random.uniform(16, 35), np.random.uniform(85, 180),
                   np.random.uniform(600, 2500), np.random.uniform(0.08, 0.16), np.random.uniform(0.1, 0.35),
                   np.random.uniform(0.1, 0.6), np.random.uniform(0.03, 0.25), np.random.uniform(0.18, 0.35),
                   np.random.uniform(0.06, 0.12), np.random.uniform(0.6, 2.8), np.random.uniform(0.7, 3.5),
                   np.random.uniform(5, 22), np.random.uniform(40, 250), np.random.uniform(0.006, 0.025),
                   np.random.uniform(0.015, 0.12), np.random.uniform(0.02, 0.15), np.random.uniform(0.008, 0.06),
                   np.random.uniform(0.01, 0.06), np.random.uniform(0.002, 0.012), np.random.uniform(14, 38),
                   np.random.uniform(20, 50), np.random.uniform(110, 260), np.random.uniform(900, 4500),
                   np.random.uniform(0.11, 0.28), np.random.uniform(0.18, 0.6), np.random.uniform(0.15, 0.9),
                   np.random.uniform(0.06, 0.35), np.random.uniform(0.2, 0.55), np.random.uniform(0.09, 0.28)]
            y.append(1)
        X.append(row)

    sample_features = X[0]
    print(f"✅ Created {len(X)} sample data points")
    return np.array(X), np.array(y)

def train_all_models():
    """Train all 8 ML models with enhanced configurations"""
    global scaler, available_models, model_loaded, pca, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X_train_pca

    print("🔄 Training 8 models with enhanced configurations...")

    try:
        # Load data
        X, y = load_or_create_data()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit PCA with variance analysis
        pca = PCA(n_components=2, random_state=42)
        pca.fit(X_train_scaled)
        X_train_pca = pca.transform(X_train_scaled)
        
        variance_exp = pca.explained_variance_ratio_
        print(f"📊 PCA Variance: PC1={variance_exp[0]*100:.1f}%, PC2={variance_exp[1]*100:.1f}%, Total={(variance_exp[0]+variance_exp[1])*100:.1f}%")

        # Train each model
        available_models = {}

        for model_id, config in MODEL_CONFIGS.items():
            print(f"  Training {config['name']}...")

            # Initialize and train
            model_class = config['class']
            model = model_class(**config['params'])
            
            # Special handling for Linear Regression
            if model_id == "linear":
                model.fit(X_train_scaled, y_train)
                # Convert regression predictions to binary
                y_pred_raw = model.predict(X_test_scaled)
                y_pred = (y_pred_raw >= 0.5).astype(int)
                y_proba = np.column_stack([1 - y_pred_raw, y_pred_raw])
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test_scaled)
                else:
                    y_proba = np.column_stack([1 - y_pred, y_pred])

            # Enhanced evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred, zero_division=0)
            precision = precision_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Calculate selectivity (True Negative Rate) and fall-out
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            selectivity = tn / (tn + fp) if (tn + fp) > 0 else 0
            fall_out = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # Calculate AUC
            auc_score = roc_auc_score(y_test, y_proba[:, 1]) if len(np.unique(y_test)) > 1 else None

            # Save model
            model_path = MODELS_DIR / f"{model_id}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            # Store comprehensive info
            available_models[model_id] = {
                "id": model_id,
                "name": config['name'],
                "accuracy": float(accuracy),
                "recall": float(recall),
                "precision": float(precision),
                "f1": float(f1),
                "selectivity": float(selectivity),
                "fall_out": float(fall_out),
                "auc": float(auc_score) if auc_score is not None else None,
                "path": str(model_path),
                "trained_at": datetime.now().isoformat(),
                "class": model_class.__name__
            }

            print(f"    Acc: {accuracy:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f if auc_score is not None else 'N/A'}")
        # Save scaler and PCA
        with open(MODELS_DIR / "scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)

        with open(MODELS_DIR / "pca.pkl", 'wb') as f:
            pickle.dump(pca, f)

        # Save model info with enhanced metrics
        with open(MODELS_DIR / "models_info.json", 'w') as f:
            json.dump({
                "models": available_models,
                "last_trained": datetime.now().isoformat(),
                "feature_names": feature_names,
                "sample_features": sample_features,
                "dataset_info": {
                    "total_samples": len(X),
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "malignant_count": int(sum(y)),
                    "benign_count": int(len(y) - sum(y))
                },
                "pca_variance": {
                    "pc1": float(variance_exp[0] * 100),
                    "pc2": float(variance_exp[1] * 100),
                    "total": float(sum(variance_exp[:2]) * 100)
                }
            }, f, indent=2)

        # Create artifact system
        create_artifact_system()

        # Set default model (best by F1 score)
        if available_models:
            # Filter out linear regression for default selection
            candidate_models = {k: v for k, v in available_models.items() if k != "linear"}
            if candidate_models:
                best_model_id = max(candidate_models.items(), key=lambda x: x[1].get('f1', 0))[0]
                set_current_model(best_model_id)
                model_loaded = True

        print(f"✅ Training completed. {len(available_models)} models available.")
        return True

    except Exception as e:
        print(f"❌ Error training models: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_artifact_system():
    """Create organized artifact system"""
    try:
        # Create artifact directories
        (ARTIFACTS_DIR / "models").mkdir(exist_ok=True, parents=True)
        (ARTIFACTS_DIR / "metrics").mkdir(exist_ok=True, parents=True)
        (ARTIFACTS_DIR / "features").mkdir(exist_ok=True, parents=True)
        (ARTIFACTS_DIR / "visualizations").mkdir(exist_ok=True, parents=True)

        # Save feature names
        pd.Series(feature_names).to_csv(
            ARTIFACTS_DIR / "features" / "feature_names.csv",
            index=False,
            header=False
        )

        # Save model registry
        model_registry = {}
        for model_id, info in available_models.items():
            model_file = f"model_{model_id}.joblib"
            model_path = ARTIFACTS_DIR / "models" / model_file

            # Load and save with joblib
            with open(info['path'], 'rb') as f:
                model = pickle.load(f)
            joblib.dump(model, model_path)

            model_registry[model_id] = {
                "name": info['name'],
                "file": model_file,
                "metrics": {
                    "accuracy": info['accuracy'],
                    "recall": info['recall'],
                    "precision": info['precision'],
                    "f1": info['f1'],
                    "auc": info['auc']
                },
                "trained_at": info['trained_at']
            }

        # Save registry
        with open(ARTIFACTS_DIR / "models_registry.json", 'w') as f:
            json.dump(model_registry, f, indent=2)

        # Save metrics
        with open(ARTIFACTS_DIR / "metrics" / "model_metrics.json", 'w') as f:
            json.dump(available_models, f, indent=2)

        # Save dataset info
        dataset_info = {
            "total_samples": len(X_train) + len(X_test) if X_train is not None else 0,
            "train_samples": len(X_train) if X_train is not None else 0,
            "test_samples": len(X_test) if X_test is not None else 0,
            "malignant_count": int(sum(y_train) + sum(y_test)) if y_train is not None else 0,
            "benign_count": int((len(y_train) + len(y_test)) - (sum(y_train) + sum(y_test))) if y_train is not None else 0,
            "feature_count": len(feature_names)
        }
        with open(ARTIFACTS_DIR / "metrics" / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)

        print(f"📦 Artifacts saved to {ARTIFACTS_DIR}")

    except Exception as e:
        print(f"⚠️ Artifact system creation failed: {e}")

def load_existing_models():
    """Load previously trained models with compatibility handling"""
    global scaler, available_models, model_loaded, pca, feature_names, sample_features, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X_train_pca

    info_path = MODELS_DIR / "models_info.json"
    scaler_path = MODELS_DIR / "scaler.pkl"
    pca_path = MODELS_DIR / "pca.pkl"

    if info_path.exists() and scaler_path.exists() and pca_path.exists():
        try:
            # Load model info
            with open(info_path, 'r') as f:
                data = json.load(f)
                available_models = data.get('models', {})
                feature_names = data.get('feature_names', [])
                sample_features = data.get('sample_features', [])

            # Load scaler and PCA
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            with open(pca_path, 'rb') as f:
                pca = pickle.load(f)

            # Load and prepare data for PCA
            X, y = load_or_create_data()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train_pca = pca.transform(X_train_scaled)

            # Add missing fields to old models for compatibility
            for model_id, model_info in available_models.items():
                # Ensure all expected fields exist
                model_info.setdefault('precision', model_info.get('accuracy', 0))
                model_info.setdefault('f1', model_info.get('accuracy', 0))
                model_info.setdefault('selectivity', 0.5)
                model_info.setdefault('fall_out', 0.5)
                model_info.setdefault('auc', None)
                model_info.setdefault('class', 'Unknown')

            # Set default model with compatibility handling
            if available_models:
                # Try to find best model by F1, fallback to accuracy
                try:
                    # Filter out models without 'f1' field (old models)
                    models_with_f1 = {k: v for k, v in available_models.items() if 'f1' in v and v.get('f1', 0) > 0}
                    if models_with_f1:
                        best_model_id = max(models_with_f1.items(), key=lambda x: x[1].get('f1', 0))[0]
                    else:
                        best_model_id = max(available_models.items(), key=lambda x: x[1].get('accuracy', 0))[0]
                except:
                    # Ultimate fallback
                    best_model_id = list(available_models.keys())[0]
                
                set_current_model(best_model_id)
                model_loaded = True

                print(f"✅ Loaded {len(available_models)} existing models")
                return True

        except Exception as e:
            print(f"❌ Error loading existing models: {e}")
            import traceback
            traceback.print_exc()

    return False

def set_current_model(model_id: str):
    """Set the current active model"""
    global current_model, model_info, shap_explainer

    if model_id in available_models:
        try:
            model_path = available_models[model_id]['path']
            with open(model_path, 'rb') as f:
                current_model = pickle.load(f)

            model_info = available_models[model_id]

            # Update metrics with fallbacks
            model_accuracy_gauge.set(model_info.get('accuracy', 0))
            model_recall_gauge.set(model_info.get('recall', 0))
            model_precision_gauge.set(model_info.get('precision', model_info.get('accuracy', 0)))
            model_f1_gauge.set(model_info.get('f1', model_info.get('accuracy', 0)))
            model_auc_gauge.set(model_info.get('auc', 0) or 0)

            # Initialize SHAP for this model
            initialize_shap_explainer()

            # Print with fallback for F1
            f1_display = model_info.get('f1', 0)
            f1_str = f"{f1_display:.4f}" if isinstance(f1_display, (int, float)) else "N/A"
            
            print(f"✅ Active model: {model_info['name']} (Acc: {model_info.get('accuracy', 0):.4f}, F1: {f1_str})")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    else:
        print(f"❌ Model ID not found: {model_id}")
        return False

def generate_pca_plot():
    """Generate PCA visualization plot"""
    try:
        if X_train_scaled is None or pca is None or X_train_pca is None:
            return None

        # Create plot
        plt.figure(figsize=(12, 10))

        # Plot benign
        benign_mask = y_train == 0
        plt.scatter(X_train_pca[benign_mask, 0], X_train_pca[benign_mask, 1],
                   alpha=0.7, c='green', label='Benign', s=50)

        # Plot malignant
        malignant_mask = y_train == 1
        plt.scatter(X_train_pca[malignant_mask, 0], X_train_pca[malignant_mask, 1],
                   alpha=0.7, c='red', label='Malignant', s=50)

        # Add labels and title
        variance_exp = pca.explained_variance_ratio_
        plt.xlabel(f'PC1 ({variance_exp[0]*100:.1f}% variance)', fontsize=12)
        plt.ylabel(f'PC2 ({variance_exp[1]*100:.1f}% variance)', fontsize=12)
        plt.title(f'PCA Visualization - Total Variance: {(variance_exp[0]+variance_exp[1])*100:.1f}%', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        # Add info box
        info_text = f"Dataset: {len(X_train)} samples\n"
        info_text += f"Features: {len(feature_names)}\n"
        info_text += f"Benign: {sum(y_train==0)}, Malignant: {sum(y_train==1)}"
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Save to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        return img_base64

    except Exception as e:
        print(f"❌ Error generating PCA plot: {e}")
        return None

def generate_roc_plot():
    """Generate ROC curve plot for current model"""
    try:
        if current_model is None or X_test_scaled is None or y_test is None:
            return None

        # Get probability scores
        if hasattr(current_model, 'predict_proba'):
            y_proba = current_model.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(current_model, 'decision_function'):
            y_proba = current_model.decision_function(X_test_scaled)
        else:
            return None

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        # Create plot
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_info.get("name", "Current Model")}', fontsize=14)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)

        # Save to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        return {
            "plot": img_base64,
            "auc": float(roc_auc),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        }

    except Exception as e:
        print(f"❌ Error generating ROC plot: {e}")
        return None

def initialize_shap_explainer():
    """Initialize SHAP explainer for the current model"""
    global shap_explainer, feature_names_shap, X_train_shap

    try:
        if current_model is None or X_train_scaled is None:
            print("⚠️ Cannot initialize SHAP: model or training data not available")
            return False

        # Use existing feature names
        feature_names_shap = feature_names

        # Store training data for SHAP (use subset for performance)
        sample_size = min(100, len(X_train_scaled))
        X_train_shap = X_train_scaled[:sample_size]

        # Determine model type and choose appropriate explainer
        model_type = type(current_model).__name__.lower()

        print(f"🔍 Initializing SHAP for {model_type}...")

        try:
            if 'randomforest' in model_type or 'forest' in model_type:
                shap_explainer = shap.TreeExplainer(current_model)
            elif 'logistic' in model_type or 'linear' in model_type:
                shap_explainer = shap.LinearExplainer(current_model, X_train_shap)
            else:
                # Default for other models
                shap_explainer = shap.KernelExplainer(current_model.predict_proba, X_train_shap)
            
            print(f"✅ SHAP explainer initialized for {model_type}")
            return True
        except:
            # Fallback explainer
            shap_explainer = shap.KernelExplainer(current_model.predict_proba, X_train_shap)
            print(f"✅ SHAP KernelExplainer (fallback) initialized for {model_type}")
            return True

    except Exception as e:
        print(f"❌ Error initializing SHAP: {e}")
        shap_explainer = None
        return False

# Setup Prometheus
instrumentator = Instrumentator()
instrumentator.instrument(app)
instrumentator.expose(app)

# Startup event
@app.on_event("startup")
async def startup_event():
    print("🚀 Starting Breast Cancer Detection API (Enhanced Version 5.0.0)...")
    print("=" * 60)
    print("📋 Features:")
    print("  • 8 ML models (MLP, k-NN, Random Forest, SVM, Logistic, Linear)")
    print("  • ROC curve visualization")
    print("  • Artifact system with model registry")
    print("  • Business Objectives tracking")
    print("  • PCA with variance analysis")
    print("  • SHAP explainability")
    print("=" * 60)

    # Try to load existing models first
    if not load_existing_models():
        # If no models exist, train new ones
        print("📊 No existing models found. Training new models...")
        train_all_models()

    # Get F1 score for display (with proper error handling)
    f1_value = model_info.get('f1', 0)
    if isinstance(f1_value, (int, float)):
        f1_display = f"{f1_value*100:.1f}%"
    else:
        f1_display = "N/A"

    print("=" * 60)
    print("✅ API is ready!")
    print(f"📊 Active model: {model_info.get('name', 'None')}")
    print(f"🎯 Accuracy: {model_info.get('accuracy', 0)*100:.1f}%")
    print(f"📈 F1 Score: {f1_display}")
    print(f"🔢 Features: {len(feature_names)}")
    print("=" * 60)

# ============ API ENDPOINTS ============

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the HTML dashboard"""
    pca_plot = generate_pca_plot()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "feature_names": feature_names,
        "sample_features": sample_features,
        "model_info": model_info,
        "available_models": available_models,
        "total_features": len(feature_names),
        "model_loaded": model_loaded,
        "pca_plot": pca_plot,
        "business_objectives": BUSINESS_OBJECTIVES,
        "data_science_objectives": DATA_SCIENCE_OBJECTIVES,
        "total_models": len(MODEL_CONFIGS)
    })

@app.get("/health")
async def health():
    """Health check endpoint"""
    uptime = int(time.time() - start_time)

    # Handle F1 score safely
    model_f1 = model_info.get('f1', 0.0)
    if not isinstance(model_f1, (int, float)):
        model_f1 = 0.0

    return {
        "status": "healthy",
        "version": "5.0.0",
        "model_loaded": model_loaded,
        "active_model": model_info.get('name', 'None'),
        "model_accuracy": model_info.get('accuracy', 0.0),
        "model_recall": model_info.get('recall', 0.0),
        "model_precision": model_info.get('precision', 0.0),
        "model_f1": model_f1,
        "total_models": len(available_models),
        "features_count": len(feature_names),
        "uptime_seconds": uptime,
        "predictions_made": predictions_counter._value.get() if hasattr(predictions_counter, '_value') else 0,
        "timestamp": datetime.now().isoformat(),
        "artifact_system": str(ARTIFACTS_DIR)
    }

@app.get("/model/info")
async def get_model_info():
    """Get current model information"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="No model loaded")

    return model_info

@app.get("/models")
async def get_models():
    """Get all available models with enhanced metrics"""
    # Calculate best models by different metrics
    best_by_metric = {}
    if available_models:
        # Filter out linear regression for metric comparisons
        candidate_models = {k: v for k, v in available_models.items() if k != "linear"}
        if candidate_models:
            best_by_metric = {
                "accuracy": max(candidate_models.items(), key=lambda x: x[1].get('accuracy', 0))[0],
                "recall": max(candidate_models.items(), key=lambda x: x[1].get('recall', 0))[0],
                "f1": max(candidate_models.items(), key=lambda x: x[1].get('f1', 0))[0],
                "auc": max([(k, v) for k, v in candidate_models.items() if v.get('auc')], 
                          key=lambda x: x[1].get('auc', 0))[0] if any(v.get('auc') for v in candidate_models.values()) else None
            }

    return {
        "available_models": available_models,
        "total_models": len(available_models),
        "expected_models": len(MODEL_CONFIGS),
        "current_model": model_info.get('id', 'None'),
        "model_loaded": model_loaded,
        "best_models": best_by_metric
    }

@app.get("/features")
async def get_features():
    """Get list of features"""
    return {
        "count": len(feature_names),
        "features": feature_names,
        "sample_values": sample_features,
        "feature_groups": {
            "mean_features": [f for f in feature_names if '_mean' in f],
            "se_features": [f for f in feature_names if '_se' in f],
            "worst_features": [f for f in feature_names if '_worst' in f]
        }
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make a prediction"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.features) != len(feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(feature_names)} features, got {len(request.features)}"
        )

    try:
        # Switch model if requested
        if request.model_id != model_info.get('id'):
            if not set_current_model(request.model_id):
                raise HTTPException(status_code=400, detail=f"Model {request.model_id} not found")

        # Prepare features
        features_array = np.array(request.features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        # Make prediction (special handling for Linear Regression)
        if request.model_id == "linear":
            prediction_raw = current_model.predict(features_scaled)[0]
            prediction = 1 if prediction_raw >= 0.5 else 0
            probability = [1 - prediction_raw, prediction_raw] if 0 <= prediction_raw <= 1 else [0.5, 0.5]
        else:
            prediction = current_model.predict(features_scaled)[0]
            if hasattr(current_model, 'predict_proba'):
                probability = current_model.predict_proba(features_scaled)[0]
            else:
                probability = [1 - prediction, prediction] if prediction in [0, 1] else [0.5, 0.5]

        # Increment counter
        predictions_counter.inc()

        # Generate PCA coordinates for this prediction
        pca_coords = None
        if pca is not None and X_train_pca is not None:
            pca_features = pca.transform(features_scaled)[0]
            
            # Calculate distances to cluster centers
            benign_center = np.mean(X_train_pca[y_train == 0], axis=0) if sum(y_train == 0) > 0 else np.zeros(2)
            malignant_center = np.mean(X_train_pca[y_train == 1], axis=0) if sum(y_train == 1) > 0 else np.zeros(2)
            
            pca_coords = {
                "pc1": float(pca_features[0]),
                "pc2": float(pca_features[1]),
                "distance_to_benign_center": float(np.linalg.norm(pca_features - benign_center)),
                "distance_to_malignant_center": float(np.linalg.norm(pca_features - malignant_center))
            }

        # Identify high-risk features
        high_risk_features = []
        if X_train is not None:
            for i, feature_name in enumerate(feature_names):
                if i < X_train.shape[1]:
                    percentile = np.percentile(X_train[:, i], 75)
                    if request.features[i] > percentile and feature_name in ['radius_worst', 'area_worst', 'concave_points_worst']:
                        high_risk_features.append({
                            "name": feature_name,
                            "value": float(request.features[i]),
                            "percentile": float(percentile),
                            "above_percentile": float(request.features[i] - percentile)
                        })

        return {
            "success": True,
            "model_used": model_info,
            "prediction": {
                "value": int(prediction),
                "class": "MALIGNANT" if prediction == 1 else "BENIGN",
                "confidence": float(max(probability)),
                "probabilities": {
                    "benign": float(probability[0]),
                    "malignant": float(probability[1])
                },
                "risk_factors": {
                    "high_risk_features": high_risk_features,
                    "risk_level": "HIGH" if prediction == 1 and max(probability) > 0.8 else "MEDIUM" if prediction == 1 else "LOW"
                }
            },
            "pca_coordinates": pca_coords,
            "timestamp": datetime.now().isoformat(),
            "business_impact": {
                "supports_bo1": prediction == 1,  # Early detection
                "supports_bo2": True,  # Faster diagnosis
                "confidence_level": "HIGH" if max(probability) > 0.9 else "MEDIUM" if max(probability) > 0.7 else "LOW",
                "recommended_action": "Immediate follow-up" if prediction == 1 and max(probability) > 0.8 else "Routine monitoring"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/switch-model")
async def switch_model(request: ModelSwitchRequest):
    """Switch to a different model"""
    try:
        if set_current_model(request.model_id):
            return {
                "success": True,
                "message": f"Switched to {model_info['name']}",
                "model_info": model_info,
                "performance_summary": {
                    "accuracy": f"{model_info.get('accuracy', 0)*100:.1f}%",
                    "recall": f"{model_info.get('recall', 0)*100:.1f}%",
                    "precision": f"{model_info.get('precision', 0)*100:.1f}%",
                    "f1": f"{model_info.get('f1', 0)*100:.1f}%" if isinstance(model_info.get('f1', 0), (int, float)) else "N/A",
                    "auc": f"{model_info.get('auc', 0)*100:.1f}%" if model_info.get('auc') else "N/A"
                }
            }
        else:
            raise HTTPException(status_code=400, detail=f"Model {request.model_id} not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain():
    """Retrain all models with enhanced configurations"""
    print("🔄 Manual retraining triggered via API...")
    try:
        if train_all_models():
            return {
                "success": True,
                "message": "All models retrained successfully with enhanced configurations",
                "models": available_models,
                "current_model": model_info,
                "artifact_location": str(ARTIFACTS_DIR),
                "training_time": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Training failed")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pca/plot")
async def get_pca_plot():
    """Get PCA plot as base64"""
    pca_plot = generate_pca_plot()
    if not pca_plot:
        raise HTTPException(status_code=404, detail="PCA plot not available")

    variance_exp = pca.explained_variance_ratio_ if pca else [0, 0]

    return {
        "plot": pca_plot,
        "format": "png",
        "variance_explained": {
            "pc1": float(variance_exp[0] * 100),
            "pc2": float(variance_exp[1] * 100),
            "total": float(sum(variance_exp[:2]) * 100)
        },
        "dataset_info": {
            "total_samples": len(X_train) if X_train is not None else 0,
            "benign_count": int(sum(y_train == 0)) if y_train is not None else 0,
            "malignant_count": int(sum(y_train == 1)) if y_train is not None else 0
        }
    }

@app.get("/analysis/roc")
async def get_roc_analysis():
    """Get ROC curve analysis for current model"""
    try:
        roc_data = generate_roc_plot()
        if not roc_data:
            raise HTTPException(status_code=404, detail="ROC analysis not available for this model")

        # Interpretation based on AUC
        auc_value = roc_data['auc']
        if auc_value >= 0.9:
            interpretation = "Excellent diagnostic performance"
        elif auc_value >= 0.8:
            interpretation = "Good diagnostic performance"
        elif auc_value >= 0.7:
            interpretation = "Fair diagnostic performance"
        else:
            interpretation = "Poor diagnostic performance"

        return {
            "success": True,
            "model": model_info.get('name', ''),
            "plot": roc_data['plot'],
            "auc": roc_data['auc'],
            "interpretation": interpretation,
            "clinical_significance": "AUC measures the model's ability to distinguish between malignant and benign cases",
            "quality": {
                "excellent": auc_value >= 0.9,
                "good": auc_value >= 0.8,
                "fair": auc_value >= 0.7,
                "poor": auc_value < 0.7
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(REGISTRY)

@app.get("/analysis", response_class=HTMLResponse)
async def analysis_dashboard(request: Request):
    """Serve the analysis dashboard"""
    return templates.TemplateResponse("analysis.html", {
        "request": request,
        "model_info": model_info,
        "model_loaded": model_loaded,
        "total_models": len(available_models),
        "business_objectives": BUSINESS_OBJECTIVES,
        "data_science_objectives": DATA_SCIENCE_OBJECTIVES,
        "artifact_location": str(ARTIFACTS_DIR)
    })

@app.get("/analysis/metrics")
async def get_analysis_metrics():
    """Get metrics for analysis dashboard"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Calculate objective achievement
    ds_objective_achievement = {}
    if available_models:
        # Filter out linear regression
        candidate_models = [m for m in available_models.values() if m.get('id') != 'linear']
        
        if candidate_models:
            ds_objective_achievement = {
                "dso1_high_recall": any(m.get('recall', 0) >= 0.95 for m in candidate_models),
                "dso2_high_selectivity": any(m.get('selectivity', 0) >= 0.90 for m in candidate_models),
                "dso4_high_accuracy": any(m.get('accuracy', 0) >= 0.95 for m in candidate_models)
            }

    # Enhanced metrics from notebook
    metrics_data = {
        "kpis": {
            "accuracy": model_info.get('accuracy', 0.0) * 100,
            "recall": model_info.get('recall', 0.0) * 100,
            "precision": model_info.get('precision', 0.0) * 100,
            "f1": model_info.get('f1', 0.0) * 100 if isinstance(model_info.get('f1', 0), (int, float)) else None,
            "selectivity": model_info.get('selectivity', 0.0) * 100,
            "auc": model_info.get('auc', 0.0) * 100 if model_info.get('auc') else None,
            "total_models": len(available_models),
            "features_count": len(feature_names),
            "predictions_made": predictions_counter._value.get() if hasattr(predictions_counter, '_value') else 0,
            "uptime_minutes": int((time.time() - start_time) / 60),
            "dataset_size": len(X_train) + len(X_test) if X_train is not None else 0
        },
        "business_objectives": BUSINESS_OBJECTIVES,
        "data_science_objectives": DATA_SCIENCE_OBJECTIVES,
        "model_performance": available_models,
        "ds_objective_achievement": ds_objective_achievement,
        "best_models": {
            "by_accuracy": max([(k, v) for k, v in available_models.items() if k != "linear"], 
                              key=lambda x: x[1].get('accuracy', 0))[0] if len(available_models) > 1 else None,
            "by_recall": max([(k, v) for k, v in available_models.items() if k != "linear"], 
                            key=lambda x: x[1].get('recall', 0))[0] if len(available_models) > 1 else None,
            "by_f1": max([(k, v) for k, v in available_models.items() if k != "linear" and isinstance(v.get('f1'), (int, float))], 
                        key=lambda x: x[1].get('f1', 0))[0] if len(available_models) > 1 else None
        }
    }

    return metrics_data

@app.post("/shap/explain")
async def get_shap_explanation(request: PredictionRequest):
    """Generate SHAP explanation for a prediction"""
    try:
        if shap_explainer is None:
            return {
                "success": False,
                "available": False,
                "message": "SHAP explainer not initialized. Please make a prediction first or retrain models.",
                "simulation": True,
                "top_features": [
                    {"name": "radius_worst", "importance": 0.85, "impact": "positive", "value": "increases risk"},
                    {"name": "concave_points_worst", "importance": 0.72, "impact": "positive", "value": "increases risk"},
                    {"name": "perimeter_worst", "importance": 0.68, "impact": "positive", "value": "increases risk"},
                    {"name": "area_worst", "importance": 0.64, "impact": "positive", "value": "increases risk"}
                ]
            }

        # Prepare features
        features_array = np.array(request.features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        # Get SHAP values
        shap_values = shap_explainer.shap_values(features_scaled)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            if len(shap_values) > 1:
                shap_vals = np.array(shap_values[1]).flatten()
                if isinstance(shap_explainer.expected_value, (list, np.ndarray)):
                    expected_value = float(shap_explainer.expected_value[1]) if len(shap_explainer.expected_value) > 1 else float(shap_explainer.expected_value[0])
                else:
                    expected_value = float(shap_explainer.expected_value)
            else:
                shap_vals = np.array(shap_values[0]).flatten()
                expected_value = float(shap_explainer.expected_value)
        else:
            shap_vals = np.array(shap_values).flatten()
            expected_value = float(shap_explainer.expected_value)

        # Ensure proper length
        if len(shap_vals) > len(feature_names):
            shap_vals = shap_vals[:len(feature_names)]
        elif len(shap_vals) < len(feature_names):
            shap_vals = np.pad(shap_vals, (0, len(feature_names) - len(shap_vals)))

        # Prepare feature contributions
        contributions = []
        for i, (feature_name, value, shap_val) in enumerate(zip(feature_names, request.features, shap_vals)):
            # Calculate percentile if training data available
            percentile = None
            if X_train is not None and i < X_train.shape[1]:
                try:
                    percentile = float(np.percentile(X_train[:, i], value))
                except:
                    percentile = 50.0
            
            contributions.append({
                "name": feature_name,
                "value": float(value),
                "importance": float(abs(shap_val)),
                "impact": float(shap_val),
                "direction": "increases risk" if shap_val > 0 else "decreases risk",
                "percentile": percentile,
                "clinical_interpretation": "High risk factor" if shap_val > 0.1 else "Moderate risk factor" if shap_val > 0 else "Protective factor"
            })

        # Sort by importance
        contributions.sort(key=lambda x: x["importance"], reverse=True)

        # Generate SHAP force plot
        plot_data = None
        try:
            # Create force plot
            plt.figure(figsize=(10, 4))
            
            # Handle base value
            base_value = expected_value
            if hasattr(base_value, '__len__'):
                if len(base_value) > 1:
                    base_value = float(base_value[1])
                else:
                    base_value = float(base_value[0]) if len(base_value) > 0 else 0.0
            
            shap.force_plot(
                base_value,
                shap_vals,
                features_scaled[0],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            
            # Convert to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            
            plot_data = {
                "type": "force_plot",
                "data": plot_base64
            }
        except Exception as plot_error:
            print(f"⚠️ SHAP plot generation failed: {plot_error}")

        return {
            "success": True,
            "available": True,
            "base_value": float(expected_value),
            "top_features": contributions[:10],
            "plot": plot_data,
            "prediction_value": float(expected_value + sum(shap_vals)),
            "clinical_interpretation": {
                "high_risk_factors": [c for c in contributions[:5] if c["impact"] > 0],
                "protective_factors": [c for c in contributions[:5] if c["impact"] < 0],
                "summary": f"Prediction influenced by {len([c for c in contributions if c['impact'] > 0])} risk-increasing factors and {len([c for c in contributions if c['impact'] < 0])} protective factors."
            }
        }

    except Exception as e:
        print(f"❌ SHAP explanation error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/artifacts/info")
async def get_artifacts_info():
    """Get information about saved artifacts"""
    try:
        registry_path = ARTIFACTS_DIR / "models_registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry = json.load(f)
            
            # Count artifacts
            artifact_count = sum(1 for _ in ARTIFACTS_DIR.rglob('*') if _.is_file())
            
            return {
                "success": True,
                "artifact_count": artifact_count,
                "models_registered": len(registry),
                "models": list(registry.keys()),
                "artifact_directory": str(ARTIFACTS_DIR),
                "last_updated": datetime.fromtimestamp(registry_path.stat().st_mtime).isoformat(),
                "subdirectories": [d.name for d in ARTIFACTS_DIR.iterdir() if d.is_dir()]
            }
        else:
            return {
                "success": False,
                "message": "Artifacts not generated yet. Run /retrain first.",
                "artifact_directory": str(ARTIFACTS_DIR)
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/objectives/status")
async def get_objectives_status():
    """Get status of Business and Data Science Objectives"""
    if not model_loaded or not available_models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Filter out linear regression
    candidate_models = [m for m in available_models.values() if m.get('id') != 'linear']
    
    if not candidate_models:
        raise HTTPException(status_code=503, detail="No valid models available")
    
    # Calculate objective achievement
    bo_status = {}
    for bo_id, bo_info in BUSINESS_OBJECTIVES.items():
        target = bo_info.get('target', 0)
        
        # Different logic for different business objectives
        if bo_id == "bo1_early_detection":
            # Based on recall of best model
            best_recall = max(m.get('recall', 0) * 100 for m in candidate_models)
            achieved = best_recall >= target
            bo_status[bo_id] = {
                "name": bo_info['name'],
                "target": target,
                "achieved": achieved,
                "current_value": float(best_recall),
                "gap": float(target - best_recall) if not achieved else 0,
                "primary_model": max(candidate_models, key=lambda x: x.get('recall', 0))['name']
            }
        elif bo_id == "bo4_prioritize_urgent":
            # Based on precision of best model
            best_precision = max(m.get('precision', 0) * 100 for m in candidate_models)
            achieved = best_precision >= target
            bo_status[bo_id] = {
                "name": bo_info['name'],
                "target": target,
                "achieved": achieved,
                "current_value": float(best_precision),
                "gap": float(target - best_precision) if not achieved else 0,
                "primary_model": max(candidate_models, key=lambda x: x.get('precision', 0))['name']
            }
        else:
            # Default to accuracy
            best_accuracy = max(m.get('accuracy', 0) * 100 for m in candidate_models)
            achieved = best_accuracy >= target
            bo_status[bo_id] = {
                "name": bo_info['name'],
                "target": target,
                "achieved": achieved,
                "current_value": float(best_accuracy),
                "gap": float(target - best_accuracy) if not achieved else 0,
                "primary_model": max(candidate_models, key=lambda x: x.get('accuracy', 0))['name']
            }
    
    dso_status = {}
    for dso_id, dso_info in DATA_SCIENCE_OBJECTIVES.items():
        metric_name = dso_info.get('metric', '')
        target_str = dso_info.get('target', '')
        
        # Parse target value
        target_value = 0
        if '≥' in target_str:
            target_value = float(target_str.split('≥')[1].replace('%', '').strip())
        
        # Find best model for this metric
        if metric_name == 'recall':
            best_value = max(m.get('recall', 0) * 100 for m in candidate_models)
            best_model = max(candidate_models, key=lambda x: x.get('recall', 0))
        elif metric_name == 'selectivity':
            best_value = max(m.get('selectivity', 0) * 100 for m in candidate_models)
            best_model = max(candidate_models, key=lambda x: x.get('selectivity', 0))
        elif metric_name == 'accuracy':
            best_value = max(m.get('accuracy', 0) * 100 for m in candidate_models)
            best_model = max(candidate_models, key=lambda x: x.get('accuracy', 0))
        else:
            best_value = 0
            best_model = candidate_models[0]
        
        achieved = best_value >= target_value if target_value > 0 else True
        
        dso_status[dso_id] = {
            "name": dso_info['name'],
            "target": target_str,
            "target_value": target_value,
            "achieved": achieved,
            "current_value": float(best_value),
            "best_model": best_model['name'],
            "gap": float(target_value - best_value) if not achieved else 0
        }
    
    return {
        "business_objectives": bo_status,
        "data_science_objectives": dso_status,
        "overall_achievement": {
            "business": sum(1 for bo in bo_status.values() if bo['achieved']) / len(bo_status) * 100,
            "data_science": sum(1 for dso in dso_status.values() if dso['achieved']) / len(dso_status) * 100
        }
    }

@app.get("/analysis/shap")
async def generate_shap_analysis_legacy():
    """Generate SHAP analysis (legacy endpoint)"""
    return {
        "available": False,
        "message": "Use POST /shap/explain for real SHAP explanations",
        "top_features": [
            {"name": "radius_worst", "importance": 0.85},
            {"name": "concave_points_worst", "importance": 0.72},
            {"name": "perimeter_worst", "importance": 0.68},
            {"name": "area_worst", "importance": 0.64}
        ],
        "base_value": 0.2345
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
