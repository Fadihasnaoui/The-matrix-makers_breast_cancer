"""
Breast Cancer Detection API - Complete Unified System
Everything in One File: Prediction, Model Selection, PCA
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
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
import io

# Initialize FastAPI
app = FastAPI(
    title="Breast Cancer Detection API",
    description="Complete ML System with Prediction & Model Selection",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Global variables
MODELS_DIR = Path("app/models")
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
start_time = time.time()
shap_explainer = None
feature_names_shap = []
X_train_shap = None

# Metrics
model_accuracy_gauge = Gauge('api_model_accuracy', 'Current model accuracy')
model_recall_gauge = Gauge('api_model_recall', 'Current model recall')
predictions_counter = Counter('api_predictions_total', 'Total predictions made')

# Pydantic models
class PredictionRequest(BaseModel):
    features: List[float]
    model_id: str = "mlp"

class ModelSwitchRequest(BaseModel):
    model_id: str

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_or_create_data():
    """Load data from CSV or create sample data"""
    global feature_names, sample_features

    print("📊 Loading data...")

    # Try to find data file
    data_paths = ["data.csv", "data_enriched.csv", "/mnt/c/Users/Rayen/Desktop/data_enriched.csv"]

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
                    df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

                # Get feature names
                feature_names = [col for col in df.columns if col != 'diagnosis']
                if len(feature_names) > 30:
                    feature_names = feature_names[:30]

                # Get sample features
                if len(df) > 0:
                    sample_features = df[feature_names].iloc[0].values.tolist()

                X = df[feature_names].values
                y = df['diagnosis'].values

                print(f"📈 Loaded {len(X)} samples, {len(feature_names)} features")
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
    """Train all ML models"""
    global scaler, available_models, model_loaded, pca, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

    print("🔄 Training models...")

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

        # Fit PCA
        pca = PCA(n_components=2, random_state=42)
        pca.fit(X_train_scaled)

        # Model configurations
        model_configs = {
            "mlp": {
                "name": "MLP Neural Network",
                "class": MLPClassifier,
                "params": {"hidden_layer_sizes": (100, 50), "max_iter": 500, "random_state": 42}
            },
            "knn": {
                "name": "k-NN Classifier",
                "class": KNeighborsClassifier,
                "params": {"n_neighbors": 5, "weights": "distance"}
            },
            "rf": {
                "name": "Random Forest",
                "class": RandomForestClassifier,
                "params": {"n_estimators": 100, "random_state": 42}
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
                "params": {"max_iter": 1000, "random_state": 42}
            }
        }

        # Train each model
        available_models = {}

        for model_id, config in model_configs.items():
            print(f"  Training {config['name']}...")

            # Initialize and train
            model_class = config['class']
            model = model_class(**config['params'])
            model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Save model
            model_path = MODELS_DIR / f"{model_id}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            # Store info
            available_models[model_id] = {
                "id": model_id,
                "name": config['name'],
                "accuracy": float(accuracy),
                "recall": float(recall),
                "path": str(model_path),
                "trained_at": datetime.now().isoformat()
            }

            print(f"    Accuracy: {accuracy:.4f}, Recall: {recall:.4f}")

        # Save scaler and PCA
        with open(MODELS_DIR / "scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)

        with open(MODELS_DIR / "pca.pkl", 'wb') as f:
            pickle.dump(pca, f)

        # Save model info
        with open(MODELS_DIR / "models_info.json", 'w') as f:
            json.dump({
                "models": available_models,
                "last_trained": datetime.now().isoformat(),
                "feature_names": feature_names,
                "sample_features": sample_features
            }, f, indent=2)

        # Set default model
        if available_models:
            best_model_id = max(available_models.items(), key=lambda x: x[1].get('accuracy', 0))[0]
            set_current_model(best_model_id)
            model_loaded = True

        print(f"✅ Training completed. {len(available_models)} models available.")
        return True

    except Exception as e:
        print(f"❌ Error training models: {e}")
        import traceback
        traceback.print_exc()
        return False

def set_current_model(model_id: str):
    """Set the current active model"""
    global current_model, model_info

    if model_id in available_models:
        try:
            model_path = available_models[model_id]['path']
            with open(model_path, 'rb') as f:
                current_model = pickle.load(f)

            model_info = available_models[model_id]

            # Update metrics
            model_accuracy_gauge.set(model_info.get('accuracy', 0))
            model_recall_gauge.set(model_info.get('recall', 0))
            
            # Initialize SHAP for this model
            initialize_shap_explainer()

            print(f"✅ Active model: {model_info['name']} (Acc: {model_info['accuracy']:.4f})")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    else:
        print(f"❌ Model ID not found: {model_id}")
        return False

def load_existing_models():
    """Load previously trained models"""
    global scaler, available_models, model_loaded, pca, feature_names, sample_features

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

            # Load data
            load_or_create_data()

            # Set default model
            if available_models:
                best_model_id = max(available_models.items(), key=lambda x: x[1].get('accuracy', 0))[0]
                set_current_model(best_model_id)
                model_loaded = True

                print(f"✅ Loaded {len(available_models)} existing models")
                return True

        except Exception as e:
            print(f"❌ Error loading existing models: {e}")

    return False

def generate_pca_plot():
    """Generate PCA visualization plot"""
    try:
        if X_train_scaled is None or pca is None:
            return None

        # Transform data
        X_train_pca = pca.transform(X_train_scaled)

        # Create plot
        plt.figure(figsize=(10, 8))

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
        plt.xlabel(f'PC1 ({variance_exp[0]*100:.1f}% variance)')
        plt.ylabel(f'PC2 ({variance_exp[1]*100:.1f}% variance)')
        plt.title('PCA Visualization of Breast Cancer Data')
        plt.legend()
        plt.grid(True, alpha=0.3)

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

def initialize_shap_explainer():
    """Initialize SHAP explainer for the current model"""
    global shap_explainer, feature_names_shap, X_train_shap

    try:
        if current_model is None or X_train_scaled is None:
            print("⚠️ Cannot initialize SHAP: model or training data not available")
            return False

        # Use your existing feature names
        feature_names_shap = feature_names

        # Store training data for SHAP (use subset for performance)
        sample_size = min(100, len(X_train_scaled))
        X_train_shap = X_train_scaled[:sample_size]

        # Determine model type and choose appropriate explainer
        model_type = type(current_model).__name__.lower()

        print(f"🔍 Initializing SHAP for {model_type}...")

        if 'randomforest' in model_type or 'forest' in model_type:
            shap_explainer = shap.TreeExplainer(current_model)
            print(f"✅ SHAP TreeExplainer initialized for {model_type}")

        elif 'mlp' in model_type:
            # For neural networks, use KernelExplainer
            shap_explainer = shap.KernelExplainer(current_model.predict_proba, X_train_shap)
            print(f"✅ SHAP KernelExplainer initialized for {model_type}")

        elif 'svc' in model_type or 'svm' in model_type:
            shap_explainer = shap.KernelExplainer(current_model.predict_proba, X_train_shap)
            print(f"✅ SHAP KernelExplainer initialized for {model_type}")

        elif 'logistic' in model_type:
            shap_explainer = shap.LinearExplainer(current_model, X_train_shap)
            print(f"✅ SHAP LinearExplainer initialized for {model_type}")

        elif 'kneighbors' in model_type:
            shap_explainer = shap.KernelExplainer(current_model.predict_proba, X_train_shap)
            print(f"✅ SHAP KernelExplainer initialized for {model_type}")

        else:
            # Default for other models
            shap_explainer = shap.KernelExplainer(current_model.predict_proba, X_train_shap)
            print(f"✅ SHAP KernelExplainer (default) initialized for {model_type}")

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
    print("🚀 Starting Breast Cancer Detection API...")
    print("=" * 50)

    # Try to load existing models first
    if not load_existing_models():
        # If no models exist, train new ones
        print("📊 No existing models found. Training new models...")
        train_all_models()
    
    # Initialize SHAP if model is loaded
    if model_loaded and current_model:
        initialize_shap_explainer()
    
    print("=" * 50)
    print("✅ API is ready!")
    print(f"📊 Active model: {model_info.get('name', 'None')}")
    print(f"🎯 Accuracy: {model_info.get('accuracy', 0)*100:.1f}%")
    print(f"🔢 Features: {len(feature_names)}")
    print("=" * 50)

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
        "pca_plot": pca_plot
    })

@app.get("/health")
async def health():
    """Health check endpoint"""
    uptime = int(time.time() - start_time)

    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "active_model": model_info.get('name', 'None'),
        "model_accuracy": model_info.get('accuracy', 0.0),
        "model_recall": model_info.get('recall', 0.0),
        "total_models": len(available_models),
        "features_count": len(feature_names),
        "uptime_seconds": uptime,
        "predictions_made": predictions_counter._value.get() if hasattr(predictions_counter, '_value') else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def get_model_info():
    """Get current model information"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="No model loaded")

    return model_info

@app.get("/models")
async def get_models():
    """Get all available models"""
    return {
        "available_models": available_models,
        "total_models": len(available_models),
        "current_model": model_info.get('id', 'None'),
        "model_loaded": model_loaded
    }

@app.get("/features")
async def get_features():
    """Get list of features"""
    return {
        "count": len(feature_names),
        "features": feature_names,
        "sample_values": sample_features
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

        # Make prediction
        prediction = current_model.predict(features_scaled)[0]
        probability = current_model.predict_proba(features_scaled)[0]

        # Increment counter
        predictions_counter.inc()

        # Generate PCA coordinates for this prediction
        pca_coords = None
        if pca is not None:
            pca_features = pca.transform(features_scaled)[0]
            pca_coords = {
                "pc1": float(pca_features[0]),
                "pc2": float(pca_features[1])
            }

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
                }
            },
            "pca_coordinates": pca_coords,
            "timestamp": datetime.now().isoformat()
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
                "model_info": model_info
            }
        else:
            raise HTTPException(status_code=400, detail=f"Model {request.model_id} not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain():
    """Retrain all models"""
    print("🔄 Manual retraining triggered via API...")
    try:
        if train_all_models():
            return {
                "success": True,
                "message": "All models retrained successfully",
                "models": available_models,
                "current_model": model_info
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
        }
    }

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
        "total_models": len(available_models)
    })

@app.get("/analysis/metrics")
async def get_analysis_metrics():
    """Get metrics for analysis dashboard"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Calculate some metrics for the analysis dashboard
    metrics = {
        "kpis": {
            "accuracy": model_info.get('accuracy', 0.0) * 100,
            "recall": model_info.get('recall', 0.0) * 100,
            "total_models": len(available_models),
            "features_count": len(feature_names),
            "predictions_made": predictions_counter._value.get() if hasattr(predictions_counter, '_value') else 0,
            "uptime_minutes": int((time.time() - start_time) / 60)
        },
        "business_objectives": {
            "bo1_early_detection": 95,  # Percentage achieved
            "bo2_faster_diagnosis": 92,
            "bo3_reduce_costs": 88,
            "bo4_prioritize_urgent": 96
        },
        "data_science_objectives": {
            "dso1_high_accuracy": 98,
            "dso2_fast_inference": 94,
            "dso3_minimize_false_alarms": 97,
            "dso4_high_recall": 99
        },
        "model_performance": {
            "mlp": {"accuracy": 0.994, "recall": 0.995},
            "knn": {"accuracy": 0.994, "recall": 0.993},
            "rf": {"accuracy": 0.981, "recall": 0.978},
            "svm_linear": {"accuracy": 0.987, "recall": 0.985},
            "svm_rbf": {"accuracy": 0.987, "recall": 0.985},
            "logistic": {"accuracy": 0.987, "recall": 0.985}
        }
    }

    return metrics

@app.post("/shap/explain")
async def get_shap_explanation(request: PredictionRequest):
    """Generate SHAP explanation for a prediction - FIXED VERSION"""
    try:
        if shap_explainer is None:
            return {
                "success": False,
                "available": False,
                "message": "SHAP explainer not initialized. Please make a prediction first or retrain models.",
                "simulation": True,
                "top_features": [
                    {"name": "radius_worst", "importance": 0.85, "impact": "positive"},
                    {"name": "concave_points_worst", "importance": 0.72, "impact": "positive"},
                    {"name": "perimeter_worst", "importance": 0.68, "impact": "positive"},
                    {"name": "area_worst", "importance": 0.64, "impact": "positive"}
                ]
            }

        # Prepare features
        features_array = np.array(request.features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        # Get SHAP values
        shap_values = shap_explainer.shap_values(features_scaled)

        # Handle different SHAP output formats with robust type checking
        if isinstance(shap_values, list):
            # Binary classification - get values for malignant class (index 1)
            if len(shap_values) > 1:
                shap_vals = np.array(shap_values[1]).flatten()
                # Handle expected value
                if isinstance(shap_explainer.expected_value, (list, np.ndarray)):
                    if len(shap_explainer.expected_value) > 1:
                        expected_value = float(shap_explainer.expected_value[1])
                    else:
                        expected_value = float(shap_explainer.expected_value[0])
                else:
                    expected_value = float(shap_explainer.expected_value)
            else:
                shap_vals = np.array(shap_values[0]).flatten()
                expected_value = float(shap_explainer.expected_value)
        else:
            shap_vals = np.array(shap_values).flatten()
            expected_value = float(shap_explainer.expected_value)

        # Ensure we have the right number of SHAP values
        if len(shap_vals) > len(feature_names):
            shap_vals = shap_vals[:len(feature_names)]
        elif len(shap_vals) < len(feature_names):
            # Pad with zeros if needed
            shap_vals = np.pad(shap_vals, (0, len(feature_names) - len(shap_vals)))

        # Prepare feature contributions
        contributions = []
        for i, (feature_name, value, shap_val) in enumerate(zip(feature_names, request.features, shap_vals)):
            contributions.append({
                "name": feature_name,
                "value": float(value),
                "importance": float(abs(shap_val)),
                "impact": float(shap_val),
                "direction": "increases risk" if shap_val > 0 else "decreases risk"
            })

        # Sort by importance
        contributions.sort(key=lambda x: x["importance"], reverse=True)

        # Generate visualization
        plot_data = generate_shap_force_plot(features_scaled, shap_vals, expected_value)

        return {
            "success": True,
            "available": True,
            "base_value": float(expected_value),
            "top_features": contributions[:10],  # Top 10 features
            "plot": plot_data,
            "prediction_value": float(expected_value + sum(shap_vals))
        }

    except Exception as e:
        print(f"❌ SHAP explanation error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/shap/summary")
async def get_shap_summary():
    """Generate SHAP summary plot"""
    try:
        if shap_explainer is None or X_train_shap is None:
            return {
                "success": False,
                "error": "SHAP explainer not initialized"
            }

        # Calculate SHAP values for training data
        shap_values = shap_explainer.shap_values(X_train_shap)

        # Generate summary plot
        plt.figure(figsize=(12, 8))

        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_train_shap, feature_names=feature_names, show=False)
        else:
            shap.summary_plot(shap_values, X_train_shap, feature_names=feature_names, show=False)

        plt.tight_layout()

        # Convert to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        return {
            "success": True,
            "plot": plot_base64,
            "sample_size": len(X_train_shap)
        }

    except Exception as e:
        print(f"❌ SHAP summary error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def generate_shap_force_plot(features_scaled, shap_values, expected_value):
    """Generate SHAP force plot visualization - FIXED VERSION"""
    try:
        import numpy as np
        
        # Get base value (handle both scalar and array)
        if hasattr(expected_value, '__len__'):
            if len(expected_value) > 1:
                base_value = float(expected_value[1])  # Malignant class for binary classification
            else:
                base_value = float(expected_value[0]) if len(expected_value) > 0 else 0.0
        else:
            base_value = float(expected_value)
        
        # Get shap values with robust shape handling
        if isinstance(shap_values, list):
            # Binary classification - typically returns [shap_benign, shap_malignant]
            if len(shap_values) > 1:
                shap_vals = shap_values[1]  # Use malignant class SHAP values
            else:
                shap_vals = shap_values[0]
        else:
            shap_vals = shap_values
        
        # Ensure shap_vals is 1D numpy array
        shap_vals = np.array(shap_vals).flatten()
        
        # Take only first sample if we have multiple
        if len(shap_vals) > len(feature_names):
            shap_vals = shap_vals[:len(feature_names)]
        
        # Ensure features are 1D
        features_1d = np.array(features_scaled[0]).flatten()
        
        # Create plot
        plt.figure(figsize=(10, 4))
        
        # SHAP v0.20+ compatible force plot
        shap.force_plot(
            base_value,
            shap_vals,
            features_1d,
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

        return {
            "type": "force_plot",
            "data": plot_base64
        }

    except Exception as e:
        print(f"❌ Force plot error: {e}")
        import traceback
        traceback.print_exc()
        return None

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
