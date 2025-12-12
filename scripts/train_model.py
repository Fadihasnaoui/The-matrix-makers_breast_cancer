"""
Train and save the breast cancer model
"""
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os

def train_and_save_model():
    print("📊 Loading breast cancer dataset...")
    
    # Load data
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names.tolist()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution: Malignant: {sum(y==1)}, Benign: {sum(y==0)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("🔄 Training SVC model...")
    model = SVC(probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    accuracy = model.score(X_test_scaled, y_test)
    print(f"✅ Model accuracy: {accuracy:.4f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'accuracy': accuracy
    }
    
    joblib.dump(model_data, "models/breast_cancer_model.pkl")
    print("💾 Model saved to models/breast_cancer_model.pkl")

if __name__ == "__main__":
    train_and_save_model()
