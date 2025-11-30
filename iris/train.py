"""
Iris Classification Model Training
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ============================================
# CREATE MODELS FOLDER
# ============================================
if not os.path.exists('models'):
    os.makedirs('models')
    print("✓ Created 'models' folder")

# ============================================
# LOAD IRIS DATASET
# ============================================
print("Loading Iris dataset...")

iris = load_iris()
X = iris.data
y = iris.target

# Create DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(f"✓ Dataset loaded: {df.shape}")
print(f"✓ Features: {iris.feature_names}")
print(f"✓ Classes: {iris.target_names}")

# ============================================
# SPLIT DATA
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Training set: {X_train.shape}")
print(f"✓ Test set: {X_test.shape}")

# ============================================
# SCALE FEATURES
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# TRAIN MODEL
# ============================================
print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=5
)

model.fit(X_train_scaled, y_train)

# ============================================
# EVALUATE MODEL
# ============================================
train_accuracy = model.score(X_train_scaled, y_train)
test_accuracy = model.score(X_test_scaled, y_test)

print(f"\n✓ Training Accuracy: {train_accuracy:.4f}")
print(f"✓ Test Accuracy: {test_accuracy:.4f}")

# ============================================
# SAVE MODEL AND SCALER
# ============================================
print("\nSaving model and scaler...")

joblib.dump(model, 'models/iris_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Save feature names and target names for reference
metadata = {
    'feature_names': iris.feature_names,
    'target_names': iris.target_names.tolist()
}
joblib.dump(metadata, 'models/metadata.pkl')

print("✓ Model saved to: models/iris_model.pkl")
print("✓ Scaler saved to: models/scaler.pkl")
print("✓ Metadata saved to: models/metadata.pkl")

# ============================================
# TEST PREDICTIONS
# ============================================
print("\n" + "="*50)
print("Testing predictions...")
print("="*50)

# Test sample
test_sample = X_test_scaled[0].reshape(1, -1)
prediction = model.predict(test_sample)
probabilities = model.predict_proba(test_sample)

print(f"\nTest Sample Features: {X_test[0]}")
print(f"Predicted Class: {iris.target_names[prediction[0]]}")
print(f"Probabilities: {probabilities[0]}")

print("\n✓ Model training complete!")
print("\nNext steps:")
print("1. Run: python app.py")
print("2. Visit: http://localhost:8000/docs")