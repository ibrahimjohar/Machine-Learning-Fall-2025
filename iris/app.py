"""
Iris Classification FastAPI Application
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
import uvicorn
from mangum import Mangum

# INITIALIZE FASTAPI APP
app = FastAPI(
    title="Iris Classification API",
    description="Predict iris flower species based on measurements",
    version="1.0.0"
)

# Mangum handler for AWS Lambda
handler = Mangum(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'iris_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'models', 'metadata.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metadata = joblib.load(METADATA_PATH)
    print("Model, scaler, and metadata loaded successfully!")
except Exception as e:
    print(f"Error loading files: {e}")
    model = None
    scaler = None
    metadata = None

# REQUEST AND RESPONSE MODELS
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm", example=5.1)
    sepal_width: float = Field(..., description="Sepal width in cm", example=3.5)
    petal_length: float = Field(..., description="Petal length in cm", example=1.4)
    petal_width: float = Field(..., description="Petal width in cm", example=0.2)

class PredictionResponse(BaseModel):
    predicted_species: str
    confidence: float
    probabilities: dict

# API ENDPOINTS

@app.get("/")
def home():
    """Welcome endpoint"""
    return {
        "message": "Welcome to Iris Classification API",
        "status": "active" if model else "model not loaded",
        "endpoints": {
            "/predict": "POST - Predict iris species",
            "/predict-batch": "POST - Predict multiple samples",
            "/model-info": "GET - Get model information",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if (model and scaler and metadata) else "unhealthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "metadata_loaded": metadata is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_species(features: IrisFeatures):
    """Predict iris species based on flower measurements"""
    if model is None or scaler is None or metadata is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Prepare input data
        input_data = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Get species name
        species_name = metadata['target_names'][prediction]
        
        # Create probability dictionary
        prob_dict = {
            metadata['target_names'][i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        return {
            "predicted_species": species_name,
            "confidence": float(max(probabilities)),
            "probabilities": prob_dict
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
def predict_batch(samples: list[IrisFeatures]):
    """Predict iris species for multiple samples"""
    if model is None or scaler is None or metadata is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if len(samples) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 samples allowed")
    
    try:
        predictions = []
        
        for sample in samples:
            input_data = np.array([[
                sample.sepal_length,
                sample.sepal_width,
                sample.petal_length,
                sample.petal_width
            ]])
            
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            
            species_name = metadata['target_names'][prediction]
            
            prob_dict = {
                metadata['target_names'][i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
            
            predictions.append({
                "input": {
                    "sepal_length": sample.sepal_length,
                    "sepal_width": sample.sepal_width,
                    "petal_length": sample.petal_length,
                    "petal_width": sample.petal_width
                },
                "predicted_species": species_name,
                "confidence": float(max(probabilities)),
                "probabilities": prob_dict
            })
        
        return {
            "predictions": predictions,
            "count": len(predictions)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
def get_model_info():
    """Get information about the model"""
    if model is None or metadata is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "features": metadata['feature_names'],
        "target_classes": metadata['target_names'],
        "n_features": len(metadata['feature_names']),
        "n_classes": len(metadata['target_names'])
    }

# RUN APPLICATION
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting Iris Classification API...")
    print("API: http://localhost:8000")
    print("Documentation: http://localhost:8000/docs")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)