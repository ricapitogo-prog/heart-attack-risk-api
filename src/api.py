"""
Flask REST API for heart attack risk prediction.

Endpoints:
- POST /predict: Make a prediction
- GET /health: Health check
- GET /model-info: Get model information
"""

from flask import Flask, request, jsonify
from src.data_validator import PatientData, PredictionResponse
from src.model_loader import ModelLoader
from src.predictor import HeartAttackPredictor
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Get project root directory (parent of src/)
project_root = Path(__file__).parent.parent
models_dir = project_root / "models"

# Load model artifacts at startup
logger.info("Loading model artifacts...")
loader = ModelLoader(models_dir=str(models_dir))
model, scaler, metadata = loader.load_artifacts()

# Validate artifacts
if not loader.validate_artifacts():
    raise Exception("Model artifacts validation failed")

# Initialize predictor
predictor = HeartAttackPredictor(model, scaler, metadata)
logger.info("API ready to serve predictions")


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    
    Returns:
        JSON response with health status
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "model_type": metadata.get('model_type')
    }), 200


@app.route('/model-info', methods=['GET'])
def model_info():
    """
    Get information about the loaded model.
    
    Returns:
        JSON response with model details
    """
    try:
        info = predictor.get_model_info()
        return jsonify(info), 200
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({"error": "Failed to get model info"}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a heart attack risk prediction.
    
    Request body should contain patient data matching PatientData schema.
    
    Returns:
        JSON response with prediction result
    """
    try:
        # Validate input data
        patient_data = PatientData(**request.json)
        
        # Convert to feature array
        features = patient_data.to_feature_array()
        
        # Make prediction
        result = predictor.predict_risk(features)
        
        # Validate response
        response = PredictionResponse(**result)
        
        return jsonify(response.dict()), 200
        
    except ValueError as e:
        # Validation error
        logger.warning(f"Validation error: {e}")
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    
    except Exception as e:
        # Unexpected error
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # Run the Flask app on port 5001 instead of 5000
    app.run(host='0.0.0.0', port=5001, debug=True)
