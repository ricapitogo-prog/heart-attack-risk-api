"""
Prediction logic for heart attack risk assessment.

Handles feature scaling and prediction generation.
"""

import numpy as np
from datetime import datetime
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeartAttackPredictor:
    """
    Makes heart attack risk predictions using loaded model.
    """
    
    def __init__(self, model, scaler, metadata):
        """
        Initialize predictor with loaded artifacts.
        
        Args:
            model: Trained ML model
            scaler: Fitted StandardScaler
            metadata: Model metadata dictionary
        """
        self.model = model
        self.scaler = scaler
        self.metadata = metadata
        self.model_type = metadata.get('model_type', 'Unknown')
        
    def predict_risk(self, features: list) -> Dict[str, Any]:
        """
        Predict heart attack risk for a patient.
        
        Args:
            features: List of 13 patient features
            
        Returns:
            Dictionary with prediction results
            
        Raises:
            ValueError: If features are invalid
            Exception: If prediction fails
        """
        try:
            # Validate input
            if len(features) != 13:
                raise ValueError(f"Expected 13 features, got {len(features)}")
            
            # Convert to numpy array
            features_array = np.array([features])
            
            # Scale features using saved scaler
            features_scaled = self.scaler.transform(features_array)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            # Get high risk probability
            high_risk_prob = float(probability[1])
            
            # Determine risk level
            risk_level = "HIGH" if prediction == 1 else "LOW"
            
            # Create response
            result = {
                "risk_level": risk_level,
                "probability": high_risk_prob,
                "model_type": self.model_type,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Log prediction
            logger.info(
                f"Prediction made: {risk_level} "
                f"(probability: {high_risk_prob:.3f})"
            )
            
            return result
            
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise Exception(f"Prediction error: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_type": self.model_type,
            "trained_date": self.metadata.get('trained_date'),
            "n_features": self.metadata.get('n_features'),
            "feature_names": self.metadata.get('feature_names'),
            "metrics": self.metadata.get('metrics', {})
        }
