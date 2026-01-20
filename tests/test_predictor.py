"""
Unit tests for prediction logic.
"""

import pytest
import numpy as np
from src.model_loader import ModelLoader
from src.predictor import HeartAttackPredictor


@pytest.fixture
def predictor():
    """Load model and create predictor for tests."""
    loader = ModelLoader(models_dir="models")
    model, scaler, metadata = loader.load_artifacts()
    return HeartAttackPredictor(model, scaler, metadata)


class TestHeartAttackPredictor:
    """Test prediction functionality."""
    
    def test_predictor_initialization(self, predictor):
        """Test that predictor initializes correctly."""
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert predictor.metadata is not None
        assert predictor.model_type == "RandomForestClassifier"
    
    def test_predict_low_risk_patient(self, predictor):
        """Test prediction for low risk patient."""
        # Young, healthy patient
        features = [30, 1, 0, 110, 180, 0, 0, 170, 0, 0.0, 2, 0, 2]
        
        result = predictor.predict_risk(features)
        
        assert "risk_level" in result
        assert "probability" in result
        assert "model_type" in result
        assert "timestamp" in result
        assert result["risk_level"] in ["LOW", "HIGH"]
        assert 0 <= result["probability"] <= 1
    
    def test_predict_high_risk_patient(self, predictor):
        """Test prediction for high risk patient."""
        # Older patient with multiple risk factors
        features = [65, 1, 3, 160, 280, 1, 2, 100, 1, 3.0, 0, 3, 1]
        
        result = predictor.predict_risk(features)
        
        assert "risk_level" in result
        assert "probability" in result
        assert 0 <= result["probability"] <= 1
    
    def test_invalid_feature_count(self, predictor):
        """Test that wrong number of features raises error."""
        # Only 5 features instead of 13
        features = [45, 1, 0, 120, 200]
        
        with pytest.raises(ValueError, match="Expected 13 features"):
            predictor.predict_risk(features)
    
    def test_get_model_info(self, predictor):
        """Test that model info is returned correctly."""
        info = predictor.get_model_info()
        
        assert "model_type" in info
        assert "trained_date" in info
        assert "n_features" in info
        assert "feature_names" in info
        assert "metrics" in info
        assert info["n_features"] == 13
        assert len(info["feature_names"]) == 13
    
    def test_prediction_consistency(self, predictor):
        """Test that same input gives same prediction."""
        features = [45, 1, 0, 120, 200, 0, 0, 150, 0, 0.0, 2, 0, 2]
        
        result1 = predictor.predict_risk(features)
        result2 = predictor.predict_risk(features)
        
        # Same features should give same prediction
        assert result1["risk_level"] == result2["risk_level"]
        assert abs(result1["probability"] - result2["probability"]) < 0.001
