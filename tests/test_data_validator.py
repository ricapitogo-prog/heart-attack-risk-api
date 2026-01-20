"""
Unit tests for data validation schemas.
"""

import pytest
from pydantic import ValidationError
from src.data_validator import PatientData, PredictionResponse


class TestPatientData:
    """Test PatientData validation schema."""
    
    def test_valid_patient_data(self):
        """Test that valid patient data passes validation."""
        data = {
            "age": 45,
            "sex": 1,
            "cp": 0,
            "trtbps": 120,
            "chol": 200,
            "fbs": 0,
            "restecg": 0,
            "thalachh": 150,
            "exng": 0,
            "oldpeak": 0.0,
            "slp": 2,
            "caa": 0,
            "thall": 2
        }
        
        patient = PatientData(**data)
        assert patient.age == 45
        assert patient.sex == 1
        assert patient.thalachh == 150
    
    def test_to_feature_array(self):
        """Test conversion to feature array."""
        data = {
            "age": 45,
            "sex": 1,
            "cp": 0,
            "trtbps": 120,
            "chol": 200,
            "fbs": 0,
            "restecg": 0,
            "thalachh": 150,
            "exng": 0,
            "oldpeak": 0.0,
            "slp": 2,
            "caa": 0,
            "thall": 2
        }
        
        patient = PatientData(**data)
        features = patient.to_feature_array()
        
        assert len(features) == 13
        assert features[0] == 45  # age
        assert features[7] == 150  # thalachh
    
    def test_age_too_young(self):
        """Test that age < 18 is rejected."""
        data = {
            "age": 15,
            "sex": 1,
            "cp": 0,
            "trtbps": 120,
            "chol": 200,
            "fbs": 0,
            "restecg": 0,
            "thalachh": 150,
            "exng": 0,
            "oldpeak": 0.0,
            "slp": 2,
            "caa": 0,
            "thall": 2
        }
        
        with pytest.raises(ValueError, match="at least 18 years"):
            PatientData(**data)
    
    def test_age_too_old(self):
        """Test that age > 100 is rejected."""
        data = {
            "age": 101,
            "sex": 1,
            "cp": 0,
            "trtbps": 120,
            "chol": 200,
            "fbs": 0,
            "restecg": 0,
            "thalachh": 150,
            "exng": 0,
            "oldpeak": 0.0,
            "slp": 2,
            "caa": 0,
            "thall": 2
        }
        
        with pytest.raises(ValueError, match="100 years or less"):
            PatientData(**data)
    
    def test_blood_pressure_too_low(self):
        """Test that blood pressure < 80 is rejected."""
        data = {
            "age": 45,
            "sex": 1,
            "cp": 0,
            "trtbps": 70,
            "chol": 200,
            "fbs": 0,
            "restecg": 0,
            "thalachh": 150,
            "exng": 0,
            "oldpeak": 0.0,
            "slp": 2,
            "caa": 0,
            "thall": 2
        }
        
        with pytest.raises(ValidationError, match="greater than or equal to 80"):
            PatientData(**data)
    
    def test_cholesterol_too_high(self):
        """Test that cholesterol > 600 is rejected."""
        data = {
            "age": 45,
            "sex": 1,
            "cp": 0,
            "trtbps": 120,
            "chol": 650,
            "fbs": 0,
            "restecg": 0,
            "thalachh": 150,
            "exng": 0,
            "oldpeak": 0.0,
            "slp": 2,
            "caa": 0,
            "thall": 2
        }
        
        with pytest.raises(ValidationError, match="less than or equal to 600"):
            PatientData(**data)
    
    def test_missing_required_field(self):
        """Test that missing required fields are rejected."""
        data = {
            "age": 45,
            "sex": 1
        }
        
        with pytest.raises(ValidationError):
            PatientData(**data)


class TestPredictionResponse:
    """Test PredictionResponse schema."""
    
    def test_valid_response(self):
        """Test valid prediction response."""
        data = {
            "risk_level": "HIGH",
            "probability": 0.75,
            "model_type": "RandomForestClassifier",
            "timestamp": "2026-01-20T23:30:45.123456"
        }
        
        response = PredictionResponse(**data)
        assert response.risk_level == "HIGH"
        assert response.probability == 0.75
        assert response.model_type == "RandomForestClassifier"
    
    def test_probability_bounds(self):
        """Test that probability must be between 0 and 1."""
        data = {
            "risk_level": "HIGH",
            "probability": 1.5,
            "model_type": "RandomForestClassifier",
            "timestamp": "2026-01-20T23:30:45.123456"
        }
        
        with pytest.raises(ValidationError):
            PredictionResponse(**data)
