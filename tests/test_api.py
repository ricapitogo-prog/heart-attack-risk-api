"""
Integration tests for Flask API.
"""

import pytest
import json
from src.api import app


@pytest.fixture
def client():
    """Create test client for Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestHealthEndpoint:
    """Test /health endpoint."""
    
    def test_health_check(self, client):
        """Test health check returns 200."""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert data["model_loaded"] == True
        assert "model_type" in data


class TestModelInfoEndpoint:
    """Test /model-info endpoint."""
    
    def test_model_info(self, client):
        """Test model info returns correct data."""
        response = client.get('/model-info')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "model_type" in data
        assert "trained_date" in data
        assert "n_features" in data
        assert "metrics" in data
        assert data["n_features"] == 13


class TestPredictEndpoint:
    """Test /predict endpoint."""
    
    def test_valid_prediction(self, client):
        """Test prediction with valid data."""
        patient_data = {
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
        
        response = client.post(
            '/predict',
            data=json.dumps(patient_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "risk_level" in data
        assert "probability" in data
        assert "model_type" in data
        assert "timestamp" in data
        assert data["risk_level"] in ["LOW", "HIGH"]
        assert 0 <= data["probability"] <= 1
    
    def test_missing_fields(self, client):
        """Test prediction with missing fields."""
        patient_data = {
            "age": 45,
            "sex": 1
        }
        
        response = client.post(
            '/predict',
            data=json.dumps(patient_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
    
    def test_invalid_age(self, client):
        """Test prediction with invalid age."""
        patient_data = {
            "age": 999,
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
        
        response = client.post(
            '/predict',
            data=json.dumps(patient_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
    
    def test_invalid_json(self, client):
        """Test prediction with invalid JSON."""
        response = client.post(
            '/predict',
            data="not valid json",
            content_type='application/json'
        )
        
        assert response.status_code in [400, 500]
    
    def test_invalid_endpoint(self, client):
        """Test that invalid endpoints return 404."""
        response = client.get('/nonexistent')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert "error" in data
