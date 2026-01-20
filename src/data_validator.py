"""
Data validation schemas using Pydantic.

This ensures incoming patient data is valid before making predictions.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List


class PatientData(BaseModel):
    """
    Schema for patient clinical data.
    
    All features required for heart attack risk prediction.
    """
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex: 0=female, 1=male")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trtbps: int = Field(..., ge=80, le=200, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=100, le=600, description="Cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalachh: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved (bpm)")
    exng: int = Field(..., ge=0, le=1, description="Exercise induced angina (1=yes, 0=no)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise")
    slp: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    caa: int = Field(..., ge=0, le=3, description="Number of major vessels (0-3)")
    thall: int = Field(..., ge=0, le=3, description="Thalassemia (0-3)")
    
    @field_validator('age')
    @classmethod
    def age_must_be_reasonable(cls, v):
        """Validate age is in reasonable range for adults."""
        if v < 18:
            raise ValueError('Age must be at least 18 years for this model')
        if v > 100:
            raise ValueError('Age must be 100 years or less for this model')
        return v
    
    @field_validator('trtbps')
    @classmethod
    def blood_pressure_must_be_reasonable(cls, v):
        """Validate blood pressure is in reasonable range."""
        if v < 80:
            raise ValueError('Blood pressure too low (< 80 mm Hg)')
        if v > 200:
            raise ValueError('Blood pressure too high (> 200 mm Hg)')
        return v
    
    @field_validator('chol')
    @classmethod
    def cholesterol_must_be_reasonable(cls, v):
        """Validate cholesterol is in reasonable range."""
        if v < 100:
            raise ValueError('Cholesterol too low (< 100 mg/dl)')
        if v > 600:
            raise ValueError('Cholesterol too high (> 600 mg/dl)')
        return v
    
    def to_feature_array(self) -> List[float]:
        """
        Convert patient data to feature array for model prediction.
        
        Returns:
            List of feature values in correct order
        """
        return [
            self.age, self.sex, self.cp, self.trtbps, self.chol, self.fbs,
            self.restecg, self.thalachh, self.exng, self.oldpeak,
            self.slp, self.caa, self.thall
        ]
    
    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    """
    Schema for prediction API response.
    """
    risk_level: str = Field(..., description="Risk level: LOW or HIGH")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of high risk (0-1)")
    model_type: str = Field(..., description="Type of model used")
    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "risk_level": "LOW",
                "probability": 0.234,
                "model_type": "RandomForestClassifier",
                "timestamp": "2026-01-20T23:30:45.123456"
            }
        }
    }
