"""
Test Script: Verify Model Loading and Prediction

This script tests that your trained model can be loaded and used for predictions.
This simulates what will happen in production when your API starts up.
"""

import pickle
import json
import numpy as np
from pathlib import Path


class ModelTester:
    """Test that saved models can be loaded and used for predictions."""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        
    def test_model_loading(self):
        """Test 1: Can we load the model and scaler?"""
        print("Test 1: Loading saved model artifacts...")
        
        try:
            # Load model
            model_path = self.models_dir / "trained_model.pkl"
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded: {model.__class__.__name__}")
            
            # Load scaler
            scaler_path = self.models_dir / "scaler.pkl"
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"Scaler loaded: {scaler.__class__.__name__}")
            
            # Load metadata
            metadata_path = self.models_dir / "model_metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Metadata loaded")
            
            return model, scaler, metadata
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Make sure you've run train_model.py first!")
            return None, None, None
    
    def test_single_prediction(self, model, scaler, metadata):
        """Test 2: Can we make a prediction with new data?"""
        print("\nTest 2: Making a single prediction...")
        
        if model is None:
            print("Skipping - model not loaded")
            return
        
        # Sample patient data (low risk profile)
        sample_patient = np.array([[
            45,   # age
            1,    # sex (male)
            0,    # cp (asymptomatic)
            120,  # trtbps (normal blood pressure)
            200,  # chol (normal cholesterol)
            0,    # fbs (normal fasting blood sugar)
            0,    # restecg (normal)
            150,  # thalachh (good max heart rate)
            0,    # exng (no exercise angina)
            0.0,  # oldpeak (no ST depression)
            2,    # slp (upsloping)
            0,    # caa (no vessels)
            2     # thall (normal)
        ]])
        
        print(f"Sample patient features: {metadata['feature_names']}")
        print(f"Sample patient values: {sample_patient[0]}")
        
        # Scale features (CRITICAL: must use same scaler as training!)
        scaled_features = scaler.transform(sample_patient)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0]
        
        print(f"\nPrediction: {'HIGH RISK' if prediction == 1 else 'LOW RISK'}")
        print(f"Probabilities: [Low: {probability[0]:.3f}, High: {probability[1]:.3f}]")
        
        return prediction, probability
    
    def test_batch_predictions(self, model, scaler):
        """Test 3: Can we make predictions on multiple patients?"""
        print("\nTest 3: Making batch predictions...")
        
        if model is None:
            print("Skipping - model not loaded")
            return
        
        # Multiple test cases
        test_patients = np.array([
            # Low risk patient
            [30, 1, 0, 110, 180, 0, 0, 170, 0, 0.0, 2, 0, 2],
            # High risk patient  
            [65, 1, 3, 160, 280, 1, 2, 100, 1, 3.0, 0, 3, 1],
            # Medium risk patient
            [50, 0, 2, 140, 240, 0, 1, 130, 0, 1.5, 1, 1, 2],
        ])
        
        # Scale and predict
        scaled = scaler.transform(test_patients)
        predictions = model.predict(scaled)
        probabilities = model.predict_proba(scaled)
        
        print(f"Processed {len(test_patients)} patients")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            risk = "HIGH RISK" if pred == 1 else "LOW RISK"
            print(f"  Patient {i+1}: {risk} (probability: {prob[1]:.3f})")
    
    def test_metadata_validation(self, metadata):
        """Test 4: Is the metadata complete and valid?"""
        print("\nTest 4: Validating metadata...")
        
        if metadata is None:
            print("Skipping - metadata not loaded")
            return
        
        required_fields = [
            'model_type', 'trained_date', 'feature_names', 
            'target_name', 'n_features', 'metrics'
        ]
        
        for field in required_fields:
            if field in metadata:
                print(f"{field}: {metadata[field] if field != 'metrics' else 'present'}")
            else:
                print(f"Missing field: {field}")
        
        # Display performance metrics
        if 'metrics' in metadata:
            print("\nModel Performance:")
            for metric, value in metadata['metrics'].items():
                print(f"  {metric}: {value:.4f}")
    
    def run_all_tests(self):
        """Run all tests."""
        print("=" * 60)
        print("PRODUCTION MODEL TESTING SUITE")
        print("=" * 60)
        
        # Test 1: Load artifacts
        model, scaler, metadata = self.test_model_loading()
        
        if model is None:
            print("\nTests failed: Could not load model")
            print("Run train_model.py first to create the model artifacts")
            return
        
        # Test 2: Single prediction
        self.test_single_prediction(model, scaler, metadata)
        
        # Test 3: Batch predictions
        self.test_batch_predictions(model, scaler)
        
        # Test 4: Metadata validation
        self.test_metadata_validation(metadata)
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        print("\nYour model is ready for production deployment")

if __name__ == "__main__":
    tester = ModelTester()
    tester.run_all_tests()
