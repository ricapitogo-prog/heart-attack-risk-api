"""
Model loader for production deployment.

Loads trained model, scaler, and metadata from disk.
"""

import pickle
import json
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np


class ModelLoader:
    """
    Loads and manages trained model artifacts.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model loader.
        
        Args:
            models_dir: Directory containing model artifacts
        """
        self.models_dir = Path(models_dir)
        self.model = None
        self.scaler = None
        self.metadata = None
        
    def load_artifacts(self) -> Tuple[Any, Any, Dict]:
        """
        Load model, scaler, and metadata from disk.
        
        Returns:
            Tuple of (model, scaler, metadata)
            
        Raises:
            FileNotFoundError: If model artifacts not found
            Exception: If loading fails
        """
        try:
            # Load model
            model_path = self.models_dir / "trained_model.pkl"
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded: {self.model.__class__.__name__}")
            
            # Load scaler
            scaler_path = self.models_dir / "scaler.pkl"
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Scaler loaded: {self.scaler.__class__.__name__}")
            
            # Load metadata
            metadata_path = self.models_dir / "model_metadata.json"
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Metadata loaded: {self.metadata['model_type']}")
            
            return self.model, self.scaler, self.metadata
            
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model artifacts not found in {self.models_dir}. "
                f"Run train_model.py first. Error: {e}"
            )
        except Exception as e:
            raise Exception(f"Failed to load model artifacts: {e}")
    
    def validate_artifacts(self) -> bool:
        """
        Validate that all artifacts are loaded and compatible.
        
        Returns:
            True if valid, False otherwise
        """
        if self.model is None or self.scaler is None or self.metadata is None:
            return False
        
        # Check that feature count matches
        expected_features = self.metadata.get('n_features', 0)
        if expected_features != 13:
            print(f"Warning: Expected 13 features, got {expected_features}")
            return False
        
        return True
    
    def get_feature_names(self) -> list:
        """
        Get expected feature names from metadata.
        
        Returns:
            List of feature names
        """
        if self.metadata:
            return self.metadata.get('feature_names', [])
        return []
    
    def get_model_info(self) -> Dict:
        """
        Get model information for API responses.
        
        Returns:
            Dictionary with model info
        """
        if self.metadata:
            return {
                'model_type': self.metadata.get('model_type'),
                'trained_date': self.metadata.get('trained_date'),
                'metrics': self.metadata.get('metrics', {})
            }
        return {}
