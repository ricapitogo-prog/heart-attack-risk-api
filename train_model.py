import pandas as pd
import numpy as np
import pickle
import json
import sklearn
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class HeartAttackModelTrainer:
    """
    Trains and serializes a heart attack prediction model for production use.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the trainer with the path to the dataset.
        
        Args:
            data_path: Path to the heart attack CSV dataset
        """
        self.data_path = data_path
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Feature names (for documentation and validation)
        self.feature_names = [
            'age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 
            'restecg', 'thalachh', 'exng', 'oldpeak', 
            'slp', 'caa', 'thall'
        ]
        self.target_name = 'output'
        
        # Initialize placeholders
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.model = None
        
    def load_data(self):
        """Load and validate the dataset."""
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        # Basic validation
        print(f"Dataset shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        # Check if all required columns exist
        required_cols = self.feature_names + [self.target_name]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    def prepare_data(self, df, test_size=0.2, random_state=42):
        """
        Split data into train/test and create feature/target arrays.
        
        Args:
            df: DataFrame with features and target
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        print("\nPreparing data...")
        
        # Separate features and target
        X = df[self.feature_names]
        y = df[self.target_name]
        
        # Split into train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Class distribution (train): {dict(self.y_train.value_counts())}")
        
    def scale_features(self):
        """
        Scale features using StandardScaler.
        
        This is critical for production: the same scaler must be used
        for both training and inference.
        """
        print("\nScaling features...")
        
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Features scaled using StandardScaler")
        
    def train_model(self, model_type='logistic_regression'):
        """
        Train the machine learning model.
        
        Args:
            model_type: Type of model to train ('logistic_regression' or 'random_forest')
        """
        print(f"\nTraining {model_type} model...")
        
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'  # Handle class imbalance
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train the model
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, self.X_train_scaled, self.y_train, 
            cv=5, scoring='roc_auc'
        )
        
        print(f"Model trained successfully")
        print(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
    def evaluate_model(self):
        """
        Evaluate model performance on test set.
        
        Returns:
            dict: Dictionary of evaluation metrics
        """
        print("\nEvaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test_scaled)
        y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Print results
        print("\nTest Set Performance:")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(f"TN: {cm[0,0]}  FP: {cm[0,1]}")
        print(f"FN: {cm[1,0]}  TP: {cm[1,1]}")
        
        return metrics
    
    def save_model(self, metrics):
        """
        Serialize the trained model, scaler, and metadata for production use.
        
        Args:
            metrics: Dictionary of evaluation metrics
        """
        print("\nSaving model artifacts...")
        
        # Save the trained model
        model_path = self.models_dir / "trained_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {model_path}")
        
        # Save the scaler (CRITICAL for production!)
        scaler_path = self.models_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to {scaler_path}")
        
        # Save metadata (for documentation and validation)
        metadata = {
            'model_type': self.model.__class__.__name__,
            'trained_date': datetime.utcnow().isoformat(),
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'n_features': len(self.feature_names),
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'metrics': metrics,
            'sklearn_version': sklearn.__version__,
        }
        
        metadata_path = self.models_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_path}")
        
        print("\nAll model artifacts saved successfully")
        print(f"Model directory: {self.models_dir.absolute()}")
    
    def run_full_pipeline(self, model_type='logistic_regression'):
        """
        Execute the complete training pipeline.
        
        Args:
            model_type: Type of model to train
        """
        print("=" * 60)
        print("PRODUCTION MODEL TRAINING PIPELINE")
        print("=" * 60)
        
        # Step 1: Load data
        df = self.load_data()
        
        # Step 2: Prepare data
        self.prepare_data(df)
        
        # Step 3: Scale features
        self.scale_features()
        
        # Step 4: Train model
        self.train_model(model_type=model_type)
        
        # Step 5: Evaluate model
        metrics = self.evaluate_model()
        
        # Step 6: Save everything
        self.save_model(metrics)
        
        print("\n" + "=" * 60)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        return metrics


# Example usage
if __name__ == "__main__":
    # Path to your heart attack dataset
    DATA_PATH = "data/heart.csv"
    
    # Initialize trainer
    trainer = HeartAttackModelTrainer(DATA_PATH)
    
    # Run the full pipeline
    # Try 'logistic_regression' or 'random_forest'
    metrics = trainer.run_full_pipeline(model_type='logistic_regression')
    
    print("\nNext Steps:")
    print("1. Review the model performance metrics above")
    print("2. Check the models/ directory for saved artifacts")
    print("3. Try training with 'random_forest' and compare results")