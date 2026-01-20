# Heart Attack Risk Monitoring System

A production-ready machine learning system that predicts heart attack risk based on clinical parameters.

## Project Status
✅ **Days 5-6 Complete** - Comprehensive testing with 85% coverage

## Features

- **Machine Learning Model**: RandomForest classifier with 93.94% recall
- **REST API**: Flask-based API for serving predictions
- **Input Validation**: Pydantic schemas ensure data quality
- **Error Handling**: Comprehensive validation and error messages
- **Testing**: 22 unit and integration tests with 85% coverage
- **Production Ready**: Model serialization, health checks, logging

## Quick Start

### 1. Setup
```bash
# Clone repository
git clone <your-repo-url>
cd heart_attack_risk_api

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Model (if needed)
```bash
python3 train_model.py
```

### 3. Run Tests
```bash
pytest tests/ -v --cov=src
```

### 4. Start API
```bash
python3 src/api.py
```

### 5. Make Predictions
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "sex": 1, "cp": 0, "trtbps": 120, "chol": 200, "fbs": 0, "restecg": 0, "thalachh": 150, "exng": 0, "oldpeak": 0.0, "slp": 2, "caa": 0, "thall": 2}'
```

## Project Structure
```
heart_attack_risk_api/
├── data/                    # Dataset (not in git)
├── models/                  # Trained models (not in git)
│   ├── trained_model.pkl
│   ├── scaler.pkl
│   └── model_metadata.json
├── src/                     # Source code
│   ├── __init__.py
│   ├── data_validator.py   # Pydantic schemas
│   ├── model_loader.py     # Model loading
│   ├── predictor.py        # Prediction logic
│   └── api.py              # Flask API
├── tests/                   # Unit tests (85% coverage)
│   ├── conftest.py
│   ├── test_data_validator.py
│   ├── test_predictor.py
│   └── test_api.py
├── train_model.py           # Model training script
├── test_model_loading.py   # Model testing script
└── requirements.txt         # Dependencies
```

## API Endpoints

- **GET** `/health` - Health check
- **GET** `/model-info` - Model information and metrics
- **POST** `/predict` - Make heart attack risk prediction

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for detailed API docs.

## Model Performance

**RandomForestClassifier** (Production Model):
- Accuracy: 81.97%
- Precision: 77.50%
- Recall: 93.94%
- F1 Score: 84.93%
- ROC-AUC: 90.37%

See [MODEL_SELECTION.md](MODEL_SELECTION.md) for model comparison details.

## Testing

22 tests with **85% code coverage**:
- Data validation tests: 8 tests (92% coverage)
- Predictor tests: 6 tests (91% coverage)
- API integration tests: 7 tests (88% coverage)

See [TESTING.md](TESTING.md) for detailed testing documentation.

## Development Timeline

-  **Days 1-2**: Model training and serialization
-  **Days 3-4**: API development and testing
-  **Days 5-6**: Comprehensive testing suite
-  **Day 7**: Documentation and deployment prep (Next)

## Documentation

- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) - API endpoint details
- [MODEL_SELECTION.md](MODEL_SELECTION.md) - Model comparison and selection
- [TESTING.md](TESTING.md) - Testing guide and best practices
- [SETUP.md](SETUP.md) - Detailed setup instructions

## Author
Rica Mae Pitogo - Master's in Data Science in Health @ UCLA

## Last Updated
January 2026
