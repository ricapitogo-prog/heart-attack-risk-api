# Heart Attack Risk Monitoring System

> A production-ready machine learning API for predicting heart attack risk based on clinical parameters

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![Test Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen.svg)](./TESTING.md)
[![Code Style](https://img.shields.io/badge/code%20style-pep8-blue.svg)](https://www.python.org/dev/peps/pep-0008/)

## Overview

This project demonstrates end-to-end machine learning system development, from model training through production-ready API deployment. The system predicts heart attack risk with **93.94% recall**, ensuring nearly all high-risk patients are identified for early intervention.

**Key Achievement:** Built a complete ML system in one week, including model training, API development, comprehensive testing (85% coverage), and production documentation.

## Problem & Solution

**Problem:** Heart disease causes 655,000 deaths annually in the US. Early risk detection enables preventive care and saves lives.

**Solution:** REST API serving a RandomForest classifier that processes 13 clinical parameters and returns real-time risk assessments. Healthcare applications can integrate this API to flag high-risk patients for physician review.

## Key Features

- **High Recall ML Model**: 93.94% recall ensures we catch nearly all high-risk patients
- **Production-Ready API**: Flask REST API with comprehensive error handling
- **Robust Validation**: Pydantic schemas validate all inputs against clinical ranges
- **Comprehensive Testing**: 22 automated tests with 85% code coverage
- **Complete Documentation**: API docs, testing guide, deployment instructions
- **Health Monitoring**: Built-in health checks and model metadata endpoints

## Quick Start

### Prerequisites
- Python 3.8+
- pip
- Virtual environment tool

### Installation
```bash
# Clone repository
git clone https://github.com/ricapitogo-prog/heart-attack-risk-api.git
cd heart-attack-risk-api

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Train Model
```bash
python3 train_model.py
```

**Output:**
- `models/trained_model.pkl` - Trained RandomForest classifier
- `models/scaler.pkl` - Fitted StandardScaler
- `models/model_metadata.json` - Model metrics and info

### Run Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

### Start API
```bash
python3 src/api.py
```

API runs on `http://localhost:5001`

### Make a Prediction
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 65,
    "sex": 1,
    "cp": 3,
    "trtbps": 160,
    "chol": 280,
    "fbs": 1,
    "restecg": 2,
    "thalachh": 100,
    "exng": 1,
    "oldpeak": 3.0,
    "slp": 0,
    "caa": 3,
    "thall": 1
  }'
```

**Response:**
```json
{
  "risk_level": "LOW",
  "probability": 0.21,
  "model_type": "RandomForestClassifier",
  "timestamp": "2026-01-20T23:10:58.586355"
}
```

## Model Performance

| Metric | Score | Why It Matters |
|--------|-------|----------------|
| **Recall** | **93.94%** | Catches 31 out of 33 high-risk patients |
| Accuracy | 81.97% | Overall prediction correctness |
| Precision | 77.50% | Reduces false alarms |
| F1 Score | 84.93% | Balanced performance |
| ROC-AUC | 90.37% | Strong discrimination ability |

### Why RandomForest?

Compared to Logistic Regression baseline:
- Better recall: 93.94% vs 87.88%
- Fewer missed high-risk patients: 2 vs 4 false negatives
- Higher accuracy: 81.97% vs 78.69%
- Better ROC-AUC: 90.37% vs 87.12%

**Critical for Healthcare:** Only 2 out of 33 high-risk patients missed (6% false negative rate)

See [MODEL_SELECTION.md](MODEL_SELECTION.md) for detailed comparison.

## Architecture
```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Client    │─────▶│  Flask API   │─────▶│  Predictor  │
│ Application │      │ (Validation) │      │   Service   │
└─────────────┘      └──────────────┘      └─────────────┘
                            │                      │
                            │                      ▼
                            │              ┌──────────────┐
                            │              │ RandomForest │
                            │              │    Model     │
                            │              └──────────────┘
                            │                      │
                            ▼                      ▼
                     ┌──────────────┐      ┌─────────────┐
                     │   Pydantic   │      │   Scaler    │
                     │  Validation  │      │  (Fitted)   │
                     └──────────────┘      └─────────────┘
```

### Components

1. **Data Validator** (`data_validator.py`): Pydantic schemas with clinical range validation
2. **Model Loader** (`model_loader.py`): Loads serialized model, scaler, and metadata
3. **Predictor** (`predictor.py`): Feature scaling and prediction logic with error handling
4. **Flask API** (`api.py`): REST endpoints with comprehensive error handling

## Project Structure
```
heart_attack_risk_api/
├── src/
│   ├── __init__.py
│   ├── data_validator.py      # Input validation (92% coverage)
│   ├── model_loader.py        # Model loading (72% coverage)
│   ├── predictor.py           # Prediction logic (91% coverage)
│   └── api.py                 # Flask API (88% coverage)
├── tests/
│   ├── conftest.py            # Pytest configuration
│   ├── test_data_validator.py # 8 validation tests
│   ├── test_predictor.py      # 6 predictor tests
│   └── test_api.py            # 7 API integration tests
├── data/
│   └── heart.csv              # Training dataset (not in repo)
├── models/
│   ├── trained_model.pkl      # Trained model (not in repo)
│   ├── scaler.pkl             # Fitted scaler (not in repo)
│   └── model_metadata.json    # Model info (not in repo)
├── train_model.py             # Model training script
├── test_model_loading.py      # Model verification
├── requirements.txt           # Dependencies
└── [Documentation files]      # See below
```

## API Documentation

### Endpoints

#### GET `/health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "RandomForestClassifier"
}
```

#### GET `/model-info`
Get model metadata and performance metrics

**Response:**
```json
{
  "model_type": "RandomForestClassifier",
  "trained_date": "2026-01-20T23:03:01.360056",
  "n_features": 13,
  "metrics": {
    "accuracy": 0.8197,
    "recall": 0.9394,
    "roc_auc": 0.9037
  }
}
```

#### POST `/predict`
Predict heart attack risk

**Request Body:**
| Field | Type | Range | Description |
|-------|------|-------|-------------|
| age | int | 18-100 | Patient age in years |
| sex | int | 0-1 | 0=female, 1=male |
| cp | int | 0-3 | Chest pain type |
| trtbps | int | 80-200 | Resting blood pressure (mm Hg) |
| chol | int | 100-600 | Cholesterol (mg/dl) |
| fbs | int | 0-1 | Fasting blood sugar > 120 mg/dl |
| restecg | int | 0-2 | Resting ECG results |
| thalachh | int | 60-220 | Maximum heart rate (bpm) |
| exng | int | 0-1 | Exercise induced angina |
| oldpeak | float | 0.0-10.0 | ST depression |
| slp | int | 0-2 | Slope of peak exercise ST segment |
| caa | int | 0-3 | Number of major vessels |
| thall | int | 0-3 | Thalassemia |

**Success Response (200):**
```json
{
  "risk_level": "HIGH",
  "probability": 0.64,
  "model_type": "RandomForestClassifier",
  "timestamp": "2026-01-20T23:10:27.018965"
}
```

**Error Response (400):**
```json
{
  "error": "Invalid input: age must be at least 18 years"
}
```

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for complete API reference.

## Testing

**Test Coverage: 85%**
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View HTML report
open htmlcov/index.html
```

### Test Suite

| Test Category | Tests | Coverage | Focus |
|---------------|-------|----------|-------|
| Data Validation | 8 | 92% | Input validation, range checking |
| Predictor Logic | 6 | 91% | Model predictions, error handling |
| API Integration | 7 | 88% | Endpoints, error responses |
| **Total** | **22** | **85%** | **Full system** |

See [TESTING.md](TESTING.md) for detailed testing documentation.

## Technologies

**Core:**
- Python 3.12
- scikit-learn 1.8.0 (ML)
- Flask 3.0.0 (API)
- Pydantic 2.5.0 (Validation)

**Testing:**
- pytest 9.0.2
- pytest-cov 7.0.0

**Data:**
- pandas 2.3.3
- numpy 2.4.1

## Documentation

- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) - Complete API reference with examples
- [MODEL_SELECTION.md](MODEL_SELECTION.md) - Model comparison and selection rationale
- [TESTING.md](TESTING.md) - Testing guide and best practices
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment guide
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Comprehensive project overview
- [LESSONS_LEARNED.md](LESSONS_LEARNED.md) - Technical learnings and insights
- [SETUP.md](SETUP.md) - Detailed setup instructions

## Development Timeline

**One-Week Sprint:**

| Days | Milestone | Deliverables |
|------|-----------|--------------|
| 1-2 | Model Training & Serialization | Trained model, comparison, serialization |
| 3-4 | API Development | Flask API, validation, error handling |
| 5-6 | Testing & QA | 22 tests, 85% coverage, documentation |
| 7 | Documentation & Deployment Prep | Complete docs, deployment guide |

## Skills Demonstrated

### Machine Learning
- Model training and evaluation
- Model comparison and selection
- Feature engineering and scaling
- Cross-validation
- Healthcare-specific metric optimization

### ML Engineering
- Model serialization (pickle)
- Production model serving
- Inference pipeline design
- Scaler consistency management

### Software Engineering
- REST API development
- Input validation
- Error handling
- Logging
- Testing (unit & integration)
- Code organization
- Documentation

### DevOps
- Git version control
- Virtual environments
- Dependency management
- Deployment preparation

## Future Enhancements

**Phase 2 (Production Hardening):**
- Prediction logging to database
- Model monitoring and drift detection
- Authentication (JWT)
- Rate limiting
- Docker containerization

**Phase 3 (Advanced Features):**
- A/B testing framework
- Feature importance visualization
- Automated retraining pipeline
- Real-time monitoring dashboard
- CI/CD with GitHub Actions

## Important Notes

- This is a **demonstration project** for portfolio purposes
- Not intended for actual medical use without validation
- Real medical applications require:
  - Clinical validation
  - Regulatory approval (FDA)
  - HIPAA compliance
  - Additional safety measures

## Author

**Rica Mae Pitogo**  
Master's in Data Science in Health

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/rica-mae-pitogo-a7aa10193/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/ricapitogo-prog)

## License

This project is available for portfolio and educational purposes.

## Acknowledgments

- Dataset: UCI Machine Learning Repository / Kaggle
- Inspiration: The need for early heart disease detection
- Framework guidance: Production ML best practices

---

**If you found this project helpful, please consider giving it a star!**

*Built to demonstrate production ML engineering skills*

