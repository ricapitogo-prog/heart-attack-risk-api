# Heart Attack Risk Monitoring System - Project Summary

## Overview

A production-ready machine learning API that predicts heart attack risk based on clinical parameters. This project demonstrates end-to-end ML system development from model training through testing and deployment preparation.

## Problem Statement

Heart disease is the leading cause of death globally. Early detection of high-risk patients enables timely intervention and preventive care. This system provides real-time risk assessment through a REST API that healthcare applications can integrate.

## Solution

A Flask-based REST API serving a RandomForest classifier trained on clinical parameters. The system achieves 93.94% recall, ensuring we catch nearly all high-risk patients while maintaining good overall accuracy (81.97%).

## Technical Architecture

### Components

1. **Machine Learning Model**
   - Algorithm: RandomForestClassifier
   - Training: Scikit-learn with cross-validation
   - Features: 13 clinical parameters
   - Serialization: Pickle for model persistence

2. **Data Validation Layer**
   - Framework: Pydantic v2
   - Validates all 13 input features
   - Custom validators for clinical ranges
   - Automatic error messages

3. **Prediction Service**
   - Feature scaling with StandardScaler
   - Error handling and logging
   - Consistent probability scoring
   - Model metadata management

4. **REST API**
   - Framework: Flask
   - 3 endpoints: /health, /model-info, /predict
   - JSON request/response format
   - Comprehensive error handling

5. **Testing Suite**
   - Framework: Pytest
   - 22 tests across all components
   - 85% code coverage
   - Integration and unit tests

## Key Features

### 1. Production-Ready Model Serving
- Model loaded once at startup (not per request)
- Scaler consistency between training and inference
- Metadata tracking for model versioning
- Performance metrics included in responses

### 2. Robust Input Validation
- All 13 features validated with clinical ranges
- Age: 18-100 years
- Blood pressure: 80-200 mm Hg
- Cholesterol: 100-600 mg/dl
- Automatic rejection of invalid inputs

### 3. Error Handling
- Validation errors (400) with detailed messages
- Server errors (500) with proper logging
- 404 handling for invalid endpoints
- Exception catching at all levels

### 4. Comprehensive Testing
- 22 automated tests
- 85% code coverage
- Unit tests for each component
- Integration tests for API
- Test fixtures for reusability

### 5. Complete Documentation
- API documentation with examples
- Model selection rationale
- Testing guide and best practices
- Setup instructions
- Deployment preparation

## Model Performance

### Training Results

**Dataset:**
- Source: UCI Machine Learning Repository / Kaggle
- Size: 303 patients
- Split: 80% train (242), 20% test (61)
- Features: 13 clinical parameters
- Target: Binary (0=low risk, 1=high risk)

**RandomForest Model:**
- Accuracy: 81.97%
- Precision: 77.50%
- Recall: 93.94% ← Most important for healthcare
- F1 Score: 84.93%
- ROC-AUC: 90.37%

**Cross-Validation:**
- 5-fold CV ROC-AUC: 90.79% (±4.59%)
- Consistent performance across folds

### Why RandomForest?

Compared to Logistic Regression:
- Better recall: 93.94% vs 87.88%
- Fewer false negatives: 2 vs 4 (out of 33 high-risk patients)
- Better overall accuracy: 81.97% vs 78.69%
- Higher ROC-AUC: 90.37% vs 87.12%

### Clinical Significance

**False Negatives (Critical):**
- RandomForest: Only 2 out of 33 high-risk patients missed (6%)
- This means 31 out of 33 high-risk patients are correctly identified
- In healthcare, missing high-risk patients is dangerous

**False Positives (Less Critical):**
- 9 out of 28 low-risk patients flagged as high-risk
- Result: Unnecessary follow-up tests (costly but not dangerous)

## Project Structure
```
heart_attack_risk_api/
│
├── data/
│   ├── heart.csv                  # Dataset (not in git)
│   └── .gitkeep
│
├── models/
│   ├── trained_model.pkl          # Serialized RandomForest (not in git)
│   ├── scaler.pkl                 # Fitted StandardScaler (not in git)
│   └── model_metadata.json        # Model info and metrics (not in git)
│
├── src/
│   ├── __init__.py                # Package initialization
│   ├── data_validator.py          # Pydantic schemas (92% coverage)
│   ├── model_loader.py            # Model loading (72% coverage)
│   ├── predictor.py               # Prediction logic (91% coverage)
│   └── api.py                     # Flask API (88% coverage)
│
├── tests/
│   ├── conftest.py                # Pytest configuration
│   ├── test_data_validator.py    # Validation tests (8 tests)
│   ├── test_predictor.py          # Predictor tests (6 tests)
│   └── test_api.py                # API tests (7 tests)
│
├── train_model.py                 # Model training script
├── test_model_loading.py          # Model verification script
├── requirements.txt               # Python dependencies
│
├── .gitignore                     # Git ignore rules
├── README.md                      # Main documentation
├── API_DOCUMENTATION.md           # API reference
├── MODEL_SELECTION.md             # Model comparison
├── TESTING.md                     # Testing guide
├── SETUP.md                       # Setup instructions
└── PROJECT_SUMMARY.md             # This file
```

## Development Timeline

### Days 1-2: Model Training & Serialization
- Downloaded and validated heart attack dataset
- Trained LogisticRegression and RandomForest models
- Selected RandomForest based on recall performance
- Implemented model serialization with pickle
- Created metadata tracking system
- Saved scaler for inference consistency

### Days 3-4: API Development
- Built Pydantic schemas for input validation
- Created model loader for production deployment
- Implemented predictor with error handling
- Developed Flask REST API with 3 endpoints
- Added comprehensive error handling
- Tested all endpoints manually with curl

### Days 5-6: Testing & Quality Assurance
- Created 22 comprehensive tests
- Achieved 85% code coverage
- Wrote unit tests for all components
- Added integration tests for API
- Generated HTML coverage reports
- Documented testing best practices

### Day 7: Documentation & Deployment Prep
- Created comprehensive documentation
- Wrote deployment guide
- Prepared Docker configuration
- Documented lessons learned
- Organized for portfolio presentation

## API Usage Examples

### Health Check
```bash
curl http://localhost:5001/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "RandomForestClassifier"
}
```

### Get Model Information
```bash
curl http://localhost:5001/model-info
```

Response:
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

### Make Prediction
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

Response:
```json
{
  "risk_level": "LOW",
  "probability": 0.21,
  "model_type": "RandomForestClassifier",
  "timestamp": "2026-01-20T23:10:58.586355"
}
```

## Technologies Used

### Core Technologies
- **Python 3.12**: Programming language
- **scikit-learn 1.8.0**: ML model training
- **Flask 3.0.0**: Web framework
- **Pydantic 2.5.0**: Data validation
- **pytest 9.0.2**: Testing framework

### Supporting Libraries
- **pandas 2.3.3**: Data manipulation
- **numpy 2.4.1**: Numerical operations
- **pickle**: Model serialization

### Development Tools
- **Git**: Version control
- **pytest-cov**: Code coverage
- **curl**: API testing

## Production Considerations

### What's Production-Ready
✅ Model serialization and loading
✅ Input validation with clear error messages
✅ Error handling at all layers
✅ Health checks for monitoring
✅ Logging for debugging
✅ Comprehensive testing
✅ Complete documentation

### What's Needed for Full Production
⏳ Database for prediction logging
⏳ Authentication and authorization
⏳ Rate limiting
⏳ Monitoring and alerting
⏳ Docker containerization
⏳ CI/CD pipeline
⏳ Load balancing
⏳ HTTPS/SSL certificates

## Next Steps for Production

1. **Monitoring**: Add prediction logging to database
2. **Security**: Implement authentication (JWT tokens)
3. **Performance**: Add caching and rate limiting
4. **Deployment**: Containerize with Docker
5. **CI/CD**: Set up GitHub Actions
6. **Scaling**: Add load balancer and multiple instances
7. **Observability**: Integrate monitoring tools (Prometheus, Grafana)

## Repository

GitHub: [Repository URL - to be added after push]

## Contact

**Rica Mae Pitogo**  
Master's in Data Science in Health  
LinkedIn: https://www.linkedin.com/in/rica-mae-pitogo-a7aa10193/

---

**Total development time:** 7 days (one week sprint)
