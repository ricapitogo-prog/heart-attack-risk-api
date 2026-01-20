# Heart Attack Risk API Documentation

## Base URL
```
http://localhost:5001
```

## Endpoints

### 1. Health Check
**GET** `/health`

Check if the API is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "RandomForestClassifier"
}
```

---

### 2. Model Information
**GET** `/model-info`

Get detailed information about the loaded model.

**Response:**
```json
{
  "model_type": "RandomForestClassifier",
  "trained_date": "2026-01-20T23:03:01.360056",
  "n_features": 13,
  "feature_names": ["age", "sex", "cp", ...],
  "metrics": {
    "accuracy": 0.8197,
    "precision": 0.7750,
    "recall": 0.9394,
    "f1_score": 0.8493,
    "roc_auc": 0.9037
  }
}
```

---

### 3. Make Prediction
**POST** `/predict`

Predict heart attack risk for a patient.

**Request Body:**
```json
{
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
```

**Field Descriptions:**
- `age`: Patient age (18-100 years)
- `sex`: 0=female, 1=male
- `cp`: Chest pain type (0-3)
- `trtbps`: Resting blood pressure (80-200 mm Hg)
- `chol`: Cholesterol (100-600 mg/dl)
- `fbs`: Fasting blood sugar > 120 mg/dl (0=false, 1=true)
- `restecg`: Resting ECG results (0-2)
- `thalachh`: Maximum heart rate achieved (60-220 bpm)
- `exng`: Exercise induced angina (0=no, 1=yes)
- `oldpeak`: ST depression (0.0-10.0)
- `slp`: Slope of peak exercise ST segment (0-2)
- `caa`: Number of major vessels (0-3)
- `thall`: Thalassemia (0-3)

**Success Response (200):**
```json
{
  "risk_level": "HIGH",
  "probability": 0.64,
  "model_type": "RandomForestClassifier",
  "timestamp": "2026-01-20T23:10:27.018965"
}
```

**Validation Error Response (400):**
```json
{
  "error": "Invalid input: age must be at least 18 years"
}
```

---

## Example Usage

### Using curl:
```bash
# Health check
curl http://localhost:5001/health

# Get model info
curl http://localhost:5001/model-info

# Make prediction
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Using Python requests:
```python
import requests

# Make prediction
response = requests.post(
    'http://localhost:5001/predict',
    json={
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
)

result = response.json()
print(f"Risk Level: {result['risk_level']}")
print(f"Probability: {result['probability']:.2%}")
```

---

## Running the API
```bash
# Activate virtual environment
source venv/bin/activate

# Start the API
python3 src/api.py
```

The API will run on `http://localhost:5001`

---

## Error Handling

The API validates all inputs and returns appropriate error messages:

- **400 Bad Request**: Invalid input data (missing fields, out of range values)
- **404 Not Found**: Invalid endpoint
- **500 Internal Server Error**: Unexpected server error

All errors return JSON with an `error` field describing the issue.

---

## Model Performance

Current production model (RandomForestClassifier):
- Accuracy: 81.97%
- Recall: 93.94% (catches 31 out of 33 high-risk patients)
- ROC-AUC: 90.37%
- False Negatives: Only 2 out of 33 high-risk patients missed

The high recall is critical for healthcare applications where missing high-risk patients is dangerous.
