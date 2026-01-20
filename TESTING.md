# Testing Documentation

## Test Suite Overview

This project includes a comprehensive test suite with **85% code coverage**.

## Test Structure
```
tests/
├── conftest.py              # Pytest configuration
├── test_data_validator.py   # Validation schema tests (92% coverage)
├── test_predictor.py        # Prediction logic tests (91% coverage)
└── test_api.py              # API endpoint tests (88% coverage)
```

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run with coverage report
```bash
pytest tests/ -v --cov=src --cov-report=term --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_api.py -v
```

### Run specific test
```bash
pytest tests/test_api.py::TestPredictEndpoint::test_valid_prediction -v
```

## Test Categories

### 1. Data Validation Tests (8 tests)
**File:** `test_data_validator.py`

Tests Pydantic schemas for input validation:
- Valid patient data acceptance
- Feature array conversion
- Age validation (too young, too old)
- Blood pressure validation
- Cholesterol validation
- Missing field detection
- Response schema validation

### 2. Predictor Tests (6 tests)
**File:** `test_predictor.py`

Tests ML prediction logic:
- Model initialization
- Low risk predictions
- High risk predictions
- Invalid feature count handling
- Model information retrieval
- Prediction consistency

### 3. API Integration Tests (7 tests)
**File:** `test_api.py`

Tests Flask API endpoints:
- Health check endpoint
- Model info endpoint
- Valid predictions
- Missing field handling
- Invalid age handling
- Invalid JSON handling
- 404 error handling

## Test Coverage Report

Current coverage: **85%**

| Module | Coverage |
|--------|----------|
| data_validator.py | 92% |
| predictor.py | 91% |
| api.py | 88% |
| model_loader.py | 72% |

View detailed HTML coverage report:
```bash
open htmlcov/index.html
```

## Continuous Testing

### Before committing code
```bash
# Run tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src --cov-report=term
```

### Test-Driven Development
1. Write test first
2. Run test (should fail)
3. Write code to make test pass
4. Refactor
5. Repeat

## Adding New Tests

### Example: Adding a new validation test
```python
# tests/test_data_validator.py

def test_heart_rate_too_low(self):
    """Test that heart rate < 60 is rejected."""
    data = {
        "age": 45,
        "sex": 1,
        "cp": 0,
        "trtbps": 120,
        "chol": 200,
        "fbs": 0,
        "restecg": 0,
        "thalachh": 50,  # Too low
        "exng": 0,
        "oldpeak": 0.0,
        "slp": 2,
        "caa": 0,
        "thall": 2
    }
    
    with pytest.raises(ValidationError):
        PatientData(**data)
```

### Example: Adding a new API test
```python
# tests/test_api.py

def test_batch_predictions(self, client):
    """Test multiple predictions in sequence."""
    patients = [
        {...},  # Patient 1
        {...},  # Patient 2
    ]
    
    for patient in patients:
        response = client.post(
            '/predict',
            data=json.dumps(patient),
            content_type='application/json'
        )
        assert response.status_code == 200
```

## Test Fixtures

### Available fixtures (in conftest.py)

**predictor**: Loads model and creates predictor instance
```python
def test_something(predictor):
    result = predictor.predict_risk([...])
    assert result is not None
```

**client**: Flask test client for API testing
```python
def test_endpoint(client):
    response = client.get('/health')
    assert response.status_code == 200
```

## Best Practices

1. **One concept per test** - Each test should verify one specific behavior
2. **Descriptive names** - Test names should clearly describe what they test
3. **Arrange-Act-Assert** - Structure tests clearly
4. **Independent tests** - Tests should not depend on each other
5. **Fast tests** - Keep tests quick to encourage frequent running

## Troubleshooting

### Import errors
```bash
# Make sure you're in project root
cd /path/to/heart_attack_risk_api

# Activate virtual environment
source venv/bin/activate

# Run tests
pytest tests/ -v
```

### Model not found errors
```bash
# Train model first
python3 train_model.py

# Then run tests
pytest tests/ -v
```

### Coverage not showing
```bash
# Install pytest-cov
pip install pytest-cov

# Run with coverage
pytest tests/ --cov=src
```

## CI/CD Integration

For automated testing in CI/CD pipelines:
```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest tests/ -v --cov=src --cov-report=xml
    
- name: Check coverage threshold
  run: |
    coverage report --fail-under=80
```

## Next Steps

- Achieve 90%+ coverage by testing error paths
- Add performance tests for API response times
- Add load testing for concurrent requests
- Add integration tests with database logging
