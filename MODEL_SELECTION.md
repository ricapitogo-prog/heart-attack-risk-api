# Model Selection Documentation

## Models Evaluated

### 1. Logistic Regression
- Accuracy: 78.69%
- Recall: 87.88%
- ROC-AUC: 87.12%
- False Negatives: 4 out of 33 high-risk patients

### 2. Random Forest (SELECTED)
- Accuracy: 81.97%
- Recall: 93.94%
- ROC-AUC: 90.37%
- False Negatives: 2 out of 33 high-risk patients

## Decision Rationale

**Selected Model: Random Forest**

**Reasons:**
1. Higher recall (93.94% vs 87.88%) - critical for healthcare applications
2. Fewer false negatives (2 vs 4) - reduces risk of missing high-risk patients
3. Better overall performance across all metrics
4. More stable cross-validation scores

## Production Considerations

For heart attack risk prediction, recall is the most important metric because:
- False negatives are dangerous (missing high-risk patients)
- False positives are less costly (unnecessary follow-up tests)

The Random Forest model's 93.94% recall means we catch nearly all high-risk patients while maintaining good precision (77.50%).

## Model Artifacts

Current production model saved in `models/`:
- `trained_model.pkl` - RandomForestClassifier
- `scaler.pkl` - StandardScaler fitted on training data
- `model_metadata.json` - Model info and performance metrics

## Training Date
2026-01-20

## Next Steps
- Build Flask API for model deployment
- Implement input validation with Pydantic
- Add prediction logging for monitoring
