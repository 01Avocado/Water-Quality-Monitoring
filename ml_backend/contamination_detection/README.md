# Contamination Cause Detection Model

## Overview
This model detects and classifies water contamination causes into 5 categories based on real-time sensor readings. It's part of the Neerwana water quality monitoring system.

## Model Performance
- **Test Accuracy**: 97.70%
- **Cross-Validation Accuracy**: 97.05% (±1.11%)
- **Out-of-Bag Score**: 97.12%
- **Model Type**: Random Forest (100 trees)
- **Anti-Overfitting Measures**: Limited depth (15), min samples split (10), OOB validation

## Contamination Classes

### 1. Safe Water
- **Description**: Water quality within safe parameters
- **Characteristics**: Normal pH, low turbidity, moderate TDS, good DO
- **Action**: Continue monitoring

### 2. Microbial/Sewage Contamination
- **Description**: Organic waste contamination
- **Characteristics**: pH 5.8-7.0, high turbidity (8-120 NTU), low DO (0-4 mg/L)
- **Severity**: HIGH
- **Action**: Check sewage systems, inspect pipes for leaks, recommend chlorination

### 3. Chemical/Detergent Contamination
- **Description**: Chemical surfactants and detergents
- **Characteristics**: High pH (7.2-9.2), elevated TDS (600-2200 mg/L), moderate turbidity
- **Severity**: MEDIUM
- **Action**: Inspect for industrial discharge, check household connections

### 4. Pipe Corrosion
- **Description**: Metal leaching from corroded pipes
- **Characteristics**: Acidic pH (5.2-7.2), very high TDS (700-2700 mg/L), low turbidity
- **Severity**: HIGH
- **Action**: Inspect pipes for corrosion, consider pipe replacement, test for heavy metals

### 5. Natural Sediment/Rust
- **Description**: Natural sediment and rust particles
- **Characteristics**: Normal pH (6.5-8.0), very high turbidity (15-120 NTU), normal DO
- **Severity**: LOW
- **Action**: Check filtration systems, inspect for sediment sources, flush pipelines

## Input Parameters
The model requires 5 sensor readings:
1. **pH**: 5.0 - 9.5 (water acidity/alkalinity)
2. **Turbidity**: 0 - 150 NTU (water clarity)
3. **TDS**: 0 - 3000 mg/L (total dissolved solids)
4. **DO**: 0 - 10 mg/L (dissolved oxygen)
5. **Temperature**: 10 - 40°C (water temperature)

## Feature Importance
1. **Turbidity**: 35.83% (Most important)
2. **TDS**: 24.93%
3. **Dissolved Oxygen**: 19.32%
4. **pH**: 18.70%
5. **Temperature**: 1.22%

## Dataset
- **Total Samples**: 5,000
- **Training Set**: 4,000 (80%)
- **Test Set**: 1,000 (20%)
- **Class Distribution**: 40% safe, 15% each contamination type
- **Anti-Overfitting Features**:
  - 7% missing values (handled with median imputation)
  - 3% outliers
  - 10% multi-contamination scenarios
  - Sensor noise (±3%)
  - Seasonal variations
  - Time-of-day variations

## Files Generated

### Model Files
- `contamination_model.pkl` - Trained Random Forest model
- `imputer.pkl` - Median imputer for missing values
- `label_encoder.pkl` - Label encoder for class names

### Data Files
- `contamination_dataset.csv` - Synthetic training dataset (5,000 samples)

### Visualizations
- `confusion_matrix.png` - Model performance visualization
- `feature_importance.png` - Feature importance chart

### Reports
- `model_performance.txt` - Detailed performance metrics

### Scripts
- `dataset_generator.py` - Generates synthetic dataset
- `train_model.py` - Trains and evaluates model
- `predict.py` - Real-time prediction script
- `run_pipeline.py` - Complete pipeline runner

## Usage

### Option 1: Run Complete Pipeline
```powershell
python ml_backend\contamination_detection\run_pipeline.py
```

### Option 2: Run Individual Steps

**Generate Dataset:**
```powershell
python ml_backend\contamination_detection\dataset_generator.py
```

**Train Model:**
```powershell
python ml_backend\contamination_detection\train_model.py
```

**Test Predictions:**
```powershell
python ml_backend\contamination_detection\predict.py
```

### Option 3: Use in Your Code

```python
from ml_backend.contamination_detection.predict import ContaminationPredictor

# Load model
predictor = ContaminationPredictor(
    model_path='ml_backend/contamination_detection/contamination_model.pkl',
    imputer_path='ml_backend/contamination_detection/imputer.pkl',
    encoder_path='ml_backend/contamination_detection/label_encoder.pkl',
    do_imputer_path='ml_backend/common/do_imputer.pkl'
)

# Single prediction
result = predictor.predict_single(
    pH=6.3,
    turbidity=55.0,
    tds=1200,
    do=1.5,
    temperature=30
)

print(f"Contamination Type: {result['contamination_type']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch prediction
import pandas as pd
df = pd.DataFrame({
    'pH': [7.2, 6.3, 8.5],
    'turbidity_ntu': [2.0, 55.0, 18.0],
    'tds_mg/l': [250, 1200, 1600],
    'do_mg/l': [7.5, 1.5, 4.0],
    'temperature_c': [22, 30, 24]
})
predictions = predictor.predict_batch(df)
```

## Integration with Real-time Sensors

The model is designed to work with real-time sensor data:

```python
# Example integration with sensor API
def monitor_water_quality(sensor_readings):
    result = predictor.predict_single(
        pH=sensor_readings['pH'],
        turbidity=sensor_readings['turbidity'],
        tds=sensor_readings['tds'],
        do=sensor_readings['do'],
        temperature=sensor_readings['temperature']
    )
    
    if result['contamination_type'] != 'safe':
        # Alert authorities
        desc = predictor.get_contamination_description(
            result['contamination_type']
        )
        send_alert(
            contamination=result['contamination_type'],
            confidence=result['confidence'],
            severity=desc['severity'],
            action=desc['action']
        )
    
    return result
```

## Model Robustness

The model is designed to handle:
- ✅ Missing sensor values (automatically imputed)
- ✅ Sensor noise and fluctuations
- ✅ Overlapping parameter ranges between classes
- ✅ Multi-contamination scenarios
- ✅ Seasonal variations
- ✅ Time-of-day variations
- ✅ Outliers and anomalies

## Performance Metrics by Class

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Chemical/Detergent | 97% | 95% | 96% | 150 |
| Microbial/Sewage | 98% | 99% | 98% | 150 |
| Natural Sediment | 95% | 97% | 96% | 150 |
| Pipe Corrosion | 96% | 97% | 97% | 150 |
| Safe Water | 99% | 99% | 99% | 400 |

**Overall Accuracy**: 97.7%

## Why Random Forest?

1. **Handles non-linear relationships** between water parameters
2. **Robust to noise** and outliers in sensor data
3. **Provides feature importance** for understanding model decisions
4. **Out-of-bag validation** for reliable performance estimates
5. **Fast inference** for real-time predictions
6. **No assumption** about data distribution
7. **Works well with missing values** after imputation

## Future Improvements

- [ ] Add confidence thresholds for alerts
- [ ] Implement online learning for model updates
- [ ] Add anomaly detection for unknown contamination types
- [ ] Include temporal patterns for degradation forecasting
- [ ] Add explainability features (SHAP values)
- [ ] Validate with real contamination data when available

## Contact & Support

For questions about the contamination detection model:
- Model Developer: NEERWANA Team
- Model Version: 1.0
- Last Updated: 2025-01-08
- Python Version: 3.12+
- Required Packages: numpy, pandas, scikit-learn, matplotlib, seaborn, joblib

---

**Note**: This model is trained on synthetic data based on research-backed contamination signatures. Validation with real contamination events is recommended before deployment in critical systems.

