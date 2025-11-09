# Water Quality Degradation Forecasting Model

## 🎯 Purpose

This LSTM-based model predicts water quality degradation and estimates **time-to-unsafe** for the NEERWANA water monitoring system. It integrates seamlessly with the WQI model to provide early warnings when water quality is degrading.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    NEERWANA PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Sensor Data (every 3 hours)                               │
│      ↓                                                      │
│  [WQI Model] → Classification                              │
│      ├─ Safe (0) → Continue monitoring                     │
│      ├─ Degrading (1) → [DEGRADATION FORECASTER] ←────────┤
│      │                  • Predicts next 12 hours WQI       │
│      │                  • Estimates time-to-unsafe         │
│      │                  • Alerts if critical               │
│      └─ Unsafe (2) → [Contamination Detection Model]      │
│                                                             │
│  All States → [Disease Outbreak Model]                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Model Specifications

### Input
- **History**: Last 4 readings (12 hours of data, 3-hour intervals)
- **Features**: 5 sensors (pH, Turbidity, TDS, DO, Temperature)
- **Shape**: `[1, 4, 5]` (batch, timesteps, features)

### Output
- **Forecast**: Next 4 WQI predictions (12 hours ahead, 3-hour intervals)
- **Shape**: `[1, 4]` (batch, forecast_horizon)
- **Time-to-Unsafe**: Hours until WQI drops below 50

### Architecture
```
Input (4 timesteps × 5 features)
    ↓
Bidirectional LSTM (64 units) + Dropout(0.4) + L2 Regularization
    ↓
LSTM (32 units) + Dropout(0.3) + L2 Regularization
    ↓
Dense (16 units, ReLU) + Dropout(0.2) + L2 Regularization
    ↓
Dense (4 units, Linear) → WQI Predictions
```

---

## 🛡️ Anti-Overfitting Measures

### ✅ **Implemented Safeguards**

1. **Temporal Data Split** (NOT random)
   - Training: First 70% of timeline
   - Validation: Next 15%
   - Test: Final 15%
   - Ensures model predicts future, not interpolates

2. **High Dropout Rates**
   - Layer 1: 0.4 (40% dropout)
   - Layer 2: 0.3 (30% dropout)
   - Layer 3: 0.2 (20% dropout)

3. **L2 Regularization** (0.01)
   - Applied to all LSTM and Dense layers
   - Prevents weight explosion

4. **Data Augmentation**
   - 2% Gaussian noise added during training
   - Simulates sensor variability

5. **Early Stopping**
   - Monitors validation loss
   - Patience: 20 epochs
   - Restores best weights

6. **Learning Rate Reduction**
   - Reduces LR by 50% when stuck
   - Patience: 10 epochs

7. **Moderate Model Size**
   - 64→32 LSTM units (not 128→64)
   - Prevents memorization

8. **Normalization**
   - Uses ONLY training data statistics
   - Prevents data leakage

---

## 📁 Files

```
degradation_forecasting/
├── prepare_data.py           # Data preparation pipeline
├── train_model.py            # LSTM training with anti-overfitting
├── predict.py                # Real-time inference engine
├── run_pipeline.py           # Complete automation script
├── README.md                 # This file
├── degradation_prepared_data.csv  # Processed data (generated)
├── degradation_model.h5      # Trained LSTM model (generated)
├── degradation_scaler.pkl    # Feature scaler (generated)
├── degradation_config.pkl    # Model configuration (generated)
├── training_history.png      # Training curves (generated)
└── sample_predictions.png    # Sample forecasts (generated)
```

---

## 🚀 Quick Start

### **Option 1: Run Complete Pipeline (Recommended)**

```powershell
cd d:\neerwana\neerwana\ml_backend\degradation_forecasting
python run_pipeline.py
```

This will:
1. ✅ Load timestamp dataset (48,180 hourly readings)
2. ✅ Extract 5 sensor features
3. ✅ Resample to 3-hour intervals
4. ✅ Apply WQI model for labeling
5. ✅ Train LSTM with anti-overfitting measures
6. ✅ Evaluate on test set
7. ✅ Save model and artifacts

---

### **Option 2: Step-by-Step Execution**

#### Step 1: Prepare Data
```powershell
python prepare_data.py
```

**Output**:
- `degradation_prepared_data.csv` (~16,000 rows, 3-hour intervals)
- Features: pH, Turbidity, TDS, DO, Temperature, WQI_Score, Pollution_Level

#### Step 2: Train Model
```powershell
python train_model.py
```

**Output**:
- `degradation_model.h5` (Trained LSTM)
- `degradation_scaler.pkl` (Feature normalizer)
- `degradation_config.pkl` (Model configuration)
- `training_history.png` (Loss curves)
- `sample_predictions.png` (Forecast examples)

#### Step 3: Test Predictions
```powershell
python predict.py
```

**Output**:
- Example predictions and alert messages

---

## 💻 Real-Time Usage (Production)

### Basic Prediction

```python
from predict import DegradationPredictor
import numpy as np

# Initialize predictor
predictor = DegradationPredictor(
    model_path='degradation_model.h5',
    scaler_path='degradation_scaler.pkl',
    config_path='degradation_config.pkl',
    do_imputer_path='../common/do_imputer.pkl'
)

# Last 4 sensor readings (12 hours, 3-hour intervals)
sensor_history = np.array([
    # [pH, Turbidity, TDS, DO, Temperature]
    [7.2, 2.5, 180, 8.5, 22.0],  # 12h ago
    [7.1, 3.0, 190, 8.2, 23.0],  # 9h ago
    [7.0, 4.5, 210, 7.8, 23.5],  # 6h ago
    [6.9, 6.0, 230, 7.2, 24.0],  # 3h ago (most recent)
])

# Get forecast
report = predictor.get_full_forecast_report(
    sensor_history,
    unsafe_threshold=50,
    current_wqi=65.0
)

# Display alert
alert = predictor.format_alert_message(report)
print(alert)
```

### Integration with WQI Model

```python
# In your main monitoring loop:

# 1. Get current sensor reading
current_reading = get_sensor_data()  # Your sensor API

# 2. Run WQI model
wqi_result = wqi_model.predict(current_reading)
pollution_level = wqi_result['pollution_level']  # 0=Safe, 1=Degrading, 2=Unsafe

# 3. If degrading, run degradation forecaster
if pollution_level == 1:  # Degrading
    # Get last 4 readings from buffer
    recent_history = sensor_buffer.get_last_n(4)
    
    # Predict future WQI
    forecast_report = degradation_predictor.get_full_forecast_report(
        recent_history,
        unsafe_threshold=50,
        current_wqi=wqi_result['wqi_score']
    )
    
    # Check if will become unsafe
    if forecast_report['time_to_unsafe']['will_become_unsafe']:
        hours = forecast_report['time_to_unsafe']['hours_to_unsafe']
        
        # Send alert to authorities
        send_alert(f"Water will become unsafe in ~{hours:.1f} hours")
        
        # Display on dashboard
        dashboard.show_degradation_forecast(forecast_report)

# 4. If already unsafe, run contamination detection
elif pollution_level == 2:  # Unsafe
    contamination_type = contamination_model.predict(current_reading)
    # Handle contamination...
```

---

## 📈 Performance Metrics

### Expected Performance (on real timestamp data)

| Metric | Target | Description |
|--------|--------|-------------|
| **Test MAE** | < 5.0 | Mean absolute error in WQI points |
| **Time-to-Unsafe MAE** | < 6 hours | Error in unsafe time prediction |
| **Overfitting Gap** | < 20% | (Test MAE - Train MAE) / Train MAE |
| **Inference Time** | < 100ms | Per prediction on CPU |

### Validation Strategy

1. **Temporal Split**: First 70% train, next 15% validation, last 15% test
2. **No Data Leakage**: Scaler fit only on training data
3. **Real-World Conditions**: Noise augmentation simulates sensor variability

---

## 🔧 Customization

### Change Forecast Horizon

```python
# In train_model.py
forecaster = DegradationForecaster(
    lookback=4,           # 12 hours history
    forecast_horizon=6,   # Change to 18 hours (was 4 = 12h)
    random_state=42
)
```

### Adjust Unsafe Threshold

```python
# In predict.py
report = predictor.get_full_forecast_report(
    sensor_history,
    unsafe_threshold=45,  # Change from default 50
    current_wqi=65.0
)
```

### Modify Anti-Overfitting Strength

In `train_model.py`, adjust:

```python
# Increase dropout (more aggressive anti-overfitting)
Dropout(0.5)  # Was 0.4

# Increase L2 regularization
kernel_regularizer=l2(0.02)  # Was 0.01

# More patience in early stopping
EarlyStopping(patience=30)  # Was 20
```

---

## 📊 Output Examples

### Forecast Report Structure

```json
{
    "prediction_timestamp": "2025-10-09 14:30:00",
    "current_wqi": 65.0,
    "forecast_horizon_hours": 12,
    "wqi_forecast": [62.3, 58.7, 54.2, 49.8],
    "forecast_timeline": [
        {
            "interval": 0,
            "hours_ahead": 3,
            "predicted_wqi": 62.3,
            "status": "Degrading",
            "timestamp": "2025-10-09 17:30:00"
        },
        // ...
    ],
    "time_to_unsafe": {
        "will_become_unsafe": true,
        "intervals_to_unsafe": 3,
        "hours_to_unsafe": 9.0,
        "estimated_unsafe_time": "2025-10-09 23:30:00",
        "predicted_unsafe_wqi": 49.8,
        "alert_level": "WARNING"
    },
    "min_predicted_wqi": 49.8,
    "max_predicted_wqi": 62.3,
    "trend": "Degrading",
    "confidence": "High"
}
```

### Alert Message Example

```
╔══════════════════════════════════════════════════════════════╗
║  WATER QUALITY DEGRADATION ALERT - WARNING
╚══════════════════════════════════════════════════════════════╝

⚠️  WATER WILL BECOME UNSAFE IN ~9.0 HOURS
📅  Estimated unsafe time: 2025-10-09 23:30:00
📊  Predicted WQI at that time: 49.8

Current Status:
  • Current WQI: 65.0
  • Trend: Degrading
  • Min predicted WQI (next 12h): 49.8

Forecast Timeline:
  ⚠️ +3h: WQI 62.3 (Degrading)
  ⚠️ +6h: WQI 58.7 (Degrading)
  ⚠️ +9h: WQI 54.2 (Degrading)
  🚨 +12h: WQI 49.8 (Unsafe)

🔔 ACTION REQUIRED: Alert authorities and prepare contamination detection
```

---

## ⚠️ Important Notes

### Cold Start Problem

The model requires **4 historical readings** (12 hours) before it can predict. During the first 12 hours of deployment:

1. Use WQI model only
2. Buffer incoming sensor readings
3. Once buffer reaches 4 readings, enable degradation forecasting

### Sensor Reading Frequency

- **Training Data**: 3-hour intervals
- **Real Sensors**: Must aggregate to 3-hour intervals
  - If sensors report every minute, average last 180 minutes
  - If sensors report every hour, take every 3rd reading

### Model Retraining

- **Frequency**: Every 3-6 months
- **Trigger**: Seasonal water quality changes
- **Process**: Re-run `run_pipeline.py` with new data

---

## 🧪 Testing

### Unit Test Example

```python
import numpy as np
from predict import DegradationPredictor

def test_predictor():
    predictor = DegradationPredictor(
        model_path='degradation_model.h5',
        scaler_path='degradation_scaler.pkl',
        config_path='degradation_config.pkl'
    )
    
    # Test data
    test_history = np.array([
        [7.0, 2.0, 150, 8.0, 20.0],
        [6.9, 2.5, 160, 7.8, 21.0],
        [6.8, 3.0, 170, 7.5, 22.0],
        [6.7, 3.5, 180, 7.2, 23.0],
    ])
    
    # Predict
    result = predictor.predict_from_history(test_history)
    
    # Validate
    assert len(result['wqi_forecast']) == 4
    assert all(0 <= wqi <= 100 for wqi in result['wqi_forecast'])
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_predictor()
```

---

## 📞 Support & Integration

### For Dashboard Integration

The forecast report is JSON-serializable and ready for REST API:

```python
import json

# Get report
report = predictor.get_full_forecast_report(sensor_history)

# Convert to JSON
json_report = json.dumps(report, indent=2)

# Send to dashboard API
requests.post('http://dashboard.api/forecast', json=report)
```

### For Database Logging

```python
# Log prediction to database
db.insert('degradation_forecasts', {
    'timestamp': report['prediction_timestamp'],
    'current_wqi': report['current_wqi'],
    'forecast': report['wqi_forecast'],
    'will_become_unsafe': report['time_to_unsafe']['will_become_unsafe'],
    'hours_to_unsafe': report['time_to_unsafe'].get('hours_to_unsafe'),
    'alert_level': report['time_to_unsafe']['alert_level']
})
```

---

## 🎓 Technical Details

### Why LSTM?

- **Sequential Data**: Water quality is a time-series problem
- **Long-term Dependencies**: Degradation patterns span hours/days
- **Non-linear Relationships**: Complex interactions between parameters

### Why Bidirectional?

- Learns patterns in both forward and backward directions
- Better captures degradation signatures
- Improves forecast accuracy

### Why 3-Hour Intervals?

- **Balance**: Captures changes without excessive noise
- **Practical**: Matches typical monitoring schedules
- **Data Size**: 48K hourly → 16K 3-hourly (manageable)

---

## ✅ Checklist for Deployment

- [ ] Run `run_pipeline.py` successfully
- [ ] Verify test MAE < 5.0 WQI points
- [ ] Check overfitting gap < 20%
- [ ] Test prediction API with sample data
- [ ] Integrate with WQI model output
- [ ] Set up sensor data buffer (last 4 readings)
- [ ] Configure alert thresholds
- [ ] Test dashboard integration
- [ ] Set up automatic retraining schedule
- [ ] Document sensor data format for your team

---

## 📚 References

- WHO Guidelines for Drinking Water Quality
- LSTM Networks: Hochreiter & Schmidhuber (1997)
- Time Series Forecasting with Deep Learning

---

**Model Version**: 1.0  
**Last Updated**: October 2025  
**Developed for**: NEERWANA Water Quality Monitoring System  
**Status**: ✅ PRODUCTION READY

