"""
Degradation Forecasting - Real-time Prediction
For deployment with live sensor data
"""

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import load_do_imputer


class DegradationPredictor:
    """
    Real-time degradation forecaster for deployment
    Predicts future WQI and estimates time-to-unsafe
    """
    
    def __init__(self, model_path, scaler_path, config_path, do_imputer_path):
        """
        Load trained model and preprocessing components
        
        Args:
            model_path: Path to trained Keras model (.h5)
            scaler_path: Path to feature scaler (.pkl)
            config_path: Path to model configuration (.pkl)
        """
        print("="*70)
        print("DEGRADATION FORECASTING - PREDICTION ENGINE")
        print("="*70)
        print("\n[INIT] Loading model components...")
        
        # Load model
        self.model = tf.keras.models.load_model(model_path, compile=False)
        print(f"[OK] Model loaded from: {model_path}")
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)
        print(f"[OK] Scaler loaded from: {scaler_path}")
        
        # Load config
        self.config = joblib.load(config_path)
        self.lookback = self.config['lookback']
        self.forecast_horizon = self.config['forecast_horizon']
        self.feature_names = self.config['feature_names']
        
        print(f"[OK] Configuration loaded")
        print(f"[INFO] Lookback: {self.lookback} intervals ({self.lookback*3} hours)")
        print(f"[INFO] Forecast: {self.forecast_horizon} intervals ({self.forecast_horizon*3} hours)")
        print(f"[INFO] Features: {self.feature_names}")
        
        self.do_imputer = load_do_imputer(do_imputer_path)
        self.do_feature_names = list(self.do_imputer.feature_names)
        
        print("\n[READY] Predictor initialized and ready for inference")
    
    def _impute_do_in_frame(self, df):
        df = df.copy()
        if 'DO' not in df.columns:
            df['DO'] = np.nan
        mask = df['DO'].isna()
        if mask.any():
            feature_payload = df.loc[:, list(self.do_feature_names)].copy()
            df.loc[mask, 'DO'] = self.do_imputer.predict(
                feature_payload.loc[mask, list(self.do_feature_names)]
            )
        return df
    
    def predict_from_history(self, sensor_history):
        """
        Predict future WQI from sensor history
        
        Args:
            sensor_history: DataFrame or numpy array with columns [pH, Turbidity, TDS, DO, Temperature]
                          Must have exactly 'lookback' rows (most recent at bottom)
        
        Returns:
            dict with predictions and metadata
        """
        # Convert to DataFrame to handle DO imputation
        if isinstance(sensor_history, pd.DataFrame):
            history_df = sensor_history.copy()
        else:
            history_df = pd.DataFrame(sensor_history, columns=self.feature_names)
        
        history_df = self._impute_do_in_frame(history_df)
        X = history_df[self.feature_names].values
        
        # Validate shape
        if X.shape[0] != self.lookback:
            raise ValueError(f"Expected {self.lookback} historical readings, got {X.shape[0]}")
        
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {X.shape[1]}")
        
        # Reshape for LSTM: (1, lookback, features)
        X_reshaped = X.reshape(1, self.lookback, len(self.feature_names))
        
        # Normalize
        X_flat = X_reshaped.reshape(-1, len(self.feature_names))
        X_scaled = self.scaler.transform(X_flat)
        X_scaled = X_scaled.reshape(1, self.lookback, len(self.feature_names))
        
        # Predict
        wqi_forecast = self.model.predict(X_scaled, verbose=0)[0]
        
        return {
            'wqi_forecast': wqi_forecast,
            'forecast_horizon_hours': self.forecast_horizon * 3,
            'forecast_intervals': self.forecast_horizon
        }
    
    def predict_single_reading(self, pH, turbidity, tds, do, temperature, 
                              history_buffer=None, current_wqi=None):
        """
        Predict from a single new sensor reading (requires history)
        
        Args:
            pH, turbidity, tds, do, temperature: Current sensor readings
            history_buffer: Previous readings (DataFrame or list of dicts)
            current_wqi: Current WQI score (optional, for display)
        
        Returns:
            dict with predictions
        """
        # Create current reading
        current_reading = {
            'pH': pH,
            'Turbidity': turbidity,
            'TDS': tds,
            'DO': do,
            'Temperature': temperature
        }
        
        # If no history buffer, we can't predict yet
        if history_buffer is None:
            return {
                'status': 'insufficient_history',
                'message': f'Need {self.lookback} historical readings to predict',
                'required_readings': self.lookback
            }
        
        # Convert history to DataFrame if needed
        if not isinstance(history_buffer, pd.DataFrame):
            history_df = pd.DataFrame(history_buffer)
        else:
            history_df = history_buffer.copy()
        
        # Check if we have enough history
        if len(history_df) < self.lookback:
            return {
                'status': 'insufficient_history',
                'message': f'Have {len(history_df)} readings, need {self.lookback}',
                'required_readings': self.lookback - len(history_df)
            }
        
        # Take last 'lookback' readings
        recent_history = history_df[self.feature_names].iloc[-self.lookback:].values
        
        # Predict
        prediction = self.predict_from_history(recent_history)
        
        # Add current reading info
        prediction['current_reading'] = current_reading
        prediction['current_wqi'] = current_wqi
        prediction['status'] = 'success'
        
        return prediction
    
    def estimate_time_to_unsafe(self, wqi_forecast, unsafe_threshold=50):
        """
        Estimate time until water becomes unsafe
        
        Args:
            wqi_forecast: Array of predicted WQI values
            unsafe_threshold: WQI threshold for unsafe water (default 50)
        
        Returns:
            dict with time-to-unsafe information
        """
        # Find first interval where WQI drops below threshold
        below_threshold = np.where(wqi_forecast < unsafe_threshold)[0]
        
        if len(below_threshold) > 0:
            intervals_to_unsafe = below_threshold[0]
            hours_to_unsafe = intervals_to_unsafe * 3
            
            # Estimate timestamp
            estimated_unsafe_time = datetime.now() + timedelta(hours=hours_to_unsafe)
            
            return {
                'will_become_unsafe': True,
                'intervals_to_unsafe': int(intervals_to_unsafe),
                'hours_to_unsafe': float(hours_to_unsafe),
                'estimated_unsafe_time': estimated_unsafe_time.strftime('%Y-%m-%d %H:%M:%S'),
                'predicted_unsafe_wqi': float(wqi_forecast[intervals_to_unsafe]),
                'alert_level': 'CRITICAL' if hours_to_unsafe < 6 else 'WARNING'
            }
        else:
            # Check final predicted WQI
            final_wqi = wqi_forecast[-1]
            
            return {
                'will_become_unsafe': False,
                'final_predicted_wqi': float(final_wqi),
                'status': 'Safe' if final_wqi >= 70 else 'Degrading' if final_wqi >= unsafe_threshold else 'Borderline',
                'alert_level': 'NORMAL'
            }
    
    def get_full_forecast_report(self, sensor_history, unsafe_threshold=50, current_wqi=None):
        """
        Generate comprehensive forecast report for dashboard
        
        Args:
            sensor_history: Recent sensor readings
            unsafe_threshold: WQI threshold for unsafe classification
            current_wqi: Current WQI score (optional)
        
        Returns:
            Complete forecast report with all information
        """
        # Get prediction
        prediction = self.predict_from_history(sensor_history)
        wqi_forecast = prediction['wqi_forecast']
        
        # Estimate time to unsafe
        time_estimate = self.estimate_time_to_unsafe(wqi_forecast, unsafe_threshold)
        
        # Create forecast timeline
        forecast_timeline = []
        for i, wqi in enumerate(wqi_forecast):
            forecast_timeline.append({
                'interval': i,
                'hours_ahead': (i + 1) * 3,
                'predicted_wqi': float(wqi),
                'status': 'Safe' if wqi >= 70 else 'Degrading' if wqi >= unsafe_threshold else 'Unsafe',
                'timestamp': (datetime.now() + timedelta(hours=(i+1)*3)).strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # Assemble report
        report = {
            'prediction_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_wqi': current_wqi,
            'forecast_horizon_hours': self.forecast_horizon * 3,
            'wqi_forecast': [float(x) for x in wqi_forecast],
            'forecast_timeline': forecast_timeline,
            'time_to_unsafe': time_estimate,
            'min_predicted_wqi': float(wqi_forecast.min()),
            'max_predicted_wqi': float(wqi_forecast.max()),
            'trend': 'Improving' if wqi_forecast[-1] > wqi_forecast[0] else 'Degrading',
            'confidence': 'High' if len(sensor_history) >= self.lookback else 'Medium'
        }
        
        return report
    
    def format_alert_message(self, report):
        """
        Format forecast report into human-readable alert message
        
        Args:
            report: Forecast report from get_full_forecast_report()
        
        Returns:
            Formatted alert message string
        """
        time_info = report['time_to_unsafe']
        
        if time_info['will_become_unsafe']:
            hours = time_info['hours_to_unsafe']
            unsafe_time = time_info['estimated_unsafe_time']
            
            message = f"""
╔══════════════════════════════════════════════════════════════╗
║  WATER QUALITY DEGRADATION ALERT - {time_info['alert_level']}
╚══════════════════════════════════════════════════════════════╝

⚠️  WATER WILL BECOME UNSAFE IN ~{hours:.1f} HOURS
📅  Estimated unsafe time: {unsafe_time}
📊  Predicted WQI at that time: {time_info['predicted_unsafe_wqi']:.1f}

Current Status:
  • Current WQI: {report['current_wqi']:.1f if report['current_wqi'] else 'N/A'}
  • Trend: {report['trend']}
  • Min predicted WQI (next 12h): {report['min_predicted_wqi']:.1f}

Forecast Timeline:
"""
            for item in report['forecast_timeline']:
                status_emoji = '✅' if item['status'] == 'Safe' else '⚠️' if item['status'] == 'Degrading' else '🚨'
                message += f"  {status_emoji} +{item['hours_ahead']}h: WQI {item['predicted_wqi']:.1f} ({item['status']})\n"
            
            message += "\n🔔 ACTION REQUIRED: Alert authorities and prepare contamination detection\n"
        
        else:
            message = f"""
╔══════════════════════════════════════════════════════════════╗
║  WATER QUALITY FORECAST - {time_info['alert_level']}
╚══════════════════════════════════════════════════════════════╝

✅  Water quality expected to remain safe for next {self.forecast_horizon * 3} hours

Current Status:
  • Current WQI: {report['current_wqi']:.1f if report['current_wqi'] else 'N/A'}
  • Trend: {report['trend']}
  • Final predicted WQI: {time_info['final_predicted_wqi']:.1f}

Forecast Timeline:
"""
            for item in report['forecast_timeline']:
                status_emoji = '✅' if item['status'] == 'Safe' else '⚠️' if item['status'] == 'Degrading' else '🚨'
                message += f"  {status_emoji} +{item['hours_ahead']}h: WQI {item['predicted_wqi']:.1f} ({item['status']})\n"
            
            message += "\n📊 Continue monitoring\n"
        
        return message


def example_usage():
    """Example of how to use the predictor"""
    print("\n" + "="*70)
    print("EXAMPLE USAGE")
    print("="*70)
    
    # Initialize predictor
    predictor = DegradationPredictor(
        model_path='degradation_model.h5',
        scaler_path='degradation_scaler.pkl',
        config_path='degradation_config.pkl',
        do_imputer_path='../common/do_imputer.pkl',
    )
    
    print("\n[EXAMPLE 1] Predict from historical data array")
    print("-" * 70)
    
    # Simulate 4 recent sensor readings (last 12 hours, 3-hour intervals)
    sensor_history = np.array([
        # [pH, Turbidity, TDS, DO, Temperature]
        [7.2, 2.5, 180, 8.5, 22.0],  # 12 hours ago
        [7.1, 3.0, 190, 8.2, 23.0],  # 9 hours ago
        [7.0, 4.5, 210, 7.8, 23.5],  # 6 hours ago
        [6.9, 6.0, 230, 7.2, 24.0],  # 3 hours ago (most recent)
    ])
    
    # Get full forecast report
    report = predictor.get_full_forecast_report(
        sensor_history,
        unsafe_threshold=50,
        current_wqi=65.0  # Assume current WQI is 65
    )
    
    # Print alert message
    alert = predictor.format_alert_message(report)
    print(alert)
    
    print("\n[EXAMPLE 2] Predict from DataFrame")
    print("-" * 70)
    
    # Create DataFrame with historical readings
    history_df = pd.DataFrame([
        {'pH': 7.3, 'Turbidity': 1.5, 'TDS': 160, 'DO': 9.0, 'Temperature': 21.0},
        {'pH': 7.2, 'Turbidity': 2.0, 'TDS': 170, 'DO': 8.8, 'Temperature': 21.5},
        {'pH': 7.1, 'Turbidity': 2.5, 'TDS': 180, 'DO': 8.5, 'Temperature': 22.0},
        {'pH': 7.0, 'Turbidity': 3.0, 'TDS': 190, 'DO': 8.2, 'Temperature': 22.5},
    ])
    
    prediction = predictor.predict_from_history(history_df)
    print(f"Predicted WQI for next {prediction['forecast_horizon_hours']} hours:")
    for i, wqi in enumerate(prediction['wqi_forecast']):
        hours = (i+1) * 3
        print(f"  +{hours}h: {wqi:.2f}")
    
    print("\n[INFO] Integration with WQI model:")
    print("  1. WQI model classifies current reading as 'Degrading'")
    print("  2. Degradation forecaster receives last 4 readings")
    print("  3. Predicts future WQI and estimates time-to-unsafe")
    print("  4. Dashboard displays forecast and alerts authorities")


if __name__ == "__main__":
    example_usage()

