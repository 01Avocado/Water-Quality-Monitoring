"""
Contamination Cause Detection - Real-time Inference
Predicts contamination cause from sensor readings
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import load_do_imputer

class ContaminationPredictor:
    """
    Real-time predictor for contamination detection
    """
    
    def __init__(self, model_path, imputer_path, encoder_path, do_imputer_path):
        """Load trained model and preprocessing objects"""
        self.model = joblib.load(model_path)
        self.imputer = joblib.load(imputer_path)
        self.label_encoder = joblib.load(encoder_path)
        self.feature_names = ['pH', 'turbidity_ntu', 'tds_mg/l', 'do_mg/l', 'temperature_c']
        self.do_imputer = load_do_imputer(do_imputer_path)
        self.do_feature_names = list(self.do_imputer.feature_names)
        
    def _maybe_impute_do(self, pH, tds, temperature, do):
        """Return (do_value, flag) where flag=1 if imputed."""
        if do is not None and not np.isnan(do):
            return do, 0
        
        feature_payload = pd.DataFrame(
            [{
                'pH': pH,
                'Temperature': temperature,
                'TDS': tds
            }]
        )
        predicted_do = float(
            self.do_imputer.predict(feature_payload[list(self.do_feature_names)])[0]
        )
        return predicted_do, 1
    
    def predict_single(self, pH, turbidity, tds, do, temperature):
        """
        Predict contamination type from sensor readings
        
        Args:
            pH: pH value (5.0 - 9.5)
            turbidity: Turbidity in NTU (0 - 150)
            tds: Total Dissolved Solids in mg/L (0 - 3000)
            do: Dissolved Oxygen in mg/L (0 - 10)
            temperature: Temperature in Celsius (10 - 40)
            
        Returns:
            dict: Prediction result with contamination type and probabilities
        """
        do_value, do_imputed = self._maybe_impute_do(pH, tds, temperature, do)
        # Create feature array
        features = np.array([[pH, turbidity, tds, do_value, temperature]])
        
        # Handle missing values (if any sensor fails)
        features_imputed = self.imputer.transform(features)
        
        # Predict
        prediction_encoded = self.model.predict(features_imputed)[0]
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get probabilities
        probabilities = self.model.predict_proba(features_imputed)[0]
        prob_dict = {
            class_name: float(prob) 
            for class_name, prob in zip(self.label_encoder.classes_, probabilities)
        }
        
        return {
            'contamination_type': prediction,
            'confidence': float(probabilities[prediction_encoded]),
            'all_probabilities': prob_dict,
            'input_values': {
                'pH': pH,
                'turbidity_ntu': turbidity,
                'tds_mg/l': tds,
                'do_mg/l': do_value,
                'temperature_c': temperature,
                'do_imputed_flag': do_imputed
            },
            'do_imputed': bool(do_imputed)
        }
    
    def predict_batch(self, df):
        """
        Predict contamination for multiple samples
        
        Args:
            df: DataFrame with columns [pH, turbidity_ntu, tds_mg/l, do_mg/l, temperature_c]
            
        Returns:
            DataFrame with predictions
        """
        df = df.copy()
        df['do_imputed_flag'] = 0
        
        missing_mask = df['do_mg/l'].isna()
        if missing_mask.any():
            do_features = df.loc[missing_mask, ['pH', 'tds_mg/l', 'temperature_c']].rename(
                columns={
                    'tds_mg/l': 'TDS',
                    'temperature_c': 'Temperature'
                }
            )
            do_features['pH'] = df.loc[missing_mask, 'pH']
            imputed_values = self.do_imputer.predict(
                do_features[list(self.do_feature_names)]
            )
            df.loc[missing_mask, 'do_mg/l'] = imputed_values
            df.loc[missing_mask, 'do_imputed_flag'] = 1
        
        # Extract features
        X = df[self.feature_names].values
        
        # Handle missing values
        X_imputed = self.imputer.transform(X)
        
        # Predict
        predictions_encoded = self.model.predict(X_imputed)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        # Get probabilities
        probabilities = self.model.predict_proba(X_imputed)
        max_probs = probabilities.max(axis=1)
        
        # Create result DataFrame
        result_df = df.copy()
        result_df['predicted_contamination'] = predictions
        result_df['confidence'] = max_probs
        
        return result_df
    
    def get_contamination_description(self, contamination_type):
        """Get description and recommendations for contamination type"""
        descriptions = {
            'safe': {
                'description': 'Water quality is within safe parameters',
                'action': 'Continue monitoring',
                'severity': 'None'
            },
            'microbial_sewage': {
                'description': 'Microbial/Sewage contamination detected - Low DO, high turbidity, organic waste',
                'action': 'ALERT: Check sewage systems, inspect pipes for leaks, recommend chlorination',
                'severity': 'HIGH'
            },
            'chemical_detergent': {
                'description': 'Chemical/Detergent contamination detected - High pH, elevated TDS from surfactants',
                'action': 'ALERT: Inspect for industrial discharge, check household connections',
                'severity': 'MEDIUM'
            },
            'pipe_corrosion': {
                'description': 'Pipe Corrosion detected - Acidic pH, high TDS from metal leaching',
                'action': 'ALERT: Inspect pipes for corrosion, consider pipe replacement, test for heavy metals',
                'severity': 'HIGH'
            },
            'natural_sediment': {
                'description': 'Natural Sediment/Rust detected - High turbidity, normal pH and DO',
                'action': 'Check filtration systems, inspect for sediment sources, flush pipelines',
                'severity': 'LOW'
            }
        }
        return descriptions.get(contamination_type, {
            'description': 'Unknown contamination type',
            'action': 'Investigate immediately',
            'severity': 'UNKNOWN'
        })


def display_prediction_result(result):
    """Display prediction result in formatted way"""
    print("\n" + "=" * 70)
    print("CONTAMINATION DETECTION - PREDICTION RESULT")
    print("=" * 70)
    
    print("\nSENSOR READINGS:")
    for param, value in result['input_values'].items():
        print(f"  {param}: {value:.2f}")
    
    print(f"\nPREDICTION: {result['contamination_type'].upper()}")
    print(f"CONFIDENCE: {result['confidence']:.2%}")
    if result.get('do_imputed'):
        print("\n[NOTE] Dissolved Oxygen value was estimated via DO imputer "
              "(sensor reading unavailable).")
    
    print("\nPROBABILITIES:")
    for class_name, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {prob:.2%}")
    
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("CONTAMINATION DETECTION - REAL-TIME PREDICTION TEST")
    print("=" * 70)
    
    # Load model
    print("\n[STEP 1] Loading trained model...")
    predictor = ContaminationPredictor(
        model_path='ml_backend/contamination_detection/contamination_model.pkl',
        imputer_path='ml_backend/contamination_detection/imputer.pkl',
        encoder_path='ml_backend/contamination_detection/label_encoder.pkl',
        do_imputer_path='ml_backend/common/do_imputer.pkl',
    )
    print("[OK] Model loaded successfully")
    
    # Test with sample inputs
    print("\n[STEP 2] Testing with sample sensor readings...")
    
    test_cases = [
        {
            'name': 'Safe Water',
            'pH': 7.2, 'turbidity': 2.0, 'tds': 250, 'do': 7.5, 'temperature': 22
        },
        {
            'name': 'Microbial/Sewage Contamination',
            'pH': 6.3, 'turbidity': 55, 'tds': 1200, 'do': 1.5, 'temperature': 30
        },
        {
            'name': 'Chemical/Detergent Contamination',
            'pH': 8.5, 'turbidity': 18, 'tds': 1600, 'do': 4.0, 'temperature': 24
        },
        {
            'name': 'Pipe Corrosion',
            'pH': 5.8, 'turbidity': 12, 'tds': 1800, 'do': 4.5, 'temperature': 25
        },
        {
            'name': 'Natural Sediment',
            'pH': 7.4, 'turbidity': 65, 'tds': 400, 'do': 7.0, 'temperature': 23
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}: {test_case['name']}")
        print('='*70)
        
        result = predictor.predict_single(
            pH=test_case['pH'],
            turbidity=test_case['turbidity'],
            tds=test_case['tds'],
            do=test_case['do'],
            temperature=test_case['temperature']
        )
        
        display_prediction_result(result)
        
        # Get description
        desc = predictor.get_contamination_description(result['contamination_type'])
        print(f"\nDESCRIPTION: {desc['description']}")
        print(f"SEVERITY: {desc['severity']}")
        print(f"ACTION: {desc['action']}")
    
    print("\n" + "=" * 70)
    print("REAL-TIME PREDICTION TEST COMPLETED")
    print("=" * 70)
    print("\nModel is ready for integration with real-time sensor data!")
    print("Use predictor.predict_single(pH, turbidity, tds, do, temperature)")
    print("=" * 70)

