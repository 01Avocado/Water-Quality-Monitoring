"""
Disease Outbreak Prediction - Real-time Inference
Predicts WHICH disease will outbreak based on water quality sensors
Output: No Disease, Diarrhea, Cholera, or Typhoid
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


class DiseaseOutbreakPredictor:
    """
    Real-time predictor for disease outbreak
    Predicts specific disease type from sensor readings
    """
    
    def __init__(self, model_path, scaler_path, config_path, do_imputer_path):
        """
        Load trained model and preprocessing components
        
        Args:
            model_path: Path to trained model (.pkl)
            scaler_path: Path to feature scaler (.pkl)
            config_path: Path to model configuration (.pkl)
        """
        print("="*70)
        print("DISEASE OUTBREAK PREDICTION - INFERENCE ENGINE")
        print("Predicts: No Disease, Diarrhea, Cholera, Typhoid")
        print("="*70)
        print("\n[INIT] Loading model components...")
        
        # Load model
        self.model = joblib.load(model_path)
        print(f"[OK] Model loaded from: {model_path}")
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)
        print(f"[OK] Scaler loaded from: {scaler_path}")
        
        # Load config
        self.config = joblib.load(config_path)
        self.feature_names = self.config['feature_names']
        self.class_names = self.config['class_names']
        
        print(f"[OK] Configuration loaded")
        print(f"[INFO] Features: {self.feature_names}")
        print(f"[INFO] Disease Classes: {self.class_names}")
        
        # Load DO imputer for missing sensor deployments
        self.do_imputer = load_do_imputer(do_imputer_path)
        self.do_feature_names = list(self.do_imputer.feature_names)
        
        print("\n[READY] Predictor ready for real-time inference!")
    
    def _maybe_impute_do(self, pH, tds, temperature, do):
        """Return DO value and 1 if imputed."""
        if do is not None and not np.isnan(do):
            return do, 0
        features = pd.DataFrame(
            [{
                'pH': pH,
                'Temperature': temperature,
                'TDS': tds
            }]
        )
        imputed = float(
            self.do_imputer.predict(features[list(self.do_feature_names)])[0]
        )
        return imputed, 1
    
    def predict_single(self, pH, turbidity, tds, do, temperature):
        """
        Predict disease outbreak from sensor readings
        
        Args:
            pH: pH value (5.0 - 9.5)
            turbidity: Turbidity in NTU (0 - 150)
            tds: Total Dissolved Solids in mg/L (0 - 3000)
            do: Dissolved Oxygen in mg/L (0 - 15)
            temperature: Temperature in Celsius (0 - 40)
            
        Returns:
            dict: Prediction with disease type, confidence, and recommendations
        """
        do_value, do_imputed = self._maybe_impute_do(pH, tds, temperature, do)
        # Calculate engineered features (same as training)
        pH_deviation = abs(pH - 7.0)
        Turb_TDS_interaction = turbidity * tds / 1000
        DO_deficit = max(0, 8.0 - do_value)
        Temp_stress = max(0, temperature - 25)
        WQ_composite = pH_deviation * 0.2 + turbidity * 0.3 + tds/1000 * 0.2 + DO_deficit * 0.3
        
        # Create feature array with ALL 10 features (must match training order!)
        # Order: pH, Turbidity, DO, Temperature, TDS, pH_deviation, Turb_TDS_interaction, DO_deficit, Temp_stress, WQ_composite
        features = np.array([[
            pH, turbidity, do_value, temperature, tds,
            pH_deviation, Turb_TDS_interaction, DO_deficit, Temp_stress, WQ_composite
        ]])
        
        # Scale features (CRITICAL for real-time data)
        features_scaled = self.scaler.transform(features)
        
        # Predict
        disease_encoded = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        disease_name = self.class_names[disease_encoded]
        
        # Get confidence and all probabilities
        confidence = float(probabilities[disease_encoded])
        all_probabilities = {
            self.class_names[i]: float(probabilities[i])
            for i in range(len(self.class_names))
        }
        
        # Get severity and recommendations
        severity = self._get_severity(disease_name)
        recommendations = self._get_recommendations(disease_name)
        health_warning = self._get_health_warning(disease_name)
        
        return {
            'predicted_disease': disease_name,
            'disease_code': int(disease_encoded),
            'confidence': confidence,
            'all_probabilities': all_probabilities,
            'severity': severity,
            'health_warning': health_warning,
            'recommendations': recommendations,
            'input_values': {
                'pH': pH,
                'turbidity': turbidity,
                'tds': tds,
                'do': do_value,
                'temperature': temperature,
                'do_imputed_flag': do_imputed
            },
            'do_imputed': bool(do_imputed)
        }
    
    def predict_batch(self, df):
        """
        Predict disease outbreak for multiple samples
        
        Args:
            df: DataFrame with columns [pH, Turbidity, TDS, DO, Temperature]
            
        Returns:
            DataFrame with predictions
        """
        df_features = df.copy()
        df_features['DO_imputed_flag'] = 0
        
        missing_mask = df_features['DO'].isna()
        if missing_mask.any():
            do_features = df_features.loc[missing_mask, ['pH', 'TDS', 'Temperature']]
            do_features = do_features.rename(columns={'Temperature': 'Temperature', 'TDS': 'TDS'})
            imputed_vals = self.do_imputer.predict(
                do_features[list(self.do_feature_names)]
            )
            df_features.loc[missing_mask, 'DO'] = imputed_vals
            df_features.loc[missing_mask, 'DO_imputed_flag'] = 1
        
        # Calculate engineered features for batch
        df_features['pH_deviation'] = np.abs(df_features['pH'] - 7.0)
        df_features['Turb_TDS_interaction'] = df_features['Turbidity'] * df_features['TDS'] / 1000
        df_features['DO_deficit'] = np.maximum(0, 8.0 - df_features['DO'])
        df_features['Temp_stress'] = np.maximum(0, df_features['Temperature'] - 25)
        df_features['WQ_composite'] = (
            df_features['pH_deviation'] * 0.2 + 
            df_features['Turbidity'] * 0.3 + 
            df_features['TDS'] / 1000 * 0.2 + 
            df_features['DO_deficit'] * 0.3
        )
        
        # Extract features in correct order
        feature_cols = ['pH', 'Turbidity', 'DO', 'Temperature', 'TDS', 
                       'pH_deviation', 'Turb_TDS_interaction', 'DO_deficit', 'Temp_stress', 'WQ_composite']
        X = df_features[feature_cols].values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        disease_encoded = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Convert to disease names
        disease_names = [self.class_names[d] for d in disease_encoded]
        confidences = probabilities.max(axis=1)
        
        # Create result DataFrame
        result_df = df.copy()
        result_df['DO_imputed_flag'] = df_features['DO_imputed_flag']
        result_df['predicted_disease'] = disease_names
        result_df['disease_code'] = disease_encoded
        result_df['confidence'] = confidences
        
        return result_df
    
    def _get_severity(self, disease_name):
        """Get severity level for disease"""
        severity_map = {
            'No Disease': 'SAFE',
            'Diarrhea': 'MEDIUM',
            'Cholera': 'CRITICAL',
            'Typhoid': 'HIGH'
        }
        return severity_map.get(disease_name, 'UNKNOWN')
    
    def _get_health_warning(self, disease_name):
        """Get health warning message"""
        warnings_map = {
            'No Disease': 'Water quality is safe. No disease outbreak expected.',
            'Diarrhea': 'ALERT: Diarrheal disease outbreak likely. Waterborne pathogen contamination detected.',
            'Cholera': 'CRITICAL ALERT: Cholera outbreak risk detected! Immediate action required!',
            'Typhoid': 'HIGH ALERT: Typhoid fever outbreak risk detected. Contamination likely.'
        }
        return warnings_map.get(disease_name, 'Unknown disease status')
    
    def _get_recommendations(self, disease_name):
        """Get recommendations based on predicted disease"""
        recommendations = {
            'No Disease': [
                'Continue routine water quality monitoring',
                'Maintain current treatment standards',
                'Regular sensor calibration',
                'Keep disease surveillance active'
            ],
            'Diarrhea': [
                'ALERT health authorities about diarrhea outbreak risk',
                'Increase water treatment and chlorination levels',
                'Issue advisory: Boil water before consumption',
                'Monitor hospitals for diarrheal cases',
                'Inspect water distribution system',
                'Test for E.coli and fecal coliforms',
                'Prepare oral rehydration supplies'
            ],
            'Cholera': [
                'EMERGENCY: Alert health authorities IMMEDIATELY',
                'CRITICAL: Issue public health emergency declaration',
                'Shut down contaminated water supply if possible',
                'Mandatory water boiling advisory',
                'Test for Vibrio cholerae bacteria',
                'Set up emergency rehydration centers',
                'Mobilize emergency medical response teams',
                'Coordinate with WHO/CDC for outbreak response',
                'Implement emergency water treatment (super-chlorination)'
            ],
            'Typhoid': [
                'URGENT: Alert health authorities',
                'Issue water safety advisory',
                'Boil water before consumption (mandatory)',
                'Test for Salmonella typhi bacteria',
                'Monitor for fever cases in community',
                'Inspect water treatment facilities',
                'Increase chlorination levels',
                'Prepare antibiotic supplies (ciprofloxacin/ceftriaxone)',
                'Screen food handlers and water workers'
            ]
        }
        return recommendations.get(disease_name, ['Investigate immediately', 'Contact health authorities'])
    
    def get_alert_message(self, prediction):
        """
        Format prediction into human-readable alert message
        
        Args:
            prediction: Result from predict_single()
        
        Returns:
            Formatted alert message string
        """
        disease = prediction['predicted_disease']
        confidence = prediction['confidence']
        severity = prediction['severity']
        
        # Create border based on severity
        if severity == 'CRITICAL':
            border_char = '='
            icon = '[CRITICAL]'
        elif severity == 'HIGH':
            border_char = '='
            icon = '[HIGH ALERT]'
        elif severity == 'MEDIUM':
            border_char = '-'
            icon = '[ALERT]'
        else:
            border_char = '-'
            icon = '[SAFE]'
        
        border = border_char * 70
        
        message = f"""
{border}
{icon} DISEASE OUTBREAK PREDICTION - {severity}
{border}

Predicted Disease: {disease}
Confidence: {confidence:.1%}

Sensor Readings:
  * pH: {prediction['input_values']['pH']:.2f}
  * Turbidity: {prediction['input_values']['turbidity']:.2f} NTU
  * TDS: {prediction['input_values']['tds']:.2f} mg/L
  * Dissolved Oxygen: {prediction['input_values']['do']:.2f} mg/L
  * Temperature: {prediction['input_values']['temperature']:.2f} C

Disease Probabilities:
"""
        
        # Sort probabilities for display
        sorted_probs = sorted(prediction['all_probabilities'].items(), 
                             key=lambda x: x[1], reverse=True)
        for disease_name, prob in sorted_probs:
            message += f"  * {disease_name}: {prob:.1%}\n"
        
        message += f"\nHealth Warning:\n  {prediction['health_warning']}\n"
        
        message += "\nRecommended Actions:\n"
        for i, rec in enumerate(prediction['recommendations'][:5], 1):
            message += f"  {i}. {rec}\n"
        
        if len(prediction['recommendations']) > 5:
            message += f"  ... and {len(prediction['recommendations']) - 5} more actions\n"
        
        message += f"\n{border}\n"
        
        return message


def display_prediction_result(result):
    """Display prediction result in formatted way"""
    print("\n" + "=" * 70)
    print("DISEASE OUTBREAK PREDICTION - RESULT")
    print("=" * 70)
    
    print("\nSENSOR READINGS:")
    for param, value in result['input_values'].items():
        print(f"  {param}: {value:.2f}")
    
    severity_markers = {
        'SAFE': '[OK]',
        'MEDIUM': '[WARNING]',
        'HIGH': '[DANGER]',
        'CRITICAL': '[EMERGENCY]'
    }
    marker = severity_markers.get(result['severity'], '[?]')
    
    print(f"\n{marker} PREDICTED DISEASE: {result['predicted_disease']}")
    print(f"CONFIDENCE: {result['confidence']:.2%}")
    print(f"SEVERITY: {result['severity']}")
    if result.get('do_imputed'):
        print("\n[NOTE] Dissolved Oxygen value supplied by DO imputer "
              "because the sensor reading was unavailable.")
    
    print("\nALL DISEASE PROBABILITIES:")
    sorted_probs = sorted(result['all_probabilities'].items(), 
                         key=lambda x: x[1], reverse=True)
    for disease, prob in sorted_probs:
        print(f"  {disease}: {prob:.2%}")
    
    print(f"\nHEALTH WARNING:")
    print(f"  {result['health_warning']}")
    
    print("\nRECOMMENDED ACTIONS:")
    for i, rec in enumerate(result['recommendations'][:3], 1):
        print(f"  {i}. {rec}")
    
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("DISEASE OUTBREAK PREDICTION - REAL-TIME TEST")
    print("=" * 70)
    
    # Load model
    print("\n[STEP 1] Loading trained model...")
    predictor = DiseaseOutbreakPredictor(
        model_path='disease_model.pkl',
        scaler_path='disease_scaler.pkl',
        config_path='disease_config.pkl',
        do_imputer_path='../common/do_imputer.pkl'
    )
    print("[OK] Model loaded successfully")
    
    # Test with sample inputs representing different diseases
    print("\n[STEP 2] Testing with sample sensor readings...")
    
    test_cases = [
        {
            'name': 'Safe/Clean Water - No Disease Expected',
            'pH': 7.2, 'turbidity': 2.0, 'tds': 250, 'do': 8.5, 'temperature': 22
        },
        {
            'name': 'Contaminated Water - Diarrhea Risk',
            'pH': 6.5, 'turbidity': 15.0, 'tds': 800, 'do': 4.5, 'temperature': 28
        },
        {
            'name': 'Severely Contaminated - Cholera Risk',
            'pH': 6.0, 'turbidity': 45.0, 'tds': 1500, 'do': 2.0, 'temperature': 32
        },
        {
            'name': 'Poor Quality - Typhoid Risk',
            'pH': 6.3, 'turbidity': 25.0, 'tds': 1200, 'do': 3.5, 'temperature': 30
        },
        {
            'name': 'Alkaline Contamination',
            'pH': 8.5, 'turbidity': 10.0, 'tds': 900, 'do': 5.0, 'temperature': 26
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
        
        # Display formatted result
        display_prediction_result(result)
        
        # Show alert message
        print("\n[ALERT MESSAGE FOR DASHBOARD]")
        alert = predictor.get_alert_message(result)
        print(alert)
    
    print("\n" + "=" * 70)
    print("REAL-TIME PREDICTION TEST COMPLETED")
    print("=" * 70)
    print("\nModel is ready for integration with NEERWANA system!")
    print("Use: predictor.predict_single(pH, turbidity, tds, do, temperature)")
    print("=" * 70)
