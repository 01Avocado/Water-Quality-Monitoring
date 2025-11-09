"""
Degradation Forecasting - Data Preparation
Loads timestamp dataset and prepares it for LSTM training
Matches 5-sensor setup: pH, Turbidity, TDS, DO, Temperature
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add parent directory to path for WQI model imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import load_do_imputer  # noqa: E402
import joblib  # noqa: E402


class DataPreparation:
    """Prepare timestamp data for degradation forecasting"""
    
    def __init__(
        self,
        timestamp_path,
        wqi_model_path,
        wqi_scaler_path,
        wqi_config_path,
        do_imputer_path,
    ):
        self.timestamp_path = timestamp_path
        self.wqi_model_path = wqi_model_path
        self.wqi_scaler_path = wqi_scaler_path
        self.wqi_config_path = wqi_config_path
        self.do_imputer_path = do_imputer_path
        
        # Load WQI model components
        print("="*70)
        print("DEGRADATION FORECASTING - DATA PREPARATION")
        print("="*70)
        print("\n[STEP 1] Loading WQI model components...")
        self.wqi_model = joblib.load(wqi_model_path)
        self.wqi_scaler = joblib.load(wqi_scaler_path)
        self.wqi_config = joblib.load(wqi_config_path)
        print("[OK] WQI model loaded successfully")
        
        # Load DO imputer
        print("[STEP 1B] Loading dissolved oxygen imputer...")
        self.do_imputer = load_do_imputer(do_imputer_path)
        print("[OK] DO imputer ready (features: "
              f"{', '.join(self.do_imputer.feature_names)})")
        
    def load_and_clean_timestamp_data(self):
        """Load timestamp dataset and extract relevant features"""
        print("\n[STEP 2] Loading timestamp dataset...")
        
        # Load CSV
        df = pd.read_csv(self.timestamp_path)
        print(f"[OK] Loaded {len(df)} hourly readings")
        print(f"[INFO] Date range: {df['timestamp_utc'].iloc[0]} to {df['timestamp_utc'].iloc[-1]}")
        
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp_utc'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Extract and rename features to match sensor kit
        print("\n[STEP 3] Extracting 5 sensor features...")
        df_clean = pd.DataFrame()
        df_clean['timestamp'] = df['timestamp']
        df_clean['Temperature'] = pd.to_numeric(df['water_temperature (deg C)'], errors='coerce')
        df_clean['pH'] = pd.to_numeric(df['pH (pH units)'], errors='coerce')
        if 'do_concentration (mg/L)' in df.columns:
            df_clean['DO'] = pd.to_numeric(df['do_concentration (mg/L)'], errors='coerce')
        else:
            df_clean['DO'] = np.nan
        df_clean['Turbidity'] = pd.to_numeric(df['turbidity (FNU)'], errors='coerce')
        
        # Convert conductance to TDS (TDS ≈ Conductance × 640 for mS/cm to mg/L)
        conductance = pd.to_numeric(df['sp_conductance (mS/cm)'], errors='coerce')
        df_clean['TDS'] = conductance * 640  # Convert mS/cm to mg/L
        
        # Impute DO where missing using learned model
        df_clean['DO_imputed'] = 0
        df_clean = self.do_imputer.impute_dataframe(
            df_clean,
            do_column='DO',
            flag_column='DO_imputed',
            only_if_missing=True,
        )
        
        print(f"[OK] Extracted features: {['pH', 'Turbidity', 'TDS', 'DO', 'Temperature']}")
        
        # Check missing values before cleaning
        missing_before = df_clean.isnull().sum()
        print(f"\n[INFO] Missing values before cleaning:")
        for col in ['pH', 'Turbidity', 'TDS', 'DO', 'Temperature']:
            missing = missing_before[col]
            if missing > 0:
                print(f"  - {col}: {missing} ({missing/len(df_clean)*100:.2f}%)")
        
        # Handle missing values
        # For turbidity, if missing, assume low (0.5 FNU)
        df_clean['Turbidity'] = df_clean['Turbidity'].fillna(0.5)
        
        # For other parameters, use forward fill then backward fill
        df_clean[['pH', 'Temperature', 'TDS']] = df_clean[['pH', 'Temperature', 'TDS']].ffill().bfill()
        
        # If DO remains missing after imputation, fall back to interpolation
        if df_clean['DO'].isna().any():
            df_clean['DO'] = df_clean['DO'].ffill().bfill()
            df_clean.loc[df_clean['DO'].notna(), 'DO_imputed'] = 1
        
        # Drop any remaining rows with missing values
        rows_before = len(df_clean)
        df_clean = df_clean.dropna()
        rows_after = len(df_clean)
        if rows_before - rows_after > 0:
            print(f"[INFO] Removed {rows_before - rows_after} rows with missing values")
        
        print(f"[OK] Clean dataset: {len(df_clean)} readings")
        
        return df_clean
    
    def resample_to_3hour_intervals(self, df):
        """Resample hourly data to 3-hour intervals"""
        print("\n[STEP 4] Resampling to 3-hour intervals...")
        
        # Set timestamp as index
        df = df.set_index('timestamp')
        
        # Resample to 3-hour intervals, taking mean of each period
        df_3h = df.resample('3h').mean()
        if 'DO_imputed' in df_3h.columns:
            df_3h['DO_imputed'] = (df_3h['DO_imputed'] > 0).astype(int)
        
        # Remove any NaN values from resampling
        df_3h = df_3h.dropna()
        
        # Reset index to make timestamp a column again
        df_3h = df_3h.reset_index()
        
        print(f"[OK] Resampled from {len(df)} hourly -> {len(df_3h)} 3-hourly readings")
        print(f"[INFO] Date range: {df_3h['timestamp'].iloc[0]} to {df_3h['timestamp'].iloc[-1]}")
        
        return df_3h
    
    def compute_wqi_score(self, X):
        """
        Compute WHO-based WQI score for each sample (matches WQI model)
        Input: numpy array [pH, Turbidity, TDS, DO, Temperature]
        Output: WQI scores (0-100)
        """
        wqi_scores = np.zeros(len(X))
        
        WHO_WEIGHTS = self.wqi_config['WHO_WEIGHTS']
        IDEAL_RANGES = self.wqi_config['IDEAL_RANGES']
        feature_names = self.wqi_config['feature_names']
        
        for i, feature in enumerate(feature_names):
            values = X[:, i]
            weight = WHO_WEIGHTS[feature]
            ideal_min, ideal_max = IDEAL_RANGES[feature]
            
            # Calculate sub-index for each parameter
            sub_index = np.zeros(len(values))
            
            for j, val in enumerate(values):
                if ideal_min <= val <= ideal_max:
                    # Within ideal range
                    ideal_center = (ideal_min + ideal_max) / 2
                    ideal_range = ideal_max - ideal_min
                    deviation = abs(val - ideal_center) / (ideal_range / 2)
                    sub_index[j] = 100 * (1 - deviation * 0.2)
                else:
                    # Outside ideal range
                    if val < ideal_min:
                        deviation = (ideal_min - val) / ideal_min if ideal_min != 0 else val
                    else:
                        deviation = (val - ideal_max) / ideal_max if ideal_max != 0 else val
                    
                    # Exponential decay
                    sub_index[j] = 100 * np.exp(-deviation)
            
            # Weight and add to total WQI
            wqi_scores += weight * sub_index
        
        return wqi_scores
    
    def apply_wqi_model(self, df):
        """Apply WQI model to label each timestep as Safe/Degrading/Unsafe"""
        print("\n[STEP 5] Applying WQI model to label sequences...")
        
        # Extract features in correct order
        X = df[['pH', 'Turbidity', 'TDS', 'DO', 'Temperature']].values
        
        # Compute WQI scores (matching WQI model logic)
        wqi_scores = self.compute_wqi_score(X)
        
        # Add WQI score as feature
        X_with_wqi = np.hstack([X, wqi_scores.reshape(-1, 1)])
        
        # Scale features (same as WQI model training)
        X_scaled = self.wqi_scaler.transform(X_with_wqi)
        
        # Predict pollution level (0=Safe, 1=Degrading, 2=Unsafe)
        pollution_level = self.wqi_model.predict(X_scaled)
        
        # Get prediction probabilities
        pollution_proba = self.wqi_model.predict_proba(X_scaled)
        
        # Add to dataframe
        df['WQI_Score'] = wqi_scores
        df['Pollution_Level'] = pollution_level
        df['Prob_Safe'] = pollution_proba[:, 0]
        df['Prob_Degrading'] = pollution_proba[:, 1]
        df['Prob_Unsafe'] = pollution_proba[:, 2]
        
        # Label names
        level_names = {0: 'Safe', 1: 'Degrading', 2: 'Unsafe'}
        df['Status'] = df['Pollution_Level'].map(level_names)
        
        # Print distribution
        print(f"[OK] WQI model applied to {len(df)} readings")
        print(f"\n[INFO] Water quality distribution:")
        for level in [0, 1, 2]:
            count = (pollution_level == level).sum()
            print(f"  - {level_names[level]}: {count} readings ({count/len(df)*100:.2f}%)")
        
        print(f"\n[INFO] WQI Score statistics:")
        print(f"  - Mean: {wqi_scores.mean():.2f}")
        print(f"  - Min: {wqi_scores.min():.2f}")
        print(f"  - Max: {wqi_scores.max():.2f}")
        
        return df
    
    def identify_degradation_sequences(self, df, min_sequence_length=24):
        """
        Identify sequences where water transitions from better to worse quality
        This helps us learn degradation patterns
        """
        print(f"\n[STEP 6] Identifying degradation sequences (min length: {min_sequence_length})...")
        
        degradation_sequences = []
        current_sequence = []
        
        for i in range(len(df)):
            current_level = df.iloc[i]['Pollution_Level']
            
            # Start new sequence if we're at Safe or Degrading
            if current_level <= 1:
                current_sequence.append(i)
            else:
                # If we hit Unsafe and have a sequence, check if it shows degradation
                if len(current_sequence) >= min_sequence_length:
                    seq_data = df.iloc[current_sequence]
                    # Check if WQI is generally declining
                    wqi_trend = seq_data['WQI_Score'].values
                    if wqi_trend[0] > wqi_trend[-1]:  # Declining WQI
                        degradation_sequences.append({
                            'start_idx': current_sequence[0],
                            'end_idx': current_sequence[-1],
                            'length': len(current_sequence),
                            'start_wqi': wqi_trend[0],
                            'end_wqi': wqi_trend[-1],
                            'wqi_drop': wqi_trend[0] - wqi_trend[-1]
                        })
                current_sequence = []
        
        # Check last sequence
        if len(current_sequence) >= min_sequence_length:
            seq_data = df.iloc[current_sequence]
            wqi_trend = seq_data['WQI_Score'].values
            if wqi_trend[0] > wqi_trend[-1]:
                degradation_sequences.append({
                    'start_idx': current_sequence[0],
                    'end_idx': current_sequence[-1],
                    'length': len(current_sequence),
                    'start_wqi': wqi_trend[0],
                    'end_wqi': wqi_trend[-1],
                    'wqi_drop': wqi_trend[0] - wqi_trend[-1]
                })
        
        print(f"[OK] Found {len(degradation_sequences)} degradation sequences")
        if len(degradation_sequences) > 0:
            total_length = sum([s['length'] for s in degradation_sequences])
            avg_length = total_length / len(degradation_sequences)
            print(f"[INFO] Average sequence length: {avg_length:.1f} readings ({avg_length*3:.1f} hours)")
        
        return degradation_sequences
    
    def save_prepared_data(self, df, output_path):
        """Save prepared data to CSV"""
        print(f"\n[STEP 7] Saving prepared data...")
        df.to_csv(output_path, index=False)
        print(f"[OK] Saved to: {output_path}")
        print(f"[INFO] Dataset shape: {df.shape}")
        print(f"[INFO] Features: {df.columns.tolist()}")
    
    def run_full_pipeline(self, output_path='degradation_prepared_data.csv'):
        """Run complete data preparation pipeline"""
        # Load and clean data
        df = self.load_and_clean_timestamp_data()
        
        # Resample to 3-hour intervals
        df = self.resample_to_3hour_intervals(df)
        
        # Apply WQI model
        df = self.apply_wqi_model(df)
        
        # Identify degradation sequences (for analysis)
        degradation_sequences = self.identify_degradation_sequences(df)
        
        # Save prepared data
        self.save_prepared_data(df, output_path)
        
        print("\n" + "="*70)
        print("DATA PREPARATION COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\n[READY] Dataset ready for LSTM training")
        print(f"[READY] Total readings: {len(df)} (3-hour intervals)")
        print(f"[READY] Time span: ~{len(df)*3/24:.1f} days")
        print(f"[READY] Features: pH, Turbidity, TDS, DO, Temperature, WQI_Score, Pollution_Level")
        
        return df, degradation_sequences


if __name__ == "__main__":
    # Paths
    timestamp_path = '../timestamp dataset.CSV.xls'
    wqi_model_path = '../wqi/wqi_model.pkl'
    wqi_scaler_path = '../wqi/wqi_scaler.pkl'
    wqi_config_path = '../wqi/wqi_config.pkl'
    do_imputer_path = '../common/do_imputer.pkl'
    output_path = 'degradation_prepared_data.csv'
    
    # Initialize and run
    prep = DataPreparation(
        timestamp_path=timestamp_path,
        wqi_model_path=wqi_model_path,
        wqi_scaler_path=wqi_scaler_path,
        wqi_config_path=wqi_config_path,
        do_imputer_path=do_imputer_path,
    )
    
    df, sequences = prep.run_full_pipeline(output_path)
    
    print("\n[INFO] You can now run: python train_model.py")

