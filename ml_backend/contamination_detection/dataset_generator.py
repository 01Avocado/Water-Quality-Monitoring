"""
Contamination Cause Detection - Dataset Generator
Generates realistic synthetic dataset with anti-overfitting measures
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ContaminationDatasetGenerator:
    """
    Generates synthetic water quality data for contamination detection
    with realistic variations, noise, and anti-overfitting measures
    """
    
    def __init__(self, n_samples=5000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Define contamination signatures (with intentional overlap)
        self.signatures = {
            'safe': {
                'pH': (6.5, 8.5, 7.2),
                'turbidity': (0, 5, 1.5),
                'tds': (50, 500, 250),
                'do': (5, 9, 7),
                'temp': (15, 30, 22)
            },
            'microbial_sewage': {
                'pH': (5.8, 7.0, 6.4),
                'turbidity': (8, 120, 45),
                'tds': (400, 1800, 900),
                'do': (0, 4, 2),
                'temp': (23, 36, 29)
            },
            'chemical_detergent': {
                'pH': (7.2, 9.2, 8.1),
                'turbidity': (3, 35, 15),
                'tds': (600, 2200, 1300),
                'do': (2.5, 7, 4.5),
                'temp': (18, 32, 25)
            },
            'pipe_corrosion': {
                'pH': (5.2, 7.2, 6.2),
                'turbidity': (1, 25, 10),
                'tds': (700, 2700, 1600),
                'do': (2.5, 7, 4.8),
                'temp': (18, 32, 25)
            },
            'natural_sediment': {
                'pH': (6.5, 8.0, 7.3),
                'turbidity': (15, 120, 55),
                'tds': (150, 900, 450),
                'do': (5.5, 8.5, 7),
                'temp': (18, 32, 25)
            }
        }
        
    def _generate_value(self, min_val, max_val, mean_val, std_factor=0.15):
        """Generate value with realistic distribution"""
        std = (max_val - min_val) * std_factor
        value = np.random.normal(mean_val, std)
        # Add occasional outliers (5% chance)
        if np.random.random() < 0.05:
            value += np.random.uniform(-std*2, std*2)
        return np.clip(value, min_val, max_val)
    
    def _add_sensor_noise(self, value, noise_level=0.03):
        """Add realistic sensor noise"""
        noise = np.random.normal(0, abs(value) * noise_level)
        return value + noise
    
    def _create_multi_contamination(self, sig1, sig2, weight1=0.6):
        """Mix two contamination signatures"""
        weight2 = 1 - weight1
        mixed = {}
        for param in ['pH', 'turbidity', 'tds', 'do', 'temp']:
            val1 = self._generate_value(*sig1[param])
            val2 = self._generate_value(*sig2[param])
            mixed[param] = val1 * weight1 + val2 * weight2
        return mixed
    
    def generate_dataset(self):
        """Generate complete dataset with all anti-overfitting measures"""
        
        data = []
        labels = []
        timestamps = []
        
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        current_time = start_time
        
        # Distribution: 40% safe, 15% each contamination type
        class_distribution = {
            'safe': 0.40,
            'microbial_sewage': 0.15,
            'chemical_detergent': 0.15,
            'pipe_corrosion': 0.15,
            'natural_sediment': 0.15
        }
        
        samples_per_class = {k: int(v * self.n_samples) for k, v in class_distribution.items()}
        
        for class_name, n_samples in samples_per_class.items():
            signature = self.signatures[class_name]
            
            for i in range(n_samples):
                # Random time interval (not predictable)
                time_delta = timedelta(
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60),
                    seconds=np.random.randint(0, 60)
                )
                current_time += time_delta
                
                # 10% chance of multi-contamination scenario
                if class_name != 'safe' and np.random.random() < 0.10:
                    other_classes = [c for c in self.signatures.keys() if c != class_name and c != 'safe']
                    other_class = np.random.choice(other_classes)
                    params = self._create_multi_contamination(
                        signature, 
                        self.signatures[other_class],
                        weight1=np.random.uniform(0.6, 0.8)
                    )
                else:
                    params = {
                        'pH': self._generate_value(*signature['pH']),
                        'turbidity': self._generate_value(*signature['turbidity']),
                        'tds': self._generate_value(*signature['tds']),
                        'do': self._generate_value(*signature['do']),
                        'temp': self._generate_value(*signature['temp'])
                    }
                
                # Add sensor noise
                params = {k: self._add_sensor_noise(v) for k, v in params.items()}
                
                # Add seasonal temperature variation
                day_of_year = current_time.timetuple().tm_yday
                seasonal_temp_adj = 3 * np.sin(2 * np.pi * day_of_year / 365)
                params['temp'] += seasonal_temp_adj
                
                # Add time-of-day DO variation
                hour = current_time.hour
                do_variation = 0.5 * np.sin(2 * np.pi * hour / 24)
                params['do'] = max(0, params['do'] + do_variation)
                
                data.append([
                    params['pH'],
                    params['turbidity'],
                    params['tds'],
                    params['do'],
                    params['temp']
                ])
                labels.append(class_name)
                timestamps.append(current_time)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=['pH', 'turbidity_ntu', 'tds_mg/l', 'do_mg/l', 'temperature_c'])
        df['timestamp'] = timestamps
        df['contamination_type'] = labels
        
        # Shuffle to remove temporal ordering bias
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        # Add missing values randomly (7% of data)
        missing_mask = np.random.random(df.shape) < 0.07
        df_with_missing = df.copy()
        for col in ['pH', 'turbidity_ntu', 'tds_mg/l', 'do_mg/l', 'temperature_c']:
            df_with_missing.loc[missing_mask[:, df.columns.get_loc(col)], col] = np.nan
        
        # Add some extreme outliers (3% of data)
        n_outliers = int(0.03 * len(df_with_missing))
        outlier_indices = np.random.choice(len(df_with_missing), n_outliers, replace=False)
        for idx in outlier_indices:
            col = np.random.choice(['pH', 'turbidity_ntu', 'tds_mg/l', 'do_mg/l'])
            current_val = df_with_missing.loc[idx, col]
            if not pd.isna(current_val):
                df_with_missing.loc[idx, col] = current_val * np.random.uniform(1.5, 2.5)
        
        return df_with_missing


if __name__ == "__main__":
    print("=" * 70)
    print("CONTAMINATION DETECTION - DATASET GENERATION")
    print("=" * 70)
    
    generator = ContaminationDatasetGenerator(n_samples=5000, random_state=42)
    print("\n[STEP 1] Generating synthetic dataset with anti-overfitting measures...")
    df = generator.generate_dataset()
    
    print(f"\n[OK] Dataset generated: {len(df)} samples")
    print(f"\n[OK] Class distribution:")
    print(df['contamination_type'].value_counts())
    
    print(f"\n[OK] Feature statistics:")
    print(df[['pH', 'turbidity_ntu', 'tds_mg/l', 'do_mg/l', 'temperature_c']].describe())
    
    print(f"\n[OK] Missing values per column:")
    print(df.isnull().sum())
    
    output_path = 'ml_backend/contamination_detection/contamination_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Dataset saved to: {output_path}")
    print("=" * 70)
