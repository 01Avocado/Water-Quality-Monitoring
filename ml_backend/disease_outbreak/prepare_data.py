"""
Disease Outbreak Prediction - IMPROVED Data Preparation
Version 2.0 with SMOTE balancing and feature engineering
GOAL: Improve accuracy from 37% to 60-70%
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

class ImprovedDiseaseDataPreparator:
    """Improved disease outbreak dataset preparation with SMOTE and feature engineering"""
    
    def __init__(self):
        """Initialize with disease classification thresholds"""
        # Thresholds for disease outbreak
        self.DIARRHEA_OUTBREAK_THRESHOLD = 200
        self.CHOLERA_OUTBREAK_THRESHOLD = 15
        self.TYPHOID_OUTBREAK_THRESHOLD = 30
        
        # Feature mapping
        self.FEATURE_MAPPING = {
            'pH Level': 'pH',
            'Turbidity (NTU)': 'Turbidity',
            'Dissolved Oxygen (mg/L)': 'DO',
            'Temperature (°C)': 'Temperature',
            'Contaminant Level (ppm)': 'TDS_proxy'
        }
        
        self.TARGET_MAPPING = {
            'Diarrheal Cases per 100,000 people': 'diarrhea_cases',
            'Cholera Cases per 100,000 people': 'cholera_cases',
            'Typhoid Cases per 100,000 people': 'typhoid_cases'
        }
        
        self.DISEASE_CLASSES = ['No_Disease', 'Diarrhea', 'Cholera', 'Typhoid']
    
    def load_data(self, filepath):
        """Load raw dataset"""
        print("="*70)
        print("IMPROVED DISEASE OUTBREAK DATA PREPARATION v2.0")
        print("Features: SMOTE Balancing + Feature Engineering")
        print("="*70)
        print(f"\n[STEP 1] Loading data from: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"[OK] Loaded {len(df)} records")
        
        return df
    
    def extract_features(self, df):
        """Extract and standardize sensor features"""
        print(f"\n[STEP 2] Extracting sensor features...")
        
        # Extract available features
        available_features = []
        for old_name, new_name in self.FEATURE_MAPPING.items():
            if old_name in df.columns:
                available_features.append((old_name, new_name))
        
        # Extract and rename
        feature_df = df[[old_name for old_name, _ in available_features]].copy()
        feature_df.columns = [new_name for _, new_name in available_features]
        
        # Create TDS from Contaminant Level
        if 'TDS_proxy' in feature_df.columns:
            feature_df['TDS'] = feature_df['TDS_proxy'] * 100 + 150
            feature_df['TDS'] = feature_df['TDS'].clip(50, 2000)
            feature_df = feature_df.drop('TDS_proxy', axis=1)
        
        # Handle missing values
        feature_df = feature_df.fillna(feature_df.median())
        
        print(f"[OK] Extracted 5 base features: {list(feature_df.columns)}")
        
        return feature_df
    
    def engineer_features(self, feature_df):
        """
        Create engineered features to improve model
        
        NEW FEATURES:
        - pH deviation from neutral (7.0)
        - Turbidity x TDS interaction
        - DO deficit (optimal - actual)
        - Temperature stress indicator
        """
        print(f"\n[STEP 3] Engineering additional features...")
        
        engineered_df = feature_df.copy()
        
        # 1. pH deviation from neutral (diseases thrive in non-neutral pH)
        engineered_df['pH_deviation'] = np.abs(engineered_df['pH'] - 7.0)
        
        # 2. Turbidity-TDS interaction (both high = severe contamination)
        engineered_df['Turb_TDS_interaction'] = (
            engineered_df['Turbidity'] * engineered_df['TDS'] / 1000
        )
        
        # 3. DO deficit (low DO = bacterial growth)
        optimal_DO = 8.0
        engineered_df['DO_deficit'] = optimal_DO - engineered_df['DO']
        engineered_df['DO_deficit'] = engineered_df['DO_deficit'].clip(lower=0)
        
        # 4. Temperature stress (diseases thrive in warm water)
        engineered_df['Temp_stress'] = np.maximum(0, engineered_df['Temperature'] - 25)
        
        # 5. Water quality index (composite indicator)
        engineered_df['WQ_composite'] = (
            engineered_df['pH_deviation'] * 0.2 +
            engineered_df['Turbidity'] * 0.3 +
            engineered_df['TDS'] / 1000 * 0.2 +
            engineered_df['DO_deficit'] * 0.3
        )
        
        print(f"[OK] Created 5 engineered features:")
        print(f"  - pH_deviation")
        print(f"  - Turb_TDS_interaction")
        print(f"  - DO_deficit")
        print(f"  - Temp_stress")
        print(f"  - WQ_composite")
        print(f"[INFO] Total features: {len(engineered_df.columns)}")
        
        return engineered_df
    
    def extract_disease_data(self, df):
        """Extract disease case counts"""
        print(f"\n[STEP 4] Extracting disease data...")
        
        target_df = df[[old_name for old_name in self.TARGET_MAPPING.keys()]].copy()
        target_df.columns = [new_name for new_name in self.TARGET_MAPPING.values()]
        
        return target_df
    
    def classify_disease_outbreaks(self, disease_df):
        """Classify which disease will outbreak based on case counts"""
        print(f"\n[STEP 5] Classifying disease outbreaks...")
        
        n_samples = len(disease_df)
        disease_labels = np.zeros(n_samples, dtype=int)
        
        diarrhea_cases = disease_df['diarrhea_cases'].values
        cholera_cases = disease_df['cholera_cases'].values
        typhoid_cases = disease_df['typhoid_cases'].values
        
        # Check outbreaks
        diarrhea_outbreak = diarrhea_cases > self.DIARRHEA_OUTBREAK_THRESHOLD
        cholera_outbreak = cholera_cases > self.CHOLERA_OUTBREAK_THRESHOLD
        typhoid_outbreak = typhoid_cases > self.TYPHOID_OUTBREAK_THRESHOLD
        
        for i in range(n_samples):
            if cholera_outbreak[i]:
                disease_labels[i] = 2  # Cholera
            elif typhoid_outbreak[i]:
                disease_labels[i] = 3  # Typhoid
            elif diarrhea_outbreak[i]:
                disease_labels[i] = 1  # Diarrhea
            else:
                disease_labels[i] = 0  # No Disease
        
        # Count distribution BEFORE balancing
        unique, counts = np.unique(disease_labels, return_counts=True)
        print(f"[INFO] BEFORE balancing:")
        for label, count in zip(unique, counts):
            print(f"  - {self.DISEASE_CLASSES[label]}: {count} ({count/n_samples*100:.1f}%)")
        
        return disease_labels
    
    def apply_smote_balancing(self, X, y):
        """
        Apply SMOTE to balance classes
        THIS IS THE KEY IMPROVEMENT!
        """
        print(f"\n[STEP 6] Applying SMOTE for class balancing...")
        print(f"[INFO] This will create synthetic samples for minority classes")
        
        # Define target distribution (SMOTE can only increase, not decrease)
        sampling_strategy = {
            0: 1200,  # No Disease: boost from 123 to 1200
            1: 1200,  # Diarrhea: boost from 201 to 1200
            # 2: Keep Cholera at 2047 (can't reduce with SMOTE)
            3: 1200   # Typhoid: boost from 629 to 1200
        }
        
        print(f"[INFO] Target distribution:")
        for label, count in sampling_strategy.items():
            print(f"  - {self.DISEASE_CLASSES[label]}: {count} samples")
        
        # Apply SMOTE
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=42,
            k_neighbors=5
        )
        
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        # Show results
        unique, counts = np.unique(y_balanced, return_counts=True)
        print(f"\n[OK] AFTER balancing:")
        for label, count in zip(unique, counts):
            print(f"  - {self.DISEASE_CLASSES[label]}: {count} ({count/len(y_balanced)*100:.1f}%)")
        
        print(f"\n[SUCCESS] Dataset balanced! Total samples: {len(y_balanced)}")
        print(f"  Original: {len(y)} -> Balanced: {len(y_balanced)}")
        
        return X_balanced, y_balanced
    
    def prepare_complete_dataset(self, filepath):
        """Complete improved data preparation pipeline"""
        # Load data
        df = self.load_data(filepath)
        
        # Extract features
        feature_df = self.extract_features(df)
        
        # Engineer new features
        feature_df = self.engineer_features(feature_df)
        
        # Extract disease data
        disease_df = self.extract_disease_data(df)
        
        # Classify diseases
        disease_labels = self.classify_disease_outbreaks(disease_df)
        
        # Apply SMOTE balancing
        X_balanced, y_balanced = self.apply_smote_balancing(
            feature_df.values, 
            disease_labels
        )
        
        # Create final DataFrame
        feature_names = feature_df.columns.tolist()
        final_df = pd.DataFrame(X_balanced, columns=feature_names)
        final_df['disease_label'] = y_balanced
        
        # Shuffle
        final_df = final_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        return final_df, feature_names
    
    def save_prepared_data(self, df, feature_names, output_path):
        """Save prepared dataset"""
        print(f"\n[STEP 7] Saving improved dataset...")
        
        df.to_csv(output_path, index=False)
        print(f"[OK] Saved to: {output_path}")
        print(f"[INFO] Shape: {df.shape}")
        print(f"[INFO] Features: {len(feature_names)} (5 base + 5 engineered)")
        
        print(f"\n[SUMMARY] Improved Dataset Ready!")
        print(f"  - Total samples: {len(df)}")
        print(f"  - Base features: 5 (pH, Turbidity, TDS, DO, Temperature)")
        print(f"  - Engineered features: 5")
        print(f"  - Total features: {len(feature_names)}")
        print(f"  - Classes: Balanced with SMOTE")
        
        return output_path


def main():
    """Run improved data preparation pipeline"""
    print("\n" + "="*70)
    print("IMPROVED DISEASE OUTBREAK PREDICTION - DATA PREP v2.0")
    print("="*70 + "\n")
    
    # Initialize preparator
    preparator = ImprovedDiseaseDataPreparator()
    
    # Prepare dataset
    df, feature_names = preparator.prepare_complete_dataset('../water_pollution_disease.csv')
    
    # Save
    output_path = preparator.save_prepared_data(
        df, feature_names, 'disease_outbreak_data.csv'
    )
    
    # Save feature names
    import joblib
    joblib.dump(feature_names, 'feature_names_improved.pkl')
    print(f"[OK] Feature names saved to: feature_names_improved.pkl")
    
    print("\n" + "="*70)
    print("IMPROVED DATA PREPARATION COMPLETED!")
    print("="*70)
    print(f"\nNext: Run train_model_v2_improved.py")
    print(f"Expected improvement: 37% -> 60-70% accuracy")
    print("="*70 + "\n")


if __name__ == "__main__":
    np.random.seed(42)
    main()

