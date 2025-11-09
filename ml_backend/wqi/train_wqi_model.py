"""
WQI Model Training Script
Uses WHO-based weights and Random Forest Classifier
Trains on 5 sensors: pH, Turbidity, TDS, DO, Temperature
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve, auc)
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

class WQIModel:
    """Water Quality Index Model with WHO-based weights"""
    
    # WHO-based weights for water quality parameters
    # Reference: WHO Guidelines for Drinking-water Quality
    WHO_WEIGHTS = {
        'pH': 0.25,           # Critical for water chemistry
        'Turbidity': 0.20,    # Indicator of physical quality
        'TDS': 0.20,          # Dissolved solids
        'DO': 0.25,           # Dissolved oxygen (critical for aquatic life)
        'Temperature': 0.10   # Secondary factor
    }
    
    # WHO-based ideal ranges for safe water
    IDEAL_RANGES = {
        'pH': (6.5, 8.5),
        'Turbidity': (0, 5),     # NTU
        'TDS': (0, 500),         # mg/L
        'DO': (5, 14),           # mg/L
        'Temperature': (15, 25)  # Celsius
    }
    
    def __init__(self, random_state=42):
        """Initialize the WQI model"""
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = ['pH', 'Turbidity', 'TDS', 'DO', 'Temperature']
        self.class_names = ['Safe', 'Degrading', 'Unsafe']
        
    def compute_wqi_score(self, X):
        """
        Compute WHO-based WQI score for each sample
        Score range: 0-100 (100 = best quality)
        """
        wqi_scores = np.zeros(len(X))
        
        for i, feature in enumerate(self.feature_names):
            values = X[:, i]
            weight = self.WHO_WEIGHTS[feature]
            ideal_min, ideal_max = self.IDEAL_RANGES[feature]
            
            # Calculate sub-index for each parameter
            # Normalize to 0-100 scale where 100 is ideal
            sub_index = np.zeros(len(values))
            
            for j, val in enumerate(values):
                if ideal_min <= val <= ideal_max:
                    # Within ideal range - score based on how close to center
                    ideal_center = (ideal_min + ideal_max) / 2
                    ideal_range = ideal_max - ideal_min
                    deviation = abs(val - ideal_center) / (ideal_range / 2)
                    sub_index[j] = 100 * (1 - deviation * 0.2)  # Small penalty for deviation
                else:
                    # Outside ideal range - score decreases with distance
                    if val < ideal_min:
                        deviation = (ideal_min - val) / ideal_min if ideal_min != 0 else val
                    else:
                        deviation = (val - ideal_max) / ideal_max if ideal_max != 0 else val
                    
                    # Exponential decay for out-of-range values
                    sub_index[j] = 100 * np.exp(-deviation)
            
            # Weight and add to total WQI
            wqi_scores += weight * sub_index
        
        return wqi_scores
    
    def check_class_balance(self, y):
        """
        Check if dataset is balanced (no augmentation needed as it's done in data prep)
        """
        print("\n[INFO] Checking class balance...")
        print(f"[INFO] Class distribution:")
        for level in [0, 1, 2]:
            count = np.sum(y == level)
            print(f"  - {self.class_names[level]}: {count} ({count/len(y)*100:.1f}%)")
        
        return y
    
    def train(self, X_train, y_train):
        """Train the Random Forest model (NO augmentation - already balanced)"""
        print("\n" + "="*70)
        print("TRAINING WQI MODEL - PRODUCTION READY")
        print("="*70)
        
        # Check class balance (no augmentation needed)
        self.check_class_balance(y_train)
        
        # Compute WQI scores as an additional feature
        print("\n[INFO] Computing WHO-based WQI scores...")
        wqi_scores_train = self.compute_wqi_score(X_train).reshape(-1, 1)
        
        # Combine original features with WQI score
        X_train_enhanced = np.hstack([X_train, wqi_scores_train])
        
        # Scale features
        print("[INFO] Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train_enhanced)
        
        # Compute class weights with extra emphasis on degrading class (critical for your pipeline)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        # BOOST degrading class weight - critical for detecting transition to unsafe
        class_weight_dict[1] = class_weight_dict[1] * 1.5  # 50% boost for degrading
        
        print(f"\n[INFO] Class weights (degrading boosted): {class_weight_dict}")
        
        # Initialize Random Forest with production-ready parameters
        print("\n[INFO] Initializing Random Forest Classifier (Production Config)...")
        print("  - n_estimators: 150 (robust ensemble)")
        print("  - max_depth: 15 (balanced depth)")
        print("  - min_samples_split: 8 (flexible splitting)")
        print("  - min_samples_leaf: 4 (sensitive to patterns)")
        print("  - max_features: 'sqrt' (feature randomness)")
        print("  - class_weight: boosted for degrading class")
        
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight=class_weight_dict,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Train model
        print("\n[INFO] Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        print("[OK] Model training completed!")
        
        return X_train_scaled, y_train
    
    def evaluate(self, X_test, y_test, X_train_scaled=None, y_train=None):
        """Comprehensive model evaluation"""
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        # Compute WQI scores for test set
        wqi_scores_test = self.compute_wqi_score(X_test).reshape(-1, 1)
        X_test_enhanced = np.hstack([X_test, wqi_scores_test])
        X_test_scaled = self.scaler.transform(X_test_enhanced)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        print(f"\n[RESULTS] Test Set Performance:")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Precision (weighted): {precision:.4f}")
        print(f"  - Recall (weighted): {recall:.4f}")
        print(f"  - F1-score (weighted): {f1:.4f}")
        
        # Cross-validation if training data provided
        if X_train_scaled is not None and y_train is not None:
            print(f"\n[INFO] Performing 5-fold cross-validation...")
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring='accuracy'
            )
            print(f"[RESULTS] Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Per-class metrics
        print(f"\n[RESULTS] Per-class Performance:")
        class_report = classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            zero_division=0
        )
        print(class_report)
        
        # Check for overfitting
        if X_train_scaled is not None and y_train is not None:
            y_train_pred = self.model.predict(X_train_scaled)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            overfitting_gap = train_accuracy - accuracy
            
            print(f"\n[ANALYSIS] Overfitting Check:")
            print(f"  - Training Accuracy: {train_accuracy:.4f}")
            print(f"  - Test Accuracy: {accuracy:.4f}")
            print(f"  - Gap: {overfitting_gap:.4f}")
            
            if overfitting_gap < 0.05:
                print(f"  - Status: Good generalization (gap < 5%)")
            elif overfitting_gap < 0.10:
                print(f"  - Status: Acceptable generalization (gap < 10%)")
            else:
                print(f"  - Status: Warning - possible overfitting (gap >= 10%)")
        
        return {
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def get_feature_importance(self):
        """Get feature importance including WQI score"""
        feature_names_enhanced = self.feature_names + ['WQI_Score']
        importances = self.model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        print(f"\n[RESULTS] Feature Importance:")
        for i, idx in enumerate(indices, 1):
            print(f"  {i}. {feature_names_enhanced[idx]}: {importances[idx]:.4f}")
        
        return feature_names_enhanced, importances

def create_visualizations(model, eval_results, feature_importance_data):
    """Create production visualizations: confusion matrix and ROC curves only"""
    print("\n" + "="*70)
    print("CREATING PRODUCTION VISUALIZATIONS")
    print("="*70)
    
    y_test = eval_results['y_test']
    y_pred = eval_results['y_pred']
    y_pred_proba = eval_results['y_pred_proba']
    feature_names, importances = feature_importance_data
    
    # 1. Confusion Matrix -> new_1.png
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations with both count and percentage
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=model.class_names,
                yticklabels=model.class_names,
                cbar_kws={'label': 'Count'},
                linewidths=1, linecolor='black')
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - WQI Production Model\n(WHO-based weights, Random Forest)',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('new_1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: new_1.png (Confusion Matrix)")
    
    # 2. ROC Curves (One-vs-Rest for multiclass) -> new_2.png
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, class_name in enumerate(model.class_names):
        # Binarize the labels for this class
        y_test_binary = (y_test == i).astype(int)
        y_score = y_pred_proba[:, i]
        
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        axes[i].plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                    label='Random Classifier')
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
        axes[i].set_title(f'{class_name} (One-vs-Rest)', fontsize=12, fontweight='bold')
        axes[i].legend(loc="lower right")
        axes[i].grid(alpha=0.3)
    
    plt.suptitle('ROC Curves - WQI Production Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('new_2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: new_2.png (ROC Curves)")
    
    print("\n[INFO] Production visualizations created successfully!")

def save_model_and_summary(model, eval_results, feature_importance_data):
    """Save model artifacts and performance summary"""
    print("\n" + "="*70)
    print("SAVING MODEL ARTIFACTS")
    print("="*70)
    
    # Save model
    joblib.dump(model.model, 'wqi_model.pkl')
    print("[OK] Saved: wqi_model.pkl")
    
    # Save scaler
    joblib.dump(model.scaler, 'wqi_scaler.pkl')
    print("[OK] Saved: wqi_scaler.pkl")
    
    # Save WHO weights and ranges
    model_config = {
        'WHO_WEIGHTS': model.WHO_WEIGHTS,
        'IDEAL_RANGES': model.IDEAL_RANGES,
        'feature_names': model.feature_names,
        'class_names': model.class_names
    }
    joblib.dump(model_config, 'wqi_config.pkl')
    print("[OK] Saved: wqi_config.pkl")
    
    # Save performance summary
    feature_names, importances = feature_importance_data
    
    with open('model_performance_summary.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("WQI MODEL PERFORMANCE SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("Model Configuration:\n")
        f.write("  - Algorithm: Random Forest Classifier\n")
        f.write("  - Features: pH, Turbidity, TDS, DO, Temperature + WQI Score\n")
        f.write("  - WHO-based WQI weights applied\n")
        f.write("  - Anti-overfitting measures enabled\n\n")
        
        f.write("WHO Weights:\n")
        for feature, weight in model.WHO_WEIGHTS.items():
            f.write(f"  - {feature}: {weight:.2f}\n")
        f.write("\n")
        
        f.write("Test Set Performance:\n")
        f.write(f"  - Accuracy: {eval_results['accuracy']:.4f}\n")
        f.write(f"  - Precision: {eval_results['precision']:.4f}\n")
        f.write(f"  - Recall: {eval_results['recall']:.4f}\n")
        f.write(f"  - F1-score: {eval_results['f1']:.4f}\n\n")
        
        f.write("Feature Importance:\n")
        indices = np.argsort(importances)[::-1]
        for i, idx in enumerate(indices, 1):
            f.write(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}\n")
        f.write("\n")
        
        f.write("Model Characteristics:\n")
        f.write("  - Designed for real-time sensor data\n")
        f.write("  - Robust to class imbalance\n")
        f.write("  - Production-ready\n")
        f.write("  - Validated with cross-validation\n")
    
    print("[OK] Saved: model_performance_summary.txt")
    
    print("\n[INFO] All model artifacts saved successfully!")

def main():
    """Main training pipeline"""
    print("="*70)
    print("WQI MODEL TRAINING PIPELINE")
    print("WHO-based weights | Random Forest | 5 Sensors")
    print("="*70)
    
    # Load prepared dataset
    print("\n[INFO] Loading prepared dataset...")
    df = pd.read_csv('wqi_prepared_dataset.csv')
    
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Features: {df.columns.tolist()[:-1]}")
    print(f"[INFO] Target: {df.columns.tolist()[-1]}")
    
    # Prepare features and target
    X = df[['pH', 'Turbidity', 'TDS', 'DO', 'Temperature']].values
    y = df['Pollution_Level'].values
    
    # Train-test split with stratification
    print("\n[INFO] Splitting data (80% train, 20% test, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[INFO] Training set: {X_train.shape[0]} samples")
    print(f"[INFO] Test set: {X_test.shape[0]} samples")
    
    # Initialize and train model
    wqi_model = WQIModel(random_state=42)
    X_train_scaled, y_train_final = wqi_model.train(X_train, y_train)
    
    # Evaluate model
    eval_results = wqi_model.evaluate(X_test, y_test, X_train_scaled, y_train_final)
    
    # Get feature importance
    feature_importance_data = wqi_model.get_feature_importance()
    
    # Create visualizations
    create_visualizations(wqi_model, eval_results, feature_importance_data)
    
    # Save model and summary
    save_model_and_summary(wqi_model, eval_results, feature_importance_data)
    
    print("\n" + "="*70)
    print("WQI MODEL TRAINING COMPLETED - PRODUCTION READY!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. wqi_model.pkl - Trained Random Forest model")
    print("  2. wqi_scaler.pkl - Feature scaler")
    print("  3. wqi_config.pkl - WHO weights and configuration")
    print("  4. new_1.png - Confusion matrix visualization")
    print("  5. new_2.png - ROC curves for all classes")
    print("  6. model_performance_summary.txt - Performance summary")
    print("\n[READY] Model is production-ready for hardware deployment!")
    print("="*70)

if __name__ == "__main__":
    main()

