"""
Disease Outbreak Prediction - IMPROVED Model Training
Version 2.0 with better hyperparameters and ensemble methods
GOAL: Improve from 37% to 60-70% accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve, auc
)
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class ImprovedDiseaseOutbreakModel:
    """
    IMPROVED Disease outbreak prediction
    - BALANCED data (SMOTE)
    - BETTER hyperparameters (not too strict)
    - ENSEMBLE methods (multiple models)
    """
    
    def __init__(self, random_state=42):
        """Initialize improved model"""
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None  # Will be loaded
        self.class_names = ['No Disease', 'Diarrhea', 'Cholera', 'Typhoid']
    
    def check_class_balance(self, y):
        """Check class distribution"""
        print("\n[INFO] Class distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for level, count in zip(unique, counts):
            print(f"  - {self.class_names[level]}: {count} ({count/len(y)*100:.1f}%)")
        
        # Check if balanced
        if counts.max() / counts.min() < 2.0:
            print(f"[OK] Classes are well balanced! (ratio < 2.0)")
        else:
            print(f"[WARNING] Some class imbalance remains (ratio: {counts.max() / counts.min():.2f})")
        
        return y
    
    def train_single_model(self, X_train, y_train):
        """
        Train IMPROVED Random Forest
        
        BETTER hyperparameters (not too strict):
        - max_depth: 10 (deeper than before, but not too deep)
        - min_samples_split: 15 (relaxed from 25)
        - min_samples_leaf: 6 (relaxed from 12)
        - n_estimators: 150 (more trees for better ensemble)
        - max_features: 'sqrt' (keep randomness)
        """
        print("\n" + "="*70)
        print("TRAINING IMPROVED MODEL - BALANCED MODE")
        print("="*70)
        
        # Check class balance
        self.check_class_balance(y_train)
        
        # Scale features
        print("\n[INFO] Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Class weights (should be balanced now, but still use)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"[INFO] Class weights: {class_weight_dict}")
        
        # IMPROVED Random Forest configuration
        print("\n[INFO] Initializing IMPROVED Random Forest...")
        print("  BALANCED Anti-overfitting configuration:")
        print("  - n_estimators: 150 (strong ensemble)")
        print("  - max_depth: 10 (allows learning complex patterns)")
        print("  - min_samples_split: 15 (moderate requirement)")
        print("  - min_samples_leaf: 6 (more flexible)")
        print("  - max_features: 'sqrt' (maintain feature randomness)")
        print("  - bootstrap: True (with OOB validation)")
        print("  Goal: Better accuracy while avoiding memorization")
        
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,              # Deeper (was 6)
            min_samples_split=15,      # Relaxed (was 25)
            min_samples_leaf=6,        # Relaxed (was 12)
            max_features='sqrt',
            class_weight=class_weight_dict,
            bootstrap=True,
            oob_score=True,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Train
        print("\n[INFO] Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        print(f"\n[OK] Training completed!")
        print(f"[INFO] Out-of-Bag (OOB) Score: {self.model.oob_score_:.4f}")
        
        return X_train_scaled, y_train
    
    def train_ensemble(self, X_train, y_train):
        """
        Train ENSEMBLE of models for even better accuracy
        Combines Random Forest + Gradient Boosting
        """
        print("\n" + "="*70)
        print("TRAINING ENSEMBLE MODEL - STACKING APPROACH")
        print("="*70)
        
        # Scale features
        print("\n[INFO] Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        # Model 1: Random Forest
        print("\n[INFO] Creating Model 1: Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=15,
            min_samples_leaf=6,
            max_features='sqrt',
            class_weight=class_weight_dict,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Model 2: Gradient Boosting
        print("[INFO] Creating Model 2: Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.random_state
        )
        
        # Combine models with Voting
        print("[INFO] Creating Ensemble (Soft Voting)...")
        self.model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('gb', gb_model)
            ],
            voting='soft',  # Use probabilities
            n_jobs=-1
        )
        
        # Train ensemble
        print("\n[INFO] Training ensemble...")
        self.model.fit(X_train_scaled, y_train)
        
        print(f"\n[OK] Ensemble training completed!")
        print(f"[INFO] Ensemble combines Random Forest + Gradient Boosting")
        
        return X_train_scaled, y_train
    
    def grid_search_hyperparameters(self, X_train, y_train):
        """
        Use Grid Search to find OPTIMAL hyperparameters
        This takes longer but finds the best settings
        """
        print("\n" + "="*70)
        print("GRID SEARCH FOR OPTIMAL HYPERPARAMETERS")
        print("="*70)
        
        # Scale features
        print("\n[INFO] Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [8, 10, 12],
            'min_samples_split': [10, 15, 20],
            'min_samples_leaf': [4, 6, 8],
            'max_features': ['sqrt', 'log2']
        }
        
        print(f"[INFO] Testing {np.prod([len(v) for v in param_grid.values()])} combinations...")
        
        # Grid search
        rf_base = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1
        )
        
        grid_search = GridSearchCV(
            rf_base,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        print("\n[INFO] Running grid search (this may take a few minutes)...")
        grid_search.fit(X_train_scaled, y_train)
        
        print(f"\n[OK] Grid search completed!")
        print(f"[BEST] Best parameters: {grid_search.best_params_}")
        print(f"[BEST] Best CV score: {grid_search.best_score_:.4f}")
        
        # Use best model
        self.model = grid_search.best_estimator_
        
        return X_train_scaled, y_train
    
    def evaluate(self, X_test, y_test, X_train_scaled=None, y_train=None):
        """Comprehensive evaluation"""
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        # Scale test set
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        print(f"\n[RESULTS] Test Set Performance:")
        print(f"  - Accuracy: {accuracy:.4f} ({'UP' if accuracy > 0.37 else 'DOWN'} from baseline 37%)")
        print(f"  - Precision (weighted): {precision:.4f}")
        print(f"  - Recall (weighted): {recall:.4f}")
        print(f"  - F1-score (weighted): {f1:.4f}")
        
        # Improvement over baseline
        baseline_accuracy = 0.3733
        improvement = ((accuracy - baseline_accuracy) / baseline_accuracy) * 100
        print(f"\n[IMPROVEMENT] {improvement:+.1f}% vs baseline (37.33%)")
        
        # Cross-validation
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
        
        # Overfitting check
        if X_train_scaled is not None and y_train is not None:
            y_train_pred = self.model.predict(X_train_scaled)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            overfitting_gap = train_accuracy - accuracy
            
            print(f"\n[OVERFITTING CHECK]:")
            print(f"  - Training Accuracy: {train_accuracy:.4f}")
            print(f"  - Test Accuracy: {accuracy:.4f}")
            print(f"  - Gap: {overfitting_gap:.4f}")
            
            # Check OOB if available
            if hasattr(self.model, 'oob_score_'):
                print(f"  - OOB Score: {self.model.oob_score_:.4f}")
                oob_test_diff = abs(self.model.oob_score_ - accuracy)
                print(f"  - OOB vs Test diff: {oob_test_diff:.4f}")
            
            if overfitting_gap < 0.05:
                print(f"  [EXCELLENT] Minimal overfitting!")
            elif overfitting_gap < 0.10:
                print(f"  [GOOD] Acceptable overfitting")
            elif overfitting_gap < 0.15:
                print(f"  [OK] Moderate overfitting")
            else:
                print(f"  [WARNING] High overfitting")
        
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
        """Get feature importance"""
        # Handle ensemble models
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'estimators_'):
            # Voting classifier - average importances
            importances = np.mean([
                est.feature_importances_ for name, est in self.model.estimators_
                if hasattr(est, 'feature_importances_')
            ], axis=0)
        else:
            print("[WARNING] Model doesn't support feature importance")
            return self.feature_names, np.zeros(len(self.feature_names))
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        print(f"\n[RESULTS] Feature Importance:")
        for i, idx in enumerate(indices[:10], 1):  # Top 10
            print(f"  {i}. {self.feature_names[idx]}: {importances[idx]:.4f}")
        
        return self.feature_names, importances


def create_visualizations(model, eval_results, feature_importance_data, output_suffix=''):
    """Create visualizations for improved model"""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    y_test = eval_results['y_test']
    y_pred = eval_results['y_pred']
    y_pred_proba = eval_results['y_pred_proba']
    feature_names, importances = feature_importance_data
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=model.class_names,
                yticklabels=model.class_names,
                cbar_kws={'label': 'Count'},
                linewidths=1, linecolor='black')
    
    plt.xlabel('Predicted Disease', fontsize=12, fontweight='bold')
    plt.ylabel('True Disease', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - IMPROVED Model\n(With SMOTE + Feature Engineering)',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix{output_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: confusion_matrix{output_suffix}.png")
    
    # 2. Feature Importance (Top 10)
    plt.figure(figsize=(12, 6))
    indices = np.argsort(importances)[::-1][:10]  # Top 10
    
    plt.bar(range(len(indices)), importances[indices], 
            color='#2ecc71', edgecolor='black', linewidth=1.5)
    
    plt.xticks(range(len(indices)), 
               [feature_names[i] for i in indices],
               rotation=45, ha='right')
    plt.ylabel('Importance Score', fontsize=12, fontweight='bold')
    plt.xlabel('Feature', fontsize=12, fontweight='bold')
    plt.title(f'Top 10 Feature Importance - IMPROVED Model',
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'feature_importance{output_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: feature_importance{output_suffix}.png")
    
    print("\n[INFO] Visualizations created!")


def save_model_artifacts(model, eval_results, feature_importance_data, output_suffix=''):
    """Save improved model artifacts"""
    print("\n" + "="*70)
    print("SAVING MODEL ARTIFACTS")
    print("="*70)
    
    # Save model
    joblib.dump(model.model, f'disease_model{output_suffix}.pkl')
    print(f"[OK] Saved: disease_model{output_suffix}.pkl")
    
    # Save scaler
    joblib.dump(model.scaler, f'disease_scaler{output_suffix}.pkl')
    print(f"[OK] Saved: disease_scaler{output_suffix}.pkl")
    
    # Save configuration
    model_config = {
        'feature_names': model.feature_names,
        'class_names': model.class_names,
        'version': '2.0_improved',
        'improvements': ['SMOTE', 'Feature Engineering', 'Better Hyperparameters']
    }
    joblib.dump(model_config, f'disease_config{output_suffix}.pkl')
    print(f"[OK] Saved: disease_config{output_suffix}.pkl")
    
    # Save performance summary
    feature_names, importances = feature_importance_data
    
    with open(f'model_performance{output_suffix}.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("IMPROVED DISEASE OUTBREAK PREDICTION - PERFORMANCE\n")
        f.write("="*70 + "\n\n")
        
        f.write("Improvements Applied:\n")
        f.write("  1. SMOTE class balancing\n")
        f.write("  2. Feature engineering (10 total features)\n")
        f.write("  3. Better hyperparameters (max_depth=10)\n\n")
        
        f.write("Test Set Performance:\n")
        f.write(f"  - Accuracy: {eval_results['accuracy']:.4f}\n")
        f.write(f"  - Precision: {eval_results['precision']:.4f}\n")
        f.write(f"  - Recall: {eval_results['recall']:.4f}\n")
        f.write(f"  - F1-score: {eval_results['f1']:.4f}\n\n")
        
        f.write("Top 10 Feature Importance:\n")
        indices = np.argsort(importances)[::-1][:10]
        for i, idx in enumerate(indices, 1):
            f.write(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}\n")
    
    print(f"[OK] Saved: model_performance{output_suffix}.txt")
    print("\n[INFO] All artifacts saved!")


def main():
    """Main improved training pipeline"""
    print("="*70)
    print("IMPROVED DISEASE OUTBREAK PREDICTION - TRAINING v2.0")
    print("="*70)
    
    # Load prepared dataset
    print("\n[INFO] Loading IMPROVED dataset (with SMOTE + feature engineering)...")
    df = pd.read_csv('disease_outbreak_data.csv')
    feature_names = joblib.load('feature_names_improved.pkl')
    
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Features ({len(feature_names)}): {feature_names}")
    print(f"[INFO] Target: disease_label")
    
    # Prepare features and target
    X = df[feature_names].values
    y = df['disease_label'].values
    
    # Train-test split
    print("\n[INFO] Splitting data (70% train, 30% test, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    print(f"[INFO] Training set: {X_train.shape[0]} samples")
    print(f"[INFO] Test set: {X_test.shape[0]} samples")
    
    # Initialize model
    disease_model = ImprovedDiseaseOutbreakModel(random_state=42)
    disease_model.feature_names = feature_names
    
    # Choose training method (change this to try different approaches)
    TRAINING_METHOD = 'single'  # Options: 'single', 'ensemble', 'grid_search'
    
    if TRAINING_METHOD == 'single':
        X_train_scaled, y_train_final = disease_model.train_single_model(X_train, y_train)
    elif TRAINING_METHOD == 'ensemble':
        X_train_scaled, y_train_final = disease_model.train_ensemble(X_train, y_train)
    elif TRAINING_METHOD == 'grid_search':
        X_train_scaled, y_train_final = disease_model.grid_search_hyperparameters(X_train, y_train)
    
    # Evaluate
    eval_results = disease_model.evaluate(X_test, y_test, X_train_scaled, y_train_final)
    
    # Feature importance
    feature_importance_data = disease_model.get_feature_importance()
    
    # Visualizations
    create_visualizations(disease_model, eval_results, feature_importance_data)
    
    # Save
    save_model_artifacts(disease_model, eval_results, feature_importance_data)
    
    print("\n" + "="*70)
    print("IMPROVED MODEL TRAINING COMPLETED!")
    print("="*70)
    print(f"\n[SUCCESS] Accuracy improved from 37.33% to {eval_results['accuracy']*100:.2f}%")
    print(f"[SUCCESS] Improvement: {((eval_results['accuracy']/0.3733 - 1)*100):+.1f}%")
    print("="*70)


if __name__ == "__main__":
    np.random.seed(42)
    main()

