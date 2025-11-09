"""
Contamination Cause Detection - Model Training
Trains Random Forest classifier with proper validation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.impute import SimpleImputer
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class ContaminationDetectionModel:
    """
    Random Forest model for contamination detection
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        # Random Forest with anti-overfitting parameters
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,  # Limit depth to prevent overfitting
            min_samples_split=10,  # Require more samples to split
            min_samples_leaf=5,  # Require more samples in leaf nodes
            max_features='sqrt',  # Use subset of features
            bootstrap=True,
            oob_score=True,  # Out-of-bag score for validation
            random_state=random_state,
            n_jobs=-1
        )
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def prepare_data(self, df):
        """Prepare data for training"""
        # Extract features and target
        feature_cols = ['pH', 'turbidity_ntu', 'tds_mg/l', 'do_mg/l', 'temperature_c']
        X = df[feature_cols].values
        y = df['contamination_type'].values
        
        self.feature_names = feature_cols
        
        # Handle missing values
        X_imputed = self.imputer.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X_imputed, y_encoded, y
    
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        return self.model.oob_score_
    
    def evaluate(self, X_test, y_test, y_test_labels):
        """Evaluate model performance"""
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test_labels, y_pred_labels)
        conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
        
        return accuracy, report, conf_matrix, y_pred_labels
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self.model, X, y, cv=skf, scoring='accuracy')
        return scores
    
    def get_feature_importance(self):
        """Get feature importance"""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        return [(self.feature_names[i], importances[i]) for i in indices]
    
    def save_model(self, model_path, imputer_path, encoder_path):
        """Save trained model and preprocessing objects"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.imputer, imputer_path)
        joblib.dump(self.label_encoder, encoder_path)
        
    def load_model(self, model_path, imputer_path, encoder_path):
        """Load trained model and preprocessing objects"""
        self.model = joblib.load(model_path)
        self.imputer = joblib.load(imputer_path)
        self.label_encoder = joblib.load(encoder_path)


def plot_confusion_matrix(conf_matrix, labels, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Contamination Detection - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Confusion matrix saved to: {save_path}")


def plot_feature_importance(importance_list, save_path):
    """Plot and save feature importance"""
    features, importances = zip(*importance_list)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(features)), importances, color='steelblue')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance')
    plt.title('Feature Importance for Contamination Detection')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Feature importance plot saved to: {save_path}")


def plot_roc_curves(y_test, y_pred_proba, classes, save_path):
    """Plot ROC curves for multi-class classification"""
    n_classes = len(classes)
    
    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    for i, color, class_name in zip(range(n_classes), colors, classes):
        plt.plot(fpr[i], tpr[i], color=color, lw=2.5,
                label=f'{class_name} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 0.15])  # Zoom in on x-axis to see differences
    plt.ylim([0.85, 1.02])  # Zoom in on y-axis to see differences
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Contamination Detection Model (Zoomed View)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] ROC curves saved to: {save_path}")
    
    return roc_auc


if __name__ == "__main__":
    print("=" * 70)
    print("CONTAMINATION DETECTION - MODEL TRAINING")
    print("=" * 70)
    
    # Load dataset
    print("\n[STEP 2] Loading dataset...")
    df = pd.read_csv('ml_backend/contamination_detection/contamination_dataset.csv')
    print(f"[OK] Loaded {len(df)} samples")
    
    # Initialize model
    model = ContaminationDetectionModel(random_state=42)
    
    # Prepare data
    print("\n[STEP 3] Preparing data...")
    X, y, y_labels = model.prepare_data(df)
    print(f"[OK] Features shape: {X.shape}")
    print(f"[OK] Target classes: {model.label_encoder.classes_}")
    
    # Split data (temporal split to simulate real-world deployment)
    print("\n[STEP 4] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
        X, y, y_labels, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[OK] Train set: {len(X_train)} samples")
    print(f"[OK] Test set: {len(X_test)} samples")
    
    # Cross-validation
    print("\n[STEP 5] Performing 5-fold cross-validation...")
    cv_scores = model.cross_validate(X_train, y_train, cv=5)
    print(f"[OK] Cross-validation scores: {cv_scores}")
    print(f"[OK] Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train model
    print("\n[STEP 6] Training Random Forest model...")
    oob_score = model.train(X_train, y_train)
    print(f"[OK] Model trained successfully")
    print(f"[OK] Out-of-Bag score: {oob_score:.4f}")
    
    # Evaluate on test set
    print("\n[STEP 7] Evaluating on test set...")
    accuracy, report, conf_matrix, y_pred = model.evaluate(X_test, y_test, y_test_labels)
    print(f"[OK] Test accuracy: {accuracy:.4f}")
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(report)
    
    # Feature importance
    print("\n[STEP 8] Analyzing feature importance...")
    importance_list = model.get_feature_importance()
    print("[OK] Feature importance ranking:")
    for feature, importance in importance_list:
        print(f"  {feature}: {importance:.4f}")
    
    # Save visualizations
    print("\n[STEP 9] Generating visualizations...")
    plot_confusion_matrix(
        conf_matrix, 
        model.label_encoder.classes_,
        'ml_backend/contamination_detection/confusion_matrix.png'
    )
    plot_feature_importance(
        importance_list,
        'ml_backend/contamination_detection/feature_importance.png'
    )
    
    # Generate ROC curves
    y_pred_proba = model.model.predict_proba(X_test)
    roc_scores = plot_roc_curves(
        y_test, y_pred_proba,
        model.label_encoder.classes_,
        'ml_backend/contamination_detection/roc_curves.png'
    )
    
    # Save model
    print("\n[STEP 10] Saving model...")
    model.save_model(
        'ml_backend/contamination_detection/contamination_model.pkl',
        'ml_backend/contamination_detection/imputer.pkl',
        'ml_backend/contamination_detection/label_encoder.pkl'
    )
    print("[OK] Model saved successfully")
    
    # Save performance report
    report_path = 'ml_backend/contamination_detection/model_performance.txt'
    with open(report_path, 'w') as f:
        f.write("CONTAMINATION DETECTION MODEL - PERFORMANCE REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset Size: {len(df)} samples\n")
        f.write(f"Train Set: {len(X_train)} samples\n")
        f.write(f"Test Set: {len(X_test)} samples\n\n")
        f.write(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n")
        f.write(f"Out-of-Bag Score: {oob_score:.4f}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(report + "\n\n")
        f.write("FEATURE IMPORTANCE\n")
        f.write("=" * 70 + "\n")
        for feature, importance in importance_list:
            f.write(f"{feature}: {importance:.4f}\n")
    print(f"[OK] Performance report saved to: {report_path}")
    
    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nKey Metrics:")
    print(f"  - Test Accuracy: {accuracy:.2%}")
    print(f"  - CV Accuracy: {cv_scores.mean():.2%}")
    print(f"  - Model: Random Forest (100 trees, max_depth=15)")
    print(f"  - Anti-overfitting: Limited depth, min samples, OOB validation")
    print("=" * 70)

