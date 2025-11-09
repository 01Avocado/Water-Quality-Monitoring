"""
Contamination Detection - Complete Pipeline
Runs dataset generation, model training, and testing in sequence
"""

import subprocess
import sys
import os

def run_script(script_path, description):
    """Run a Python script and check for errors"""
    print(f"\n{'='*70}")
    print(f"RUNNING: {description}")
    print(f"{'='*70}\n")
    
    # Use Python 3.12 which has the required packages
    python_exe = r"C:\Users\RIDDHI KULKARNI\AppData\Local\Programs\Python\Python312\python.exe"
    
    result = subprocess.run(
        [python_exe, script_path],
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print(f"\n[ERROR] Failed to run {description}")
        return False
    else:
        print(f"\n[SUCCESS] Completed {description}")
        return True

def main():
    """Run complete contamination detection pipeline"""
    print("="*70)
    print("CONTAMINATION DETECTION - COMPLETE PIPELINE")
    print("="*70)
    print("\nThis will:")
    print("  1. Generate synthetic dataset with anti-overfitting measures")
    print("  2. Train Random Forest model with cross-validation")
    print("  3. Test model with sample predictions")
    print("="*70)
    
    base_dir = "ml_backend/contamination_detection"
    
    # Step 1: Generate dataset
    if not run_script(
        os.path.join(base_dir, "dataset_generator.py"),
        "Dataset Generation"
    ):
        return False
    
    # Step 2: Train model
    if not run_script(
        os.path.join(base_dir, "train_model.py"),
        "Model Training"
    ):
        return False
    
    # Step 3: Test predictions
    if not run_script(
        os.path.join(base_dir, "predict.py"),
        "Real-time Prediction Test"
    ):
        return False
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated Files:")
    print(f"  - {base_dir}/contamination_dataset.csv")
    print(f"  - {base_dir}/contamination_model.pkl")
    print(f"  - {base_dir}/imputer.pkl")
    print(f"  - {base_dir}/label_encoder.pkl")
    print(f"  - {base_dir}/confusion_matrix.png")
    print(f"  - {base_dir}/feature_importance.png")
    print(f"  - {base_dir}/model_performance.txt")
    print("="*70)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

