"""
Degradation Forecasting - Complete Pipeline Runner
Automates: Data Preparation → Training → Evaluation → Deployment
"""

import sys
import os
from datetime import datetime

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")

def main():
    """Run complete degradation forecasting pipeline"""
    
    print_header("NEERWANA DEGRADATION FORECASTING PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # =========================================================================
    # STEP 1: DATA PREPARATION
    # =========================================================================
    print_header("STEP 1: DATA PREPARATION")
    print("[INFO] Loading timestamp dataset and applying WQI model...")
    print("[INFO] This may take a few minutes...\n")
    
    try:
        from prepare_data import DataPreparation
        
        prep = DataPreparation(
            timestamp_path='../timestamp dataset.CSV.xls',
            wqi_model_path='../wqi/wqi_model.pkl',
            wqi_scaler_path='../wqi/wqi_scaler.pkl',
            wqi_config_path='../wqi/wqi_config.pkl',
            do_imputer_path='../common/do_imputer.pkl',
        )
        
        df, sequences = prep.run_full_pipeline('degradation_prepared_data.csv')
        
        print("\n[OK] Data preparation completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] Required file not found: {e}")
        print("\n[HELP] Make sure you have:")
        print("  1. timestamp dataset.CSV.xls in ml_backend/")
        print("  2. Trained WQI model files in ml_backend/wqi/")
        print("\nRun the WQI model training first if needed:")
        print("  cd ../wqi")
        print("  python train_wqi_model.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] During data preparation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # =========================================================================
    # STEP 2: MODEL TRAINING
    # =========================================================================
    print_header("STEP 2: LSTM MODEL TRAINING")
    print("[INFO] Training degradation forecasting model...")
    print("[INFO] This will take several minutes...\n")
    
    try:
        from train_model import DegradationForecaster
        
        # Initialize
        forecaster = DegradationForecaster(
            lookback=4,  # 12 hours of history
            forecast_horizon=4,  # Predict next 12 hours
            random_state=42
        )
        
        # Load data
        df = forecaster.load_prepared_data('degradation_prepared_data.csv')
        
        # Create sequences with noise augmentation
        X, y, ts = forecaster.create_sequences(df, add_noise=True, noise_level=0.02)
        
        # Temporal split
        train_data, val_data, test_data = forecaster.temporal_train_test_split(X, y, ts)
        X_train, y_train, ts_train = train_data
        X_val, y_val, ts_val = val_data
        X_test, y_test, ts_test = test_data
        
        # Normalize
        X_train_scaled, X_val_scaled, X_test_scaled = forecaster.normalize_features(
            X_train, X_val, X_test
        )
        
        # Build model
        input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
        forecaster.build_model(input_shape)
        
        # Train
        history = forecaster.train_model(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            epochs=100,
            batch_size=32
        )
        
        print("\n[OK] Model training completed successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] During training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # =========================================================================
    # STEP 3: MODEL EVALUATION
    # =========================================================================
    print_header("STEP 3: MODEL EVALUATION")
    print("[INFO] Evaluating model performance...\n")
    
    try:
        # Evaluate
        results = forecaster.evaluate_model(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            X_test_scaled, y_test
        )
        
        # Evaluate time-to-unsafe
        ttu_results = forecaster.evaluate_time_to_unsafe(y_test, results['y_test_pred'])
        
        # Plot results
        forecaster.plot_training_history(history)
        forecaster.plot_sample_predictions(X_test_scaled, y_test, results['y_test_pred'])
        
        print("\n[OK] Model evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] During evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # =========================================================================
    # STEP 4: SAVE MODEL ARTIFACTS
    # =========================================================================
    print_header("STEP 4: SAVING MODEL ARTIFACTS")
    print("[INFO] Saving model for deployment...\n")
    
    try:
        forecaster.save_model_and_artifacts(
            model_path='degradation_model.h5',
            scaler_path='degradation_scaler.pkl',
            config_path='degradation_config.pkl'
        )
        
        print("\n[OK] Model artifacts saved successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] Saving model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # =========================================================================
    # STEP 5: GENERATE SUMMARY REPORT
    # =========================================================================
    print_header("STEP 5: GENERATING SUMMARY REPORT")
    
    try:
        with open('model_performance_summary.txt', 'w') as f:
            f.write("="*70 + "\n")
            f.write("DEGRADATION FORECASTING MODEL - PERFORMANCE SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Model Configuration:\n")
            f.write("  - Architecture: Bidirectional LSTM\n")
            f.write("  - Input: 4 timesteps (12 hours) × 5 features\n")
            f.write("  - Output: 4 WQI predictions (next 12 hours)\n")
            f.write("  - Features: pH, Turbidity, TDS, DO, Temperature\n")
            f.write("  - Sensors: 3-hour intervals\n\n")
            
            f.write("Dataset Information:\n")
            f.write(f"  - Total sequences: {len(X)}\n")
            f.write(f"  - Training: {len(X_train)} sequences\n")
            f.write(f"  - Validation: {len(X_val)} sequences\n")
            f.write(f"  - Test: {len(X_test)} sequences\n")
            f.write(f"  - Time range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}\n\n")
            
            f.write("Performance Metrics:\n")
            f.write(f"  - Train MAE: {results['train_mae']:.3f} WQI points\n")
            f.write(f"  - Val MAE: {results['val_mae']:.3f} WQI points\n")
            f.write(f"  - Test MAE: {results['test_mae']:.3f} WQI points\n")
            
            overfitting_gap = results['test_mae'] - results['train_mae']
            overfitting_pct = (overfitting_gap / results['train_mae']) * 100
            f.write(f"  - Overfitting gap: {overfitting_pct:.1f}%\n")
            
            if overfitting_pct < 20:
                f.write("  - Generalization: ✓ GOOD (gap < 20%)\n")
            elif overfitting_pct < 40:
                f.write("  - Generalization: ⚠ ACCEPTABLE (gap < 40%)\n")
            else:
                f.write("  - Generalization: ✗ OVERFITTING (gap >= 40%)\n")
            
            f.write("\n")
            
            if ttu_results:
                f.write("Time-to-Unsafe Prediction:\n")
                f.write(f"  - MAE: {ttu_results['mae_hours']:.2f} hours\n")
                f.write(f"  - Analyzed cases: {len(ttu_results['true_ttu'])}\n")
                f.write(f"  - Mean true time: {ttu_results['true_ttu'].mean():.1f} hours\n")
                f.write(f"  - Mean pred time: {ttu_results['pred_ttu'].mean():.1f} hours\n\n")
            
            f.write("Anti-Overfitting Measures:\n")
            f.write("  ✓ Temporal train/val/test split (not random)\n")
            f.write("  ✓ High dropout rates (0.2-0.4)\n")
            f.write("  ✓ L2 regularization (0.01)\n")
            f.write("  ✓ Data augmentation (2% noise)\n")
            f.write("  ✓ Early stopping (patience=20)\n")
            f.write("  ✓ Learning rate reduction\n")
            f.write("  ✓ Moderate model size (64→32 LSTM units)\n")
            f.write("  ✓ Bidirectional LSTM\n\n")
            
            f.write("Generated Files:\n")
            f.write("  - degradation_prepared_data.csv (Processed dataset)\n")
            f.write("  - degradation_model.h5 (Trained LSTM model)\n")
            f.write("  - degradation_scaler.pkl (Feature scaler)\n")
            f.write("  - degradation_config.pkl (Model configuration)\n")
            f.write("  - training_history.png (Training curves)\n")
            f.write("  - sample_predictions.png (Forecast examples)\n")
            f.write("  - model_performance_summary.txt (This file)\n\n")
            
            f.write("Deployment Status:\n")
            if results['test_mae'] < 5.0 and overfitting_pct < 30:
                f.write("  [OK] PRODUCTION READY\n")
                f.write("  - Test MAE within acceptable range (< 5.0)\n")
                f.write("  - Good generalization (gap < 30%)\n")
                f.write("  - Model ready for real-time deployment\n")
            elif results['test_mae'] < 8.0 and overfitting_pct < 40:
                f.write("  [WARNING] ACCEPTABLE FOR DEPLOYMENT\n")
                f.write("  - Test MAE acceptable (< 8.0)\n")
                f.write("  - Monitor performance on live data\n")
                f.write("  - Consider retraining with more data\n")
            else:
                f.write("  [ERROR] NEEDS IMPROVEMENT\n")
                f.write("  - Test MAE too high or overfitting detected\n")
                f.write("  - Review model architecture or data quality\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print("[OK] Performance summary saved to: model_performance_summary.txt")
        print("\n[OK] Summary report generated successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] Generating summary: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # PIPELINE COMPLETE
    # =========================================================================
    print_header("PIPELINE EXECUTION COMPLETED!")
    
    print("[RESULTS SUMMARY]:")
    print(f"  - Test MAE: {results['test_mae']:.3f} WQI points")
    print(f"  - Overfitting gap: {overfitting_pct:.1f}%")
    if ttu_results:
        print(f"  - Time-to-unsafe MAE: {ttu_results['mae_hours']:.2f} hours")
    
    print("\n[GENERATED FILES]:")
    print("  - degradation_prepared_data.csv")
    print("  - degradation_model.h5")
    print("  - degradation_scaler.pkl")
    print("  - degradation_config.pkl")
    print("  - training_history.png")
    print("  - sample_predictions.png")
    print("  - model_performance_summary.txt")
    
    print("\n[NEXT STEPS]:")
    print("  1. Review training_history.png for training curves")
    print("  2. Review sample_predictions.png for forecast quality")
    print("  3. Read model_performance_summary.txt for detailed metrics")
    print("  4. Test predictions: python predict.py")
    print("  5. Integrate with your WQI model pipeline")
    
    print("\n[INTEGRATION EXAMPLE]:")
    print("  from predict import DegradationPredictor")
    print("  predictor = DegradationPredictor(")
    print("      model_path='degradation_model.h5',")
    print("      scaler_path='degradation_scaler.pkl',")
    print("      config_path='degradation_config.pkl'")
    print("  )")
    print("  report = predictor.get_full_forecast_report(sensor_history)")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[FATAL ERROR]: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

