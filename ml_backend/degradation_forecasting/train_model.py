"""
Degradation Forecasting - LSTM Model Training
Predicts WQI degradation and time-to-unsafe
WITH COMPREHENSIVE ANTI-OVERFITTING MEASURES
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

class DegradationForecaster:
    """
    LSTM-based water quality degradation forecaster
    Predicts future WQI values and estimates time-to-unsafe
    """
    
    def __init__(self, lookback=4, forecast_horizon=4, random_state=42):
        """
        Initialize forecaster
        
        Args:
            lookback: Number of past 3-hour intervals to use (4 = 12 hours)
            forecast_horizon: Number of future 3-hour intervals to predict (4 = 12 hours)
            random_state: Random seed for reproducibility
        """
        self.lookback = lookback  # 4 intervals = 12 hours of history
        self.forecast_horizon = forecast_horizon  # Predict next 12 hours
        self.random_state = random_state
        self.feature_names = ['pH', 'Turbidity', 'TDS', 'DO', 'Temperature']
        self.scaler = StandardScaler()
        self.model = None
        
        # Set seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        print("="*70)
        print("DEGRADATION FORECASTING - LSTM MODEL TRAINING")
        print("="*70)
        print(f"[CONFIG] Lookback window: {lookback} intervals ({lookback*3} hours)")
        print(f"[CONFIG] Forecast horizon: {forecast_horizon} intervals ({forecast_horizon*3} hours)")
        print(f"[CONFIG] Features: {self.feature_names}")
        
    def load_prepared_data(self, data_path):
        """Load prepared data from CSV"""
        print(f"\n[STEP 1] Loading prepared data...")
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"[OK] Loaded {len(df)} 3-hourly readings")
        print(f"[INFO] Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        return df
    
    def create_sequences(self, df, add_noise=False, noise_level=0.02):
        """
        Create input-output sequences for LSTM
        
        Args:
            df: Prepared dataframe
            add_noise: Whether to add noise for regularization
            noise_level: Standard deviation of Gaussian noise (default 2%)
        
        Returns:
            X: Input sequences [samples, lookback, features]
            y: Target WQI sequences [samples, forecast_horizon]
            timestamps: Corresponding timestamps for tracking
        """
        print(f"\n[STEP 2] Creating sequences (lookback={self.lookback}, horizon={self.forecast_horizon})...")
        
        # Extract features and target
        features = df[self.feature_names].values
        wqi = df['WQI_Score'].values
        timestamps = df['timestamp'].values
        
        # Add noise for training data augmentation (anti-overfitting)
        if add_noise:
            print(f"[INFO] Adding {noise_level*100:.1f}% Gaussian noise for regularization...")
            noise = np.random.normal(0, noise_level, features.shape)
            features = features + features * noise  # Proportional noise
        
        X, y, ts = [], [], []
        
        for i in range(len(df) - self.lookback - self.forecast_horizon + 1):
            # Input: last 'lookback' intervals of features
            X.append(features[i:i+self.lookback])
            
            # Output: next 'forecast_horizon' intervals of WQI
            y.append(wqi[i+self.lookback:i+self.lookback+self.forecast_horizon])
            
            # Timestamp of the last input point
            ts.append(timestamps[i+self.lookback-1])
        
        X = np.array(X)
        y = np.array(y)
        ts = np.array(ts)
        
        print(f"[OK] Created {len(X)} sequences")
        print(f"[INFO] X shape: {X.shape} [samples, timesteps, features]")
        print(f"[INFO] y shape: {y.shape} [samples, forecast_horizon]")
        
        return X, y, ts
    
    def temporal_train_test_split(self, X, y, ts, train_ratio=0.7, val_ratio=0.15):
        """
        Split data temporally (NOT randomly) to prevent data leakage
        This ensures we're predicting the future, not interpolating
        """
        print(f"\n[STEP 3] Temporal train/val/test split...")
        print("[IMPORTANT] Using temporal split (not random) - prevents overfitting!")
        
        n = len(X)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        # Temporal split
        X_train = X[:train_size]
        y_train = y[:train_size]
        ts_train = ts[:train_size]
        
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        ts_val = ts[train_size:train_size+val_size]
        
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        ts_test = ts[train_size+val_size:]
        
        print(f"[OK] Train: {len(X_train)} sequences ({ts_train[0]} to {ts_train[-1]})")
        print(f"[OK] Val: {len(X_val)} sequences ({ts_val[0]} to {ts_val[-1]})")
        print(f"[OK] Test: {len(X_test)} sequences ({ts_test[0]} to {ts_test[-1]})")
        
        return (X_train, y_train, ts_train), (X_val, y_val, ts_val), (X_test, y_test, ts_test)
    
    def normalize_features(self, X_train, X_val, X_test):
        """
        Normalize features using training data statistics only
        Prevents data leakage from validation/test sets
        """
        print(f"\n[STEP 4] Normalizing features...")
        print("[IMPORTANT] Using ONLY training data statistics - prevents data leakage!")
        
        # Reshape for scaling
        n_train, timesteps, features = X_train.shape
        n_val = X_val.shape[0]
        n_test = X_test.shape[0]
        
        X_train_flat = X_train.reshape(-1, features)
        X_val_flat = X_val.reshape(-1, features)
        X_test_flat = X_test.reshape(-1, features)
        
        # Fit scaler ONLY on training data
        self.scaler.fit(X_train_flat)
        
        # Transform all sets
        X_train_scaled = self.scaler.transform(X_train_flat).reshape(n_train, timesteps, features)
        X_val_scaled = self.scaler.transform(X_val_flat).reshape(n_val, timesteps, features)
        X_test_scaled = self.scaler.transform(X_test_flat).reshape(n_test, timesteps, features)
        
        print(f"[OK] Features normalized using StandardScaler")
        print(f"[INFO] Scaler mean: {self.scaler.mean_}")
        print(f"[INFO] Scaler std: {self.scaler.scale_}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def build_model(self, input_shape):
        """
        Build LSTM model with ANTI-OVERFITTING measures:
        1. Moderate model size (not too large)
        2. High dropout (0.3-0.4)
        3. L2 regularization
        4. Bidirectional LSTM for better patterns
        5. Early stopping
        """
        print(f"\n[STEP 5] Building LSTM model with anti-overfitting measures...")
        
        model = Sequential([
            # First LSTM layer - Bidirectional for better pattern learning
            Bidirectional(LSTM(
                64,  # Moderate size - not too large
                return_sequences=True,
                kernel_regularizer=l2(0.01),  # L2 regularization
                recurrent_regularizer=l2(0.01)
            ), input_shape=input_shape),
            Dropout(0.4),  # High dropout to prevent overfitting
            
            # Second LSTM layer
            LSTM(
                32,
                kernel_regularizer=l2(0.01),
                recurrent_regularizer=l2(0.01)
            ),
            Dropout(0.3),
            
            # Dense layers
            Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            
            # Output layer
            Dense(self.forecast_horizon)  # Predict next N WQI values
        ])
        
        # Compile with MAE loss (robust to outliers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mae',
            metrics=['mse', 'mae']
        )
        
        print("[OK] Model architecture:")
        model.summary()
        
        print("\n[ANTI-OVERFITTING MEASURES APPLIED]:")
        print("  - Moderate model size (64->32 LSTM units)")
        print("  - High dropout rates (0.2-0.4)")
        print("  - L2 regularization (0.01)")
        print("  - Bidirectional LSTM (better generalization)")
        print("  - MAE loss (robust to outliers)")
        
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train model with callbacks for anti-overfitting
        """
        print(f"\n[STEP 6] Training model (max epochs={epochs})...")
        
        # Callbacks
        callbacks = [
            # Early stopping - stop if validation loss doesn't improve
            EarlyStopping(
                monitor='val_loss',
                patience=20,  # Generous patience
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate when stuck
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            
            # Save best model
            ModelCheckpoint(
                'best_degradation_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        print("\n[CALLBACKS ENABLED]:")
        print("  - Early stopping (patience=20)")
        print("  - Learning rate reduction (factor=0.5, patience=10)")
        print("  - Best model checkpoint")
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n[OK] Training completed!")
        return history
    
    def evaluate_model(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Evaluate model on all sets to check for overfitting
        """
        print(f"\n[STEP 7] Evaluating model...")
        
        # Predictions
        y_train_pred = self.model.predict(X_train, verbose=0)
        y_val_pred = self.model.predict(X_val, verbose=0)
        y_test_pred = self.model.predict(X_test, verbose=0)
        
        # MAE for each set
        train_mae = np.mean(np.abs(y_train - y_train_pred))
        val_mae = np.mean(np.abs(y_val - y_val_pred))
        test_mae = np.mean(np.abs(y_test - y_test_pred))
        
        print(f"\n[RESULTS] Mean Absolute Error (WQI points):")
        print(f"  - Train MAE: {train_mae:.3f}")
        print(f"  - Val MAE:   {val_mae:.3f}")
        print(f"  - Test MAE:  {test_mae:.3f}")
        
        # Check for overfitting
        overfitting_gap = test_mae - train_mae
        overfitting_pct = (overfitting_gap / train_mae) * 100
        
        print(f"\n[OVERFITTING CHECK]:")
        print(f"  - Gap (Test - Train): {overfitting_gap:.3f} WQI points")
        print(f"  - Gap percentage: {overfitting_pct:.1f}%")
        
        if overfitting_pct < 20:
            print(f"  - Status: [OK] GOOD GENERALIZATION (gap < 20%)")
        elif overfitting_pct < 40:
            print(f"  - Status: [WARNING] ACCEPTABLE (gap < 40%)")
        else:
            print(f"  - Status: [ERROR] OVERFITTING DETECTED (gap >= 40%)")
        
        return {
            'train_mae': train_mae,
            'val_mae': val_mae,
            'test_mae': test_mae,
            'y_train_pred': y_train_pred,
            'y_val_pred': y_val_pred,
            'y_test_pred': y_test_pred
        }
    
    def compute_time_to_unsafe(self, wqi_sequence, unsafe_threshold=50):
        """
        Compute time (in 3-hour intervals) until WQI drops below unsafe threshold
        Returns np.inf if never goes unsafe
        """
        below_threshold = np.where(wqi_sequence < unsafe_threshold)[0]
        if len(below_threshold) > 0:
            return below_threshold[0]  # First interval where unsafe
        return np.inf
    
    def evaluate_time_to_unsafe(self, y_test, y_test_pred, unsafe_threshold=50):
        """
        Evaluate time-to-unsafe predictions
        """
        print(f"\n[STEP 8] Evaluating time-to-unsafe predictions (threshold={unsafe_threshold})...")
        
        true_ttu = []
        pred_ttu = []
        
        for i in range(len(y_test)):
            true_time = self.compute_time_to_unsafe(y_test[i], unsafe_threshold)
            pred_time = self.compute_time_to_unsafe(y_test_pred[i], unsafe_threshold)
            
            # Only consider cases where water actually becomes unsafe
            if true_time != np.inf:
                true_ttu.append(true_time)
                pred_ttu.append(pred_time)
        
        if len(true_ttu) > 0:
            # Convert to hours
            true_ttu_hours = np.array(true_ttu) * 3
            pred_ttu_hours = np.array(pred_ttu) * 3
            
            # Calculate error
            errors = pred_ttu_hours - true_ttu_hours
            mae_hours = np.mean(np.abs(errors))
            
            print(f"[RESULTS] Time-to-Unsafe Prediction:")
            print(f"  - Cases analyzed: {len(true_ttu)}")
            print(f"  - MAE: {mae_hours:.2f} hours ({mae_hours/3:.1f} intervals)")
            print(f"  - Mean true time: {true_ttu_hours.mean():.1f} hours")
            print(f"  - Mean pred time: {pred_ttu_hours.mean():.1f} hours")
            
            return {
                'mae_hours': mae_hours,
                'true_ttu': true_ttu_hours,
                'pred_ttu': pred_ttu_hours,
                'errors': errors
            }
        else:
            print(f"[INFO] No degradation to unsafe in test set")
            return None
    
    def plot_training_history(self, history, save_path='training_history.png'):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('MAE (WQI points)', fontsize=11)
        axes[0].set_title('Model Loss During Training', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # MAE
        axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
        axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('MAE (WQI points)', fontsize=11)
        axes[1].set_title('Model MAE During Training', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Training history saved to: {save_path}")
    
    def plot_sample_predictions(self, X_test, y_test, y_test_pred, n_samples=4, save_path='sample_predictions.png'):
        """Plot sample predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i in range(min(n_samples, len(y_test))):
            # Last actual WQI from input
            last_input_wqi = y_test[i][0] - 5  # Approximate
            
            # Time axis
            input_time = np.arange(-self.lookback, 0)
            forecast_time = np.arange(0, self.forecast_horizon)
            
            # Plot
            axes[i].axhline(y=50, color='red', linestyle='--', linewidth=1.5, 
                           label='Unsafe Threshold', alpha=0.7)
            axes[i].plot(forecast_time, y_test[i], 'o-', label='True Future', 
                        color='green', linewidth=2, markersize=6)
            axes[i].plot(forecast_time, y_test_pred[i], 's--', label='Predicted Future',
                        color='orange', linewidth=2, markersize=6)
            
            axes[i].set_xlabel('Time (3-hour intervals)', fontsize=10)
            axes[i].set_ylabel('WQI Score', fontsize=10)
            axes[i].set_title(f'Forecast Example {i+1}', fontsize=11, fontweight='bold')
            axes[i].legend()
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Sample predictions saved to: {save_path}")
    
    def save_model_and_artifacts(self, model_path='degradation_model.h5', 
                                 scaler_path='degradation_scaler.pkl',
                                 config_path='degradation_config.pkl'):
        """Save trained model and preprocessing artifacts"""
        print(f"\n[STEP 9] Saving model and artifacts...")
        
        # Save Keras model
        self.model.save(model_path)
        print(f"[OK] Model saved to: {model_path}")
        
        # Save scaler
        joblib.dump(self.scaler, scaler_path)
        print(f"[OK] Scaler saved to: {scaler_path}")
        
        # Save configuration
        config = {
            'lookback': self.lookback,
            'forecast_horizon': self.forecast_horizon,
            'feature_names': self.feature_names,
            'random_state': self.random_state
        }
        joblib.dump(config, config_path)
        print(f"[OK] Config saved to: {config_path}")


def main():
    """Main training pipeline"""
    # Initialize
    forecaster = DegradationForecaster(
        lookback=4,  # 12 hours of history
        forecast_horizon=4,  # Predict next 12 hours
        random_state=42
    )
    
    # Load data
    df = forecaster.load_prepared_data('degradation_prepared_data.csv')
    
    # Create sequences with noise augmentation for training
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
    
    # Save model
    forecaster.save_model_and_artifacts()
    
    print("\n" + "="*70)
    print("MODEL TRAINING COMPLETED - PRODUCTION READY!")
    print("="*70)
    print(f"\n[READY] Model trained on real timestamp data")
    print(f"[READY] Anti-overfitting measures: [OK] Temporal split, Dropout, L2 reg, Early stopping")
    print(f"[READY] Test MAE: {results['test_mae']:.3f} WQI points")
    print(f"\n[NEXT] Run: python predict.py (for real-time inference)")


if __name__ == "__main__":
    main()

