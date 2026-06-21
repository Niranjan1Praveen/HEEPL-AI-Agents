"""
Time Series Forecasting Model for Industrial Wastewater Parameters
-----------------------------------------------------------------
This model uses Random Forest Regressors wrapped in sklearn Pipelines to predict
future values of wastewater parameters based on historical patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add server to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.csv_loader import CSVLoader
from config.mappings import get_all_industry_ids

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib

class WastewaterForecaster:
    """
    Forecasting model for wastewater parameters.
    Predicts future values of parameters like BOD, COD, TSS, etc.
    """
    
    def __init__(self, forecast_horizon=3, lookback_window=5):
        """
        Args:
            forecast_horizon: Number of future steps to predict (samples)
            lookback_window: Number of past samples to use for prediction
        """
        self.forecast_horizon = forecast_horizon
        self.lookback_window = lookback_window
        self.models = {}  # Store model and scaler per parameter
        self.parameter_columns = []
        self.model_type = 'random_forest'
        
    def _get_unit(self, parameter):
        """Get unit for a parameter"""
        if 'BOD' in parameter or 'COD' in parameter or 'TSS' in parameter or 'TDS' in parameter:
            return 'mg/L'
        elif 'pH' in parameter:
            return 'pH'
        elif 'Temperature' in parameter:
            return '°C'
        else:
            return 'units'
            
    def train_random_forest(self, df, parameter):
        """Train Random Forest model for forecasting - FIXED for Data Leakage"""
        values = df[parameter].values
        # Remove NaN values
        values = values[~np.isnan(values)]
        
        # Check if enough data
        min_required = self.lookback_window + self.forecast_horizon
        if len(values) < min_required * 2:  # Make sure we have enough to split train/test
            print(f"⚠️ Not enough data for {parameter}. Have {len(values)}, need at least {min_required * 2}")
            return None
            
        # 1. SPLIT FIRST before scaling to avoid Data Leakage
        split_idx = int(len(values) * 0.8)
        train_values = values[:split_idx]
        test_values = values[split_idx:]
        
        # 2. FIT SCALER ONLY ON TRAINING SET
        scaler = MinMaxScaler()
        scaler.fit(train_values.reshape(-1, 1))
        
        # 3. TRANSFORM train and test set separately
        train_scaled = scaler.transform(train_values.reshape(-1, 1)).flatten()
        test_scaled = scaler.transform(test_values.reshape(-1, 1)).flatten()
        
        # 4. PREPARE SEQUENCES
        X_train, y_train = [], []
        for i in range(len(train_scaled) - self.lookback_window - self.forecast_horizon + 1):
            X_train.append(train_scaled[i:i + self.lookback_window])
            y_train.append(train_scaled[i + self.lookback_window:i + self.lookback_window + self.forecast_horizon])
            
        X_test, y_test = [], []
        for i in range(len(test_scaled) - self.lookback_window - self.forecast_horizon + 1):
            X_test.append(test_scaled[i:i + self.lookback_window])
            y_test.append(test_scaled[i + self.lookback_window:i + self.lookback_window + self.forecast_horizon])
            
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"⚠️ Could not create valid sequences for {parameter}")
            return None
            
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Build training pipeline
        pipeline = Pipeline([
            ('model', RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        # Train pipeline
        pipeline.fit(X_train, y_train)
        
        # Predict on test data
        y_pred_scaled = pipeline.predict(X_test)
        
        # Evaluate on the first step prediction in unscaled space
        # Decode first step prediction
        y_test_first = y_test[:, 0].reshape(-1, 1)
        y_pred_first = y_pred_scaled[:, 0].reshape(-1, 1)
        
        y_test_original = scaler.inverse_transform(y_test_first).flatten()
        y_pred_original = scaler.inverse_transform(y_pred_first).flatten()
        
        mae = mean_absolute_error(y_test_original, y_pred_original)
        mse = mean_squared_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_original, y_pred_original)
        
        # MAPE calculation (ignoring zero values)
        non_zero = y_test_original != 0
        if np.any(non_zero):
            mape = np.mean(np.abs((y_test_original[non_zero] - y_pred_original[non_zero]) / y_test_original[non_zero])) * 100
        else:
            mape = 0.0
            
        unit = self._get_unit(parameter)
        print(f"   📊 {parameter:<22} - MAE: {mae:.2f} {unit:<5} | RMSE: {rmse:.2f} | R²: {r2:+.3f} | MAPE: {mape:.2f}%")
        
        return {
            'pipeline': pipeline,
            'scaler': scaler,
            'metrics': {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'r2': float(r2),
                'mape': float(mape)
            },
            'test_predictions': y_pred_original,
            'test_actual': y_test_original
        }
        
    def train(self, df, parameter, method='random_forest'):
        """Train forecasting model for a specific parameter"""
        print(f"\n🏋️ Training forecasting model for: {parameter}")
        print("-" * 40)
        
        if method == 'random_forest':
            result = self.train_random_forest(df, parameter)
        else:
            print(f"❌ Unknown method: {method}")
            return None
            
        if result:
            self.models[parameter] = result
            self.parameter_columns.append(parameter)
            
        return result
        
    def train_all_parameters(self, df, parameters=None, min_samples=50):
        """Train models for all available parameters"""
        if parameters is None:
            # Auto-detect numeric parameters
            parameters = df.select_dtypes(include=[np.number]).columns.tolist()
            parameters = [p for p in parameters if p not in ['Sample_ID', 'index', 'sample_num', 'industry']]
            
        results = {}
        for param in parameters:
            if param in df.columns:
                non_nan_count = df[param].notna().sum()
                if non_nan_count < min_samples:
                    print(f"⚠️ Skipping {param}: Only {non_nan_count} samples (need {min_samples}+)")
                    continue
                result = self.train(df, param, method='random_forest')
                if result:
                    results[param] = result
                    
        print(f"\n✅ Successfully trained {len(results)} forecasting models")
        return results
        
    def predict(self, parameter, recent_values):
        """Predict future values for a parameter"""
        if parameter not in self.models:
            raise ValueError(f"No model trained for {parameter}")
            
        model_data = self.models[parameter]
        pipeline = model_data['pipeline']
        scaler = model_data['scaler']
        
        # Scale recent values
        scaled_values = scaler.transform(np.array(recent_values).reshape(-1, 1)).flatten()
        
        # Reshape for prediction
        X_pred = np.array(scaled_values).reshape(1, -1)
        
        # Predict scaled values
        scaled_pred = pipeline.predict(X_pred)[0]
        
        # Inverse transform
        predictions = scaler.inverse_transform(scaled_pred.reshape(-1, 1)).flatten()
        
        return predictions.tolist()
        
    def save_model(self, path):
        """Save all trained models and configuration"""
        model_data = {
            'models': self.models,
            'parameter_columns': self.parameter_columns,
            'forecast_horizon': self.forecast_horizon,
            'lookback_window': self.lookback_window,
            'model_type': self.model_type
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, path)
        print(f"✅ Forecasting models saved to: {path}")
        
    def load_model(self, path):
        """Load trained models"""
        model_data = joblib.load(path)
        self.models = model_data.get('models')
        self.parameter_columns = model_data.get('parameter_columns')
        self.forecast_horizon = model_data.get('forecast_horizon')
        self.lookback_window = model_data.get('lookback_window')
        self.model_type = model_data.get('model_type')
        print(f"✅ Forecasting models loaded from: {path}")

def train_forecasting_models():
    """Main function to train forecasting models on all industries"""
    print("\n" + "="*70)
    print("🔮 FORECASTING MODEL TRAINING")
    print("="*70)
    
    loader = CSVLoader()
    all_industries = get_all_industry_ids()
    all_data = []
    
    for industry_id in all_industries:
        df = loader.load_industry_data(industry_id)
        if df is not None and len(df) > 0:
            # Ensure numeric columns are properly typed
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['industry'] = industry_id
            all_data.append(df)
            
    if not all_data:
        print("❌ No data loaded.")
        return None
        
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Combined dataset: {len(combined_df)} samples")
    
    # Sort by sample number if available
    if 'Sample_ID' in combined_df.columns:
        combined_df['sample_num'] = combined_df['Sample_ID'].str.extract('(\d+)').astype(float)
        combined_df = combined_df.sort_values('sample_num').dropna(subset=['sample_num'])
        
    parameters_to_forecast = [
        'BOD (mg/L)', 'COD (mg/L)', 'TSS (mg/L)', 
        'TDS (mg/L)', 'pH', 'Oil & Grease (mg/L)',
        'Ammonia (mg/L)', 'Temperature (°C)'
    ]
    
    available_params = []
    for param in parameters_to_forecast:
        if param in combined_df.columns:
            non_null_count = combined_df[param].notna().sum()
            if non_null_count >= 100:
                available_params.append(param)
                
    forecaster = WastewaterForecaster(
        forecast_horizon=3,
        lookback_window=5
    )
    
    results = forecaster.train_all_parameters(combined_df, available_params, min_samples=100)
    
    model_path = Path(__file__).parent.parent / "models" / "forecasting_models.pkl"
    forecaster.save_model(model_path)
    
    return forecaster

if __name__ == "__main__":
    train_forecasting_models()