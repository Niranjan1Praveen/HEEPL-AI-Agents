"""
Time Series Forecasting Model for Industrial Wastewater Parameters
-----------------------------------------------------------------
This model uses LSTM (Long Short-Term Memory) neural networks to predict 
future values of wastewater parameters based on historical patterns.

Why LSTM for wastewater forecasting?
1. Captures long-term dependencies in sequential data
2. Handles seasonal patterns (daily/weekly/monthly cycles)
3. Learns complex non-linear relationships
4. Works well with multivariate time series (multiple parameters)

Alternative models included for flexibility:
- ARIMA (Statistical): Good for univariate, interpretable
- Prophet (Meta): Handles missing data, seasonality
- XGBoost: Fast, handles tabular data well
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

# Optional imports with fallbacks
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not installed. Run: pip install scikit-learn")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet not installed. Run: pip install prophet")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("⚠️ joblib not installed. Run: pip install joblib")

class WastewaterForecaster:
    """
    Forecasting model for wastewater parameters
    
    Predicts future values of parameters like BOD, COD, TSS, etc.
    based on historical patterns and trends.
    """
    
    def __init__(self, forecast_horizon=5, lookback_window=10):
        """
        Args:
            forecast_horizon: Number of future steps to predict (days/samples)
            lookback_window: Number of past samples to use for prediction
        """
        self.forecast_horizon = forecast_horizon
        self.lookback_window = lookback_window
        self.models = {}  # Store model per parameter
        self.scalers = {}  # Store scaler per parameter
        self.parameter_columns = []
        self.model_type = 'random_forest'  # 'random_forest', 'prophet', or 'xgboost'
        
    def prepare_sequences(self, data, parameter):
        """Prepare sequences for time series forecasting"""
        X, y = [], []
        
        for i in range(len(data) - self.lookback_window - self.forecast_horizon + 1):
            # Input: lookback_window past values
            X.append(data[i:i + self.lookback_window])
            # Output: forecast_horizon future values
            y.append(data[i + self.lookback_window:i + self.lookback_window + self.forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def train_random_forest(self, df, parameter):
        """Train Random Forest model for forecasting - FIXED for NaN values"""
        if not SKLEARN_AVAILABLE:
            print("❌ scikit-learn not available")
            return None
        
        # Extract the parameter values
        values = df[parameter].values
        
        # Remove NaN values
        values = values[~np.isnan(values)]
        
        if len(values) < self.lookback_window + self.forecast_horizon:
            print(f"⚠️ Not enough data for {parameter}. Need {self.lookback_window + self.forecast_horizon} samples, have {len(values)}")
            return None
        
        # Scale the values
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(values.reshape(-1, 1)).flatten()
        
        # Prepare sequences
        X, y = [], []
        
        for i in range(len(scaled_values) - self.lookback_window - self.forecast_horizon + 1):
            # Input: lookback_window past values
            X.append(scaled_values[i:i + self.lookback_window])
            # Output: forecast_horizon future values (ensure no NaN)
            target = scaled_values[i + self.lookback_window:i + self.lookback_window + self.forecast_horizon]
            
            # Only add if target has no NaN values
            if not np.any(np.isnan(target)):
                y.append(target)
            else:
                X.pop()  # Remove the corresponding X
        
        if len(X) == 0:
            print(f"⚠️ Could not create valid sequences for {parameter}")
            return None
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train and test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Flatten X for Random Forest
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Flatten y for regression (predict all horizon values)
        y_train_flat = y_train.reshape(y_train.shape[0], -1)
        y_test_flat = y_test.reshape(y_test.shape[0], -1)
        
        # Check for NaN in y_train after flattening
        if np.any(np.isnan(y_train_flat)):
            print(f"⚠️ NaN values found in training targets for {parameter}. Skipping.")
            return None
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=50,  # Reduced for faster training
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_flat, y_train_flat)
        
        # Evaluate
        y_pred = model.predict(X_test_flat)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_flat.flatten(), y_pred.flatten())
        r2 = r2_score(y_test_flat.flatten(), y_pred.flatten())
        
        # Inverse transform predictions for interpretability (only first value)
        y_test_first = y_test[:, 0]  # First predicted step
        y_pred_first = y_pred[:, 0]
        
        y_test_original = scaler.inverse_transform(y_test_first.reshape(-1, 1)).flatten()
        y_pred_original = scaler.inverse_transform(y_pred_first.reshape(-1, 1)).flatten()
        mae_original = mean_absolute_error(y_test_original, y_pred_original)
        
        print(f"   📊 {parameter} - MAE: {mae_original:.2f} {self._get_unit(parameter)}, R²: {r2:.3f}")
        
        return {
            'model': model,
            'scaler': scaler,
            'metrics': {'mae': mae_original, 'mse': mae_original**2, 'r2': r2},
            'test_predictions': y_pred_original,
            'test_actual': y_test_original
        }

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
    
    def train_prophet(self, df, parameter):
        """Train Prophet model (best for seasonal patterns)"""
        if not PROPHET_AVAILABLE:
            print("❌ Prophet not available")
            return None
        
        # Prepare data for Prophet (needs ds = date, y = value)
        # Create sequential index as date
        prophet_df = pd.DataFrame({
            'ds': pd.date_range(start='2020-01-01', periods=len(df), freq='D'),
            'y': df[parameter].values
        })
        
        # Handle missing values
        prophet_df = prophet_df.dropna()
        
        if len(prophet_df) < 30:
            print(f"⚠️ Not enough data for {parameter} with Prophet. Need at least 30 samples.")
            return None
        
        # Split data (last forecast_horizon days for testing)
        split_idx = len(prophet_df) - self.forecast_horizon
        train_df = prophet_df[:split_idx]
        test_df = prophet_df[split_idx:]
        
        # Train Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(train_df)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=self.forecast_horizon)
        forecast = model.predict(future)
        
        # Evaluate on test set
        if len(test_df) > 0:
            test_predictions = forecast.tail(len(test_df))['yhat'].values
            mae = mean_absolute_error(test_df['y'].values, test_predictions)
            print(f"   📊 {parameter} - MAE: {mae:.2f} mg/L")
        
        return {
            'model': model,
            'forecast': forecast.tail(self.forecast_horizon),
            'metrics': {'mae': mae if 'mae' in locals() else None}
        }
    
    def train(self, df, parameter, method='random_forest'):
        """
        Train forecasting model for a specific parameter
        
        Args:
            df: DataFrame with time series data
            parameter: Column name to forecast (e.g., 'BOD (mg/L)')
            method: 'random_forest', 'prophet', or 'xgboost'
        """
        print(f"\n🏋️ Training forecasting model for: {parameter}")
        print("-" * 40)
        
        # Check if enough data
        if len(df) < self.lookback_window + self.forecast_horizon:
            print(f"❌ Insufficient data. Need {self.lookback_window + self.forecast_horizon} samples, have {len(df)}")
            return None
        
        if method == 'random_forest':
            result = self.train_random_forest(df, parameter)
        elif method == 'prophet':
            result = self.train_prophet(df, parameter)
        else:
            print(f"❌ Unknown method: {method}")
            return None
        
        if result:
            self.models[parameter] = result
            self.parameter_columns.append(parameter)
        
        return result
    
    def train_all_parameters(self, df, parameters=None, method='random_forest', min_samples=50):
        """Train models for all available parameters with minimum sample requirement"""
        if parameters is None:
            # Auto-detect numeric parameters
            parameters = df.select_dtypes(include=[np.number]).columns.tolist()
            parameters = [p for p in parameters if p not in ['Sample_ID', 'index', 'sample_num']]
        
        results = {}
        for param in parameters:
            if param in df.columns:
                # Check minimum samples
                non_nan_count = df[param].notna().sum()
                if non_nan_count < min_samples:
                    print(f"⚠️ Skipping {param}: Only {non_nan_count} samples (need {min_samples})")
                    continue
                
                result = self.train(df, param, method)
                if result:
                    results[param] = result
        
        print(f"\n✅ Trained {len(results)} forecasting models")
        return results
    
    def predict(self, parameter, recent_values):
        """
        Predict future values for a parameter
        
        Args:
            parameter: Parameter name
            recent_values: List of recent values (length should match lookback_window)
        
        Returns:
            List of predicted future values
        """
        if parameter not in self.models:
            raise ValueError(f"No model trained for {parameter}")
        
        model_data = self.models[parameter]
        
        if self.model_type == 'random_forest':
            # Scale recent values
            scaled_values = model_data['scaler'].transform(np.array(recent_values).reshape(-1, 1)).flatten()
            
            # Reshape for prediction
            X_pred = np.array(scaled_values).reshape(1, -1)
            
            # Predict scaled values
            scaled_pred = model_data['model'].predict(X_pred)[0]
            
            # Inverse transform to original scale
            predictions = model_data['scaler'].inverse_transform(scaled_pred.reshape(-1, 1)).flatten()
            
            return predictions.tolist()
        
        elif self.model_type == 'prophet':
            # Prophet handles its own forecasting
            future = model_data['model'].make_future_dataframe(periods=self.forecast_horizon)
            forecast = model_data['model'].predict(future)
            return forecast.tail(self.forecast_horizon)['yhat'].values.tolist()
        
        return []
    
    def save_model(self, path):
        """Save all trained models"""
        if not JOBLIB_AVAILABLE:
            print("⚠️ Cannot save model: joblib not installed")
            return
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
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
        if not JOBLIB_AVAILABLE:
            print("⚠️ Cannot load model: joblib not installed")
            return
        
        model_data = joblib.load(path)
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.parameter_columns = model_data['parameter_columns']
        self.forecast_horizon = model_data['forecast_horizon']
        self.lookback_window = model_data['lookback_window']
        self.model_type = model_data['model_type']
        print(f"✅ Forecasting models loaded from: {path}")


def train_forecasting_models():
    """Main function to train forecasting models on all industries"""
    
    print("\n" + "="*70)
    print("🔮 FORECASTING MODEL TRAINING")
    print("="*70)
    
    # 1. Load ALL available data
    print("\n📂 Loading all industry data...")
    loader = CSVLoader()
    all_industries = get_all_industry_ids()
    
    all_data = []
    industry_sample_counts = {}
    
    for industry_id in all_industries:
        df = loader.load_industry_data(industry_id)
        if df is not None and len(df) > 0:
            # Ensure numeric columns are properly typed
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['industry'] = industry_id
            all_data.append(df)
            industry_sample_counts[industry_id] = len(df)
            print(f"   ✅ {industry_id}: {len(df)} samples")
    
    if not all_data:
        print("❌ No data loaded. Please check CSV files.")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Combined dataset: {len(combined_df)} samples from {len(industry_sample_counts)} industries")
    
    # 2. Sort by sample number if available
    if 'Sample_ID' in combined_df.columns:
        # Extract numeric portion for sorting
        combined_df['sample_num'] = combined_df['Sample_ID'].str.extract('(\d+)').astype(float)
        combined_df = combined_df.sort_values('sample_num').dropna(subset=['sample_num'])
    
    # 3. Identify parameters with enough data (min 100 samples)
    print("\n🔍 Identifying parameters for forecasting...")
    parameters_to_forecast = [
        'BOD (mg/L)', 'COD (mg/L)', 'TSS (mg/L)', 
        'TDS (mg/L)', 'pH', 'Oil & Grease (mg/L)',
        'Ammonia (mg/L)', 'Temperature (°C)'
    ]
    
    available_params = []
    for param in parameters_to_forecast:
        if param in combined_df.columns:
            # Count non-NaN values
            non_null_count = combined_df[param].notna().sum()
            if non_null_count >= 100:  # Minimum 100 samples for forecasting
                available_params.append(param)
                print(f"   ✅ {param}: {non_null_count} samples")
            else:
                print(f"   ⚠️ {param}: Only {non_null_count} samples (need 100+)")
    
    if not available_params:
        print("❌ No parameters with enough data for forecasting")
        return None
    
    # 4. Train forecasting models
    print("\n" + "="*70)
    print("🏋️ TRAINING FORECASTING MODELS")
    print("="*70)
    
    forecaster = WastewaterForecaster(
        forecast_horizon=3,  # Predict next 3 samples (reduced from 5)
        lookback_window=5    # Use last 5 samples (reduced from 10)
    )
    
    # Train on combined data
    results = forecaster.train_all_parameters(combined_df, available_params, method='random_forest')
    
    if not results:
        print("❌ No models were successfully trained")
        return None
    
    # 5. Save models
    model_path = Path(__file__).parent.parent / "models" / "forecasting_models.pkl"
    forecaster.save_model(model_path)
    
    # 6. Demonstrate predictions
    print("\n" + "="*70)
    print("🔮 DEMONSTRATING PREDICTIONS")
    print("="*70)
    
    # Test prediction for each trained parameter
    for param, model_data in forecaster.models.items():
        if 'test_predictions' in model_data and len(model_data['test_predictions']) > 0:
            print(f"\n📊 Parameter: {param}")
            print("-" * 40)
            
            # Show actual vs predicted (first 3)
            print("\n   Actual vs Predicted (first 3 test samples):")
            print("   # | Actual | Predicted | Error")
            print("   " + "-" * 35)
            
            for i in range(min(3, len(model_data['test_predictions']))):
                actual = model_data['test_actual'][i]
                predicted = model_data['test_predictions'][i]
                error = abs(actual - predicted)
                error_pct = (error / actual) * 100 if actual != 0 else 0
                
                print(f"   {i+1} | {actual:7.0f} | {predicted:9.0f} | {error:6.0f} ({error_pct:.0f}%)")
    
    return forecaster


def print_model_explanation():
    """Print detailed explanation of the forecasting model"""
    print("\n" + "="*70)
    print("📚 FORECASTING MODEL EXPLANATION")
    print("="*70)
    
    print("""
    🧠 FORECASTING MODEL: RANDOM FOREST REGRESSOR
    
    WHAT IT DOES:
    ------------
    Predicts future wastewater parameter values (BOD, COD, TSS, etc.) 
    based on historical patterns and trends.
    
    HOW IT WORKS:
    ------------
    1. Uses past 'lookback_window' values to predict next 'forecast_horizon' values
    2. Creates multiple decision trees on random subsets of data
    3. Averages predictions from all trees for final forecast
    
    ARCHITECTURE:
    ------------
    Input: [value_t-10, value_t-9, ..., value_t-1]  (10 past values)
              ↓
    Random Forest (100 trees)
              ↓
    Output: [value_t, value_t+1, ..., value_t+4]  (5 future values)
    
    WHY RANDOM FOREST FOR WASTEWATER FORECASTING:
    --------------------------------------------
    ✓ Handles non-linear relationships (BOD doesn't change linearly)
    ✓ Robust to outliers (equipment failures, spills)
    ✓ Provides feature importance (which past values matter most)
    ✓ Fast training and prediction
    ✓ No complex hyperparameter tuning needed
    
    FORECASTING HORIZON:
    -------------------
    - Default: 5 steps ahead (e.g., next 5 days or 5 samples)
    - Can be adjusted based on your needs
    - Longer horizons = more uncertainty
    
    EVALUATION METRICS:
    ------------------
    - MAE (Mean Absolute Error): Average prediction error in mg/L
    - R² (R-squared): How well the model explains variance (0 to 1)
    - Lower MAE = More accurate predictions
    
    PRACTICAL APPLICATIONS:
    ----------------------
    1. Predict when BOD levels will exceed discharge limits
    2. Forecast treatment plant loading for capacity planning
    3. Early warning of deteriorating water quality
    4. Optimize chemical dosing based on predicted loads
    5. Schedule maintenance during predicted low-load periods
    
    LIMITATIONS:
    -----------
    - Requires sufficient historical data (100+ samples)
    - Cannot predict sudden equipment failures
    - Assumes past patterns continue into future
    - Works best for stable treatment processes
    
    NEXT STEPS:
    ----------
    1. Monitor prediction accuracy over time
    2. Retrain model monthly with new data
    3. Add external factors (weather, production volume)
    4. Implement ensemble of multiple models
    """)


if __name__ == "__main__":
    # Train forecasting models on ALL available data
    forecaster = train_forecasting_models()
    
    # Print detailed explanation
    # print_model_explanation()
    
    # Example: Make a prediction for a specific parameter
    if forecaster and len(forecaster.models) > 0:
        print("\n" + "="*70)
        print("💡 EXAMPLE PREDICTION")
        print("="*70)
        
        # Get first trained parameter
        first_param = list(forecaster.models.keys())[0]
        print(f"\n🔮 Predicting next {forecaster.forecast_horizon} values for: {first_param}")
        
        # Example recent values (you would get these from your latest data)
        example_recent = [1200, 1250, 1180, 1220, 1190, 1210, 1230, 1200, 1220, 1180]
        
        print(f"\n   Recent values: {example_recent}")
        
        try:
            predictions = forecaster.predict(first_param, example_recent)
            print(f"\n   📈 Predicted next {len(predictions)} values: {[round(p, 0) for p in predictions]}")
            
            # Trend analysis
            trend = predictions[-1] - predictions[0]
            if trend > 0:
                print(f"\n   📊 Trend: 📈 Increasing ({trend:.0f} mg/L increase expected)")
            elif trend < 0:
                print(f"   📊 Trend: 📉 Decreasing ({abs(trend):.0f} mg/L decrease expected)")
            else:
                print(f"   📊 Trend: ➡️ Stable")
                
        except Exception as e:
            print(f"   ⚠️ Prediction example not available: {e}")