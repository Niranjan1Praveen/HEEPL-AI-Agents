"""
Analysis Agent - Centralized Model Management
---------------------------------------------
This agent orchestrates all AI models for wastewater analysis:
- Anomaly Detection: Identifies unusual patterns
- Classification: Categorizes sample severity
- Forecasting: Predicts future parameter values
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.csv_loader import CSVLoader
from config.thresholds import WASTEWATER_THRESHOLDS, check_parameter_violations

class ModelType(Enum):
    """Available model types"""
    ANOMALY = "anomaly"
    CLASSIFICATION = "classification"
    FORECASTING = "forecasting"

@dataclass
class AnalysisResult:
    """Standardized result format from analysis"""
    sample_id: str
    industry_id: str
    anomaly_score: Optional[float]
    is_anomaly: Optional[bool]
    predicted_class: Optional[str]
    class_confidence: Optional[float]
    forecast: Optional[List[float]]
    violations: List[Dict]
    raw_data: Dict[str, float]
    
    @staticmethod
    def _safe(val):
        """Convert numpy scalars to JSON-serializable Python natives."""
        if isinstance(val, (np.bool_,)):
            return bool(val)
        if isinstance(val, np.floating):
            return float(val)
        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, list):
            return [AnalysisResult._safe(v) for v in val]
        if isinstance(val, dict):
            return {k: AnalysisResult._safe(v) for k, v in val.items()}
        return val

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'sample_id': self.sample_id,
            'industry_id': self.industry_id,
            'anomaly_score': self._safe(self.anomaly_score),
            'is_anomaly': self._safe(self.is_anomaly),
            'predicted_class': self.predicted_class,
            'class_confidence': self._safe(self.class_confidence),
            'forecast': self._safe(self.forecast),
            'violations': self._safe(self.violations),
            'raw_data': self._safe(self.raw_data)
        }

class ModelRegistry:
    """
    Registry for managing multiple models
    Allows easy swapping and loading of different model implementations
    """
    
    def __init__(self, models_dir: Path = None):
        self.models_dir = models_dir or Path(__file__).parent.parent / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self._models = {
            ModelType.ANOMALY: None,
            ModelType.CLASSIFICATION: None,
            ModelType.FORECASTING: None
        }
        
        self.model_configs = {
            ModelType.ANOMALY: {
                'file': 'anomaly_model.pkl',
                'fallback_enabled': True
            },
            ModelType.CLASSIFICATION: {
                'file': 'classification_model.pkl',
                'fallback_enabled': True
            },
            ModelType.FORECASTING: {
                'file': 'forecasting_models.pkl',
                'fallback_enabled': True
            }
        }
    
    def _load_model(self, model_type: ModelType):
        """Lazy load a model when needed"""
        if self._models[model_type] is not None:
            return self._models[model_type]
        
        config = self.model_configs[model_type]
        model_path = self.models_dir / config['file']
        
        if not model_path.exists():
            print(f"⚠️ Model not found: {model_path}")
            if config['fallback_enabled']:
                print(f"   Using fallback for {model_type.value}")
                return None
            else:
                raise FileNotFoundError(f"Model {model_type.value} not found at {model_path}")
        
        try:
            import joblib
            model_data = joblib.load(model_path)
            self._models[model_type] = model_data
            print(f"✅ Loaded {model_type.value} model from {model_path}")
            return model_data
        except Exception as e:
            print(f"❌ Failed to load {model_type.value} model: {e}")
            if config['fallback_enabled']:
                return None
            raise
    
    def get_anomaly_model(self):
        """Get anomaly detection model"""
        return self._load_model(ModelType.ANOMALY)
    
    def get_classification_model(self):
        """Get classification model"""
        return self._load_model(ModelType.CLASSIFICATION)
    
    def get_forecasting_model(self):
        """Get forecasting model"""
        return self._load_model(ModelType.FORECASTING)

class AnalysisAgent:
    """
    Main Analysis Agent that orchestrates all models
    Provides a unified interface for all analysis operations
    """
    
    def __init__(self, use_models: bool = True):
        """
        Initialize the Analysis Agent
        
        Args:
            use_models: If False, only use rule-based analysis (no ML models)
        """
        self.use_models = use_models
        self.registry = ModelRegistry() if use_models else None
        self.loader = CSVLoader()
        self.current_industry: Optional[str] = None
        self.current_data: Optional[pd.DataFrame] = None
    
    def load_industry_data(self, industry_id: str) -> bool:
        """Load data for a specific industry"""
        df = self.loader.load_industry_data(industry_id)
        if df is not None:
            self.current_industry = industry_id
            self.current_data = df
            return True
        return False
    
    def analyze_sample(self, sample: Dict[str, Any], industry_id: str = None) -> AnalysisResult:
        """
        Analyze a single sample using all available models and rules
        """
        industry = industry_id or self.current_industry or "unknown"
        sample_id = sample.get('Sample_ID', 'unknown')
        
        # Check violations using centralized config helper
        violations = check_parameter_violations(sample)
        
        # Run anomaly detection
        anomaly_score, is_anomaly = self._detect_anomaly(sample)
        
        # Run classification
        predicted_class, class_confidence = self._classify_sample(sample)
        
        # Run forecasting (if forecaster trained and BOD available)
        forecast = self._forecast_parameter(sample, 'BOD (mg/L)')
        
        return AnalysisResult(
            sample_id=sample_id,
            industry_id=industry,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            predicted_class=predicted_class,
            class_confidence=class_confidence,
            forecast=forecast,
            violations=violations,
            raw_data=sample
        )
    
    def analyze_dataframe(self, df: pd.DataFrame, limit: int = None) -> List[AnalysisResult]:
        """Analyze multiple samples from a DataFrame"""
        results = []
        samples = df.head(limit) if limit else df
        
        for idx, row in samples.iterrows():
            result = self.analyze_sample(row.to_dict())
            results.append(result)
        
        return results
    
    def _detect_anomaly(self, sample: Dict) -> tuple:
        """Detect if sample is anomalous using pipeline or rules fallback"""
        if not self.use_models or not self.registry:
            return self._fallback_anomaly_detection(sample)
        
        model_data = self.registry.get_anomaly_model()
        if model_data is None:
            return self._fallback_anomaly_detection(sample)
        
        try:
            pipeline = model_data.get('pipeline')
            feature_columns = model_data.get('feature_columns', [])
            
            # Prepare features in exact order
            features = []
            for col in feature_columns:
                val = sample.get(col, np.nan)
                features.append(float(val) if pd.notna(val) else np.nan)
            
            X = np.array(features).reshape(1, -1)
            
            if pipeline is not None:
                # Use unified pipeline
                prediction = pipeline.predict(X)[0]
                # Extract preprocessed intermediate values to compute anomaly score
                scaler = pipeline.named_steps['scaler']
                imputer = pipeline.named_steps['imputer']
                forest = pipeline.named_steps['model']
                X_trans = scaler.transform(imputer.transform(X))
                score = forest.score_samples(X_trans)[0]
                return float(score), bool(prediction == -1)
            else:
                # Fallback to old format if pipeline is missing
                model = model_data.get('model')
                scaler = model_data.get('scaler')
                imputer = model_data.get('imputer')
                
                # Preprocess manually
                if imputer:
                    X = imputer.transform(X)
                if scaler:
                    X = scaler.transform(X)
                    
                score = model.score_samples(X)[0]
                prediction = model.predict(X)[0]
                return float(score), bool(prediction == -1)
                
        except Exception as e:
            print(f"⚠️ Anomaly detection model prediction failed: {e}, using fallback")
            return self._fallback_anomaly_detection(sample)
            
    def _fallback_anomaly_detection(self, sample: Dict) -> tuple:
        """Rule-based anomaly detection when model unavailable"""
        violation_count = len(check_parameter_violations(sample))
        if violation_count >= 2:
            return -0.7, True
        elif violation_count >= 1:
            return -0.4, True
        else:
            return 0.2, False
            
    def _classify_sample(self, sample: Dict) -> tuple:
        """Classify sample using pipeline or rules fallback"""
        if not self.use_models or not self.registry:
            return self._fallback_classification(sample)
        
        model_data = self.registry.get_classification_model()
        if model_data is None:
            return self._fallback_classification(sample)
            
        try:
            pipeline = model_data.get('pipeline')
            feature_columns = model_data.get('feature_columns', [])
            label_encoder = model_data.get('label_encoder')
            classes = model_data.get('classes_', ['Normal', 'Warning', 'Critical'])
            
            # Prepare features
            features = []
            for col in feature_columns:
                val = sample.get(col, np.nan)
                features.append(float(val) if pd.notna(val) else np.nan)
                
            X = np.array(features).reshape(1, -1)
            
            if pipeline is not None:
                # Use unified pipeline
                prediction = pipeline.predict(X)[0]
                probabilities = pipeline.predict_proba(X)[0]
            else:
                model = model_data.get('model')
                scaler = model_data.get('scaler')
                imputer = model_data.get('imputer')
                
                if imputer:
                    X = imputer.transform(X)
                if scaler:
                    X = scaler.transform(X)
                    
                prediction = model.predict(X)[0]
                probabilities = model.predict_proba(X)[0]
                
            if label_encoder:
                label = label_encoder.inverse_transform([prediction])[0]
            else:
                label = classes[prediction] if prediction < len(classes) else "Unknown"
                
            confidence = max(probabilities) if len(probabilities) > 0 else 0.0
            return str(label), float(confidence)
            
        except Exception as e:
            print(f"⚠️ Classification model prediction failed: {e}, using fallback")
            return self._fallback_classification(sample)
            
    def _fallback_classification(self, sample: Dict) -> tuple:
        """Rule-based classification when model unavailable"""
        violations = check_parameter_violations(sample)
        critical_count = sum(1 for v in violations if v['severity'] == 'critical')
        warning_count = sum(1 for v in violations if v['severity'] == 'warning')
        
        if critical_count >= 1:
            return "Critical", 0.8
        elif warning_count >= 1:
            return "Warning", 0.6
        else:
            return "Normal", 0.9
            
    def _forecast_parameter(self, sample: Dict, parameter: str) -> Optional[List[float]]:
        """Forecast future values using the pipeline-based forecasting models"""
        if not self.use_models or not self.registry:
            return None
            
        model_data = self.registry.get_forecasting_model()
        if model_data is None:
            return None
            
        try:
            models_dict = model_data.get('models', {})
            if parameter not in models_dict:
                return None
                
            param_model = models_dict[parameter]
            pipeline = param_model.get('pipeline')
            scaler = param_model.get('scaler')
            lookback_window = model_data.get('lookback_window', 5)
            
            # Since forecasting requires historical values, if historical data is not available,
            # we simulate lookback values by slightly varying the current sample value
            current_value = sample.get(parameter)
            if current_value is None or pd.isna(current_value):
                return None
                
            # Simulate historical sequence
            simulated_history = [float(current_value) * (0.95 + 0.01 * i) for i in range(lookback_window)]
            
            # Scale history
            scaled_history = scaler.transform(np.array(simulated_history).reshape(-1, 1)).flatten()
            
            # Predict
            X_pred = scaled_history.reshape(1, -1)
            scaled_pred = pipeline.predict(X_pred)[0]
            
            # Inverse scale
            predictions = scaler.inverse_transform(scaled_pred.reshape(-1, 1)).flatten()
            return [float(p) for p in predictions]
            
        except Exception as e:
            print(f"⚠️ Forecasting prediction failed: {e}")
            return None
            
    def get_summary_stats(self, df: pd.DataFrame = None) -> Dict:
        """Get summary statistics for current data"""
        data = df if df is not None else self.current_data
        if data is None:
            return {"error": "No data loaded"}
            
        summary = {
            "total_samples": len(data),
            "parameters": {},
            "status_distribution": {}
        }
        
        for col in ['BOD (mg/L)', 'COD (mg/L)', 'TSS (mg/L)', 'TDS (mg/L)', 'pH']:
            if col in data.columns:
                series = data[col].dropna()
                if len(series) > 0:
                    summary["parameters"][col] = {
                        "mean": round(float(series.mean()), 2),
                        "median": round(float(series.median()), 2),
                        "min": round(float(series.min()), 2),
                        "max": round(float(series.max()), 2),
                        "std": round(float(series.std()), 2)
                    }
                    
        if 'Status' in data.columns:
            summary["status_distribution"] = data['Status'].value_counts().to_dict()
            
        return summary

if __name__ == "__main__":
    agent = AnalysisAgent(use_models=True)
    test_sample = {
        'Sample_ID': 'TEST_001',
        'BOD (mg/L)': 350.0,
        'COD (mg/L)': 800.0,
        'TSS (mg/L)': 120.0,
        'TDS (mg/L)': 2500.0,
        'pH': 7.2
    }
    res = agent.analyze_sample(test_sample)
    print("Analysis agent test completed successfully:", res.to_dict())