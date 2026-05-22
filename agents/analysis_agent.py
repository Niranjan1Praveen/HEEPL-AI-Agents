"""
Analysis Agent - Centralized Model Management
---------------------------------------------
This agent orchestrates all AI models for wastewater analysis:
- Anomaly Detection: Identifies unusual patterns
- Classification: Categorizes sample severity
- Forecasting: Predicts future parameter values

Design Principles:
- Model-agnostic: Easy to swap models
- Lazy loading: Models loaded only when needed
- Fallback strategies: If model fails, use rules
- Extensible: Add new models easily
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.csv_loader import CSVLoader


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
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'sample_id': self.sample_id,
            'industry_id': self.industry_id,
            'anomaly_score': self.anomaly_score,
            'is_anomaly': self.is_anomaly,
            'predicted_class': self.predicted_class,
            'class_confidence': self.class_confidence,
            'forecast': self.forecast,
            'violations': self.violations,
            'raw_data': self.raw_data
        }


class ModelRegistry:
    """
    Registry for managing multiple models
    Allows easy swapping and loading of different model implementations
    """
    
    def __init__(self, models_dir: Path = None):
        self.models_dir = models_dir or Path(__file__).parent.parent / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model instances (lazy loaded)
        self._models = {
            ModelType.ANOMALY: None,
            ModelType.CLASSIFICATION: None,
            ModelType.FORECASTING: None
        }
        
        # Model configurations
        self.model_configs = {
            ModelType.ANOMALY: {
                'file': 'anomaly_model.pkl',
                'class': 'IsolationForest',
                'fallback_enabled': True
            },
            ModelType.CLASSIFICATION: {
                'file': 'classification_model.pkl',
                'class': 'RandomForestClassifier',
                'fallback_enabled': True
            },
            ModelType.FORECASTING: {
                'file': 'forecasting_models.pkl',
                'class': 'RandomForestRegressor',
                'fallback_enabled': False  # Forecasting requires model
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
    
    def set_model(self, model_type: ModelType, model_data: Any):
        """Manually set a model (for testing or custom models)"""
        self._models[model_type] = model_data
        print(f"✅ Manually set {model_type.value} model")


class AnalysisAgent:
    """
    Main Analysis Agent that orchestrates all models
    Provides a unified interface for all analysis operations
    """
    
    # Regulatory thresholds for fallback (when models unavailable)
    FALLBACK_THRESHOLDS = {
        'BOD (mg/L)': {'critical': 500, 'warning': 100},
        'COD (mg/L)': {'critical': 1000, 'warning': 500},
        'TSS (mg/L)': {'critical': 300, 'warning': 150},
        'TDS (mg/L)': {'critical': 5000, 'warning': 3000},
        'pH': {'critical_low': 4, 'critical_high': 10, 'warning_low': 5.5, 'warning_high': 9},
        'Oil & Grease (mg/L)': {'critical': 50, 'warning': 20},
        'Ammonia (mg/L)': {'critical': 100, 'warning': 50},
    }
    
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
        Analyze a single sample using all available models
        
        Args:
            sample: Dictionary of parameter values
            industry_id: Industry ID (optional, uses current if not provided)
        
        Returns:
            AnalysisResult with all analysis outputs
        """
        industry = industry_id or self.current_industry or "unknown"
        sample_id = sample.get('Sample_ID', 'unknown')
        
        # Get violations using rules
        violations = self._check_violations(sample)
        
        # Run anomaly detection
        anomaly_score, is_anomaly = self._detect_anomaly(sample)
        
        # Run classification
        predicted_class, class_confidence = self._classify_sample(sample)
        
        # Run forecasting (optional, requires historical data)
        forecast = self._forecast_parameter(sample.get('BOD (mg/L)'))
        
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
        """
        Analyze multiple samples from a DataFrame
        
        Args:
            df: DataFrame with samples
            limit: Maximum number of samples to analyze
        
        Returns:
            List of AnalysisResult objects
        """
        results = []
        samples = df.head(limit) if limit else df
        
        for idx, row in samples.iterrows():
            result = self.analyze_sample(row.to_dict())
            results.append(result)
        
        return results
    
    def _detect_anomaly(self, sample: Dict) -> tuple:
        """Detect if sample is anomalous using model or rules"""
        if not self.use_models or not self.registry:
            return self._fallback_anomaly_detection(sample)
        
        model_data = self.registry.get_anomaly_model()
        if model_data is None:
            return self._fallback_anomaly_detection(sample)
        
        try:
            # Extract features in the order model expects
            model = model_data.get('model')
            scaler = model_data.get('scaler')
            feature_columns = model_data.get('feature_columns', [])
            
            # Prepare features
            features = []
            for col in feature_columns:
                if col in sample and pd.notna(sample[col]):
                    features.append(float(sample[col]))
                else:
                    features.append(0.0)
            
            # Scale and predict
            X = np.array(features).reshape(1, -1)
            if scaler:
                X = scaler.transform(X)
            
            score = model.score_samples(X)[0]
            prediction = model.predict(X)[0]
            
            return float(score), prediction == -1
            
        except Exception as e:
            print(f"⚠️ Anomaly detection failed: {e}, using fallback")
            return self._fallback_anomaly_detection(sample)
    
    def _fallback_anomaly_detection(self, sample: Dict) -> tuple:
        """Rule-based anomaly detection when model unavailable"""
        violation_count = len(self._check_violations(sample))
        
        # Simple rule: more violations = more anomalous
        if violation_count >= 2:
            return -0.7, True  # Strong anomaly
        elif violation_count >= 1:
            return -0.4, True  # Weak anomaly
        else:
            return 0.2, False  # Normal
    
    def _classify_sample(self, sample: Dict) -> tuple:
        """Classify sample using model or rules"""
        if not self.use_models or not self.registry:
            return self._fallback_classification(sample)
        
        model_data = self.registry.get_classification_model()
        if model_data is None:
            return self._fallback_classification(sample)
        
        try:
            model = model_data.get('model')
            scaler = model_data.get('scaler')
            imputer = model_data.get('imputer')
            label_encoder = model_data.get('label_encoder')
            feature_columns = model_data.get('feature_columns', [])
            classes = model_data.get('classes_', ['Normal', 'Warning', 'Critical'])
            
            # Prepare features
            features = []
            for col in feature_columns:
                if col in sample and pd.notna(sample[col]):
                    features.append(float(sample[col]))
                else:
                    features.append(0.0)
            
            # Transform
            X = np.array(features).reshape(1, -1)
            if imputer:
                X = imputer.transform(X)
            if scaler:
                X = scaler.transform(X)
            
            # Predict
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            # Decode prediction
            if label_encoder:
                label = label_encoder.inverse_transform([prediction])[0]
            else:
                label = classes[prediction] if prediction < len(classes) else "Unknown"
            
            confidence = max(probabilities) if len(probabilities) > 0 else 0
            
            return label, float(confidence)
            
        except Exception as e:
            print(f"⚠️ Classification failed: {e}, using fallback")
            return self._fallback_classification(sample)
    
    def _fallback_classification(self, sample: Dict) -> tuple:
        """Rule-based classification when model unavailable"""
        violations = self._check_violations(sample)
        
        critical_count = sum(1 for v in violations if v['severity'] == 'critical')
        warning_count = sum(1 for v in violations if v['severity'] == 'warning')
        
        if critical_count >= 1:
            return "Critical", 0.8
        elif warning_count >= 1:
            return "Warning", 0.6
        else:
            return "Normal", 0.9
    
    def _forecast_parameter(self, current_value: float = None) -> Optional[List[float]]:
        """
        Forecast future values (simplified - would use actual model)
        For full forecasting, use the dedicated forecasting model
        """
        if not self.use_models or not self.registry:
            return None
        
        model_data = self.registry.get_forecasting_model()
        if model_data is None:
            return None
        
        # Simplified: return None for now
        # Full implementation would use the forecasting model
        return None
    
    def _check_violations(self, sample: Dict) -> List[Dict]:
        """Check parameter violations against thresholds"""
        violations = []
        
        for param, thresholds in self.FALLBACK_THRESHOLDS.items():
            if param not in sample:
                continue
            
            value = sample[param]
            if pd.isna(value):
                continue
            
            if param == 'pH':
                if value <= thresholds.get('critical_low', 0) or value >= thresholds.get('critical_high', 14):
                    violations.append({
                        'parameter': param,
                        'value': value,
                        'severity': 'critical',
                        'message': f"pH is {value} - outside safe range ({thresholds.get('critical_low')}-{thresholds.get('critical_high')})"
                    })
                elif value <= thresholds.get('warning_low', 0) or value >= thresholds.get('warning_high', 14):
                    violations.append({
                        'parameter': param,
                        'value': value,
                        'severity': 'warning',
                        'message': f"pH is {value} - approaching limit"
                    })
            else:
                critical = thresholds.get('critical', float('inf'))
                warning = thresholds.get('warning', float('inf'))
                
                if value >= critical:
                    violations.append({
                        'parameter': param,
                        'value': value,
                        'severity': 'critical',
                        'message': f"{param} is {value} mg/L - exceeds critical limit ({critical})"
                    })
                elif value >= warning:
                    violations.append({
                        'parameter': param,
                        'value': value,
                        'severity': 'warning',
                        'message': f"{param} is {value} mg/L - exceeds warning limit ({warning})"
                    })
        
        return violations
    
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
        
        # Parameter statistics
        for col in ['BOD (mg/L)', 'COD (mg/L)', 'TSS (mg/L)', 'TDS (mg/L)', 'pH']:
            if col in data.columns:
                series = data[col].dropna()
                if len(series) > 0:
                    summary["parameters"][col] = {
                        "mean": round(series.mean(), 2),
                        "median": round(series.median(), 2),
                        "min": round(series.min(), 2),
                        "max": round(series.max(), 2),
                        "std": round(series.std(), 2)
                    }
        
        # Status distribution
        if 'Status' in data.columns:
            summary["status_distribution"] = data['Status'].value_counts().to_dict()
        
        return summary


# Quick test
if __name__ == "__main__":
    print("="*60)
    print("🔬 ANALYSIS AGENT TEST")
    print("="*60)
    
    # Initialize agent
    agent = AnalysisAgent(use_models=True)
    
    # Test sample
    test_sample = {
        'Sample_ID': 'TEST_001',
        'BOD (mg/L)': 350000,
        'COD (mg/L)': 800000,
        'TSS (mg/L)': 12000,
        'TDS (mg/L)': 2500000,
        'pH': 7.2,
        'Dairy (mg/L)': 25,
    }
    
    print("\n📊 Testing with sample:")
    for k, v in test_sample.items():
        print(f"   {k}: {v}")
    
    # Analyze
    result = agent.analyze_sample(test_sample, industry_id="test_industry")
    
    print(f"\n📈 Analysis Results:")
    print(f"   Sample ID: {result.sample_id}")
    print(f"   Is Anomaly: {result.is_anomaly} (score: {result.anomaly_score:.2f})")
    print(f"   Predicted Class: {result.predicted_class} (confidence: {result.class_confidence:.2%})")
    print(f"   Violations: {len(result.violations)}")
    
    for v in result.violations:
        print(f"      • {v['message']}")
    
    print("\n✅ Analysis Agent ready!")