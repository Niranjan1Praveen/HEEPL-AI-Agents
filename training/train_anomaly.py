"""
Anomaly Detection Model for Industrial Wastewater Data
------------------------------------------------------
This model uses Isolation Forest algorithm to detect anomalies in wastewater parameters.
Isolation Forest works by randomly isolating observations - anomalies are easier to isolate
because they are few and different from normal data points.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path
import sys
import os

# Add server to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.mappings import get_all_industry_ids
from utils.csv_loader import CSVLoader

class AnomalyDetector:
    """
    Anomaly Detection using Isolation Forest Algorithm wrapped in an sklearn Pipeline.
    """
    
    def __init__(self, contamination=0.1, random_state=42):
        """
        Args:
            contamination: Expected proportion of anomalies (0.1 = 10%)
            random_state: For reproducible results
        """
        self.contamination = contamination
        self.random_state = random_state
        self.feature_columns = None
        self.pipeline = None
        
        # Estimators for compatibility
        self.model = None
        self.scaler = None
        self.imputer = None
        
    def train(self, df):
        """Train the anomaly detection pipeline"""
        print("\n" + "="*60)
        print("🏋️ TRAINING ANOMALY DETECTION PIPELINE")
        print("="*60)
        
        # Define all possible features
        all_features = [
            'BOD (mg/L)', 'COD (mg/L)', 'TSS (mg/L)', 
            'TDS (mg/L)', 'pH', 'Oil & Grease (mg/L)',
            'Ammonia (mg/L)', 'Conductivity (uS/cm)', 'Temperature (°C)'
        ]
        
        # Keep only columns that exist in the dataframe
        available_features = [col for col in all_features if col in df.columns]
        self.feature_columns = available_features
        
        if len(available_features) < 3:
            print(f"⚠️ Warning: Only {len(available_features)} features available.")
        else:
            print(f"📋 Using features: {available_features}")
            
        X = df[available_features].values
        
        # Build Pipeline
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=100,
                max_samples='auto',
                bootstrap=False
            ))
        ])
        
        print(f"\n📊 Training Data Shape: {X.shape}")
        print(f"🎯 Expected Contamination Rate: {self.contamination * 100}%")
        
        # Fit the pipeline
        self.pipeline.fit(X)
        
        # Keep references for backward compatibility
        self.model = self.pipeline.named_steps['model']
        self.scaler = self.pipeline.named_steps['scaler']
        self.imputer = self.pipeline.named_steps['imputer']
        
        # Evaluate on training data
        predictions, scores = self.predict(df)
        anomaly_count = sum(predictions == -1)
        anomaly_rate = anomaly_count / len(predictions)
        
        print(f"\n📈 Training Results:")
        print(f"   ✅ Normal samples: {sum(predictions == 1)} ({100 - anomaly_rate*100:.1f}%)")
        print(f"   ⚠️ Anomalies detected: {anomaly_count} ({anomaly_rate*100:.1f}%)")
        print(f"   📊 Anomaly scores range: {scores.min():.2f} to {scores.max():.2f}")
        print(f"   📊 Mean anomaly score: {scores.mean():.4f} (StdDev: {scores.std():.4f})")
        
        # Calculate training metrics: correlation of each feature with the anomaly score
        # Highly negative correlation means higher parameter value corresponds to lower score (more anomalous)
        print(f"\n🔑 Feature Impact on Anomaly Score:")
        imputed_X = self.imputer.transform(X)
        for i, col in enumerate(self.feature_columns):
            corr = np.corrcoef(imputed_X[:, i], scores)[0, 1]
            print(f"   • {col:<22}: Correlation = {corr:+.4f} ({'Anomalous at High values' if corr < 0 else 'Normal/Low impact'})")
            
        return self
    
    def predict(self, df):
        """Predict anomalies in new data using the pipeline"""
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train() or load_model() first.")
            
        X = df[self.feature_columns].values
        
        # Pipeline prediction (-1 = anomaly, 1 = normal)
        predictions = self.pipeline.predict(X)
        
        # Get anomaly scores (need to pass through preprocessing first)
        X_preprocessed = self.pipeline.named_steps['scaler'].transform(
            self.pipeline.named_steps['imputer'].transform(X)
        )
        scores = self.pipeline.named_steps['model'].score_samples(X_preprocessed)
        
        return predictions, scores
    
    def get_anomaly_details(self, df):
        """Get detailed anomaly information for each row"""
        predictions, scores = self.predict(df)
        
        results = []
        for idx, (pred, score) in enumerate(zip(predictions, scores)):
            results.append({
                'sample_id': df.iloc[idx].get('Sample_ID', f'sample_{idx}'),
                'is_anomaly': pred == -1,
                'anomaly_score': float(score),
                'confidence': float(1 - (score + 0.5)) if pred == -1 else float(score + 0.5)
            })
        
        return results
    
    def save_model(self, path):
        """Save the trained model and pipeline to disk"""
        model_data = {
            'pipeline': self.pipeline,
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_columns': self.feature_columns,
            'contamination': self.contamination
        }
        joblib.dump(model_data, path)
        print(f"✅ Model and Pipeline saved to: {path}")
    
    def load_model(self, path):
        """Load a trained model from disk"""
        model_data = joblib.load(path)
        self.pipeline = model_data.get('pipeline')
        self.model = model_data.get('model')
        self.scaler = model_data.get('scaler')
        self.imputer = model_data.get('imputer')
        self.feature_columns = model_data.get('feature_columns')
        self.contamination = model_data.get('contamination')
        
        # Reconstruct pipeline if missing
        if self.pipeline is None and self.model is not None:
            self.pipeline = Pipeline([
                ('imputer', self.imputer),
                ('scaler', self.scaler),
                ('model', self.model)
            ])
        print(f"✅ Pipeline loaded from: {path}")
    
    def explain_anomaly(self, sample_row):
        """Explain why a specific sample is anomalous"""
        if self.feature_columns is None:
            return "Model not trained yet"
        
        # Get feature values
        features = [sample_row.get(col, np.nan) for col in self.feature_columns]
        
        # Feed through pipeline components
        X = np.array(features).reshape(1, -1)
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        score = self.model.score_samples(X_scaled)[0]
        prediction = self.model.predict(X_scaled)[0]
        
        mean_values = self.scaler.mean_
        std_values = self.scaler.scale_
        
        deviations = []
        for i, col in enumerate(self.feature_columns):
            value = X_imputed[0, i]
            # Get typical range (mean ± 2 std deviations)
            actual_scaled = (value - mean_values[i]) / std_values[i]
            
            if abs(actual_scaled) > 2:
                direction = "high" if actual_scaled > 0 else "low"
                deviations.append({
                    'parameter': col,
                    'value': float(value),
                    'deviation': f"Unusually {direction} (z-score: {actual_scaled:.2f})"
                })
        
        return {
            'is_anomaly': prediction == -1,
            'anomaly_score': float(score),
            'deviations': deviations
        }

def train_and_test_model():
    """Main function to train and test the anomaly detection model"""
    print("\n" + "="*60)
    print("🔍 ANOMALY DETECTION MODEL TRAINING")
    print("="*60)
    
    # 1. Load data
    print("\n📂 Loading wastewater data...")
    loader = CSVLoader()
    industries_to_train = get_all_industry_ids()
    
    all_data = []
    for industry_id in industries_to_train:
        df = loader.load_industry_data(industry_id)
        if df is not None:
            df['industry'] = industry_id
            all_data.append(df)
            
    if not all_data:
        print("❌ No data loaded. Check your CSV file paths.")
        return
        
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Combined dataset: {len(combined_df)} samples from {len(industries_to_train)} industries")
    
    # 2. Train anomaly detection model
    detector = AnomalyDetector(contamination=0.15)
    detector.train(combined_df)
    
    # 3. Save the model
    model_path = Path(__file__).parent.parent / "models" / "anomaly_model.pkl"
    model_path.parent.mkdir(exist_ok=True)
    detector.save_model(model_path)
    
    # 4. Test on a specific industry's data (dairy)
    print("\n" + "="*60)
    print("🧪 TESTING MODEL ON NEW DATA")
    print("="*60)
    
    test_industry = "dairy"
    test_df = loader.load_industry_data(test_industry)
    
    if test_df is not None:
        print(f"\n📊 Testing on: {test_industry} ({len(test_df)} samples)")
        predictions, scores = detector.predict(test_df)
        test_df['is_anomaly'] = predictions == -1
        test_df['anomaly_score'] = scores
        
        anomaly_count = test_df['is_anomaly'].sum()
        print(f"\n📈 Test Results:")
        print(f"   ✅ Normal samples: {len(test_df) - anomaly_count}")
        print(f"   ⚠️ Anomalies detected: {anomaly_count} ({anomaly_count/len(test_df)*100:.1f}%)")
        
        anomalies = test_df[test_df['is_anomaly'] == True]
        if len(anomalies) > 0:
            print(f"\n🔍 Sample Anomalies (first 5):")
            print("-" * 80)
            for idx, row in anomalies.head(5).iterrows():
                print(f"\n   Sample: {row.get('Sample_ID', 'Unknown')}")
                print(f"   Status (original): {row.get('Status', 'N/A')}")
                print(f"   Anomaly Score: {row['anomaly_score']:.2f}")
                print(f"   Key Parameters: BOD={row.get('BOD (mg/L)', 'N/A')}, COD={row.get('COD (mg/L)', 'N/A')}, pH={row.get('pH', 'N/A')}")
                
        # 5. Explain a specific anomaly
        if len(anomalies) > 0:
            print(f"\n💡 EXPLANATION OF FIRST ANOMALY:")
            print("-" * 80)
            first_anomaly = anomalies.iloc[0]
            explanation = detector.explain_anomaly(first_anomaly)
            
            print(f"\n   Sample ID: {first_anomaly.get('Sample_ID', 'Unknown')}")
            print(f"   Is Anomaly: {'Yes' if explanation['is_anomaly'] else 'No'}")
            print(f"   Anomaly Score: {explanation['anomaly_score']:.2f}")
            
            if explanation['deviations']:
                print(f"\n   🔍 Deviations detected:")
                for dev in explanation['deviations']:
                    print(f"      • {dev['parameter']}: {dev['value']} - {dev['deviation']}")
                    
    return detector

if __name__ == "__main__":
    detector = train_and_test_model()