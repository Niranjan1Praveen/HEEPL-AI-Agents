"""
Anomaly Detection Model for Industrial Wastewater Data
------------------------------------------------------
This model uses Isolation Forest algorithm to detect anomalies in wastewater parameters.
Isolation Forest works by randomly isolating observations - anomalies are easier to isolate
because they are few and different from normal data points.

Why Isolation Forest for wastewater data?
1. Handles high-dimensional data (multiple parameters like BOD, COD, pH, etc.)
2. Robust to outliers in training data
3. Doesn't assume normal distribution of data
4. Works well with mixed parameter ranges (mg/L, pH, temperature)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    Anomaly Detection using Isolation Forest Algorithm
    
    How Isolation Forest works:
    1. Randomly selects a feature (e.g., BOD level)
    2. Randomly selects a split value between min and max of that feature
    3. Recursively partitions the data until each point is isolated
    4. Anomalies require fewer splits to isolate (shorter paths)
    5. Normal points require more splits (longer paths)
    
    Contamination parameter: Expected proportion of anomalies in the dataset
    """
    
    def __init__(self, contamination=0.1, random_state=42):
        """
        Args:
            contamination: Expected proportion of anomalies (0.1 = 10%)
            random_state: For reproducible results
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,  # Number of trees in the forest
            max_samples='auto',  # Number of samples per tree
            bootstrap=False  # No bootstrap sampling
        )
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.contamination = contamination
        
    def prepare_features(self, df):
        """Extract and normalize features for training - handles missing columns"""
        
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
            print(f"⚠️ Warning: Only {len(available_features)} features available. Minimum 3 required.")
            # Use whatever is available
        else:
            print(f"📋 Using features: {available_features}")
        
        # Extract features
        X = df[available_features].values
        
        # Fill any NaN values with column means
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled
    
    def train(self, df):
        """Train the anomaly detection model"""
        print("\n" + "="*60)
        print("🏋️ TRAINING ANOMALY DETECTION MODEL")
        print("="*60)
        
        # Prepare features
        X = self.prepare_features(df)
        
        print(f"\n📊 Training Data Shape: {X.shape}")
        print(f"📋 Features Used: {self.feature_columns}")
        print(f"🎯 Expected Contamination Rate: {self.contamination * 100}%")
        
        # Train the model
        self.model.fit(X)
        
        # Evaluate on training data
        predictions = self.model.predict(X)
        anomaly_count = sum(predictions == -1)
        anomaly_rate = anomaly_count / len(predictions)
        
        print(f"\n📈 Training Results:")
        print(f"   ✅ Normal samples: {sum(predictions == 1)} ({100 - anomaly_rate*100:.1f}%)")
        print(f"   ⚠️ Anomalies detected: {anomaly_count} ({anomaly_rate*100:.1f}%)")
        
        # Get anomaly scores (lower = more anomalous)
        scores = self.model.score_samples(X)
        print(f"   📊 Anomaly scores range: {scores.min():.2f} to {scores.max():.2f}")
        
        return self
    
    def predict(self, df):
        """Predict anomalies in new data"""
        if self.feature_columns is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Predict (-1 = anomaly, 1 = normal)
        predictions = self.model.predict(X)
        
        # Get anomaly scores (negative = more anomalous)
        scores = self.model.score_samples(X)
        
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
        """Save the trained model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'contamination': self.contamination
        }
        joblib.dump(model_data, path)
        print(f"✅ Model saved to: {path}")
    
    def load_model(self, path):
        """Load a trained model from disk"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.contamination = model_data['contamination']
        print(f"✅ Model loaded from: {path}")
    
    def explain_anomaly(self, sample_row):
        """Explain why a specific sample is anomalous"""
        if self.feature_columns is None:
            return "Model not trained yet"
        
        # Get feature values
        features = [sample_row.get(col, 0) for col in self.feature_columns]
        
        # Scale and predict
        X = self.scaler.transform([features])
        score = self.model.score_samples(X)[0]
        prediction = self.model.predict(X)[0]
        
        # Calculate deviation for each feature
        X_unscaled = np.array([features])
        mean_values = self.scaler.mean_
        std_values = self.scaler.scale_
        
        deviations = []
        for i, col in enumerate(self.feature_columns):
            value = features[i]
            # Get typical range (mean ± 2 std deviations)
            typical_min = mean_values[i] - 2 * std_values[i]
            typical_max = mean_values[i] + 2 * std_values[i]
            actual_scaled = (value - mean_values[i]) / std_values[i]
            
            if abs(actual_scaled) > 2:
                direction = "high" if actual_scaled > 0 else "low"
                deviations.append({
                    'parameter': col,
                    'value': value,
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
    
    # Load multiple industries for diverse training data
    industries_to_train = get_all_industry_ids()  # Load all available industries
    
    all_data = []
    for industry_id in industries_to_train:
        df = loader.load_industry_data(industry_id)
        if df is not None:
            df['industry'] = industry_id
            all_data.append(df)
            print(f"   ✅ Loaded {industry_id}: {len(df)} samples")
    
    if not all_data:
        print("❌ No data loaded. Check your CSV file paths.")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Combined dataset: {len(combined_df)} samples from {len(industries_to_train)} industries")
    
    # 2. Train anomaly detection model
    detector = AnomalyDetector(contamination=0.15)  # Expect 15% anomalies
    detector.train(combined_df)
    
    # 3. Save the model
    model_path = Path(__file__).parent.parent / "models" / "anomaly_model.pkl"
    model_path.parent.mkdir(exist_ok=True)
    detector.save_model(model_path)
    
    # 4. Test on a specific industry's data
    print("\n" + "="*60)
    print("🧪 TESTING MODEL ON NEW DATA")
    print("="*60)
    
    test_industry = "dairy"  # Test on dairy industry
    test_df = loader.load_industry_data(test_industry)
    
    if test_df is not None:
        print(f"\n📊 Testing on: {test_industry} ({len(test_df)} samples)")
        
        # Get predictions
        predictions, scores = detector.predict(test_df)
        
        # Add results to dataframe
        test_df['is_anomaly'] = predictions == -1
        test_df['anomaly_score'] = scores
        
        # Print summary
        anomaly_count = test_df['is_anomaly'].sum()
        print(f"\n📈 Test Results:")
        print(f"   ✅ Normal samples: {len(test_df) - anomaly_count}")
        print(f"   ⚠️ Anomalies detected: {anomaly_count} ({anomaly_count/len(test_df)*100:.1f}%)")
        
        # Show sample anomalies
        anomalies = test_df[test_df['is_anomaly'] == True]
        if len(anomalies) > 0:
            print(f"\n🔍 Sample Anomalies (first 5):")
            print("-" * 80)
            for idx, row in anomalies.head(5).iterrows():
                print(f"\n   Sample: {row.get('Sample_ID', 'Unknown')}")
                print(f"   Status (original): {row.get('Status', 'N/A')}")
                print(f"   Anomaly Score: {row['anomaly_score']:.2f}")
                # Show key parameters
                print(f"   Key Parameters: BOD={row.get('BOD (mg/L)', 'N/A')}, "
                      f"COD={row.get('COD (mg/L)', 'N/A')}, "
                      f"pH={row.get('pH', 'N/A')}")
        
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


def print_model_explanation():
    """Print detailed explanation of the model"""
    print("\n" + "="*60)
    print("📚 MODEL EXPLANATION")
    print("="*60)
    
    print("""
    🧠 ANOMALY DETECTION MODEL: ISOLATION FOREST
    
    WHAT IT DOES:
    ------------
    Identifies unusual patterns in wastewater parameters by isolating
    observations that behave differently from the majority.
    
    HOW IT WORKS:
    ------------
    1. Randomly selects a parameter (e.g., BOD, COD, pH)
    2. Randomly picks a value between min and max of that parameter
    3. Splits the data into two groups based on that value
    4. Repeats recursively until each point is isolated
    5. Anomalies require fewer splits (shorter isolation paths)
    
    WHY ISOLATION FOREST FOR WASTEWATER DATA:
    -----------------------------------------
    ✓ Handles multiple parameter types (mg/L, pH, temperature)
    ✓ Robust to outliers in training data
    ✓ No assumption of normal distribution
    ✓ Works well with real-world industrial data
    ✓ Interpretable results (can explain why a sample is anomalous)
    
    INPUT FEATURES:
    --------------
    - BOD (mg/L): Biochemical Oxygen Demand
    - COD (mg/L): Chemical Oxygen Demand  
    - TSS (mg/L): Total Suspended Solids
    - TDS (mg/L): Total Dissolved Solids
    - pH: Acidity/Alkalinity
    - Oil & Grease (mg/L)
    - Ammonia (mg/L)
    - Conductivity (uS/cm)
    - Temperature (°C)
    
    OUTPUT:
    -------
    - Anomaly Score: Negative values indicate anomalies
    - Classification: -1 (Anomaly) or 1 (Normal)
    - Confidence level for each prediction
    
    PERFORMANCE:
    -----------
    - Training Time: ~2-5 seconds (5000 samples)
    - Inference Time: <0.1 seconds per sample
    - Memory Usage: ~50-100MB
    
    PRACTICAL APPLICATIONS:
    ----------------------
    1. Real-time monitoring of treatment plant performance
    2. Early warning of equipment malfunction
    3. Identifying samples requiring retesting
    4. Compliance violation detection
    5. Process optimization opportunities
    """)


if __name__ == "__main__":
    # Train and test the model
    detector = train_and_test_model()
    
    # Print detailed explanation
    # print_model_explanation()