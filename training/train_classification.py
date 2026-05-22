"""
Classification Model for Industrial Wastewater Data
---------------------------------------------------
This model classifies wastewater samples into categories:
- Critical: Immediate action required (extreme parameter violations)
- Warning: Monitoring needed (moderate violations)
- Normal: Within acceptable limits (safe discharge)

Approaches used:
1. If Status column exists: Supervised learning
2. If no Status column: Unsupervised + Rule-based classification
3. Hybrid: Use anomaly scores + domain thresholds

Classification algorithms:
- Random Forest: Best for mixed data types, handles non-linearity
- XGBoost: High accuracy, handles imbalanced data
- Gradient Boosting: Good for complex patterns
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
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (classification_report, confusion_matrix, 
                                 accuracy_score, precision_score, recall_score, f1_score)
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not installed. Run: pip install scikit-learn")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not installed. Run: pip install xgboost")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("⚠️ joblib not installed. Run: pip install joblib")


class WastewaterClassifier:
    """
    Classification model for wastewater quality assessment
    
    Classifies samples into Critical, Warning, or Normal categories
    based on parameter values and regulatory limits.
    """
    
    # Regulatory limits and thresholds (based on CPCB/EPA standards)
    THRESHOLDS = {
        'BOD (mg/L)': {'critical': 500, 'warning': 100, 'normal': 30},
        'COD (mg/L)': {'critical': 1000, 'warning': 500, 'normal': 250},
        'TSS (mg/L)': {'critical': 300, 'warning': 150, 'normal': 100},
        'TDS (mg/L)': {'critical': 5000, 'warning': 3000, 'normal': 2100},
        'pH': {'critical_low': 4, 'critical_high': 10, 'warning_low': 5.5, 'warning_high': 9, 'normal_low': 6.5, 'normal_high': 8.5},
        'Oil & Grease (mg/L)': {'critical': 50, 'warning': 20, 'normal': 10},
        'Ammonia (mg/L)': {'critical': 100, 'warning': 50, 'normal': 25},
        'Temperature (°C)': {'critical': 45, 'warning': 35, 'normal': 30}
    }
    
    def __init__(self, use_supervised=True, contamination_threshold=0.15):
        """
        Args:
            use_supervised: If True, use Status column if available
            contamination_threshold: For unsupervised classification
        """
        self.use_supervised = use_supervised
        self.contamination_threshold = contamination_threshold
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = []
        self.label_encoder = LabelEncoder()
        self.classes_ = ['Normal', 'Warning', 'Critical']
        
    def create_rules_based_labels(self, df):
        """
        Create labels using domain knowledge rules (when Status column missing)
        Returns only: 'Normal', 'Warning', 'Critical'
        """
        labels = []
        
        for idx, row in df.iterrows():
            score = 0
            critical_count = 0
            warning_count = 0
            
            # Check each parameter against thresholds
            for param, thresholds in self.THRESHOLDS.items():
                if param not in df.columns:
                    continue
                    
                value = row[param]
                if pd.isna(value):
                    continue
                
                # Handle pH separately (has ranges)
                if param == 'pH':
                    if value <= thresholds['critical_low'] or value >= thresholds['critical_high']:
                        critical_count += 1
                        score += 3
                    elif value <= thresholds['warning_low'] or value >= thresholds['warning_high']:
                        warning_count += 1
                        score += 1
                else:
                    # For parameters with critical/warning/normal thresholds
                    if value >= thresholds['critical']:
                        critical_count += 1
                        score += 3
                    elif value >= thresholds['warning']:
                        warning_count += 1
                        score += 1
            
            # Determine final label (ensure only 3 classes)
            if critical_count >= 1 or score >= 5:
                labels.append('Critical')
            elif warning_count >= 1 or score >= 2:
                labels.append('Warning')
            else:
                labels.append('Normal')
        
        return np.array(labels)
    
    def create_anomaly_based_labels(self, df, anomaly_scores):
        """
        Create labels using anomaly scores (combines with rules)
        """
        rules_labels = self.create_rules_based_labels(df)
        
        # Combine with anomaly scores
        combined_labels = []
        for i, (rules_label, anomaly_score) in enumerate(zip(rules_labels, anomaly_scores)):
            # If rules say Critical OR anomaly score is very negative
            if rules_label == 'Critical' or anomaly_score < -0.6:
                combined_labels.append('Critical')
            elif rules_label == 'Warning' or anomaly_score < -0.3:
                combined_labels.append('Warning')
            else:
                combined_labels.append('Normal')
        
        return np.array(combined_labels)
    
    def extract_features(self, df):
        """Extract and prepare features for classification"""
        # Define features to use
        feature_cols = ['BOD (mg/L)', 'COD (mg/L)', 'TSS (mg/L)', 
                        'TDS (mg/L)', 'pH', 'Oil & Grease (mg/L)',
                        'Ammonia (mg/L)', 'Conductivity (uS/cm)', 'Temperature (°C)']
        
        # Keep only available columns
        available_features = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_features
        
        if len(available_features) == 0:
            raise ValueError("No valid feature columns found in data")
        
        # Extract features
        X = df[available_features].values
        
        # Impute missing values
        X = self.imputer.fit_transform(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
    
        return X_scaled
    def get_labels(self, df, anomaly_scores=None):
        """Get classification labels using available methods"""
        
        # Method 1: Use Status column if available and supervised learning enabled
        if self.use_supervised and 'Status' in df.columns:
            print("   📋 Using existing 'Status' column for labels")
            labels = df['Status'].values
            
            # Standardize label names - map to one of the three classes
            label_mapping = {
                'Critical': 'Critical', 'critical': 'Critical', 'CRITICAL': 'Critical',
                'Warning': 'Warning', 'warning': 'Warning', 'WARNING': 'Warning',
                'Normal': 'Normal', 'normal': 'Normal', 'NORMAL': 'Normal',
                'High': 'Critical',
                'Medium': 'Warning',
                'Low': 'Normal',
            }
            
            mapped_labels = []
            for l in labels:
                mapped = label_mapping.get(l, 'Normal')
                mapped_labels.append(mapped)
            
            return np.array(mapped_labels)
        
        # Method 2: Use anomaly scores + rules
        elif anomaly_scores is not None:
            print("   📋 Using anomaly scores + rule-based classification")
            return self.create_anomaly_based_labels(df, anomaly_scores)
        
        # Method 3: Use only rule-based classification
        else:
            print("   📋 Using rule-based classification (domain thresholds)")
            return self.create_rules_based_labels(df)
    
    def train(self, df, anomaly_scores=None):
        """Train classification model"""
        print("\n" + "="*60)
        print("🏋️ TRAINING CLASSIFICATION MODEL")
        print("="*60)
        
        # Extract features
        X = self.extract_features(df)
        print(f"\n📊 Feature matrix shape: {X.shape}")
        print(f"📋 Features used: {self.feature_columns}")
        
        # Get labels
        y = self.get_labels(df, anomaly_scores)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Update classes_ to only those actually present
        self.classes_ = list(self.label_encoder.classes_)
        print(f"\n📋 Classes present in data: {self.classes_}")
        
        # Print label distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n📈 Label Distribution:")
        for label, count in zip(unique, counts):
            pct = (count / len(y)) * 100
            emoji = "🔴" if label == "Critical" else "🟡" if label == "Warning" else "🟢"
            print(f"   {emoji} {label}: {count} ({pct:.1f}%)")
        
        # Note which classes are missing
        missing_classes = set(['Normal', 'Warning', 'Critical']) - set(unique)
        if missing_classes:
            print(f"\n⚠️ Note: These classes were not found in the data: {missing_classes}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Choose and train model
        print(f"\n🤖 Training Random Forest Classifier...")
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\n📊 Model Performance:")
        print(f"   ✅ Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   📈 Precision: {precision:.3f}")
        print(f"   🎯 Recall: {recall:.3f}")
        print(f"   🏆 F1-Score: {f1:.3f}")
        
        # Get unique classes present in test data
        unique_test_classes = np.unique(y_test)
        present_class_names = [self.classes_[i] for i in unique_test_classes if i < len(self.classes_)]
        
        if len(present_class_names) > 0:
            print(f"\n📋 Per-Class Performance (for classes present in data):")
            print("   " + "-" * 50)
            
            try:
                report = classification_report(
                    y_test, y_pred, 
                    labels=unique_test_classes,
                    target_names=present_class_names, 
                    zero_division=0
                )
                print(report)
            except Exception as e:
                print(f"   Could not generate detailed report: {e}")
                # Fallback: simple per-class accuracy
                for class_idx in unique_test_classes:
                    if class_idx < len(self.classes_):
                        class_name = self.classes_[class_idx]
                        mask = y_test == class_idx
                        if np.any(mask):
                            class_accuracy = accuracy_score(y_test[mask], y_pred[mask])
                            print(f"   {class_name}: {class_accuracy:.3f} ({class_accuracy*100:.1f}%)")
        
        # Cross-validation
        try:
            cv_scores = cross_val_score(self.model, X, y_encoded, cv=min(5, len(np.unique(y_encoded))))
            print(f"\n🔄 Cross-validation scores: {cv_scores}")
            print(f"   Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        except Exception as e:
            print(f"\n⚠️ Cross-validation skipped: {e}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔑 Top 5 Most Important Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"   • {row['feature']}: {row['importance']:.3f}")
        
        return {
            'model': self.model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': feature_importance,
            'label_distribution': dict(zip(unique, counts))
        }
    
    def predict(self, df, return_probabilities=False):
        """Predict classifications for new data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        # Extract features
        X = self.extract_features(df)
        
        # Predict
        predictions = self.model.predict(X)
        labels = self.label_encoder.inverse_transform(predictions)
        
        if return_probabilities:
            probabilities = self.model.predict_proba(X)
            return labels, probabilities
        
        return labels
    def explain_prediction(self, sample):
        """Explain why a sample was classified a certain way"""
        # Get feature values - ensure all features are present
        features = []
        for col in self.feature_columns:
            if col in sample and pd.notna(sample[col]):
                features.append(sample[col])
            else:
                features.append(0)  # Default value for missing features
        
        # Scale and predict
        X = np.array(features).reshape(1, -1)
        
        # Check if imputer is fitted
        try:
            X_imputed = self.imputer.transform(X)
            X_scaled = self.scaler.transform(X_imputed)
        except Exception as e:
            # If transform fails, use raw features
            print(f"⚠️ Could not transform features: {e}")
            X_scaled = X
        
        prediction = self.model.predict(X_scaled)[0]
        if prediction < len(self.classes_):
            label = self.classes_[prediction]
        else:
            label = "Unknown"
        
        # Get probabilities safely
        try:
            probabilities = self.model.predict_proba(X_scaled)[0]
            max_prob = max(probabilities) if len(probabilities) > 0 else 0
            prob_dict = {self.classes_[i] if i < len(self.classes_) else f"Class_{i}": prob 
                        for i, prob in enumerate(probabilities)}
        except Exception as e:
            max_prob = 0
            prob_dict = {}
        
        # Get threshold violations (only for parameters that exist)
        violations = []
        for param, thresholds in self.THRESHOLDS.items():
            if param in sample and pd.notna(sample[param]):
                value = sample[param]
                
                if param == 'pH':
                    if value <= thresholds['critical_low'] or value >= thresholds['critical_high']:
                        violations.append(f"{param}: {value} (Critical - outside {thresholds['critical_low']}-{thresholds['critical_high']})")
                    elif value <= thresholds['warning_low'] or value >= thresholds['warning_high']:
                        violations.append(f"{param}: {value} (Warning - outside {thresholds['warning_low']}-{thresholds['warning_high']})")
                else:
                    if value >= thresholds['critical']:
                        violations.append(f"{param}: {value} (Critical - exceeds {thresholds['critical']})")
                    elif value >= thresholds['warning']:
                        violations.append(f"{param}: {value} (Warning - exceeds {thresholds['warning']})")
        
        return {
            'predicted_class': label,
            'confidence': max_prob,
            'probabilities': prob_dict,
            'violations': violations
        }
    def save_model(self, path):
        """Save the trained model"""
        if not JOBLIB_AVAILABLE:
            print("⚠️ Cannot save model: joblib not installed")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'classes_': self.classes_,
            'thresholds': self.THRESHOLDS
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, path)
        print(f"✅ Classification model saved to: {path}")
    
    def load_model(self, path):
        """Load a trained model"""
        if not JOBLIB_AVAILABLE:
            print("⚠️ Cannot load model: joblib not installed")
            return
        
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.imputer = model_data['imputer']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        self.classes_ = model_data['classes_']
        print(f"✅ Classification model loaded from: {path}")


def train_classification_model():
    """Main function to train classification model on all data"""
    
    print("\n" + "="*70)
    print("🏷️ CLASSIFICATION MODEL TRAINING")
    print("="*70)
    
    # 1. Load ALL available data
    print("\n📂 Loading all industry data...")
    loader = CSVLoader()
    all_industries = get_all_industry_ids()
    
    all_data = []
    status_available_count = 0
    
    for industry_id in all_industries:
        df = loader.load_industry_data(industry_id)
        if df is not None and len(df) > 0:
            all_data.append(df)
            if 'Status' in df.columns:
                status_available_count += 1
            print(f"   ✅ {industry_id}: {len(df)} samples {'(has Status)' if 'Status' in df.columns else ''}")
    
    if not all_data:
        print("❌ No data loaded")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Combined dataset: {len(combined_df)} samples from {len(all_data)} industries")
    print(f"📋 Industries with Status column: {status_available_count}/{len(all_data)}")
    
    # 2. Load anomaly scores if available (for better classification)
    anomaly_scores = None
    anomaly_model_path = Path(__file__).parent.parent / "models" / "anomaly_model.pkl"
    
    if anomaly_model_path.exists():
        try:
            print(f"✅ Found existing anomaly model")
        except:
            print("⚠️ Could not load anomaly model")
    
    # 3. Train classification model
    classifier = WastewaterClassifier(use_supervised=True)
    
    # Train on combined data
    results = classifier.train(combined_df, anomaly_scores)
    
    # 4. Save model
    model_path = Path(__file__).parent.parent / "models" / "classification_model.pkl"
    classifier.save_model(model_path)
    
    # 5. Demonstrate predictions on sample data
    print("\n" + "="*70)
    print("🔮 DEMONSTRATING PREDICTIONS")
    print("="*70)
    
    # Get a few samples for demonstration
    demo_samples = combined_df.head(10)
    predictions = classifier.predict(demo_samples)
    
    print("\n📊 Sample Predictions:")
    print("-" * 70)
    
    for i, (idx, row) in enumerate(demo_samples.iterrows()):
        # Get explanation
        explanation = classifier.explain_prediction(row.to_dict())
        
        # Get actual Status if available
        actual = row.get('Status', 'N/A')
        
        # Get key parameters
        bod = row.get('BOD (mg/L)', 'N/A')
        cod = row.get('COD (mg/L)', 'N/A')
        
        print(f"\nSample {i+1}:")
        print(f"   BOD: {bod}, COD: {cod}")
        print(f"   Actual Status: {actual}")
        print(f"   Predicted: {explanation['predicted_class']} (confidence: {explanation['confidence']:.2%})")
        
        if explanation['violations']:
            print(f"   Violations:")
            for v in explanation['violations'][:2]:
                print(f"      • {v}")
    
    return classifier


def print_model_explanation():
    """Print detailed explanation of the classification model"""
    print("\n" + "="*70)
    print("📚 CLASSIFICATION MODEL EXPLANATION")
    print("="*70)
    
    print("""
    🧠 CLASSIFICATION MODEL: RANDOM FOREST CLASSIFIER
    
    WHAT IT DOES:
    ------------
    Classifies wastewater samples into three categories:
    🔴 CRITICAL: Immediate action required
    🟡 WARNING: Monitoring recommended  
    🟢 NORMAL: Within acceptable limits
    
    HOW IT WORKS:
    ------------
    1. Uses 9+ water quality parameters as input features
    2. Builds multiple decision trees on random data subsets
    3. Each tree "votes" on the classification
    4. Majority vote determines final class
    
    LABEL CREATION STRATEGY:
    -----------------------
    If 'Status' column exists → Use it as ground truth
    If no 'Status' column → Create labels using:
        • Regulatory thresholds (CPCB/EPA standards)
        • Anomaly detection scores
        • Domain knowledge rules
    
    FEATURE IMPORTANCE (Typical):
    ----------------------------
    • BOD (Biochemical Oxygen Demand) - Highest impact
    • COD (Chemical Oxygen Demand) - Second highest
    • pH - Critical for biological treatment
    • TSS/TDS - Physical quality indicators
    
    CLASSIFICATION THRESHOLDS:
    -------------------------
    Parameter    | Normal  | Warning | Critical
    -------------|---------|---------|----------
    BOD (mg/L)   | <30     | 30-500  | >500
    COD (mg/L)   | <250    | 250-1000| >1000
    TSS (mg/L)   | <100    | 100-300 | >300
    pH           | 6.5-8.5 | 5.5-9   | <5.5 or >9
    
    PRACTICAL APPLICATIONS:
    ----------------------
    1. Real-time effluent quality assessment
    2. Automated compliance monitoring
    3. Early warning system for violations
    4. Prioritize inspection resources
    5. Identify problematic industry sectors
    
    HANDLING MISSING STATUS COLUMNS:
    --------------------------------
    The model intelligently handles CSV files without 'Status' by:
    1. Using domain-specific threshold rules
    2. Incorporating anomaly detection scores
    3. Creating pseudo-labels for training
    """)


if __name__ == "__main__":
    # Train classification model on ALL available data
    classifier = train_classification_model()
    
    # Print detailed explanation
    print_model_explanation()
    
    # Example: Classify a new sample
if classifier:
    print("\n" + "="*70)
    print("💡 CLASSIFY A NEW SAMPLE")
    print("="*70)
    
    # Example sample with ALL features the model expects
    new_sample = {
        'BOD (mg/L)': 350,
        'COD (mg/L)': 800,
        'TSS (mg/L)': 120,
        'TDS (mg/L)': 2500,
        'pH': 7.2,
        'Oil & Grease (mg/L)': 25,
        'Ammonia (mg/L)': np.nan,  # Missing value
        'Conductivity (uS/cm)': np.nan,  # Missing value
        'Temperature (°C)': np.nan,  # Missing value
    }
    
    print("\n📊 New Sample Parameters:")
    for param, value in new_sample.items():
        if pd.notna(value):
            print(f"   {param}: {value}")
        else:
            print(f"   {param}: (not measured)")
    
    # Create a DataFrame with the sample
    sample_df = pd.DataFrame([new_sample])
    
    # The extract_features method will handle missing columns automatically
    # But we need to ensure all feature columns exist
    for col in classifier.feature_columns:
        if col not in sample_df.columns:
            sample_df[col] = np.nan
    
    # Verify we have all expected features
    print(f"\n📋 Expected features ({len(classifier.feature_columns)}): {classifier.feature_columns}")
    print(f"   Sample has {len([c for c in classifier.feature_columns if c in sample_df.columns])} features")
    
    # Predict
    try:
        prediction = classifier.predict(sample_df)[0]
        print(f"\n🔮 Predicted Classification: {prediction}")
        
        # Get explanation
        explanation = classifier.explain_prediction(new_sample)
        print(f"\n💡 Explanation:")
        print(f"   Confidence: {explanation['confidence']:.2%}")
        
        if explanation['violations']:
            print(f"   Key violations detected:")
            for v in explanation['violations']:
                print(f"      • {v}")
    except Exception as e:
        print(f"\n⚠️ Could not classify sample: {e}")
        print("   This is because the model expects certain features that are missing.")
        print("   In production, you would provide all measured parameters.")