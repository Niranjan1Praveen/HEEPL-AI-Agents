"""
Classification Model for Industrial Wastewater Data
---------------------------------------------------
This model classifies wastewater samples into categories:
- Critical: Immediate action required (extreme parameter violations)
- Warning: Monitoring needed (moderate violations)
- Normal: Within acceptable limits (safe discharge)
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
from config.thresholds import WASTEWATER_THRESHOLDS

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

class WastewaterClassifier:
    """
    Classification model for wastewater quality assessment.
    Classifies samples into Critical, Warning, or Normal categories.
    """
    
    def __init__(self, use_supervised=True, contamination_threshold=0.15):
        """
        Args:
            use_supervised: If True, use Status column if available
            contamination_threshold: For unsupervised classification
        """
        self.use_supervised = use_supervised
        self.contamination_threshold = contamination_threshold
        self.pipeline = None
        
        # Preprocessors and estimators for backward compatibility
        self.model = None
        self.scaler = None
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoder = LabelEncoder()
        
        self.feature_columns = []
        self.classes_ = ['Normal', 'Warning', 'Critical']
        
    def create_rules_based_labels(self, df):
        """Create labels using domain knowledge rules from config/thresholds.py"""
        labels = []
        
        for idx, row in df.iterrows():
            score = 0
            critical_count = 0
            warning_count = 0
            
            # Check each parameter against thresholds
            for param, thresholds in WASTEWATER_THRESHOLDS.items():
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
            
            # Determine final label
            if critical_count >= 1 or score >= 5:
                labels.append('Critical')
            elif warning_count >= 1 or score >= 2:
                labels.append('Warning')
            else:
                labels.append('Normal')
        
        return np.array(labels)
    
    def create_anomaly_based_labels(self, df, anomaly_scores):
        """Create labels using anomaly scores combined with rules"""
        rules_labels = self.create_rules_based_labels(df)
        combined_labels = []
        for i, (rules_label, anomaly_score) in enumerate(zip(rules_labels, anomaly_scores)):
            if rules_label == 'Critical' or anomaly_score < -0.6:
                combined_labels.append('Critical')
            elif rules_label == 'Warning' or anomaly_score < -0.3:
                combined_labels.append('Warning')
            else:
                combined_labels.append('Normal')
        return np.array(combined_labels)
    
    def get_labels(self, df, anomaly_scores=None):
        """Get classification labels using available methods"""
        if self.use_supervised and 'Status' in df.columns:
            print("   📋 Using existing 'Status' column for labels")
            labels = df['Status'].values
            
            # Standardize labels
            label_mapping = {
                'Critical': 'Critical', 'critical': 'Critical', 'CRITICAL': 'Critical',
                'Warning': 'Warning', 'warning': 'Warning', 'WARNING': 'Warning',
                'Normal': 'Normal', 'normal': 'Normal', 'NORMAL': 'Normal',
                'High': 'Critical', 'Medium': 'Warning', 'Low': 'Normal',
            }
            mapped_labels = [label_mapping.get(l, 'Normal') for l in labels]
            return np.array(mapped_labels)
        elif anomaly_scores is not None:
            print("   📋 Using anomaly scores + rule-based classification")
            return self.create_anomaly_based_labels(df, anomaly_scores)
        else:
            print("   📋 Using rule-based classification (domain thresholds)")
            return self.create_rules_based_labels(df)
            
    def train(self, df, anomaly_scores=None):
        """Train the classification model pipeline"""
        print("\n" + "="*60)
        print("🏋️ TRAINING CLASSIFICATION PIPELINE")
        print("="*60)
        
        feature_cols = ['BOD (mg/L)', 'COD (mg/L)', 'TSS (mg/L)', 
                        'TDS (mg/L)', 'pH', 'Oil & Grease (mg/L)',
                        'Ammonia (mg/L)', 'Conductivity (uS/cm)', 'Temperature (°C)']
        
        available_features = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_features
        
        X = df[available_features].values
        y = self.get_labels(df, anomaly_scores)
        
        # Fit Label Encoder
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = list(self.label_encoder.classes_)
        print(f"\n📋 Classes: {self.classes_}")
        
        # Label distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"📈 Label Distribution:")
        for label, count in zip(unique, counts):
            pct = (count / len(y)) * 100
            emoji = "🔴" if label == "Critical" else "🟡" if label == "Warning" else "🟢"
            print(f"   {emoji} {label:<8}: {count:<6} ({pct:.1f}%)")
            
        # Build Pipeline
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Fit pipeline on training data
        self.pipeline.fit(X_train, y_train)
        
        # Keep references for compatibility
        self.model = self.pipeline.named_steps['model']
        self.scaler = self.pipeline.named_steps['scaler']
        self.imputer = self.pipeline.named_steps['imputer']
        
        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\n📊 Test Metrics Report:")
        print(f"   ✅ Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"   📈 Precision: {precision:.4f}")
        print(f"   🎯 Recall: {recall:.4f}")
        print(f"   🏆 F1-Score: {f1:.4f}")
        
        unique_test_classes = np.unique(y_test)
        present_class_names = [self.classes_[i] for i in unique_test_classes]
        
        print(f"\n📋 Per-Class Performance:")
        print("-" * 55)
        print(classification_report(y_test, y_pred, labels=unique_test_classes, target_names=present_class_names, zero_division=0))
        
        # Cross-validation
        try:
            cv_scores = cross_val_score(self.pipeline, X, y_encoded, cv=5)
            print(f"🔄 5-Fold Cross-validation accuracy scores: {cv_scores}")
            print(f"   Mean CV accuracy: {cv_scores.mean():.4f} (StdDev: {cv_scores.std():.4f})")
        except Exception as e:
            print(f"⚠️ CV calculation skipped: {e}")
            
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔑 Feature Importances:")
        for idx, row in feature_importance.iterrows():
            print(f"   • {row['feature']:<22}: {row['importance']:.4f}")
            
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': feature_importance
        }
        
    def predict(self, df, return_probabilities=False):
        """Predict classifications for new data"""
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train() or load_model() first.")
            
        # Ensure all columns are present
        X_df = df.copy()
        for col in self.feature_columns:
            if col not in X_df.columns:
                X_df[col] = np.nan
                
        X = X_df[self.feature_columns].values
        
        # Predict using pipeline
        predictions = self.pipeline.predict(X)
        labels = self.label_encoder.inverse_transform(predictions)
        
        if return_probabilities:
            probabilities = self.pipeline.predict_proba(X)
            return labels, probabilities
            
        return labels
        
    def explain_prediction(self, sample):
        """Explain why a sample was classified in a certain category"""
        features = [sample.get(col, np.nan) for col in self.feature_columns]
        X = np.array(features).reshape(1, -1)
        
        # Impute/scale and predict
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        prediction = self.model.predict(X_scaled)[0]
        label = self.label_encoder.inverse_transform([prediction])[0]
        
        probabilities = self.model.predict_proba(X_scaled)[0]
        prob_dict = {self.label_encoder.inverse_transform([i])[0]: float(prob) for i, prob in enumerate(probabilities)}
        
        # Check rule violations
        from config.thresholds import check_parameter_violations
        violations = check_parameter_violations(sample)
        violations_messages = [v['message'] for v in violations]
        
        return {
            'predicted_class': label,
            'confidence': float(max(probabilities)),
            'probabilities': prob_dict,
            'violations': violations_messages
        }
        
    def save_model(self, path):
        """Save pipeline and encoder objects"""
        model_data = {
            'pipeline': self.pipeline,
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'classes_': self.classes_
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, path)
        print(f"✅ Classification model saved to: {path}")
        
    def load_model(self, path):
        """Load pipeline and encoder objects"""
        model_data = joblib.load(path)
        self.pipeline = model_data.get('pipeline')
        self.model = model_data.get('model')
        self.scaler = model_data.get('scaler')
        self.imputer = model_data.get('imputer')
        self.label_encoder = model_data.get('label_encoder')
        self.feature_columns = model_data.get('feature_columns')
        self.classes_ = model_data.get('classes_')
        
        if self.pipeline is None and self.model is not None:
            self.pipeline = Pipeline([
                ('imputer', self.imputer),
                ('scaler', self.scaler),
                ('model', self.model)
            ])
        print(f"✅ Classification Pipeline loaded from: {path}")

def train_classification_model():
    """Main function to train classification model on all data"""
    print("\n" + "="*70)
    print("🏷️ CLASSIFICATION MODEL TRAINING")
    print("="*70)
    
    loader = CSVLoader()
    all_industries = get_all_industry_ids()
    all_data = []
    
    for industry_id in all_industries:
        df = loader.load_industry_data(industry_id)
        if df is not None and len(df) > 0:
            all_data.append(df)
            
    if not all_data:
        print("❌ No data loaded.")
        return None
        
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Combined dataset: {len(combined_df)} samples")
    
    classifier = WastewaterClassifier(use_supervised=True)
    metrics = classifier.train(combined_df)
    
    model_path = Path(__file__).parent.parent / "models" / "classification_model.pkl"
    classifier.save_model(model_path)
    
    return classifier

if __name__ == "__main__":
    train_classification_model()