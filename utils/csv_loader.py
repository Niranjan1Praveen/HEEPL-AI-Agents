import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.mappings import INDUSTRY_CSV_MAPPING, get_csv_path, get_all_industry_ids

# Path to CSV files in server folder
SERVER_DATA_PATH = Path(__file__).parent.parent / "data" / "water-characteristics-data"

class CSVLoader:
    """Load and preprocess CSV files for all industries with dynamic columns"""
    
    # Define expected column categories
    CORE_COLUMNS = ['Sample_ID', 'pH', 'BOD (mg/L)', 'COD (mg/L)', 'TSS (mg/L)', 'TDS (mg/L)']
    OPTIONAL_COLUMNS = ['Oil & Grease (mg/L)', 'Ammonia (mg/L)', 'Conductivity (uS/cm)', 
                        'Temperature (°C)', 'Status']
    
    @classmethod
    def get_csv_path(cls, industry_id: str) -> Optional[Path]:
        """Get the full path for a given industry ID"""
        relative_path = get_csv_path(industry_id)
        if relative_path is None:
            return None
        return SERVER_DATA_PATH / relative_path
    
    @classmethod
    def load_industry_data(cls, industry_id: str) -> Optional[pd.DataFrame]:
        """Load CSV data for a specific industry"""
        csv_path = cls.get_csv_path(industry_id)
        if not csv_path or not csv_path.exists():
            print(f"⚠️ CSV not found: {csv_path}")
            return None
        
        try:
            # Try reading with different strategies
            df = cls._read_csv_with_fallback(csv_path)
            
            if df is None or len(df) == 0:
                return None
            
            # Standardize columns
            df = cls._standardize_columns(df)
            
            # Preprocess (convert types, handle NaN)
            df = cls._preprocess_data(df)
            
            if len(df) == 0:
                return None
            
            # Add metadata
            df.attrs['industry_id'] = industry_id
            df.attrs['source_file'] = csv_path.name
            
            print(f"✅ Loaded {industry_id}: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            print(f"❌ Error loading {industry_id}: {e}")
            return None
    
    @classmethod
    def _read_csv_with_fallback(cls, csv_path: Path) -> Optional[pd.DataFrame]:
        """Try multiple strategies to read CSV"""
        strategies = [
            lambda: pd.read_csv(csv_path),
            lambda: pd.read_csv(csv_path, on_bad_lines='skip'),
            lambda: pd.read_csv(csv_path, engine='python'),
            lambda: pd.read_csv(csv_path, encoding='latin1'),
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                df = strategy()
                if len(df) > 0:
                    return df
            except Exception:
                continue
        return None
    
    @classmethod
    def _standardize_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names"""
        column_mapping = {
            'Sample ID': 'Sample_ID', 'sample_id': 'Sample_ID',
            'BOD': 'BOD (mg/L)', 'bod': 'BOD (mg/L)',
            'COD': 'COD (mg/L)', 'cod': 'COD (mg/L)',
            'TSS': 'TSS (mg/L)', 'tss': 'TSS (mg/L)',
            'TDS': 'TDS (mg/L)', 'tds': 'TDS (mg/L)',
            'pH': 'pH', 'ph': 'pH',
            'Oil and Grease': 'Oil & Grease (mg/L)',
            'Oil & Grease': 'Oil & Grease (mg/L)',
            'Ammonia': 'Ammonia (mg/L)', 'ammonia': 'Ammonia (mg/L)',
            'Conductivity': 'Conductivity (uS/cm)',
            'Temperature': 'Temperature (°C)', 'temp': 'Temperature (°C)',
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure core columns exist
        for col in cls.CORE_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        
        return df
    
    @classmethod
    def _preprocess_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data - FIXED for string dtype issue"""
        df = df.copy()
        
        # Define all numeric columns
        all_numeric_cols = [col for col in cls.CORE_COLUMNS + cls.OPTIONAL_COLUMNS 
                           if col != 'Sample_ID' and col != 'Status']
        
        # Convert each numeric column properly
        for col in all_numeric_cols:
            if col in df.columns:
                # First, try to convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # For columns that are completely string/object type, drop them
        for col in all_numeric_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    # Try one more time with different approach
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows where all core numeric columns are NaN
        core_numeric = [col for col in cls.CORE_COLUMNS if col != 'Sample_ID']
        df = df.dropna(subset=core_numeric, how='all')
        
        # Fill remaining NaN with column median (only if column is numeric)
        for col in all_numeric_cols:
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                if df[col].isna().any():
                    median_val = df[col].median()
                    if not np.isnan(median_val):
                        df[col] = df[col].fillna(median_val)
                    else:
                        df[col] = df[col].fillna(0)
        
        # Clean up Sample_ID - ensure it's string
        if 'Sample_ID' in df.columns:
            df['Sample_ID'] = df['Sample_ID'].astype(str)
        
        return df
    
    @classmethod
    def get_available_columns(cls, industry_id: str) -> Dict[str, List[str]]:
        """Get information about available columns in a CSV"""
        csv_path = cls.get_csv_path(industry_id)
        if not csv_path or not csv_path.exists():
            return {"error": "File not found"}
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                header = f.readline().strip().split(',')
            
            available_core = [col for col in cls.CORE_COLUMNS if col in header]
            available_optional = [col for col in cls.OPTIONAL_COLUMNS if col in header]
            missing_core = [col for col in cls.CORE_COLUMNS if col not in header]
            
            return {
                "industry_id": industry_id,
                "all_columns": header,
                "available_core": available_core,
                "available_optional": available_optional,
                "missing_core": missing_core,
                "has_status": "Status" in header
            }
        except Exception as e:
            return {"error": str(e)}
    
    @classmethod
    def load_all_industries(cls) -> Dict[str, pd.DataFrame]:
        """Load data for all industries"""
        all_data = {}
        for industry_id in get_all_industry_ids():
            df = cls.load_industry_data(industry_id)
            if df is not None and len(df) > 0:
                all_data[industry_id] = df
        return all_data
    
    @classmethod
    def get_industry_summary(cls, industry_id: str) -> dict:
        """Get comprehensive summary of industry data"""
        df = cls.load_industry_data(industry_id)
        if df is None:
            return {"error": f"Industry {industry_id} not found"}
        
        summary = {
            "industry_id": industry_id,
            "total_samples": len(df),
            "columns_available": list(df.columns),
            "statistics": {}
        }
        
        # Calculate statistics for numeric columns only
        for col in df.columns:
            if col != 'Sample_ID' and df[col].dtype in ['float64', 'int64']:
                summary["statistics"][col] = {
                    "mean": round(df[col].mean(), 2),
                    "std": round(df[col].std(), 2),
                    "min": round(df[col].min(), 2),
                    "max": round(df[col].max(), 2),
                }
        
        if 'Status' in df.columns:
            summary["status_distribution"] = df['Status'].value_counts().to_dict()
        
        return summary
    
    @classmethod
    def list_available_industries(cls) -> List[str]:
        """List all industries with CSV files available"""
        available = []
        for industry_id in get_all_industry_ids():
            csv_path = cls.get_csv_path(industry_id)
            if csv_path and csv_path.exists():
                available.append(industry_id)
        return available


# Test the loader
if __name__ == "__main__":
    print("="*60)
    print("📊 CSV LOADER TEST")
    print("="*60)
    
    print(f"\n📁 Data path: {SERVER_DATA_PATH}")
    print(f"📁 Path exists: {SERVER_DATA_PATH.exists()}\n")
    
    # Test industries
    test_industries = ["dairy", "grain", "molasses", "cotton", "api-bulk"]
    
    for industry_id in test_industries:
        print(f"\n🔍 Testing: {industry_id}")
        print("-" * 40)
        
        df = CSVLoader.load_industry_data(industry_id)
        if df is not None and len(df) > 0:
            print(f"   ✅ Loaded: {len(df)} rows")
            print(f"   📋 Numeric columns: {list(df.select_dtypes(include=['float64', 'int64']).columns)}")
            
            # Show sample
            print(f"\n   📊 Sample data (first row):")
            for col in ['Sample_ID', 'BOD (mg/L)', 'COD (mg/L)', 'pH']:
                if col in df.columns:
                    val = df[col].iloc[0]
                    print(f"      {col}: {val}")