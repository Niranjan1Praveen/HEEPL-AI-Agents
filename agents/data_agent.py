import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from utils.csv_loader import CSVLoader

class DataAgent:
    """Responsible for loading, validating, and providing data statistics"""
    
    def __init__(self):
        self.loader = CSVLoader()
        self.current_data: Optional[pd.DataFrame] = None
        self.current_industry: Optional[str] = None
    
    def load_industry(self, industry_id: str) -> Dict[str, Any]:
        """Load data for a specific industry"""
        self.current_industry = industry_id
        self.current_data = self.loader.load_industry_data(industry_id)
        
        if self.current_data is None:
            return {"error": f"Failed to load data for {industry_id}"}
        
        return self.get_basic_stats()
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """Calculate basic statistics for current data"""
        if self.current_data is None:
            return {"error": "No data loaded"}
        
        df = self.current_data
        
        # Define parameter columns
        param_cols = {
            "BOD": "BOD (mg/L)",
            "COD": "COD (mg/L)", 
            "TSS": "TSS (mg/L)",
            "TDS": "TDS (mg/L)",
            "pH": "pH",
            "Oil_Grease": "Oil & Grease (mg/L)",
            "Ammonia": "Ammonia (mg/L)",
            "Conductivity": "Conductivity (uS/cm)",
            "Temperature": "Temperature (°C)"
        }
        
        stats = {
            "industry_id": self.current_industry,
            "total_samples": len(df),
            "parameters": {}
        }
        
        # Calculate stats for each parameter
        for name, col in param_cols.items():
            if col in df.columns:
                series = df[col]
                stats["parameters"][name] = {
                    "mean": round(series.mean(), 2),
                    "median": round(series.median(), 2),
                    "std": round(series.std(), 2),
                    "min": round(series.min(), 2),
                    "max": round(series.max(), 2),
                    "q1": round(series.quantile(0.25), 2),
                    "q3": round(series.quantile(0.75), 2),
                    "iqr": round(series.quantile(0.75) - series.quantile(0.25), 2)
                }
        
        # Status distribution if available
        if "Status" in df.columns:
            status_counts = df["Status"].value_counts()
            stats["status_distribution"] = {
                "critical": int(status_counts.get("Critical", 0)),
                "warning": int(status_counts.get("Warning", 0)),
                "normal": int(status_counts.get("Normal", 0))
            }
            stats["critical_percentage"] = round(
                stats["status_distribution"]["critical"] / len(df) * 100, 2
            )
        
        # Sample data preview
        stats["sample_preview"] = df.head(5).to_dict(orient="records")
        
        return stats
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get more detailed statistics for analysis"""
        if self.current_data is None:
            return {"error": "No data loaded"}
        
        df = self.current_data
        stats = self.get_basic_stats()
        
        # Add correlation matrix for key parameters
        param_cols = ['BOD (mg/L)', 'COD (mg/L)', 'TSS (mg/L)', 'TDS (mg/L)', 'pH']
        available_cols = [c for c in param_cols if c in df.columns]
        
        if len(available_cols) > 1:
            corr_matrix = df[available_cols].corr().round(2)
            stats["correlation_matrix"] = corr_matrix.to_dict()
        
        # Add summary statistics for anomalies detection
        for param_name, param_stats in stats["parameters"].items():
            # Flag potential outliers using IQR method
            iqr = param_stats["iqr"]
            q1 = param_stats["q1"]
            q3 = param_stats["q3"]
            
            outlier_threshold_high = q3 + 1.5 * iqr
            outlier_threshold_low = q1 - 1.5 * iqr
            
            param_stats["outlier_threshold_high"] = round(outlier_threshold_high, 2)
            param_stats["outlier_threshold_low"] = round(outlier_threshold_low, 2)
        
        return stats
    
    def get_time_series_data(self, parameter: str) -> Dict[str, Any]:
        """Get time series data for a specific parameter"""
        if self.current_data is None:
            return {"error": "No data loaded"}
        
        df = self.current_data
        
        # Map parameter name to column
        param_map = {
            "BOD": "BOD (mg/L)",
            "COD": "COD (mg/L)",
            "TSS": "TSS (mg/L)",
            "TDS": "TDS (mg/L)",
            "pH": "pH"
        }
        
        col = param_map.get(parameter, parameter)
        
        if col not in df.columns:
            return {"error": f"Parameter {parameter} not found"}
        
        # Return values in order (assuming Sample_ID indicates order)
        return {
            "parameter": parameter,
            "values": df[col].tolist(),
            "sample_ids": df["Sample_ID"].tolist() if "Sample_ID" in df.columns else list(range(len(df))),
            "status": df["Status"].tolist() if "Status" in df.columns else None
        }
    
    def print_summary(self):
        """Print a formatted summary of the data"""
        if self.current_data is None:
            print("No data loaded. Use load_industry() first.")
            return
        
        stats = self.get_basic_stats()
        
        print("\n" + "="*60)
        print(f"📊 DATA AGENT SUMMARY: {stats['industry_id'].upper()}")
        print("="*60)
        
        print(f"\n📈 Total Samples: {stats['total_samples']}")
        
        if "status_distribution" in stats:
            print(f"\n⚠️  Status Distribution:")
            print(f"   🔴 Critical: {stats['status_distribution']['critical']} ({stats['critical_percentage']}%)")
            print(f"   🟡 Warning: {stats['status_distribution']['warning']}")
            print(f"   🟢 Normal: {stats['status_distribution']['normal']}")
        
        print(f"\n📊 Parameter Statistics:")
        print("-" * 50)
        print(f"{'Parameter':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'IQR':>10}")
        print("-" * 50)
        
        for param, param_stats in stats["parameters"].items():
            print(f"{param:<12} {param_stats['mean']:>10,.0f} {param_stats['std']:>10,.0f} "
                  f"{param_stats['min']:>10,.0f} {param_stats['max']:>10,.0f} {param_stats['iqr']:>10,.0f}")
        
        print("\n" + "="*60)
        
        # Show sample preview
        print("\n📋 Sample Preview (first 3 rows):")
        print("-" * 60)
        for i, sample in enumerate(stats["sample_preview"][:3]):
            print(f"\nSample {i+1}:")
            for key, value in sample.items():
                if key in ['Sample_ID', 'BOD (mg/L)', 'COD (mg/L)', 'TSS (mg/L)', 'pH', 'Status']:
                    print(f"   {key}: {value}")
        
        print("\n" + "="*60 + "\n")


# Test the Data Agent
if __name__ == "__main__":
    # Initialize Data Agent
    agent = DataAgent()
    
    # Test with dairy-processing industry
    print("\n🔍 Testing Data Agent for: dairy-processing")
    result = agent.load_industry("cane-crushing")
    
    if "error" not in result:
        agent.print_summary()
        
        # Get detailed stats
        detailed = agent.get_detailed_stats()
        if "correlation_matrix" in detailed:
            print("\n📈 Correlation Matrix (BOD, COD, TSS, TDS, pH):")
            print("-" * 40)
            for key, values in detailed["correlation_matrix"].items():
                print(f"   {key}: {values}")
    else:
        print(f"❌ Error: {result['error']}")