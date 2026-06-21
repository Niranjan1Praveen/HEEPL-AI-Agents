# config/thresholds.py
"""
Centralized Regulatory Thresholds (based on CPCB / EPA Standards)
Provides a single source of truth for parameter warnings, critical levels,
and normal ranges.
"""

WASTEWATER_THRESHOLDS = {
    'BOD (mg/L)': {
        'critical': 500.0,
        'warning': 100.0,
        'normal': 30.0
    },
    'COD (mg/L)': {
        'critical': 1000.0,
        'warning': 500.0,
        'normal': 250.0
    },
    'TSS (mg/L)': {
        'critical': 300.0,
        'warning': 150.0,
        'normal': 100.0
    },
    'TDS (mg/L)': {
        'critical': 5000.0,
        'warning': 3000.0,
        'normal': 2100.0
    },
    'pH': {
        'critical_low': 4.0,
        'critical_high': 10.0,
        'warning_low': 5.5,
        'warning_high': 9.0,
        'normal_low': 6.5,
        'normal_high': 8.5
    },
    'Oil & Grease (mg/L)': {
        'critical': 50.0,
        'warning': 20.0,
        'normal': 10.0
    },
    'Ammonia (mg/L)': {
        'critical': 100.0,
        'warning': 50.0,
        'normal': 25.0
    },
    'Temperature (°C)': {
        'critical': 45.0,
        'warning': 35.0,
        'normal': 30.0
    }
}

def check_parameter_violations(sample: dict) -> list:
    """
    Check sample parameters against centralized thresholds.
    Returns a list of violation dictionaries containing details about parameter,
    value, severity, and message.
    """
    violations = []
    
    for param, limits in WASTEWATER_THRESHOLDS.items():
        if param not in sample:
            continue
            
        value = sample[param]
        # Ignore null/NaN values
        try:
            import pandas as pd
            if pd.isna(value):
                continue
        except ImportError:
            if value is None:
                continue
                
        try:
            val_float = float(value)
        except (ValueError, TypeError):
            continue
            
        if param == 'pH':
            if val_float <= limits.get('critical_low', 0.0) or val_float >= limits.get('critical_high', 14.0):
                violations.append({
                    'parameter': param,
                    'value': val_float,
                    'severity': 'critical',
                    'message': f"pH is {val_float} - outside critical safety limits ({limits.get('critical_low')}-{limits.get('critical_high')})"
                })
            elif val_float <= limits.get('warning_low', 0.0) or val_float >= limits.get('warning_high', 14.0):
                violations.append({
                    'parameter': param,
                    'value': val_float,
                    'severity': 'warning',
                    'message': f"pH is {val_float} - approaching limits; monitor closely"
                })
        else:
            critical = limits.get('critical', float('inf'))
            warning = limits.get('warning', float('inf'))
            
            if val_float >= critical:
                violations.append({
                    'parameter': param,
                    'value': val_float,
                    'severity': 'critical',
                    'message': f"{param} is {val_float} mg/L - exceeds critical limit ({critical})"
                })
            elif val_float >= warning:
                violations.append({
                    'parameter': param,
                    'value': val_float,
                    'severity': 'warning',
                    'message': f"{param} is {val_float} mg/L - exceeds warning limit ({warning})"
                })
                
    return violations
