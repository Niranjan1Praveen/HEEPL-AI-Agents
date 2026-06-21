"""
config/equipment_thresholds.py
-------------------------------
ISO 10816 / maintenance engineering standards for pump and blower health.
Provides thresholds for sound (dB), vibration (mm/s RMS), and temperature (°C).
"""

EQUIPMENT_THRESHOLDS = {
    "Pump": {
        "sound": {
            "normal": 80,    # dB — all-clear
            "warning": 90,   # dB — elevated noise, inspect soon
            "critical": 100, # dB — bearing/cavitation failure imminent
        },
        "vibration": {
            "normal": 2.8,   # mm/s RMS — ISO 10816 Class II acceptable
            "warning": 4.5,  # mm/s — allowable limit
            "critical": 7.1, # mm/s — damage threshold
        },
        "temperature": {
            "normal": 60,    # °C — bearing temperature
            "warning": 75,   # °C — elevated, check lubrication
            "critical": 90,  # °C — risk of seizure
        },
    },
    "Blower": {
        "sound": {
            "normal": 88,
            "warning": 95,
            "critical": 105,
        },
        "vibration": {
            "normal": 3.5,
            "warning": 6.3,
            "critical": 10.0,
        },
        "temperature": {
            "normal": 65,
            "warning": 80,
            "critical": 100,
        },
    },
}

# Default baseline parameters used for roster initialization (healthy mid-range values)
EQUIPMENT_DEFAULTS = {
    "Pump":   {"sound": 72.0, "vibration": 2.1, "temperature": 48.0},
    "Blower": {"sound": 85.0, "vibration": 2.8, "temperature": 58.0},
}


def check_equipment_violations(equipment_type: str, parameters: dict) -> list:
    """
    Check equipment sensor readings against ISO 10816 thresholds.
    Returns list of {parameter, value, severity, message} — same pattern
    as check_parameter_violations() in thresholds.py.
    """
    violations = []
    limits = EQUIPMENT_THRESHOLDS.get(equipment_type, EQUIPMENT_THRESHOLDS["Pump"])
    param_units = {"sound": "dB", "vibration": "mm/s", "temperature": "°C"}

    for param, thresholds in limits.items():
        value = parameters.get(param)
        if value is None:
            continue
        try:
            val = float(value)
        except (ValueError, TypeError):
            continue

        unit = param_units.get(param, "")
        critical = thresholds["critical"]
        warning = thresholds["warning"]

        if val >= critical:
            violations.append({
                "parameter": param,
                "value": val,
                "severity": "critical",
                "message": f"{param.capitalize()} is {val} {unit} — exceeds critical limit ({critical} {unit}) for {equipment_type}",
            })
        elif val >= warning:
            violations.append({
                "parameter": param,
                "value": val,
                "severity": "warning",
                "message": f"{param.capitalize()} is {val} {unit} — exceeds warning limit ({warning} {unit}) for {equipment_type}",
            })

    return violations
