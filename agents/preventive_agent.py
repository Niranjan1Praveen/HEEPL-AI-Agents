"""
agents/preventive_agent.py
----------------------------
PreventiveAgent — orchestrates equipment health analysis for the Preventive
Maintenance tab. Mirrors the AnalysisAgent pattern:
  - Loads equipment_anomaly_model.pkl (IsolationForest pipeline)
  - Runs anomaly detection on sound / vibration / temperature
  - Checks ISO 10816 threshold violations
  - Falls back to rule-based analysis if model file is missing
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.equipment_thresholds import check_equipment_violations, EQUIPMENT_DEFAULTS
from config.equipment_registry import get_equipment_for_industry

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH  = MODELS_DIR / "equipment_anomaly_model.pkl"

ANOMALY_THRESHOLD = 0.0   # IsolationForest: score < 0 → anomalous


@dataclass
class EquipmentAnalysisResult:
    equipment_id:    str
    equipment_name:  str
    equipment_type:  str        # "Pump" | "Blower"
    location:        str
    parameters:      dict       # {sound, vibration, temperature}
    anomaly_score:   float      # IsolationForest score (negative = anomalous)
    is_anomaly:      bool
    health_status:   str        # "Healthy" | "Warning" | "Critical"
    violations:      list       # [{parameter, value, severity, message}]

    def to_dict(self) -> dict:
        d = asdict(self)
        # Sanitise numpy types
        d["anomaly_score"] = float(self.anomaly_score)
        return d


@dataclass
class FleetAnalysisResult:
    industry_id:       str
    equipment_results: List[EquipmentAnalysisResult]
    fleet_health:      str    # worst-case across all units
    anomaly_count:     int
    critical_count:    int
    warning_count:     int

    def to_dict(self) -> dict:
        return {
            "industry_id":       self.industry_id,
            "fleet_health":      self.fleet_health,
            "anomaly_count":     self.anomaly_count,
            "critical_count":    self.critical_count,
            "warning_count":     self.warning_count,
            "equipment_results": [r.to_dict() for r in self.equipment_results],
        }


class PreventiveAgent:
    """
    Analyses equipment health using an IsolationForest model + ISO 10816 thresholds.
    """

    def __init__(self, use_models: bool = True):
        self.use_models = use_models
        self._pipeline = None
        self._feature_columns = ["sound", "vibration", "temperature"]
        self._model_loaded = False
        if use_models:
            self._load_model()

    # ------------------------------------------------------------------ #
    # Model loading                                                        #
    # ------------------------------------------------------------------ #

    def _load_model(self):
        if not MODEL_PATH.exists():
            print(f"[PreventiveAgent] Model not found at {MODEL_PATH}. Using rule-based fallback.")
            self._model_loaded = False
            return
        try:
            data = joblib.load(MODEL_PATH)
            self._pipeline        = data["pipeline"]
            self._feature_columns = data.get("feature_columns", self._feature_columns)
            self._model_loaded    = True
            print(f"[PreventiveAgent] Loaded equipment anomaly model from {MODEL_PATH}")
        except Exception as e:
            print(f"[PreventiveAgent] Failed to load model: {e}. Using rule-based fallback.")
            self._model_loaded = False

    # ------------------------------------------------------------------ #
    # Equipment roster                                                     #
    # ------------------------------------------------------------------ #

    def get_equipment_roster(self, industry_id: str) -> list:
        """Return industry-specific equipment list with ISO-standard default parameters."""
        roster = get_equipment_for_industry(industry_id)
        result = []
        for eq in roster:
            defaults = EQUIPMENT_DEFAULTS.get(eq["type"], EQUIPMENT_DEFAULTS["Pump"])
            result.append({
                **eq,
                "default_parameters": dict(defaults),
            })
        return result

    # ------------------------------------------------------------------ #
    # Single unit analysis                                                 #
    # ------------------------------------------------------------------ #

    def analyze_unit(self, equipment: dict, parameters: dict) -> EquipmentAnalysisResult:
        """
        Analyse a single equipment unit.
        `equipment` — {id, name, type, location}
        `parameters` — {sound, vibration, temperature}
        """
        eq_type    = equipment.get("type", "Pump")
        violations = check_equipment_violations(eq_type, parameters)

        if self._model_loaded:
            anomaly_score, is_anomaly = self._detect_anomaly(parameters, eq_type)
        else:
            anomaly_score, is_anomaly = self._fallback_anomaly(violations)

        health_status = self._determine_health_status(violations, is_anomaly)

        return EquipmentAnalysisResult(
            equipment_id   = equipment.get("id", "unknown"),
            equipment_name = equipment.get("name", "Unknown Equipment"),
            equipment_type = eq_type,
            location       = equipment.get("location", ""),
            parameters     = {k: float(v) for k, v in parameters.items() if k in self._feature_columns},
            anomaly_score  = float(anomaly_score),
            is_anomaly     = bool(is_anomaly),
            health_status  = health_status,
            violations     = violations,
        )

    # ------------------------------------------------------------------ #
    # Fleet analysis                                                       #
    # ------------------------------------------------------------------ #

    def analyze_fleet(self, industry_id: str, equipment_list: list) -> FleetAnalysisResult:
        """
        Analyse all equipment units sent by the frontend.
        `equipment_list` — list of {id, name, type, location, parameters: {sound, vibration, temperature}}
        """
        results = []
        for item in equipment_list:
            params = item.get("parameters", {})
            results.append(self.analyze_unit(item, params))

        health_priority = {"Critical": 2, "Warning": 1, "Healthy": 0}
        fleet_health = max(
            (r.health_status for r in results),
            key=lambda s: health_priority.get(s, 0),
            default="Healthy",
        )

        return FleetAnalysisResult(
            industry_id       = industry_id,
            equipment_results = results,
            fleet_health      = fleet_health,
            anomaly_count     = sum(1 for r in results if r.is_anomaly),
            critical_count    = sum(1 for r in results if r.health_status == "Critical"),
            warning_count     = sum(1 for r in results if r.health_status == "Warning"),
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _detect_anomaly(self, parameters: dict, equipment_type: str) -> tuple:
        """Run IsolationForest on the 3 sensor parameters. Returns (score, is_anomaly)."""
        try:
            X = pd.DataFrame([{
                "sound":       float(parameters.get("sound",       0)),
                "vibration":   float(parameters.get("vibration",   0)),
                "temperature": float(parameters.get("temperature", 0)),
            }])
            score       = float(self._pipeline.decision_function(X)[0])
            is_anomaly  = score < ANOMALY_THRESHOLD
            return score, is_anomaly
        except Exception as e:
            print(f"[PreventiveAgent] Anomaly detection failed: {e}")
            return self._fallback_anomaly([])

    def _fallback_anomaly(self, violations: list) -> tuple:
        """Rule-based fallback: anomaly = any critical violation exists."""
        is_anomaly   = any(v.get("severity") == "critical" for v in violations)
        anomaly_score = -0.5 if is_anomaly else 0.1
        return anomaly_score, is_anomaly

    def _determine_health_status(self, violations: list, is_anomaly: bool) -> str:
        """
        Priority: Critical violation → Critical
                  Warning violation OR anomaly → Warning
                  Otherwise → Healthy
        """
        severities = {v.get("severity") for v in violations}
        if "critical" in severities:
            return "Critical"
        if "warning" in severities or is_anomaly:
            return "Warning"
        return "Healthy"


if __name__ == "__main__":
    agent = PreventiveAgent(use_models=True)
    roster = agent.get_equipment_roster("molasses")
    print(f"Distillery roster: {len(roster)} items")

    test_fleet = [
        {**eq, "parameters": {**eq["default_parameters"]}} for eq in roster
    ]
    # Inject one degraded pump
    test_fleet[0]["parameters"] = {"sound": 98, "vibration": 8.0, "temperature": 92}

    fleet = agent.analyze_fleet("molasses", test_fleet)
    print(f"Fleet health: {fleet.fleet_health}  |  "
          f"Anomalies: {fleet.anomaly_count}  |  "
          f"Critical: {fleet.critical_count}")
    for r in fleet.equipment_results:
        print(f"  {r.equipment_name}: {r.health_status}  score={r.anomaly_score:+.3f}  violations={len(r.violations)}")
