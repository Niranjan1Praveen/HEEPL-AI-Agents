# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Flask REST API server for industrial wastewater analysis and equipment preventive maintenance using ML models and a Gemini LLM agent. It analyzes wastewater samples across 60+ industry types (anomaly detection, classification, NLP insights) and monitors equipment health (IsolationForest on sound/vibration/temperature, ISO 10816 thresholds, Gemini maintenance prescriptions).

## Commands

### Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Run (development)
```bash
python app.py
```

### Run (production)
```bash
gunicorn app:app --bind 0.0.0.0:5000 --workers 2 --timeout 120
```

### Train ML models
```bash
python training/train_anomaly.py
python training/train_classification.py
python training/train_forecasting.py
python training/train_equipment_anomaly.py   # Equipment health IsolationForest
```

### Test individual agents
```bash
python agents/analysis_agent.py
python agents/data_agent.py
python agents/llm_agent.py
python agents/preventive_agent.py
python utils/csv_loader.py
```

## Environment Variables

Copy `.env.example` to `.env`:
- `GEMINI_API_KEY` — Required for LLM insights; both `LLMAgent.generate_insights()` and `LLMAgent.generate_maintenance_insights()` fall back to templates if absent.
- `USE_ML_MODELS` — Set to `false` to skip ML model loading and use rule-based fallbacks only (applies to both wastewater and equipment models).
- `PORT` — Defaults to `5000`.

## Architecture

### Agent Pipeline

Requests flow through four lazy-loaded, globally shared agents:

1. **`AnalysisAgent`** (`agents/analysis_agent.py`) — Core orchestrator for wastewater. Runs anomaly detection, classification, and forecasting via `ModelRegistry`. Violations checked via `check_parameter_violations()` from `config/thresholds.py`. Rule-based fallback if `.pkl` missing. `AnalysisResult.to_dict()` sanitizes numpy types via `_safe()`.

2. **`DataAgent`** (`agents/data_agent.py`) — Loads CSV data and computes descriptive statistics. Thin wrapper around `CSVLoader`.

3. **`LLMAgent`** (`agents/llm_agent.py`) — Gemini-powered NLP. Two separate pipelines:
   - **Qualitative**: `InsightSchema` Pydantic model with `response_mime_type="application/json"`. Returns `parameter_treatments` (one per violated parameter — chemical, dosage, process, cost band).
   - **Preventive**: `MaintenanceSchema` Pydantic model with `response_mime_type="application/json"` (no `response_schema=` due to nested model `$defs` issue with some Gemini variants). Returns `maintenance_actions` per Warning/Critical equipment unit.
   - Both fall back to template responses if API unavailable.

4. **`PreventiveAgent`** (`agents/preventive_agent.py`) — Equipment health orchestrator.
   - Loads `models/equipment_anomaly_model.pkl` (IsolationForest trained on synthetic sensor data).
   - `get_equipment_roster(industry_id)` returns industry-specific equipment list with ISO baseline defaults.
   - `analyze_fleet(industry_id, equipment_list)` runs IsolationForest + threshold checks per unit → `FleetAnalysisResult`.
   - Falls back to rule-based analysis if model file missing.

### Config Layer

- **`config/thresholds.py`** — CPCB/EPA regulatory limits for 8 wastewater parameters. `check_parameter_violations(sample)` helper. **Edit thresholds only here.**
  ```python
  'pH': { 'critical_low': 5.5, 'critical_high': 9.5, 'warning_low': 6.0, 'warning_high': 9.0, 'normal_low': 6.5, 'normal_high': 8.5 }
  ```

- **`config/equipment_thresholds.py`** — ISO 10816 limits for Pump and Blower (sound dB, vibration mm/s RMS, temperature °C). `check_equipment_violations(equipment_type, parameters)` helper. **Edit equipment thresholds only here.**
  ```python
  "Pump": { "sound": {"normal":80,"warning":90,"critical":100}, "vibration": {"normal":2.8,"warning":4.5,"critical":7.1}, "temperature": {"normal":60,"warning":75,"critical":90} }
  ```

- **`config/equipment_registry.py`** — 17 industry-specific equipment rosters (pumps and blowers with realistic names and locations). `get_equipment_for_industry(industry_id)` resolves sub-category IDs (e.g. `"grain"` → `"distillery"` roster) via `SUB_TO_ROOT` mapping. Falls back to generic 6-unit roster.

- **`config/mappings.py`** — Maps short industry IDs to CSV paths. Single source of truth for all valid wastewater industry IDs.

### ML Models (`models/`)

- `anomaly_model.pkl` — Wastewater anomaly: `{pipeline: {imputer → scaler → IsolationForest}, feature_columns}`
- `classification_model.pkl` — Wastewater classification: `{pipeline: {imputer → scaler → RandomForestClassifier}, label_encoder, classes_, feature_columns}`
- `forecasting_models.pkl` — `{models: {param → {pipeline, scaler}}, lookback_window}`
- `equipment_anomaly_model.pkl` — Equipment health: `{pipeline: {imputer → scaler → IsolationForest}, feature_columns: ["sound","vibration","temperature"], contamination: 0.30}`
  - Trained on synthetic data: 70% healthy / 20% warning / 10% critical, for both Pump and Blower types.
  - Score < 0 → anomalous; score ≥ 0 → normal. The more negative, the more anomalous.
  - Pass `pd.DataFrame` (not a plain list) to `pipeline.decision_function()` to avoid sklearn feature-name warnings.

### LLM Pydantic Schemas

**Qualitative:**
```python
class ParameterTreatment(BaseModel):
    parameter, current_value, issue, chemical, dosage, process, expected_outcome, cost_band

class InsightSchema(BaseModel):
    summary, key_findings: List[str], recommendations: List[str], severity_level, parameter_treatments: Optional[List[ParameterTreatment]]
```

**Preventive:**
```python
class EquipmentMaintenance(BaseModel):
    equipment_id, equipment_name, health_status, issue, action, components_to_check: str,  # comma-separated string, not List[str]
    estimated_downtime, urgency, cost_band

class MaintenanceSchema(BaseModel):
    fleet_summary, critical_units: List[str], maintenance_actions: Optional[List[EquipmentMaintenance]],
    overall_recommendation, shutdown_risk
```

> **Note:** `components_to_check` is a `str` (comma-separated) not `List[str]` because Gemini's schema validator rejects `$defs` references from nested `List[str]` inside a nested Pydantic model. The frontend splits on `","` when rendering component chips.

> **Note:** `MaintenanceSchema` is passed with `response_mime_type="application/json"` only (no `response_schema=`) because the nested `EquipmentMaintenance` model generates `$defs` references that some Gemini model variants reject. The prompt fully describes the schema instead.

### Key Data Shapes

Wastewater parameters: `BOD (mg/L)`, `COD (mg/L)`, `TSS (mg/L)`, `TDS (mg/L)`, `pH`, `Oil & Grease (mg/L)`, `Ammonia (mg/L)`, `Temperature (°C)`. These column names must match exactly — `CSVLoader._standardize_columns` handles common variants.

Equipment parameters: `sound` (dB), `vibration` (mm/s RMS), `temperature` (°C).

## API Endpoints

### Wastewater Analysis
- `POST /analyze/with-insights` — Full analysis + LLM insights. Response includes `parameter_treatments` array.
- `POST /analyze/sample` — Analysis only (no LLM).
- `GET /industries/<industry_id>/stats` — CSV statistics for an industry.
- `GET /industries/<industry_id>/analyze` — Batch analysis on CSV rows.

### Preventive Maintenance
- `GET /preventive/equipment/<industry_id>` — Returns industry-specific equipment roster with `default_parameters` (ISO baseline values) for seeding frontend sliders.
- `POST /preventive/analyze` — Fleet analysis. Request: `{ industry_id, mode: "collective"|"individual", equipment: [{id, name, type, location, parameters: {sound, vibration, temperature}}] }`. Response includes `fleet_health`, per-unit `equipment_results`, and `insights.maintenance_actions`.

### Diagnostics
- `GET /health` — Health check. Checks all 4 model files + Gemini API key.
- `POST /api/v1/test/data` — Validates DataAgent.
- `POST /api/v1/test/analysis` — Validates AnalysisAgent pipelines.
- `POST /api/v1/test/llm` — Validates LLMAgent schema parsing.

### Training (on-demand)
- `POST /train/anomaly`, `/train/classification`, `/train/forecasting`, `/train/all`

## Deployment

Deployed on Render.com. `render.yaml` configures Python 3.11, trains anomaly + classification + equipment anomaly models during build, starts with gunicorn.
