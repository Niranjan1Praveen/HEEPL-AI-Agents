# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Flask REST API server for wastewater analysis using ML models and an LLM agent. It analyzes industrial wastewater samples across 60+ industry types, performing anomaly detection, classification, and NLP-based insights generation.

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
```

### Test individual agents
```bash
python agents/analysis_agent.py
python agents/data_agent.py
python agents/llm_agent.py
python utils/csv_loader.py
```

## Environment Variables

Copy `.env.example` to `.env`:
- `GEMINI_API_KEY` — Required for LLM insights; the `LLMAgent` falls back to templates if absent.
- `USE_ML_MODELS` — Set to `false` to skip ML model loading and use rule-based fallbacks only.
- `PORT` — Defaults to `5000`.

## Architecture

### Agent Pipeline

Requests flow through three lazy-loaded, globally shared agents:

1. **`AnalysisAgent`** (`agents/analysis_agent.py`) — Core orchestrator. Runs anomaly detection (Isolation Forest), classification (Random Forest), and forecasting (Random Forest Regressor) via a `ModelRegistry`. If a `.pkl` model is missing, it falls back to rule-based logic using hardcoded thresholds in `FALLBACK_THRESHOLDS`. Returns `AnalysisResult` dataclass.

2. **`DataAgent`** (`agents/data_agent.py`) — Loads CSV data and computes descriptive statistics. Thin wrapper around `CSVLoader`.

3. **`LLMAgent`** (`agents/llm_agent.py`) — Converts `AnalysisResult` dicts into natural language using Google Gemini (`google-genai`). Tries each model in `MODEL_NAMES` list at startup. Falls back to template-based responses if API is unavailable.

### Data Layer

- **`CSVLoader`** (`utils/csv_loader.py`) — Loads CSVs from `data/water-characteristics-data/<industry-subfolder>/`. Standardizes column names, fills NaN with column medians, and drops rows where all core numeric columns are null.
- **`config/mappings.py`** — Maps short industry IDs (e.g., `"dairy"`, `"cotton"`) to relative CSV paths. This is the single source of truth for all valid industry IDs.

### ML Models

Trained models are saved as `.pkl` files in `models/`:
- `anomaly_model.pkl` — `{model, scaler, feature_columns}`
- `classification_model.pkl` — `{model, scaler, imputer, label_encoder, feature_columns, classes_}`
- `forecasting_models.pkl`

Models are trained via scripts in `training/` and must be present before the server receives analysis requests (or `USE_ML_MODELS=false` must be set). The `render.yaml` build command trains anomaly and classification models at deploy time.

### Key Data Shapes

The primary wastewater parameters are: `BOD (mg/L)`, `COD (mg/L)`, `TSS (mg/L)`, `TDS (mg/L)`, `pH`, `Oil & Grease (mg/L)`, `Ammonia (mg/L)`. These column names must match exactly — `CSVLoader._standardize_columns` handles common variants.

## Deployment

Deployed on Render.com. `render.yaml` configures Python 3.11, runs model training during build, and starts with gunicorn. The `wsgi.py` file is an alternative entry point for WSGI servers.
