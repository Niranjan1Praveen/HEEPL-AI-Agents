"""
Flask Application for Wastewater Analysis AI Agents
---------------------------------------------------
Exposes REST APIs for:
- Data analysis (anomaly detection, classification, forecasting)
- Industry data management
- Natural language insights generation

Deployment: Ready for Render.com
"""

import os
import sys
import json
from pathlib import Path
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import agents
from agents.analysis_agent import AnalysisAgent, AnalysisResult
from agents.data_agent import DataAgent
from agents.llm_agent import LLMAgent
from utils.csv_loader import CSVLoader
from config.mappings import get_all_industry_ids, get_csv_path

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global agent instances (lazy loaded)
_analysis_agent = None
_data_agent = None
_llm_agent = None

def get_analysis_agent():
    """Get or create AnalysisAgent instance"""
    global _analysis_agent
    if _analysis_agent is None:
        use_models = os.environ.get('USE_ML_MODELS', 'true').lower() == 'true'
        _analysis_agent = AnalysisAgent(use_models=use_models)
    return _analysis_agent

def get_data_agent():
    """Get or create DataAgent instance"""
    global _data_agent
    if _data_agent is None:
        _data_agent = DataAgent()
    return _data_agent

def get_llm_agent():
    """Get or create LLMAgent instance"""
    global _llm_agent
    if _llm_agent is None:
        api_key = os.environ.get('GEMINI_API_KEY')
        _llm_agent = LLMAgent(api_key=api_key)
    return _llm_agent

# ==================== HEALTH CHECK ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        'status': 'healthy',
        'service': 'Wastewater Analysis AI Agents',
        'version': '1.0.0'
    })

@app.route('/', methods=['GET'])
def index():
    """API documentation"""
    return jsonify({
        'service': 'Wastewater Analysis AI Agents API',
        'version': '1.0.0',
        'endpoints': {
            '/health': 'GET - Health check',
            '/industries': 'GET - List all industries',
            '/industries/<industry_id>': 'GET - Get industry summary',
            '/industries/<industry_id>/stats': 'GET - Get industry statistics',
            '/industries/<industry_id>/analyze': 'POST - Analyze a sample',
            '/industries/<industry_id>/analyze/batch': 'POST - Analyze multiple samples',
            '/industries/<industry_id}/insights': 'POST - Get natural language insights',
            '/analyze/sample': 'POST - Analyze a single sample (JSON body)',
            '/train/all': 'POST - Retrain all models (admin)'
        }
    })

# ==================== INDUSTRY ENDPOINTS ====================

@app.route('/industries', methods=['GET'])
def list_industries():
    """List all available industries"""
    industries = get_all_industry_ids()
    
    # Get additional info for each industry
    loader = CSVLoader()
    industry_info = []
    
    for industry_id in industries:
        csv_path = loader.get_csv_path(industry_id)
        exists = csv_path.exists() if csv_path else False
        
        # Try to load sample count
        sample_count = 0
        if exists:
            df = loader.load_industry_data(industry_id)
            if df is not None:
                sample_count = len(df)
        
        industry_info.append({
            'id': industry_id,
            'has_data': exists,
            'sample_count': sample_count
        })
    
    return jsonify({
        'total': len(industries),
        'industries': industry_info
    })

@app.route('/industries/<industry_id>', methods=['GET'])
def get_industry_summary(industry_id):
    """Get summary for a specific industry"""
    agent = get_data_agent()
    result = agent.load_industry(industry_id)
    
    if 'error' in result:
        return jsonify({'error': result['error']}), 404
    
    return jsonify(result)

@app.route('/industries/<industry_id>/stats', methods=['GET'])
def get_industry_stats(industry_id):
    """Get detailed statistics for an industry"""
    loader = CSVLoader()
    summary = loader.get_industry_summary(industry_id)
    
    if 'error' in summary:
        return jsonify(summary), 404
    
    return jsonify(summary)

# ==================== ANALYSIS ENDPOINTS ====================

@app.route('/industries/<industry_id>/analyze', methods=['POST'])
def analyze_sample(industry_id):
    """
    Analyze a single sample from an industry
    
    Request body:
    {
        "sample": {
            "Sample_ID": "TEST_001",
            "BOD (mg/L)": 350,
            "COD (mg/L)": 800,
            "TSS (mg/L)": 120,
            "TDS (mg/L)": 2500,
            "pH": 7.2,
            "Oil & Grease (mg/L)": 25
        }
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    sample = data.get('sample')
    if not sample:
        return jsonify({'error': 'Missing "sample" field'}), 400
    
    agent = get_analysis_agent()
    
    # Optionally load industry data for context
    agent.load_industry_data(industry_id)
    
    # Analyze the sample
    result = agent.analyze_sample(sample, industry_id=industry_id)
    
    return jsonify(result.to_dict())

@app.route('/industries/<industry_id>/analyze/batch', methods=['POST'])
def analyze_batch(industry_id):
    """
    Analyze multiple samples from an industry
    
    Request body:
    {
        "samples": [...array of samples...],
        "limit": 100 (optional)
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    samples = data.get('samples')
    limit = data.get('limit')
    
    if not samples:
        return jsonify({'error': 'Missing "samples" field'}), 400
    
    # Convert to DataFrame
    df = pd.DataFrame(samples)
    
    agent = get_analysis_agent()
    agent.load_industry_data(industry_id)
    
    # Analyze batch
    results = agent.analyze_dataframe(df, limit=limit)
    
    return jsonify({
        'industry_id': industry_id,
        'total_analyzed': len(results),
        'results': [r.to_dict() for r in results]
    })

@app.route('/analyze/sample', methods=['POST'])
def analyze_any_sample():
    """
    Analyze a sample without industry context
    
    Request body:
    {
        "sample": {...},
        "industry_id": "optional_industry_id"
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    sample = data.get('sample')
    industry_id = data.get('industry_id', 'unknown')
    
    if not sample:
        return jsonify({'error': 'Missing "sample" field'}), 400
    
    agent = get_analysis_agent()
    
    if industry_id != 'unknown':
        agent.load_industry_data(industry_id)
    
    result = agent.analyze_sample(sample, industry_id=industry_id)
    
    return jsonify(result.to_dict())

# ==================== INSIGHTS ENDPOINTS ====================

@app.route('/industries/<industry_id>/insights', methods=['POST'])
def get_insights(industry_id):
    """
    Get natural language insights for a sample
    
    Request body:
    {
        "sample": {...} (same as analyze endpoint)
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    sample = data.get('sample')
    
    if not sample:
        return jsonify({'error': 'Missing "sample" field'}), 400
    
    # First analyze the sample
    analysis_agent = get_analysis_agent()
    result = analysis_agent.analyze_sample(sample, industry_id=industry_id)
    
    # Then generate insights
    llm_agent = get_llm_agent()
    insights = llm_agent.generate_insights(result.to_dict(), industry_id=industry_id)
    
    return jsonify({
        'analysis': result.to_dict(),
        'insights': {
            'summary': insights.summary,
            'key_findings': insights.key_findings,
            'recommendations': insights.recommendations,
            'severity_level': insights.severity_level
        }
    })

@app.route('/analyze/with-insights', methods=['POST'])
def analyze_with_insights():
    """
    Complete analysis with natural language insights
    
    Request body:
    {
        "sample": {...},
        "industry_id": "optional_industry_id"
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    sample = data.get('sample')
    industry_id = data.get('industry_id', 'unknown')
    
    if not sample:
        return jsonify({'error': 'Missing "sample" field'}), 400
    
    # Analyze
    analysis_agent = get_analysis_agent()
    result = analysis_agent.analyze_sample(sample, industry_id=industry_id)
    
    # Generate insights
    llm_agent = get_llm_agent()
    insights = llm_agent.generate_insights(result.to_dict(), industry_id=industry_id)
    
    return jsonify({
        'sample_id': result.sample_id,
        'industry_id': result.industry_id,
        'analysis': {
            'anomaly_score': result.anomaly_score,
            'is_anomaly': result.is_anomaly,
            'predicted_class': result.predicted_class,
            'class_confidence': result.class_confidence,
            'violations': result.violations
        },
        'insights': {
            'summary': insights.summary,
            'key_findings': insights.key_findings,
            'recommendations': insights.recommendations,
            'severity_level': insights.severity_level
        }
    })

# ==================== BATCH INSIGHTS ====================

@app.route('/insights/batch', methods=['POST'])
def batch_insights():
    """
    Generate summary insights for multiple samples
    
    Request body:
    {
        "samples": [...array of analysis results...],
        "industry_id": "optional_industry_id"
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    samples = data.get('samples')
    industry_id = data.get('industry_id', 'unknown')
    
    if not samples:
        return jsonify({'error': 'Missing "samples" field'}), 400
    
    llm_agent = get_llm_agent()
    summary = llm_agent.generate_batch_insights(samples, industry_id=industry_id)
    
    return jsonify({
        'industry_id': industry_id,
        'total_samples': len(samples),
        'summary': summary
    })

# ==================== TRAINING ENDPOINTS ====================

@app.route('/train/anomaly', methods=['POST'])
def train_anomaly():
    """Retrain the anomaly detection model"""
    try:
        from training.train_anomaly import train_and_test_model
        detector = train_and_test_model()
        return jsonify({
            'status': 'success',
            'message': 'Anomaly detection model trained successfully',
            'model_path': 'models/anomaly_model.pkl'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/train/classification', methods=['POST'])
def train_classification():
    """Retrain the classification model"""
    try:
        from training.train_classification import train_classification_model
        classifier = train_classification_model()
        return jsonify({
            'status': 'success',
            'message': 'Classification model trained successfully',
            'model_path': 'models/classification_model.pkl'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/train/forecasting', methods=['POST'])
def train_forecasting():
    """Retrain the forecasting model"""
    try:
        from training.train_forecasting import train_forecasting_models
        forecaster = train_forecasting_models()
        return jsonify({
            'status': 'success',
            'message': 'Forecasting models trained successfully',
            'model_path': 'models/forecasting_models.pkl'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/train/all', methods=['POST'])
def train_all():
    """Retrain all models"""
    results = {}
    
    try:
        from training.train_anomaly import train_and_test_model
        detector = train_and_test_model()
        results['anomaly'] = 'success'
    except Exception as e:
        results['anomaly'] = str(e)
    
    try:
        from training.train_classification import train_classification_model
        classifier = train_classification_model()
        results['classification'] = 'success'
    except Exception as e:
        results['classification'] = str(e)
    
    try:
        from training.train_forecasting import train_forecasting_models
        forecaster = train_forecasting_models()
        results['forecasting'] = 'success'
    except Exception as e:
        results['forecasting'] = str(e)
    
    return jsonify({
        'status': 'completed',
        'results': results
    })

# ==================== MISC ENDPOINTS ====================

@app.route('/columns/<industry_id>', methods=['GET'])
def get_columns(industry_id):
    """Get available columns for an industry"""
    loader = CSVLoader()
    columns_info = loader.get_available_columns(industry_id)
    
    if 'error' in columns_info:
        return jsonify(columns_info), 404
    
    return jsonify(columns_info)

# ==================== MAIN ====================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    
    print("="*60)
    print("🚀 Wastewater Analysis AI Agents API")
    print("="*60)
    print(f"📍 Running on: http://localhost:{port}")
    print(f"🔧 Debug mode: {debug}")
    print(f"📊 Industries available: {len(get_all_industry_ids())}")
    print("="*60)
    
    app.run(host='0.0.0.0', port=port, debug=debug)