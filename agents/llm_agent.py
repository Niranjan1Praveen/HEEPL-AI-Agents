"""
LLM Agent - Natural Language Generation
----------------------------------------
This agent converts analysis results into human-readable insights
using Google's Gemini API (free tier).

Uses the new google.genai package with structured schemas for reliable outputs.
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import sys
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("⚠️ Google GenAI not installed. Run: pip install google-genai")

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class InsightSchema(BaseModel):
    """Pydantic schema for structured Gemini insights output"""
    summary: str = Field(description="2-3 sentence overview of the water quality status and severity level.")
    key_findings: List[str] = Field(description="List of specific parameters exceeding thresholds, anomalies, or notable patterns.")
    recommendations: List[str] = Field(description="Actionable treatment protocols, process adjustments, or monitoring instructions.")
    severity_level: str = Field(description="One of: low, medium, high, critical")

@dataclass
class LLMResponse:
    """Structured response from LLM Agent"""
    summary: str
    key_findings: List[str]
    recommendations: List[str]
    severity_level: str  # 'low', 'medium', 'high', 'critical'
    raw_response: str

class LLMAgent:
    """
    LLM Agent for generating natural language insights from analysis results
    Uses Gemini 2.0 Flash for structured, schema-validated JSON outputs.
    """
    
    # List of model names to try (in order of preference)
    MODEL_NAMES = [
        "gemini-2.0-flash",        # Fast, good for text generation
        "gemini-2.5-flash",        # Newer version
        "gemini-flash-latest",     # Latest flash version
    ]
    
    # Pre-defined templates for fallback (if API fails)
    SEVERITY_TEMPLATES = {
        'critical': {
            'prefix': "🚨 **CRITICAL ISSUES DETECTED** 🚨\n\n",
            'urgency': "Immediate action required!",
            'icon': "🔴"
        },
        'high': {
            'prefix': "⚠️ **High Priority Issues** ⚠️\n\n",
            'urgency': "Action needed soon.",
            'icon': "🟠"
        },
        'medium': {
            'prefix': "📊 **Medium Priority Findings**\n\n",
            'urgency': "Monitor and plan corrective actions.",
            'icon': "🟡"
        },
        'low': {
            'prefix': "✅ **Normal Operation**\n\n",
            'urgency': "Continue regular monitoring.",
            'icon': "🟢"
        }
    }
    
    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize LLM Agent
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = None
        self.model_name = None
        
        if GENAI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                if model_name:
                    self.model_name = model_name
                else:
                    self.model_name = self._find_working_model()
                
                if self.model_name:
                    print(f"✅ LLM Agent initialized with {self.model_name}")
                else:
                    print(f"⚠️ No working model found. Using fallback templates.")
                    self.client = None
            except Exception as e:
                print(f"⚠️ Failed to initialize Gemini: {e}")
                self.client = None
        else:
            print("⚠️ Gemini not available or API key missing. Using fallback templates.")
        
        self.industry_context = self._load_industry_context()
    
    def _find_working_model(self) -> Optional[str]:
        """Find a working Gemini model"""
        for name in self.MODEL_NAMES:
            try:
                # Try simple validation call
                response = self.client.models.generate_content(
                    model=name,
                    contents="Test connection. Respond short."
                )
                if response and response.text:
                    return name
            except Exception:
                continue
        return None
    
    def _load_industry_context(self) -> Dict:
        """Load industry-specific context for better insights"""
        return {
            "dairy": {
                "name": "Dairy Processing",
                "common_issues": ["High BOD/COD", "FOG (Fats, Oils, Grease)", "Seasonal variations"],
                "typical_solutions": ["DAF system", "Equalization tank", "Anaerobic digestion"]
            },
            "textile": {
                "name": "Textile & Dyeing",
                "common_issues": ["Color removal", "High TDS", "Reactive dyes"],
                "typical_solutions": ["Membrane filtration", "Chemical coagulation", "Biological treatment"]
            },
            "distillery": {
                "name": "Distillery",
                "common_issues": ["Very high COD (100k+)", "Low pH", "Dark color"],
                "typical_solutions": ["Biogas recovery", "Evaporation", "ZLD systems"]
            },
            "pharma": {
                "name": "Pharmaceutical",
                "common_issues": ["Refractory compounds", "Solvents", "Antibiotics"],
                "typical_solutions": ["Advanced oxidation", "Activated carbon", "Membrane bioreactors"]
            },
            "tannery": {
                "name": "Tannery",
                "common_issues": ["Chromium", "High TSS", "Sulfides"],
                "typical_solutions": ["Chromium recovery", "Primary clarifiers", "Chemical precipitation"]
            },
            "pulp_paper": {
                "name": "Pulp & Paper",
                "common_issues": ["Lignin", "AOX compounds", "High color"],
                "typical_solutions": ["Biological treatment", "Coagulation", "Membrane filtration"]
            },
            "oil_refinery": {
                "name": "Oil Refinery",
                "common_issues": ["Oil & grease", "Sulfides", "Phenols"],
                "typical_solutions": ["API separators", "Dissolved air flotation", "Biological treatment"]
            }
        }
    
    def _get_industry_context(self, industry_id: str) -> Dict:
        """Get context for a specific industry"""
        for key, context in self.industry_context.items():
            if key in industry_id.lower() or industry_id.lower() in key:
                return context
        return {
            "name": industry_id.replace("_", " ").title(),
            "common_issues": ["Parameter violations", "Treatment efficiency"],
            "typical_solutions": ["Review treatment process", "Optimize operations"]
        }
    
    def generate_insights(self, analysis_result: Dict, industry_id: str = None) -> LLMResponse:
        """
        Generate natural language insights from analysis results
        """
        industry = industry_id or analysis_result.get('industry_id', 'unknown')
        industry_context = self._get_industry_context(industry)
        severity = self._determine_severity(analysis_result)
        
        if self.client and self.model_name:
            try:
                return self._generate_with_gemini(analysis_result, industry_context, severity)
            except Exception as e:
                print(f"⚠️ Gemini API structured generation error: {e}, using fallback")
        
        return self._generate_with_templates(analysis_result, industry_context, severity)
    
    def _determine_severity(self, result: Dict) -> str:
        """Determine severity level from analysis results"""
        predicted_class = result.get('predicted_class', '')
        if predicted_class == 'Critical':
            return 'critical'
        elif predicted_class == 'Warning':
            return 'medium'
            
        violations = result.get('violations', [])
        critical_count = sum(1 for v in violations if v.get('severity') == 'critical')
        warning_count = sum(1 for v in violations if v.get('severity') == 'warning')
        
        if critical_count >= 2:
            return 'critical'
        elif critical_count >= 1:
            return 'high'
        elif warning_count >= 2:
            return 'medium'
        elif warning_count >= 1:
            return 'low'
            
        anomaly_score = result.get('anomaly_score', 0)
        if anomaly_score and anomaly_score < -0.6:
            return 'high'
        elif anomaly_score and anomaly_score < -0.3:
            return 'medium'
            
        return 'low'
    
    def _generate_with_gemini(self, result: Dict, industry_context: Dict, severity: str) -> LLMResponse:
        """Generate insights using Gemini API with strict structured schema validation"""
        prompt = self._build_prompt(result, industry_context, severity)
        
        # Configure the structured JSON output config
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=InsightSchema,
            temperature=0.2
        )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        
        # Parse the structured JSON output directly
        data = json.loads(response.text)
        
        severity_val = data.get('severity_level', severity).lower()
        severity_info = self.SEVERITY_TEMPLATES.get(severity_val, self.SEVERITY_TEMPLATES['low'])
        summary_with_icon = f"{severity_info['icon']} {data.get('summary', '')}"
        
        return LLMResponse(
            summary=summary_with_icon,
            key_findings=data.get('key_findings', []),
            recommendations=data.get('recommendations', []),
            severity_level=severity_val,
            raw_response=response.text
        )
    
    def _build_prompt(self, result: Dict, industry_context: Dict, severity: str) -> str:
        """Build prompt for Gemini"""
        sample_id = result.get('sample_id', 'Unknown')
        predicted_class = result.get('predicted_class', 'Not available')
        class_confidence = result.get('class_confidence', 0)
        is_anomaly = result.get('is_anomaly', False)
        anomaly_score = result.get('anomaly_score', 0)
        violations = result.get('violations', [])
        raw_data = result.get('raw_data', {})
        
        violations_text = ""
        for v in violations:
            violations_text += f"- {v.get('message', '')}\n"
            
        if not violations_text:
            violations_text = "No parameter violations detected.\n"
            
        params_text = ""
        for param, value in raw_data.items():
            if param != 'Sample_ID' and value is not None:
                params_text += f"- {param}: {value}\n"
                
        prompt = f"""You are an industrial wastewater treatment engineering expert. 
Analyze the following sample analysis data and supply insights.

**INDUSTRY CONTEXT:**
- Industry: {industry_context.get('name', 'Unknown')}
- Common issues in this sector: {', '.join(industry_context.get('common_issues', []))}
- Standard treatment protocols: {', '.join(industry_context.get('typical_solutions', []))}

**SAMPLE DATA:**
- Sample ID: {sample_id}
- Measured parameters:
{params_text}

**ML MODEL OUTPUTS:**
- Classification: {predicted_class} (confidence: {class_confidence:.1%})
- Is Anomaly: {is_anomaly} (anomaly score: {anomaly_score:.2f})
- Severity Threshold Level: {severity.upper()}

**VIOLATIONS DETECTED:**
{violations_text}

Provide your analysis strictly matching the requested JSON schema.
"""
        return prompt
    
    def _generate_with_templates(self, result: Dict, industry_context: Dict, severity: str) -> LLMResponse:
        """Fallback: Generate insights using templates"""
        severity_info = self.SEVERITY_TEMPLATES.get(severity, self.SEVERITY_TEMPLATES['low'])
        predicted_class = result.get('predicted_class', 'Normal')
        violation_count = len(result.get('violations', []))
        
        if severity == 'critical':
            summary = f"{severity_info['prefix']}Sample shows CRITICAL violations requiring immediate attention. "
            summary += f"Classification: {predicted_class}. {violation_count} parameter violations detected."
        elif severity == 'high':
            summary = f"{severity_info['prefix']}Sample shows significant deviations from acceptable limits. "
            summary += f"Classification: {predicted_class}. Immediate action recommended."
        elif severity == 'medium':
            summary = f"{severity_info['prefix']}Sample shows moderate deviations. "
            summary += f"Classification: {predicted_class}. Monitor and plan corrective actions."
        else:
            summary = f"{severity_info['prefix']}Sample is within acceptable limits. "
            summary += f"Classification: {predicted_class}. Continue regular monitoring."
            
        key_findings = []
        for v in result.get('violations', [])[:3]:
            key_findings.append(v.get('message', ''))
            
        if result.get('is_anomaly'):
            key_findings.append(f"Sample identified as anomalous (anomaly score: {result.get('anomaly_score', 0):.2f})")
            
        if not key_findings:
            key_findings.append("All parameters within acceptable ranges")
            key_findings.append("No unusual patterns detected")
            
        recommendations = []
        if severity in ['critical', 'high']:
            recommendations.append("⚠️ IMMEDIATE ACTION: Investigate source of violations")
            recommendations.append("📊 Review treatment process parameters")
            recommendations.append("🔬 Conduct additional sampling for verification")
        elif severity == 'medium':
            recommendations.append("📈 Increase monitoring frequency")
            recommendations.append("🔧 Optimize treatment process parameters")
        else:
            recommendations.append("✅ Maintain current treatment operations")
            recommendations.append("📊 Continue regular monitoring schedule")
            
        if industry_context.get('typical_solutions'):
            recommendations.append(f"💡 Consider solution: {industry_context['typical_solutions'][0]}")
            
        return LLMResponse(
            summary=summary,
            key_findings=key_findings,
            recommendations=recommendations,
            severity_level=severity,
            raw_response=f"Template fallback response for {severity}"
        )
    
    def generate_batch_insights(self, results: List[Dict], industry_id: str = None) -> str:
        """Generate a summary report for multiple samples"""
        if not results:
            return "No data to analyze."
            
        total = len(results)
        critical_count = sum(1 for r in results if r.get('predicted_class') == 'Critical')
        warning_count = sum(1 for r in results if r.get('predicted_class') == 'Warning')
        anomaly_count = sum(1 for r in results if r.get('is_anomaly'))
        
        if self.client and self.model_name:
            prompt = f"""Generate a brief summary report for {total} wastewater samples from {industry_id or 'industrial'} facility.
Statistics:
- Total samples: {total}
- Critical: {critical_count} ({critical_count/total*100:.1f}%)
- Warning: {warning_count} ({warning_count/total*100:.1f}%)
- Anomalies detected: {anomaly_count} ({anomaly_count/total*100:.1f}%)

Provide a 2-3 sentence executive summary of the overall water quality status.
"""
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                return response.text
            except Exception as e:
                print(f"⚠️ Batch summary failed: {e}")
                
        return f"📊 **Summary Report**\n\n- Total Samples: {total}\n- Critical: {critical_count} ({critical_count/total*100:.1f}%)\n- Warning: {warning_count} ({warning_count/total*100:.1f}%)\n- Anomalies: {anomaly_count}\n\n{critical_count} samples require immediate attention."
    
    def to_json(self, response: LLMResponse) -> Dict:
        """Convert LLMResponse to JSON-serializable dict"""
        return {
            'summary': response.summary,
            'key_findings': response.key_findings,
            'recommendations': response.recommendations,
            'severity_level': response.severity_level,
            'raw_response': response.raw_response
        }

if __name__ == "__main__":
    agent = LLMAgent()
    print("LLM Agent test module compiled successfully.")