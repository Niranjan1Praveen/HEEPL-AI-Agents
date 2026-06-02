"""
LLM Agent - Natural Language Generation
----------------------------------------
This agent converts analysis results into human-readable insights
using Google's Gemini API (free tier).

Uses the new google.genai package (replaces deprecated google.generativeai)
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("⚠️ Google GenAI not installed. Run: pip install google-genai")

# Optional: Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


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
    
    Uses Gemini 2.0 Flash (free) for cost-effective, fast responses
    """
    
    # List of model names to try (in order of preference)
    MODEL_NAMES = [
        "models/gemini-2.0-flash",        # Fast, good for text generation
        "models/gemini-2.5-flash",        # Newer version
        "models/gemini-2.0-flash-001",    # Specific version
        "models/gemini-flash-latest",     # Latest flash version
        "models/gemini-2.5-pro",          # More capable (if needed)
        "models/gemini-pro-latest",       # Latest pro version
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
        
        Args:
            api_key: Gemini API key (if None, tries env variable GEMINI_API_KEY)
            model_name: Gemini model to use (auto-detected if None)
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = None
        self.model_name = None
        
        if GENAI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                
                # Try to find a working model
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
            print("⚠️ Gemini not available. Using fallback templates.")
        
        # Industry-specific context (for better responses)
        self.industry_context = self._load_industry_context()
    
    def _find_working_model(self) -> Optional[str]:
        """Find a working Gemini model"""
        for model_name in self.MODEL_NAMES:
            try:
                # Try a simple test generation
                response = self.client.models.generate_content(
                    model=model_name,
                    contents="Test"
                )
                if response and response.text:
                    print(f"✅ Found working model: {model_name}")
                    return model_name
            except Exception as e:
                print(f"   Model {model_name} failed: {e}")
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
        
        Args:
            analysis_result: Dictionary from AnalysisAgent.analyze_sample()
            industry_id: Industry ID for context
        
        Returns:
            LLMResponse with summary, findings, and recommendations
        """
        industry = industry_id or analysis_result.get('industry_id', 'unknown')
        industry_context = self._get_industry_context(industry)
        
        # Determine severity
        severity = self._determine_severity(analysis_result)
        
        # Try Gemini API first
        if self.client and self.model_name:
            try:
                return self._generate_with_gemini(analysis_result, industry_context, severity)
            except Exception as e:
                print(f"⚠️ Gemini API error: {e}, using fallback")
        
        # Fallback to template-based generation
        return self._generate_with_templates(analysis_result, industry_context, severity)
    
    def _determine_severity(self, result: Dict) -> str:
        """Determine severity level from analysis results"""
        # Check classification first
        predicted_class = result.get('predicted_class', '')
        if predicted_class == 'Critical':
            return 'critical'
        elif predicted_class == 'Warning':
            return 'medium'
        
        # Check violations
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
        
        # Check anomaly score
        anomaly_score = result.get('anomaly_score', 0)
        if anomaly_score and anomaly_score < -0.6:
            return 'high'
        elif anomaly_score and anomaly_score < -0.3:
            return 'medium'
        
        return 'low'
    
    def _generate_with_gemini(self, result: Dict, industry_context: Dict, severity: str) -> LLMResponse:
        """Generate insights using Gemini API"""
        
        # Build prompt
        prompt = self._build_prompt(result, industry_context, severity)
        
        # Call Gemini
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        
        raw_text = response.text
        
        # Parse response
        return self._parse_response(raw_text, severity)
    
    def _build_prompt(self, result: Dict, industry_context: Dict, severity: str) -> str:
        """Build prompt for Gemini"""
        
        # Extract data
        sample_id = result.get('sample_id', 'Unknown')
        predicted_class = result.get('predicted_class', 'Not available')
        class_confidence = result.get('class_confidence', 0)
        is_anomaly = result.get('is_anomaly', False)
        anomaly_score = result.get('anomaly_score', 0)
        violations = result.get('violations', [])
        raw_data = result.get('raw_data', {})
        
        # Format violations
        violations_text = ""
        for v in violations:
            violations_text += f"- {v.get('message', '')}\n"
        
        if not violations_text:
            violations_text = "No parameter violations detected.\n"
        
        # Format parameters
        params_text = ""
        for param, value in raw_data.items():
            if param != 'Sample_ID' and value is not None:
                params_text += f"- {param}: {value}\n"
        
        prompt = f"""You are a wastewater treatment expert. Analyze this sample and provide insights.

**INDUSTRY CONTEXT:**
- Industry: {industry_context.get('name', 'Unknown')}
- Common issues: {', '.join(industry_context.get('common_issues', []))}
- Typical solutions: {', '.join(industry_context.get('typical_solutions', []))}

**SAMPLE DATA:**
- Sample ID: {sample_id}
- Parameters measured:
{params_text}

**ANALYSIS RESULTS:**
- Classification: {predicted_class} (confidence: {class_confidence:.1%})
- Is Anomaly: {is_anomaly} (score: {anomaly_score:.2f})
- Severity Level: {severity.upper()}

**VIOLATIONS DETECTED:**
{violations_text}

Please provide a response in the following EXACT format:

SUMMARY: (2-3 sentences summarizing the overall situation)

KEY FINDINGS:
- Finding 1
- Finding 2
- Finding 3

RECOMMENDATIONS:
- Recommendation 1
- Recommendation 2
- Recommendation 3

Keep the language professional but accessible. Focus on actionable insights.
"""
        return prompt
    
    def _parse_response(self, raw_response: str, severity: str) -> LLMResponse:
        """Parse Gemini response into structured format"""
        
        summary = ""
        key_findings = []
        recommendations = []
        
        lines = raw_response.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('SUMMARY:'):
                current_section = 'summary'
                summary = line.replace('SUMMARY:', '').strip()
            elif line.startswith('KEY FINDINGS:'):
                current_section = 'findings'
            elif line.startswith('RECOMMENDATIONS:'):
                current_section = 'recommendations'
            elif line.startswith('-') and current_section == 'findings':
                key_findings.append(line.lstrip('- '))
            elif line.startswith('-') and current_section == 'recommendations':
                recommendations.append(line.lstrip('- '))
            elif current_section == 'summary' and not summary:
                summary = line
        
        # Ensure we have at least some content
        if not summary:
            summary = f"Analysis complete. Sample classified as {severity.upper()} severity."
        
        if not key_findings:
            key_findings = ["Review the detailed analysis for specific findings."]
        
        if not recommendations:
            recommendations = ["Consult with treatment plant operator for next steps."]
        
        # Add severity icon to summary
        severity_info = self.SEVERITY_TEMPLATES.get(severity, self.SEVERITY_TEMPLATES['low'])
        summary = f"{severity_info['icon']} {summary}"
        
        return LLMResponse(
            summary=summary,
            key_findings=key_findings,
            recommendations=recommendations,
            severity_level=severity,
            raw_response=raw_response
        )
    
    def _generate_with_templates(self, result: Dict, industry_context: Dict, severity: str) -> LLMResponse:
        """Fallback: Generate insights using templates"""
        
        severity_info = self.SEVERITY_TEMPLATES.get(severity, self.SEVERITY_TEMPLATES['low'])
        
        # Build summary
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
        
        # Build key findings
        key_findings = []
        
        # Add violation findings
        for v in result.get('violations', [])[:3]:
            key_findings.append(v.get('message', ''))
        
        if result.get('is_anomaly'):
            key_findings.append(f"Sample identified as anomalous (score: {result.get('anomaly_score', 0):.2f})")
        
        if not key_findings:
            key_findings.append("All parameters within acceptable ranges")
            key_findings.append("No unusual patterns detected")
            key_findings.append("Continue regular monitoring schedule")
        
        # Build recommendations
        recommendations = []
        
        if severity in ['critical', 'high']:
            recommendations.append("⚠️ IMMEDIATE ACTION: Investigate source of violations")
            recommendations.append("📊 Review treatment process parameters")
            recommendations.append("🔬 Conduct additional sampling for verification")
        elif severity == 'medium':
            recommendations.append("📈 Increase monitoring frequency")
            recommendations.append("🔧 Optimize treatment process parameters")
            recommendations.append("📝 Document findings for trend analysis")
        else:
            recommendations.append("✅ Maintain current treatment operations")
            recommendations.append("📊 Continue regular monitoring schedule")
            recommendations.append("📈 Track trends for early warning")
        
        # Add industry-specific recommendations
        if industry_context.get('typical_solutions'):
            recommendations.append(f"💡 Consider: {industry_context['typical_solutions'][0]}")
        
        return LLMResponse(
            summary=summary,
            key_findings=key_findings[:5],
            recommendations=recommendations[:5],
            severity_level=severity,
            raw_response=f"Template-based response for {severity} severity"
        )
    
    def generate_batch_insights(self, results: List[Dict], industry_id: str = None) -> str:
        """
        Generate a summary report for multiple samples
        
        Args:
            results: List of analysis results
            industry_id: Industry ID for context
        
        Returns:
            Summary report as string
        """
        if not results:
            return "No data to analyze."
        
        # Calculate statistics
        total = len(results)
        critical_count = sum(1 for r in results if r.get('predicted_class') == 'Critical')
        warning_count = sum(1 for r in results if r.get('predicted_class') == 'Warning')
        anomaly_count = sum(1 for r in results if r.get('is_anomaly'))
        
        # Build prompt for batch summary
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
        
        # Fallback summary
        return f"📊 **Summary Report**\n\n- Total Samples: {total}\n- Critical: {critical_count} ({critical_count/total*100:.1f}%)\n- Warning: {warning_count} ({warning_count/total*100:.1f}%)\n- Anomalies: {anomaly_count}\n\n{critical_count} samples require immediate attention. Review individual sample details for specific violations and recommendations."
    
    def to_json(self, response: LLMResponse) -> Dict:
        """Convert LLMResponse to JSON-serializable dict"""
        return {
            'summary': response.summary,
            'key_findings': response.key_findings,
            'recommendations': response.recommendations,
            'severity_level': response.severity_level,
            'raw_response': response.raw_response
        }


# Quick test
if __name__ == "__main__":
    print("="*60)
    print("🤖 LLM AGENT TEST")
    print("="*60)
    
    # Initialize agent
    agent = LLMAgent()
    
    # Sample analysis result
    test_result = {
        'sample_id': 'DAIR_0001',
        'industry_id': 'dairy',
        'predicted_class': 'Critical',
        'class_confidence': 0.94,
        'is_anomaly': True,
        'anomaly_score': -0.54,
        'violations': [
            {'severity': 'critical', 'message': 'BOD: 1694.78 mg/L - exceeds critical limit (500)'},
            {'severity': 'critical', 'message': 'COD: 6918.58 mg/L - exceeds critical limit (1000)'}
        ],
        'raw_data': {
            'Sample_ID': 'DAIR_0001',
            'BOD (mg/L)': 1694.78,
            'COD (mg/L)': 6918.58,
            'TSS (mg/L)': 1021.97,
            'pH': 7.59
        }
    }
    
    print("\n📊 Analysis Result:")
    print(f"   Sample: {test_result['sample_id']}")
    print(f"   Classification: {test_result['predicted_class']}")
    print(f"   Anomaly: {test_result['is_anomaly']}")
    
    print("\n🔮 Generating Insights...\n")
    
    # Generate insights
    insights = agent.generate_insights(test_result, industry_id="dairy")
    
    print("="*60)
    print("📝 NATURAL LANGUAGE INSIGHTS")
    print("="*60)
    
    print(f"\n{insights.summary}\n")
    
    print("🔍 KEY FINDINGS:")
    for finding in insights.key_findings:
        print(f"   • {finding}")
    
    print("\n💡 RECOMMENDATIONS:")
    for rec in insights.recommendations:
        print(f"   • {rec}")
    
    print("\n✅ LLM Agent ready!")