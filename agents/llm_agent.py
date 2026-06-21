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
from dataclasses import dataclass, field
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

class ParameterTreatment(BaseModel):
    """Per-parameter chemical treatment guidance for a single violation"""
    parameter: str = Field(description="Parameter name, e.g. 'BOD', 'pH', 'COD'")
    current_value: float = Field(description="The measured value of the parameter")
    issue: str = Field(description="One sentence describing the violation and its impact for this industry")
    chemical: str = Field(description="Specific chemical reagent with IUPAC name in parentheses, e.g. 'Lime (Ca(OH)₂)' or 'Alum (Al₂(SO₄)₃)'")
    dosage: str = Field(description="Realistic dosage range with units scaled to the measured value, e.g. '200-400 mg/L' or '5-10 kg/m³'")
    process: str = Field(description="Treatment unit operation and location, e.g. 'Chemical neutralization in equalization tank'")
    expected_outcome: str = Field(description="Expected parameter value range after treatment, e.g. 'pH raised to 6.5-8.5'")
    cost_band: str = Field(description="One of: Low, Medium, High — Low for simple pH chemicals, Medium for biological/coagulation, High for RO/advanced oxidation/ZLD")


class InsightSchema(BaseModel):
    """Pydantic schema for structured Gemini insights output"""
    summary: str = Field(description="2-3 sentence overview of the water quality status and severity level.")
    key_findings: List[str] = Field(description="List of specific parameters exceeding thresholds, anomalies, or notable patterns.")
    recommendations: List[str] = Field(description="Actionable treatment protocols, process adjustments, or monitoring instructions.")
    severity_level: str = Field(description="One of: low, medium, high, critical")
    parameter_treatments: Optional[List[ParameterTreatment]] = Field(
        default=[],
        description="Per-parameter chemical treatment guidance. Include one entry per violated parameter only. Return empty list when no violations exist."
    )


class EquipmentMaintenance(BaseModel):
    """Per-equipment maintenance prescription for a Warning/Critical unit"""
    equipment_id: str = Field(description="The equipment ID, e.g. 'pump-1'")
    equipment_name: str = Field(description="Human-readable equipment name")
    health_status: str = Field(description="'Warning' or 'Critical'")
    issue: str = Field(description="Plain-language fault description, 1-2 sentences")
    action: str = Field(description="Immediate step the maintenance engineer should take")
    components_to_check: str = Field(description="Comma-separated list of specific parts to inspect, e.g. 'radial bearings, mechanical seal, impeller clearance, coupling alignment'")
    estimated_downtime: str = Field(description="e.g. '4-6 hours' or 'No downtime - in-service inspection'")
    urgency: str = Field(description="One of: Immediate, Within 24h, Schedule within 1 week, Monitor")
    cost_band: str = Field(description="One of: Low, Medium, High")


class MaintenanceSchema(BaseModel):
    """Pydantic schema for structured Gemini maintenance output"""
    fleet_summary: str = Field(description="2-3 sentence overview of the fleet health status")
    critical_units: List[str] = Field(description="List of equipment names in Critical state")
    maintenance_actions: Optional[List[EquipmentMaintenance]] = Field(
        default=[],
        description="One entry per Warning or Critical unit. Return empty list if all units are Healthy."
    )
    overall_recommendation: str = Field(description="Single most important action for the facility manager right now")
    shutdown_risk: str = Field(description="One of: Low, Medium, High — likelihood of unplanned shutdown in next 7 days")


@dataclass
class MaintenanceResponse:
    """Structured response from LLM maintenance analysis"""
    fleet_summary: str
    critical_units: List[str]
    maintenance_actions: List[dict] = field(default_factory=list)
    overall_recommendation: str = ""
    shutdown_risk: str = "Low"
    raw_response: str = ""


@dataclass
class LLMResponse:
    """Structured response from LLM Agent"""
    summary: str
    key_findings: List[str]
    recommendations: List[str]
    severity_level: str  # 'low', 'medium', 'high', 'critical'
    raw_response: str
    parameter_treatments: List[dict] = field(default_factory=list)

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
                    print(f"[LLMAgent] Initialized with {self.model_name}")
                else:
                    print("[LLMAgent] No working model found. Using fallback templates.")
                    self.client = None
            except Exception as e:
                print(f"[LLMAgent] Failed to initialize Gemini: {e}")
                self.client = None
        else:
            print("[LLMAgent] Gemini not available or API key missing. Using fallback templates.")
        
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
            raw_response=response.text,
            parameter_treatments=data.get('parameter_treatments', [])
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
                
        industry_name = industry_context.get('name', 'Unknown')
        prompt = f"""You are an industrial wastewater treatment engineering expert specializing in {industry_name} effluent treatment.
Analyze the following sample analysis data and supply insights.

**INDUSTRY CONTEXT:**
- Industry: {industry_name}
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

**TREATMENT GUIDANCE INSTRUCTIONS:**
For each parameter listed in VIOLATIONS DETECTED above, populate the `parameter_treatments` array with one entry.
Follow these rules strictly:
- `parameter`: Use the parameter name exactly as listed (e.g. "BOD (mg/L)", "pH", "COD (mg/L)")
- `current_value`: Use the numeric value from SAMPLE DATA
- `issue`: One sentence specific to {industry_name} — explain why this value is problematic for this industry type
- `chemical`: Be scientifically precise with IUPAC names in parentheses. Choose the most appropriate for {industry_name}:
  * pH too low (acidic) → "Lime (Ca(OH)₂)" for large flows, or "Sodium Hydroxide (NaOH)" for precise control
  * pH too high (alkaline) → "Sulfuric Acid (H₂SO₄)" diluted, or "Carbon Dioxide (CO₂) sparging" for sensitive systems
  * High BOD → "Ferric Chloride (FeCl₃) as coagulant + activated sludge inoculation" or "Polyaluminium Chloride (PAC)"
  * High COD (refractory) → "Fenton's Reagent (H₂O₂/FeSO₄)" or "Ozone (O₃)" for advanced oxidation
  * High TSS → "Alum (Al₂(SO₄)₃·18H₂O)" or "Polyacrylamide (PAM) anionic flocculant"
  * High TDS → "Reverse Osmosis (RO) membranes — no chemical addition" or "Electrodialysis (ED)"
  * High Oil & Grease → "Cationic polymer (polyDADMAC) + Dissolved Air Flotation (DAF)"
- `dosage`: Provide a realistic dosage range calibrated to the actual measured value and flow scale of {industry_name}. Include units (mg/L, kg/m³, etc.)
- `process`: Name the specific unit operation and tank, e.g. "Flash mixing tank → Coagulation chamber", "Equalization tank pH correction"
- `expected_outcome`: State the target value range after treatment, e.g. "pH corrected to 6.5–8.5", "BOD reduced to < 30 mg/L"
- `cost_band`: Assign based on technology complexity:
  * "Low" — simple chemical dosing (pH neutralization, basic coagulation)
  * "Medium" — biological treatment, multi-step coagulation/flocculation, DAF
  * "High" — Reverse Osmosis, advanced oxidation (Fenton/Ozone), ZLD, membrane bioreactors
If there are NO violations, return an empty array [] for `parameter_treatments`.

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
            raw_response=f"Template fallback response for {severity}",
            parameter_treatments=[]
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
    
    # ------------------------------------------------------------------ #
    # Preventive maintenance LLM methods                                  #
    # ------------------------------------------------------------------ #

    def generate_maintenance_insights(self, fleet_result: dict, industry_id: str = None) -> MaintenanceResponse:
        """Generate Gemini-powered maintenance prescriptions for the fleet analysis result."""
        if self.client and self.model_name:
            try:
                return self._generate_maintenance_with_gemini(fleet_result, industry_id)
            except Exception as e:
                print(f"[LLMAgent] Maintenance Gemini error: {e}, using fallback")
        return self._generate_maintenance_with_templates(fleet_result)

    def _generate_maintenance_with_gemini(self, fleet_result: dict, industry_id: str = None) -> MaintenanceResponse:
        prompt = self._build_maintenance_prompt(fleet_result, industry_id)
        # Use mime-type only (no response_schema) — the nested EquipmentMaintenance array
        # with $defs causes validation errors on some Gemini model variants.
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        data = json.loads(response.text)
        actions = []
        for a in (data.get('maintenance_actions') or []):
            if isinstance(a, dict):
                actions.append(a)
            elif hasattr(a, 'model_dump'):
                actions.append(a.model_dump())
        return MaintenanceResponse(
            fleet_summary=data.get('fleet_summary', ''),
            critical_units=data.get('critical_units', []),
            maintenance_actions=actions,
            overall_recommendation=data.get('overall_recommendation', ''),
            shutdown_risk=data.get('shutdown_risk', 'Low'),
            raw_response=response.text,
        )

    def _build_maintenance_prompt(self, fleet_result: dict, industry_id: str = None) -> str:
        industry_name = (industry_id or fleet_result.get('industry_id', 'industrial')).replace('-', ' ').title()
        fleet_health   = fleet_result.get('fleet_health', 'Unknown')
        anomaly_count  = fleet_result.get('anomaly_count', 0)
        critical_count = fleet_result.get('critical_count', 0)
        warning_count  = fleet_result.get('warning_count', 0)

        # Summarise each non-Healthy unit for the prompt
        equipment_lines = ""
        for eq in fleet_result.get('equipment_results', []):
            status = eq.get('health_status', 'Healthy')
            if status in ('Warning', 'Critical'):
                params = eq.get('parameters', {})
                violations = eq.get('violations', [])
                v_text = '; '.join(v.get('message', '') for v in violations) or 'Elevated sensor readings'
                equipment_lines += (
                    f"- [{status}] {eq.get('equipment_name', '?')} "
                    f"(ID: {eq.get('equipment_id', '?')}, Location: {eq.get('location', '?')})\n"
                    f"  Parameters: sound={params.get('sound','?')} dB, "
                    f"vibration={params.get('vibration','?')} mm/s, "
                    f"temperature={params.get('temperature','?')} °C\n"
                    f"  Violations: {v_text}\n"
                    f"  Anomaly score: {eq.get('anomaly_score', 0):.3f}\n"
                )

        if not equipment_lines:
            equipment_lines = "  All units operating within normal parameters.\n"

        return f"""You are a senior rotating equipment maintenance engineer for a {industry_name} facility.
Analyse the following ETP (Effluent Treatment Plant) equipment health data and provide structured maintenance prescriptions.

**FLEET SUMMARY:**
- Overall fleet health: {fleet_health}
- Anomalous units detected: {anomaly_count}
- Critical units: {critical_count}
- Warning units: {warning_count}

**EQUIPMENT REQUIRING ATTENTION:**
{equipment_lines}

**MAINTENANCE PRESCRIPTION INSTRUCTIONS:**
For each Warning or Critical unit above, produce one entry in `maintenance_actions`:
- `equipment_id`: exact ID as shown
- `equipment_name`: exact name as shown
- `health_status`: "Warning" or "Critical"
- `issue`: 1-2 sentences — explain the fault mode implied by the sensor pattern (e.g. bearing wear, cavitation, impeller imbalance, seal failure, motor overheating)
- `action`: Specific immediate maintenance step with ISO/maintenance engineering terminology
- `components_to_check`: Comma-separated string of 3-5 specific parts, e.g. "radial bearings, mechanical seal, impeller clearance, coupling alignment"
- `estimated_downtime`: Realistic estimate, e.g. "2–4 hours" (Warning) or "8–12 hours with parts on site" (Critical)
- `urgency`:
  * Critical → "Immediate" (shutdown now if safe) or "Within 24h"
  * Warning → "Schedule within 1 week" or "Monitor"
- `cost_band`:
  * Low — lubrication service, alignment check, visual inspection
  * Medium — bearing replacement, seal replacement, impeller inspection
  * High — major overhaul, rotor replacement, pump replacement

Return empty `maintenance_actions` array if all units are Healthy.
Provide your response strictly in the requested JSON schema with no emojis or markdown.
"""

    def _generate_maintenance_with_templates(self, fleet_result: dict) -> MaintenanceResponse:
        fleet_health  = fleet_result.get('fleet_health', 'Healthy')
        critical_units = [
            eq.get('equipment_name', '?')
            for eq in fleet_result.get('equipment_results', [])
            if eq.get('health_status') == 'Critical'
        ]
        warning_units = [
            eq.get('equipment_name', '?')
            for eq in fleet_result.get('equipment_results', [])
            if eq.get('health_status') == 'Warning'
        ]
        if fleet_health == 'Critical':
            fleet_summary = (
                f"Fleet is in Critical condition — {len(critical_units)} unit(s) require immediate attention. "
                f"Unplanned shutdown risk is elevated. Isolate affected equipment and schedule urgent maintenance."
            )
            shutdown_risk = "High"
            recommendation = f"Immediately isolate and inspect: {', '.join(critical_units)}."
        elif fleet_health == 'Warning':
            fleet_summary = (
                f"Fleet shows Warning-level deviations on {len(warning_units)} unit(s). "
                f"Schedule maintenance within the next maintenance window to prevent escalation."
            )
            shutdown_risk = "Medium"
            recommendation = f"Schedule inspection within 1 week for: {', '.join(warning_units)}."
        else:
            fleet_summary = "All equipment units are operating within normal parameters. Continue regular monitoring and preventive maintenance schedule."
            shutdown_risk = "Low"
            recommendation = "Maintain current lubrication and inspection intervals. No corrective action required."

        actions = []
        for eq in fleet_result.get('equipment_results', []):
            if eq.get('health_status') in ('Warning', 'Critical'):
                actions.append({
                    "equipment_id":       eq.get('equipment_id', '?'),
                    "equipment_name":     eq.get('equipment_name', '?'),
                    "health_status":      eq.get('health_status'),
                    "issue":              "Sensor readings indicate abnormal operating conditions.",
                    "action":             "Perform a full inspection of bearings, seals, and impeller clearance.",
                    "components_to_check": "bearings, mechanical seal, impeller, coupling alignment",
                    "estimated_downtime": "4–8 hours",
                    "urgency":            "Immediate" if eq.get('health_status') == 'Critical' else "Schedule within 1 week",
                    "cost_band":          "Medium",
                })

        return MaintenanceResponse(
            fleet_summary=fleet_summary,
            critical_units=critical_units,
            maintenance_actions=actions,
            overall_recommendation=recommendation,
            shutdown_risk=shutdown_risk,
            raw_response="Template fallback",
        )

    def maintenance_to_json(self, response: MaintenanceResponse) -> Dict:
        """Convert MaintenanceResponse to JSON-serializable dict."""
        actions = []
        for a in (response.maintenance_actions or []):
            if isinstance(a, dict):
                actions.append(a)
            elif hasattr(a, 'model_dump'):
                actions.append(a.model_dump())
            else:
                actions.append(vars(a))
        return {
            'fleet_summary':          response.fleet_summary,
            'critical_units':         response.critical_units,
            'maintenance_actions':    actions,
            'overall_recommendation': response.overall_recommendation,
            'shutdown_risk':          response.shutdown_risk,
        }

    def to_json(self, response: LLMResponse) -> Dict:
        """Convert LLMResponse to JSON-serializable dict"""
        treatments = []
        for t in (response.parameter_treatments or []):
            if isinstance(t, dict):
                treatments.append(t)
            elif hasattr(t, 'model_dump'):
                treatments.append(t.model_dump())
            else:
                treatments.append(vars(t))
        return {
            'summary': response.summary,
            'key_findings': response.key_findings,
            'recommendations': response.recommendations,
            'severity_level': response.severity_level,
            'raw_response': response.raw_response,
            'parameter_treatments': treatments,
        }

if __name__ == "__main__":
    agent = LLMAgent()
    print("LLM Agent test module compiled successfully.")