from agno.tools import BaseTool
import requests
from typing import Dict, Any, Optional, List
import os
import json
import datetime

class GuidelineCheckTool(BaseTool):
    """
    Tool for retrieving and checking clinical guidelines for specific conditions.
    """
    
    name = "guideline_check"
    description = "Check and retrieve clinical guidelines for specific medical conditions"
    
    def __init__(self):
        super().__init__()
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "guidelines_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _run(self, condition: str, **kwargs) -> Any:
        """
        Run the guideline check with the given condition.
        
        Args:
            condition: The medical condition to look up guidelines for
            
        Returns:
            Dict containing guideline information
        """
        # Check cache first
        cache_key = self._generate_cache_key(condition)
        cached_results = self._check_cache(cache_key)
        if cached_results:
            return cached_results
            
        # If not in cache, perform the lookup
        try:
            # In a real implementation, this would call guideline databases
            # For now, we'll simulate the response
            results = self._simulate_guideline_lookup(condition)
            
            # Cache the results
            self._cache_results(cache_key, results)
            
            return results
        except Exception as e:
            return {
                "error": str(e),
                "condition": condition,
                "guidelines": []
            }
    
    def _generate_cache_key(self, condition: str) -> str:
        """Generate a cache key for the lookup."""
        return f"{condition}".replace(" ", "_").lower()
    
    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check if results are in cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    # Check if cache is older than 30 days
                    if "cache_date" in data:
                        cache_date = datetime.datetime.fromisoformat(data["cache_date"])
                        now = datetime.datetime.now()
                        if (now - cache_date).days < 30:
                            return data
            except:
                return None
        return None
    
    def _cache_results(self, cache_key: str, results: Dict[str, Any]):
        """Cache the guideline results."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        results["cache_date"] = datetime.datetime.now().isoformat()
        with open(cache_file, "w") as f:
            json.dump(results, f, indent=2)
    
    def _simulate_guideline_lookup(self, condition: str) -> Dict[str, Any]:
        """
        Simulate a guideline lookup for development/testing purposes.
        
        In a real implementation, this would query guideline databases.
        """
        # Normalize condition for matching
        condition_lower = condition.lower()
        
        # Common conditions and their guidelines
        guidelines_database = {
            "pneumonia": [
                {
                    "title": "Diagnosis and Treatment of Adults with Community-acquired Pneumonia (CAP)",
                    "organization": "American Thoracic Society / Infectious Diseases Society of America",
                    "year": 2019,
                    "summary": [
                        "Perform microbiological testing for patients with severe CAP",
                        "Use empiric antibiotic therapy based on local susceptibility patterns",
                        "Consider early discharge for patients meeting clinical stability criteria",
                        "Standard empiric regimens for inpatients: beta-lactam plus macrolide OR respiratory fluoroquinolone",
                        "Duration of therapy typically 5-7 days for patients who show clinical improvement"
                    ],
                    "url": "https://www.idsociety.org/practice-guideline/community-acquired-pneumonia-cap-in-adults/"
                }
            ],
            "copd": [
                {
                    "title": "Global Strategy for the Diagnosis, Management, and Prevention of COPD",
                    "organization": "Global Initiative for Chronic Obstructive Lung Disease (GOLD)",
                    "year": 2023,
                    "summary": [
                        "Diagnosis requires spirometry with post-bronchodilator FEV1/FVC < 0.7",
                        "COPD assessment includes symptoms, exacerbation history, and comorbidities",
                        "Pharmacologic treatment includes bronchodilators and anti-inflammatory agents based on GOLD groups",
                        "Manage exacerbations with bronchodilators, systemic corticosteroids, and antibiotics when indicated",
                        "Pulmonary rehabilitation recommended for symptomatic patients"
                    ],
                    "url": "https://goldcopd.org/gold-reports/"
                }
            ],
            "myocardial infarction": [
                {
                    "title": "2020 ESC Guidelines for the management of acute coronary syndromes in patients presenting without persistent ST-segment elevation",
                    "organization": "European Society of Cardiology",
                    "year": 2020,
                    "summary": [
                        "Use high-sensitivity cardiac troponin for diagnosis",
                        "Perform early invasive strategy for high-risk patients",
                        "Dual antiplatelet therapy recommended for 12 months in most patients",
                        "Administer high-intensity statin therapy early after admission",
                        "Long-term prevention includes lifestyle changes and medication adherence"
                    ],
                    "url": "https://www.escardio.org/Guidelines/Clinical-Practice-Guidelines/Acute-Coronary-Syndromes-ACS-in-patients-presenting-without-persistent-ST-segment-elevation"
                }
            ],
            "heart failure": [
                {
                    "title": "2022 AHA/ACC/HFSA Guideline for the Management of Heart Failure",
                    "organization": "American Heart Association / American College of Cardiology / Heart Failure Society of America",
                    "year": 2022,
                    "summary": [
                        "Classify HF by LVEF: HFrEF (≤40%), HFmrEF (41-49%), HFpEF (≥50%)",
                        "First-line therapies for HFrEF include ARNI/ACEI/ARB, beta-blockers, MRAs, and SGLT2i",
                        "Diuretics recommended for congestion and volume overload",
                        "Consider device therapy (ICD, CRT) in appropriate patients",
                        "Multidisciplinary care and cardiac rehabilitation improve outcomes"
                    ],
                    "url": "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001063"
                }
            ],
            "diabetes": [
                {
                    "title": "Standards of Medical Care in Diabetes",
                    "organization": "American Diabetes Association",
                    "year": 2023,
                    "summary": [
                        "Screen for diabetes in adults age ≥35 years with BMI ≥25 kg/m²",
                        "HbA1c target < 7.0% for most adults with diabetes",
                        "Consider GLP-1 RA or SGLT2i with established CVD, high risk, HF, or CKD",
                        "Screen for complications: retinopathy, nephropathy, neuropathy",
                        "Comprehensive approach includes lifestyle modifications, psychosocial care, and medication management"
                    ],
                    "url": "https://diabetesjournals.org/care/issue/46/Supplement_1"
                }
            ]
        }
        
        # Search for matching guidelines
        matched_guidelines = []
        for key, guidelines in guidelines_database.items():
            if key in condition_lower or condition_lower in key:
                matched_guidelines.extend(guidelines)
        
        # If no specific matches, return general clinical reasoning guidelines
        if not matched_guidelines:
            matched_guidelines = [
                {
                    "title": "Clinical Reasoning in Medical Practice",
                    "organization": "Society to Improve Diagnosis in Medicine",
                    "year": 2022,
                    "summary": [
                        "Use structured approaches to differential diagnosis",
                        "Consider common diagnoses first, then rare conditions ('common things are common')",
                        "Document clinical reasoning process in patient records",
                        "Apply metacognition to recognize cognitive biases",
                        "Implement diagnostic time-outs for complex cases"
                    ],
                    "url": "https://www.improvediagnosis.org/"
                }
            ]
        
        return {
            "condition": condition,
            "result_count": len(matched_guidelines),
            "guidelines": matched_guidelines
        } 