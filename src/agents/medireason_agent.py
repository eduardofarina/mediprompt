from agno.agent import Agent
from agno.tools import tool
import json
import os
from typing import Dict, List, Optional, Any
from .agent_differential import DifferentialDiagnosisAgent

class MediReasonAgent(Agent):
    """
    MediReason is an advanced clinical reasoning system built with Python and the Agno agent framework.
    
    This agent implements a Clinical Reasoning Learning pattern based on Karpathy's 
    system prompt learning concept. It not only answers medical queries but also
    documents and improves its own reasoning processes.
    """
    
    def __init__(self):
        super().__init__()
        self.reasoning_library = self._load_reasoning_library()
        self.diff_agent = DifferentialDiagnosisAgent()
    
    def _load_reasoning_library(self) -> Dict[str, Any]:
        """Load the clinical reasoning pattern library."""
        patterns = {}
        # Load patterns from storage
        pattern_files = [f for f in os.listdir("src/patterns") if f.endswith(".json")]
        for file in pattern_files:
            with open(f"src/patterns/{file}", "r") as f:
                pattern = json.load(f)
                patterns[pattern["id"]] = pattern
        return patterns
    
    def update_reasoning_library(self, pattern_id: str, updated_pattern: Dict[str, Any]):
        """Update a reasoning pattern in the library."""
        self.reasoning_library[pattern_id] = updated_pattern
        with open(f"src/patterns/{pattern_id}.json", "w") as f:
            json.dump(updated_pattern, f, indent=2)
    
    def add_reasoning_pattern(self, pattern: Dict[str, Any]):
        """Add a new reasoning pattern to the library."""
        pattern_id = pattern["id"]
        self.reasoning_library[pattern_id] = pattern
        with open(f"src/patterns/{pattern_id}.json", "w") as f:
            json.dump(pattern, f, indent=2)
    
    def search_applicable_patterns(self, case_data: Dict[str, Any]) -> List[str]:
        """Find patterns that might apply to the given case."""
        applicable_patterns = []
        for pattern_id, pattern in self.reasoning_library.items():
            # Simple keyword matching for now - could be enhanced with embeddings
            trigger_keywords = set(pattern["trigger"].lower().split())
            case_text = " ".join([str(v) for v in case_data.values()]).lower()
            
            matches = sum(1 for keyword in trigger_keywords if keyword in case_text)
            if matches / len(trigger_keywords) > 0.5:  # If more than half the keywords match
                applicable_patterns.append(pattern_id)
                
        return applicable_patterns
        
    def analyze_case(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a clinical case using the MediReason system.
        
        Args:
            case_data: Dictionary containing the clinical case information
                Should include keys like: presenting_symptoms, patient_demographics,
                vital_signs, physical_exam, past_medical_history, etc.
                
        Returns:
            Dictionary with the analysis results
        """
        # Step 1: Initial assessment
        assessment = self._perform_initial_assessment(case_data)
        
        # Step 2: Find applicable reasoning patterns
        applicable_patterns = self.search_applicable_patterns(case_data)
        
        # Step 3: Apply the patterns to the case
        reasoning_results = []
        for pattern_id in applicable_patterns:
            result = self.apply_pattern(pattern_id, case_data)
            reasoning_results.append(result)
            
        # Step 4: Synthesize results and generate recommendations
        if not reasoning_results:
            # Use DifferentialDiagnosisAgent if no patterns found
            diff_result = self.diff_agent.get_differential_and_next_steps(case_data)
            # If one diagnosis is clearly most likely, focus on it
            differentials = diff_result.get("differential", [])
            if differentials and len(differentials) == 1:
                main_dx = differentials[0]
                return {
                    "assessment": f"High confidence in single diagnosis: {main_dx.get('diagnosis')}",
                    "differential_diagnosis": [main_dx],
                    "recommendations": diff_result.get("next_steps", []),
                    "ask_user": diff_result.get("ask_user", ""),
                    "confidence": 0.95
                }
            # If one diagnosis is much more likely than others, focus on it
            if differentials and len(differentials) > 1:
                high_likelihood = [d for d in differentials if d.get("likelihood", "").lower() == "high"]
                if len(high_likelihood) == 1:
                    return {
                        "assessment": f"High confidence in primary diagnosis: {high_likelihood[0].get('diagnosis')}",
                        "differential_diagnosis": [high_likelihood[0]],
                        "recommendations": diff_result.get("next_steps", []),
                        "ask_user": diff_result.get("ask_user", ""),
                        "confidence": 0.9
                    }
            # Otherwise, provide a broader differential
            return {
                "assessment": "Differential diagnosis generated by agent.",
                "differential_diagnosis": differentials,
                "recommendations": diff_result.get("next_steps", []),
                "ask_user": diff_result.get("ask_user", ""),
                "confidence": 0.5
            }
        synthesized_result = self._synthesize_results(reasoning_results, case_data)
        # If the synthesized result is too broad or not actionable, use diff agent
        if not synthesized_result.get("differential_diagnosis") or len(synthesized_result.get("differential_diagnosis", [])) < 2:
            diff_result = self.diff_agent.get_differential_and_next_steps(case_data)
            synthesized_result["differential_diagnosis"] = diff_result.get("differential", [])
            synthesized_result["recommendations"] = diff_result.get("next_steps", [])
            synthesized_result["ask_user"] = diff_result.get("ask_user", "")
        return synthesized_result
    
    def apply_pattern(self, pattern_id: str, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a specific reasoning pattern to a case."""
        pattern = self.reasoning_library.get(pattern_id)
        if not pattern:
            return {"error": f"Pattern {pattern_id} not found"}
            
        result = {
            "pattern_applied": pattern_id,
            "reasoning_steps": [],
            "conclusion": "",
            "confidence": 0.0
        }
        
        # Apply each step in the pattern
        for step in pattern["strategy"]:
            step_result = self._execute_reasoning_step(step, case_data)
            result["reasoning_steps"].append(step_result)
            
        # Generate conclusion
        result["conclusion"] = self._generate_conclusion(result["reasoning_steps"], pattern)
        result["confidence"] = self._calculate_confidence(result["reasoning_steps"])
        
        return result
    
    def _perform_initial_assessment(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform initial assessment of the case.
        
        In a real implementation, this would likely call an LLM to analyze the case.
        """
        # Extract core clinical findings
        core_findings = self._extract_core_findings(case_data)
        
        # Generate initial differential diagnosis
        differential = self._generate_differential(core_findings, case_data)
        
        return {
            "core_findings": core_findings,
            "initial_differential": differential
        }
    
    def _extract_core_findings(self, case_data: Dict[str, Any]) -> List[str]:
        """Extract core clinical findings from case data."""
        # In a real implementation, this would use an LLM or structured parsing
        # For now, we'll use a simple placeholder implementation
        findings = []
        
        if "presenting_symptoms" in case_data:
            symptoms = case_data["presenting_symptoms"].split(",")
            findings.extend([s.strip() for s in symptoms])
            
        if "vital_signs" in case_data:
            # Extract abnormal vitals
            vitals = case_data["vital_signs"]
            if "fever" in vitals.lower() or "temperature" in vitals.lower() and "38" in vitals:
                findings.append("Fever")
            if "tachycardia" in vitals.lower() or "HR" in vitals and any(hr in vitals for hr in ["100", "110", "120", "130"]):
                findings.append("Tachycardia")
                
        if "physical_exam" in case_data:
            # Extract key physical exam findings
            exam = case_data["physical_exam"].lower()
            if "rash" in exam:
                findings.append("Rash")
            if "murmur" in exam:
                findings.append("Heart murmur")
            if "breath sounds" in exam and ("decreased" in exam or "absent" in exam):
                findings.append("Decreased breath sounds")
                
        return findings
    
    def _generate_differential(self, findings: List[str], case_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate initial differential diagnosis."""
        # In a real implementation, this would use an LLM or medical knowledge base
        # For now, using a simple placeholder implementation
        differential = []
        
        # Very basic pattern matching
        if "Fever" in findings:
            if "cough" in str(findings).lower():
                differential.append({
                    "diagnosis": "Pneumonia",
                    "likelihood": "High",
                    "key_supporting_findings": ["Fever", "Cough", "Decreased breath sounds"],
                    "next_steps": ["Chest X-ray", "CBC", "Blood cultures"]
                })
                differential.append({
                    "diagnosis": "Bronchitis",
                    "likelihood": "Medium",
                    "key_supporting_findings": ["Cough"],
                    "next_steps": ["Chest X-ray", "Symptom management"]
                })
            
            if "rash" in str(findings).lower():
                differential.append({
                    "diagnosis": "Viral exanthem",
                    "likelihood": "Medium",
                    "key_supporting_findings": ["Fever", "Rash"],
                    "next_steps": ["CBC", "Viral panel"]
                })
                
        if "Heart murmur" in findings:
            differential.append({
                "diagnosis": "Endocarditis",
                "likelihood": "Medium",
                "key_supporting_findings": ["Heart murmur", "Fever"],
                "next_steps": ["Blood cultures", "Echocardiogram"]
            })
            
        # Add a default diagnosis if differential is empty
        if not differential:
            differential.append({
                "diagnosis": "Undetermined",
                "likelihood": "Unknown",
                "key_supporting_findings": [],
                "next_steps": ["Complete history and physical", "Basic laboratory workup"]
            })
            
        return differential
    
    def _execute_reasoning_step(self, step: str, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single reasoning step."""
        # In a real implementation, this would use LLM to reason about the step
        # For now, just return a placeholder
        return {
            "step": step,
            "observation": f"Observed relevant data for: {step}",
            "reasoning": f"Reasoning about: {step}",
            "conclusion": f"Step conclusion for: {step}"
        }
    
    def _generate_conclusion(self, reasoning_steps: List[Dict[str, Any]], pattern: Dict[str, Any]) -> str:
        """Generate an overall conclusion from reasoning steps."""
        # In a real implementation, this would use LLM to synthesize conclusions
        return f"Conclusion based on pattern {pattern['id']}"
    
    def _calculate_confidence(self, reasoning_steps: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the conclusion."""
        # In a real implementation, this would use more sophisticated logic
        return 0.8  # Placeholder
    
    def _synthesize_results(self, reasoning_results: List[Dict[str, Any]], case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from multiple reasoning patterns."""
        # In a real implementation, this would use LLM to synthesize results
        
        if not reasoning_results:
            return {
                "assessment": "No applicable reasoning patterns found",
                "differential_diagnosis": [],
                "recommendations": ["Conduct standard workup", "Monitor patient condition"],
                "confidence": 0.0
            }
        
        # For now, just use the highest confidence result
        highest_confidence_result = max(reasoning_results, key=lambda x: x.get("confidence", 0))
        
        return {
            "assessment": f"Analysis based on {len(reasoning_results)} reasoning patterns",
            "primary_pattern_applied": highest_confidence_result.get("pattern_applied", "Unknown"),
            "reasoning_steps": highest_confidence_result.get("reasoning_steps", []),
            "conclusion": highest_confidence_result.get("conclusion", "No conclusion available"),
            "confidence": highest_confidence_result.get("confidence", 0.0),
            "differential_diagnosis": self._extract_differential_from_results(reasoning_results),
            "recommendations": self._extract_recommendations_from_results(reasoning_results)
        }
    
    def _extract_differential_from_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract differential diagnoses from reasoning results."""
        # In a real implementation, this would consolidate diagnoses across patterns
        # For now, just return a placeholder
        return [
            {
                "diagnosis": "Diagnosis X",
                "likelihood": "High",
                "supporting_evidence": ["Evidence 1", "Evidence 2"],
                "patterns_supporting": [result.get("pattern_applied") for result in results if result.get("confidence", 0) > 0.7]
            }
        ]
    
    def _extract_recommendations_from_results(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract recommendations from reasoning results."""
        # In a real implementation, this would consolidate recommendations across patterns
        # For now, just return placeholders
        return ["Recommendation 1", "Recommendation 2"]
    
    def _extract_new_pattern(self, case_data: Dict[str, Any], result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract a new clinical reasoning pattern from the case analysis.
        
        In a real implementation, this would use an LLM to identify novel reasoning patterns.
        For now, return None to indicate no new pattern was identified.
        """
        return None
    
    # Define tools this agent can use
    @tool(show_result=True)
    def literature_search_tool(self, query: str) -> str:
        """Search medical literature for information relevant to the case"""
        # Implementation for literature search tool
        return f"Found 5 relevant articles for {query}"
    
    @tool(show_result=True)
    def guideline_check_tool(self, condition: str) -> str:
        """Check clinical guidelines for the management of a condition"""
        # Implementation for guideline check tool
        return f"Latest guidelines for {condition} recommend..."
    
    @tool(show_result=True)
    def pattern_search_tool(self, query: str) -> str:
        """Search for a clinical reasoning pattern in the library"""
        # Basic search implementation
        matching_patterns = []
        for pattern_id, pattern in self.reasoning_library.items():
            if query.lower() in pattern["trigger"].lower():
                matching_patterns.append(pattern_id)
        
        if matching_patterns:
            return f"Found patterns: {', '.join(matching_patterns)}"
        return "No matching patterns found"
    
    @tool(show_result=True)
    def pattern_update_tool(self, pattern_data: str) -> str:
        """Update or add a clinical reasoning pattern"""
        try:
            pattern = json.loads(pattern_data)
            pattern_id = pattern.get("id")
            
            if not pattern_id:
                return "Error: Pattern must have an ID"
                
            if pattern_id in self.reasoning_library:
                self.update_reasoning_library(pattern_id, pattern)
                return f"Updated pattern {pattern_id}"
            else:
                self.add_reasoning_pattern(pattern)
                return f"Added new pattern {pattern_id}"
                
        except json.JSONDecodeError:
            return "Error: Invalid JSON data"
        except Exception as e:
            return f"Error updating pattern: {str(e)}"

# Add a main function to make the script runnable
def main():
    # Create the agent
    agent = MediReasonAgent()
    
    # Load a sample case
    sample_case_path = "src/cases/sample_chest_pain_case.json"
    if os.path.exists(sample_case_path):
        with open(sample_case_path, "r") as f:
            case_data = json.load(f)
        
        print(f"Loaded sample case: {case_data.get('id', 'Unknown')}")
        
        # Find applicable patterns
        applicable_patterns = agent.search_applicable_patterns(case_data)
        print(f"Found {len(applicable_patterns)} applicable patterns: {applicable_patterns}")
        
        # Apply the first applicable pattern if any
        if applicable_patterns:
            result = agent.apply_pattern(applicable_patterns[0], case_data)
            print("\nApplied pattern result:")
            print(json.dumps(result, indent=2))
        else:
            print("No applicable patterns found for this case.")
    else:
        print(f"Sample case file not found: {sample_case_path}")
        print("Please create a sample case file or specify a different path.")

if __name__ == "__main__":
    main() 