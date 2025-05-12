from agno import Agent, Tool
from agno.tools import BaseTool
import json
import os
from typing import Dict, List, Optional, Any

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
        self.register_tools([
            self.literature_search_tool(),
            self.guideline_check_tool(),
            self.pattern_search_tool(),
            self.pattern_update_tool()
        ])
        
    def _load_reasoning_library(self) -> Dict[str, Any]:
        """Load the clinical reasoning pattern library."""
        patterns = {}
        pattern_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "patterns")
        
        # Create the patterns directory if it doesn't exist
        os.makedirs(pattern_dir, exist_ok=True)
        
        # Load patterns from storage
        pattern_files = [f for f in os.listdir(pattern_dir) if f.endswith(".json")]
        for file in pattern_files:
            with open(os.path.join(pattern_dir, file), "r") as f:
                pattern = json.load(f)
                patterns[pattern["id"]] = pattern
        return patterns
    
    def update_reasoning_library(self, pattern_id: str, updated_pattern: Dict[str, Any]):
        """Update a reasoning pattern in the library."""
        self.reasoning_library[pattern_id] = updated_pattern
        pattern_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "patterns")
        with open(os.path.join(pattern_dir, f"{pattern_id}.json"), "w") as f:
            json.dump(updated_pattern, f, indent=2)
    
    def add_reasoning_pattern(self, pattern: Dict[str, Any]):
        """Add a new reasoning pattern to the library."""
        pattern_id = pattern["id"]
        self.reasoning_library[pattern_id] = pattern
        pattern_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "patterns")
        with open(os.path.join(pattern_dir, f"{pattern_id}.json"), "w") as f:
            json.dump(pattern, f, indent=2)
    
    def search_applicable_patterns(self, case_data: Dict[str, Any]) -> List[str]:
        """Find patterns that might apply to the given case."""
        applicable_patterns = []
        for pattern_id, pattern in self.reasoning_library.items():
            # Simple keyword matching for now - could be enhanced with embeddings
            trigger_keywords = set(pattern["trigger"].lower().split())
            case_text = " ".join([str(v) for v in case_data.values()]).lower()
            
            matches = sum(1 for keyword in trigger_keywords if keyword in case_text)
            if matches / max(len(trigger_keywords), 1) > 0.5:  # If more than half the keywords match
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
        synthesized_result = self._synthesize_results(reasoning_results, case_data)
        
        # Step 5: Check for new patterns to add to the library
        new_pattern = self._extract_new_pattern(case_data, synthesized_result)
        if new_pattern:
            self.add_reasoning_pattern(new_pattern)
            synthesized_result["new_pattern_added"] = new_pattern["id"]
            
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
            step_result = self._execute_reasoning_step(step, case_data, pattern)
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
    
    def _execute_reasoning_step(self, step: str, case_data: Dict[str, Any], pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single reasoning step."""
        # In a real implementation, this would use LLM to reason about the step
        # For now, just return a placeholder
        return {
            "step": step,
            "observation": f"Observed relevant data for: {step}",
            "reasoning": f"Reasoning about: {step} based on patient data and medical knowledge",
            "conclusion": f"Step conclusion for: {step}"
        }
    
    def _generate_conclusion(self, reasoning_steps: List[Dict[str, Any]], pattern: Dict[str, Any]) -> str:
        """Generate an overall conclusion from reasoning steps."""
        # In a real implementation, this would use LLM to synthesize conclusions
        return f"Based on the application of pattern {pattern['id']}, the most likely diagnosis is X. Recommended next steps include Y and Z."
    
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
    def literature_search_tool(self) -> BaseTool:
        """Tool for searching medical literature."""
        return Tool(
            name="literature_search",
            description="Search medical literature for information relevant to the case",
            function=self._literature_search_implementation
        )
        
    def _literature_search_implementation(self, query: str) -> str:
        """Implementation for literature search tool."""
        # In a real implementation, this would call PubMed API or similar
        return f"Found 5 relevant articles for {query}"
    
    def guideline_check_tool(self) -> BaseTool:
        """Tool for checking clinical guidelines."""
        return Tool(
            name="guideline_check",
            description="Check clinical guidelines for the management of a condition",
            function=self._guideline_check_implementation
        )
        
    def _guideline_check_implementation(self, condition: str) -> str:
        """Implementation for guideline check tool."""
        # In a real implementation, this would query a guidelines database
        return f"Latest guidelines for {condition} recommend..."
    
    def pattern_search_tool(self) -> BaseTool:
        """Tool for searching the reasoning pattern library."""
        return Tool(
            name="pattern_search",
            description="Search for a clinical reasoning pattern in the library",
            function=self._pattern_search_implementation
        )
        
    def _pattern_search_implementation(self, query: str) -> str:
        """Implementation for pattern search tool."""
        # Basic search implementation
        matching_patterns = []
        for pattern_id, pattern in self.reasoning_library.items():
            if query.lower() in pattern["trigger"].lower():
                matching_patterns.append(pattern_id)
        
        if matching_patterns:
            return f"Found patterns: {', '.join(matching_patterns)}"
        return "No matching patterns found"
    
    def pattern_update_tool(self) -> BaseTool:
        """Tool for updating reasoning patterns."""
        return Tool(
            name="pattern_update",
            description="Update or add a clinical reasoning pattern",
            function=self._pattern_update_implementation
        )
        
    def _pattern_update_implementation(self, pattern_data: str) -> str:
        """Implementation for pattern update tool."""
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