"""
Pattern Handler for MediReason

This module provides functions for working with Clinical Reasoning Patterns
in a structured way, allowing patterns to be applied to cases and managed.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple

class PatternHandler:
    """Handles the loading, application, and management of Clinical Reasoning Patterns."""
    
    def __init__(self, patterns_dir: Optional[str] = None):
        """
        Initialize the PatternHandler.
        
        Args:
            patterns_dir: Directory containing pattern JSON files. If None, use default.
        """
        if patterns_dir is None:
            self.patterns_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "patterns")
        else:
            self.patterns_dir = patterns_dir
            
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, Any]:
        """Load all patterns from the patterns directory."""
        patterns = {}
        
        if not os.path.isdir(self.patterns_dir):
            os.makedirs(self.patterns_dir, exist_ok=True)
            return patterns
            
        for filename in os.listdir(self.patterns_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(self.patterns_dir, filename), "r") as f:
                        pattern = json.load(f)
                        if "id" in pattern:
                            patterns[pattern["id"]] = pattern
                except Exception as e:
                    print(f"Error loading pattern from {filename}: {e}")
        
        return patterns
    
    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific pattern by ID."""
        return self.patterns.get(pattern_id)
    
    def save_pattern(self, pattern: Dict[str, Any]) -> bool:
        """
        Save a pattern to the patterns directory.
        
        Args:
            pattern: Pattern data to save
            
        Returns:
            True if successful, False otherwise
        """
        if "id" not in pattern:
            return False
            
        pattern_id = pattern["id"]
        filename = os.path.join(self.patterns_dir, f"{pattern_id}.json")
        
        try:
            with open(filename, "w") as f:
                json.dump(pattern, f, indent=2)
            
            # Update the in-memory patterns
            self.patterns[pattern_id] = pattern
            return True
        except Exception as e:
            print(f"Error saving pattern {pattern_id}: {e}")
            return False
    
    def search_patterns(self, query: str) -> List[str]:
        """
        Search patterns that match the query.
        
        Args:
            query: Search terms
            
        Returns:
            List of matching pattern IDs
        """
        query_lower = query.lower()
        matching_patterns = []
        
        for pattern_id, pattern in self.patterns.items():
            # Search in trigger field
            if "trigger" in pattern and query_lower in pattern["trigger"].lower():
                matching_patterns.append(pattern_id)
                continue
                
            # Search in strategy steps
            if "strategy" in pattern:
                for step in pattern["strategy"]:
                    if query_lower in step.lower():
                        matching_patterns.append(pattern_id)
                        break
        
        return matching_patterns
    
    def find_applicable_patterns(self, case_data: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Find patterns that might apply to the given case with relevance scores.
        
        Args:
            case_data: Dictionary containing the case information
            
        Returns:
            List of tuples with (pattern_id, relevance_score)
        """
        applicable_patterns = []
        
        # Create a single string from all case data values for matching
        case_text = " ".join(str(v) for v in case_data.values() if isinstance(v, (str, int, float)))
        case_text_lower = case_text.lower()
        
        for pattern_id, pattern in self.patterns.items():
            # Skip patterns without a trigger
            if "trigger" not in pattern:
                continue
                
            # Split trigger into keywords
            trigger_keywords = set(pattern["trigger"].lower().split())
            
            # Count matching keywords
            matches = sum(1 for keyword in trigger_keywords if keyword in case_text_lower)
            
            # Calculate relevance score (0.0 to 1.0)
            relevance = matches / max(len(trigger_keywords), 1)
            
            # Add to results if above threshold
            if relevance > 0.3:  # 30% match threshold
                applicable_patterns.append((pattern_id, relevance))
        
        # Sort by relevance score (descending)
        return sorted(applicable_patterns, key=lambda x: x[1], reverse=True)
    
    def apply_pattern(self, pattern_id: str, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a specific pattern to a case.
        
        Args:
            pattern_id: ID of the pattern to apply
            case_data: Dictionary containing the case information
            
        Returns:
            Dictionary with reasoning results
        """
        pattern = self.get_pattern(pattern_id)
        if not pattern:
            return {"error": f"Pattern {pattern_id} not found"}
        
        # Initialize result structure
        result = {
            "pattern_id": pattern_id,
            "pattern_name": pattern.get("name", pattern_id),
            "reasoning_steps": [],
            "conclusion": "",
            "confidence": 0.0
        }
        
        # Handle each strategy step
        for step in pattern.get("strategy", []):
            step_result = self._handle_strategy_step(step, case_data, pattern)
            result["reasoning_steps"].append(step_result)
        
        # Generate conclusion and confidence
        conclusion, confidence = self._generate_conclusion(result["reasoning_steps"], pattern, case_data)
        result["conclusion"] = conclusion
        result["confidence"] = confidence
        
        return result
    
    def _handle_strategy_step(self, step: str, case_data: Dict[str, Any], pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a single strategy step from a pattern.
        
        Args:
            step: The strategy step to execute
            case_data: Dictionary containing the case information
            pattern: The full pattern being applied
            
        Returns:
            Dictionary with the step execution results
        """
        # In a real implementation with LLMs, this would pass to a reasoning engine
        # For now, we'll use a simplified implementation
        
        # Determine what kind of step this is
        step_lower = step.lower()
        step_type = "general"
        
        if "assess" in step_lower or "evaluate" in step_lower:
            step_type = "assessment"
        elif "identify" in step_lower or "find" in step_lower:
            step_type = "identification"
        elif "classify" in step_lower or "categorize" in step_lower:
            step_type = "classification"
        elif "compare" in step_lower:
            step_type = "comparison"
        elif "stratify" in step_lower or "risk" in step_lower:
            step_type = "risk_stratification"
        
        # Handle each step type
        if step_type == "assessment":
            return self._handle_assessment_step(step, case_data)
        elif step_type == "identification":
            return self._handle_identification_step(step, case_data)
        elif step_type == "classification":
            return self._handle_classification_step(step, case_data)
        elif step_type == "comparison":
            return self._handle_comparison_step(step, case_data)
        elif step_type == "risk_stratification":
            return self._handle_risk_stratification_step(step, case_data)
        else:
            return self._handle_general_step(step, case_data)
    
    def _handle_assessment_step(self, step: str, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an assessment step."""
        # For now, return a placeholder implementation
        return {
            "step": step,
            "observation": "Assessment of available clinical data",
            "reasoning": "Analyzed patient presentation and vital signs",
            "conclusion": "Assessment completed with key findings noted"
        }
    
    def _handle_identification_step(self, step: str, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an identification step."""
        # For now, return a placeholder implementation
        return {
            "step": step,
            "observation": "Searched for relevant clinical factors",
            "reasoning": "Considered patient history and risk factors",
            "conclusion": "Identified key clinical factors that inform diagnosis"
        }
    
    def _handle_classification_step(self, step: str, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a classification step."""
        # For now, return a placeholder implementation
        return {
            "step": step,
            "observation": "Evaluated classification criteria",
            "reasoning": "Applied standard classification frameworks",
            "conclusion": "Classified findings according to relevant guidelines"
        }
    
    def _handle_comparison_step(self, step: str, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a comparison step."""
        # For now, return a placeholder implementation
        return {
            "step": step,
            "observation": "Compared clinical presentation to known patterns",
            "reasoning": "Analyzed similarities and differences from typical presentations",
            "conclusion": "Determined pattern matches and deviations"
        }
    
    def _handle_risk_stratification_step(self, step: str, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a risk stratification step."""
        # For now, return a placeholder implementation
        return {
            "step": step,
            "observation": "Applied risk stratification tools",
            "reasoning": "Calculated risk scores based on clinical variables",
            "conclusion": "Determined risk level and implications for management"
        }
    
    def _handle_general_step(self, step: str, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a general step."""
        # For now, return a placeholder implementation
        return {
            "step": step,
            "observation": "Gathered relevant clinical information",
            "reasoning": "Applied clinical reasoning to available data",
            "conclusion": "Generated clinical insight based on reasoning"
        }
    
    def _generate_conclusion(
        self, reasoning_steps: List[Dict[str, Any]], pattern: Dict[str, Any], case_data: Dict[str, Any]
    ) -> Tuple[str, float]:
        """
        Generate a conclusion from reasoning steps.
        
        Args:
            reasoning_steps: List of reasoning step results
            pattern: The pattern being applied
            case_data: Dictionary containing the case information
            
        Returns:
            Tuple of (conclusion, confidence)
        """
        # In a real implementation, this would use an LLM to generate
        # For now, return a placeholder conclusion
        pattern_id = pattern.get("id", "Unknown")
        
        conclusion = (
            f"Based on the application of pattern {pattern_id}, the analysis "
            f"suggests further diagnostic and treatment considerations. "
            f"The pattern was applied through {len(reasoning_steps)} reasoning steps, "
            f"each providing structured analysis of the clinical situation."
        )
        
        # Calculate a simple confidence (0.0 to 1.0)
        # In a real implementation, this would use more sophisticated logic
        confidence = 0.7  # Default moderate confidence
        
        return conclusion, confidence


# Helper functions for integration with other components

def matches_trigger(case_data: Dict[str, Any], trigger: str) -> bool:
    """Check if a case matches a pattern trigger."""
    if not trigger:
        return False
        
    trigger_keywords = set(trigger.lower().split())
    case_text = " ".join(str(v) for v in case_data.values() if isinstance(v, (str, int, float)))
    case_text_lower = case_text.lower()
    
    matches = sum(1 for keyword in trigger_keywords if keyword in case_text_lower)
    return matches / max(len(trigger_keywords), 1) > 0.5  # If more than half the keywords match

def execute_reasoning_step(case_data: Dict[str, Any], step: str) -> Dict[str, Any]:
    """Execute a reasoning step for a case."""
    # This would normally invoke more complex reasoning logic
    # For demonstration, use a simplified implementation
    return {
        "step": step,
        "observation": f"Observed relevant data for: {step}",
        "reasoning": f"Applied clinical reasoning to evaluate: {step}",
        "conclusion": f"Generated conclusion based on {step}",
        "invalidates_pattern": False  # Whether this step invalidates the pattern
    }

def evaluate_cautions(case_data: Dict[str, Any], cautions: List[str]) -> List[str]:
    """Evaluate which cautions apply to the case."""
    # In a real implementation, this would apply more sophisticated logic
    # For now, just return all cautions
    return cautions

def generate_recommendation(case_data: Dict[str, Any], reasoning_steps: List[Dict[str, Any]]) -> str:
    """Generate a recommendation based on reasoning steps."""
    # In a real implementation, this would use an LLM to generate
    return "Recommendation based on clinical reasoning pattern application"


# Example handler functions for specific patterns

def handle_pattern_CRP_001(case_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler for Clinical Reasoning Pattern CRP-001 (Acute Respiratory symptoms)
    
    Args:
        case_data: Dictionary containing the case information
        
    Returns:
        Dictionary with reasoning results
    """
    # Check if this pattern applies
    if not matches_trigger(case_data, "Fever cough shortness of breath respiratory symptoms acute onset"):
        return None
    
    # Apply the strategy steps
    reasoning_steps = []
    
    # Step 1: Assess vital signs
    step1 = execute_reasoning_step(case_data, "Assess vital signs for evidence of respiratory distress")
    reasoning_steps.append(step1)
    
    # Step 2: Identify risk factors
    step2 = execute_reasoning_step(case_data, "Identify risk factors for common etiologies")
    reasoning_steps.append(step2)
    
    # Continue with remaining steps...
    
    # Check cautions
    cautions = [
        "Elderly patients and immunocompromised individuals may present atypically without fever",
        "COVID-19 may present with minimal respiratory symptoms but significant hypoxemia",
        "Heart failure can mimic or coexist with respiratory infections"
    ]
    applicable_cautions = evaluate_cautions(case_data, cautions)
    
    # Generate recommendation
    recommendation = generate_recommendation(case_data, reasoning_steps)
    
    return {
        "pattern_id": "CRP-001",
        "applicable": True,
        "reasoning_steps": reasoning_steps,
        "cautions": applicable_cautions,
        "recommendation": recommendation
    }

def handle_pattern_CRP_002(case_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler for Clinical Reasoning Pattern CRP-002 (Chest pain evaluation)
    
    Args:
        case_data: Dictionary containing the case information
        
    Returns:
        Dictionary with reasoning results
    """
    # Check if this pattern applies
    if not matches_trigger(case_data, "Chest pain atypical presentation non-cardiac cardiac differential diagnosis"):
        return None
    
    # Apply the strategy steps
    reasoning_steps = []
    
    # Step 1: Classify chest pain characteristics
    step1 = execute_reasoning_step(case_data, "Classify chest pain characteristics")
    reasoning_steps.append(step1)
    
    # Step 2: Identify red flags
    step2 = execute_reasoning_step(case_data, "Identify red flags for life-threatening causes")
    reasoning_steps.append(step2)
    
    # Continue with remaining steps...
    
    # Check cautions
    cautions = [
        "Women, elderly, and diabetic patients may present with atypical symptoms of ACS",
        "Normal ECG and initial troponin do not exclude ACS",
        "Gastroesophageal and musculoskeletal causes often diagnoses of exclusion"
    ]
    applicable_cautions = evaluate_cautions(case_data, cautions)
    
    # Generate recommendation
    recommendation = generate_recommendation(case_data, reasoning_steps)
    
    return {
        "pattern_id": "CRP-002",
        "applicable": True,
        "reasoning_steps": reasoning_steps,
        "cautions": applicable_cautions,
        "recommendation": recommendation
    } 