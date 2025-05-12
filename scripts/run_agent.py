#!/usr/bin/env python
"""
Script to run the MediReason agent on a clinical case.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.agents.medireason_agent import MediReasonAgent
except ImportError as e:
    print(f"Error importing MediReason agent: {e}")
    sys.exit(1)

def load_case(case_path: str) -> Dict[str, Any]:
    """Load a clinical case from a JSON file."""
    try:
        with open(case_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading case file: {e}")
        sys.exit(1)

def save_result(result: Dict[str, Any], output_path: str):
    """Save analysis result to a JSON file."""
    try:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {output_path}")
    except Exception as e:
        print(f"Error saving result: {e}")

def format_result_for_display(result: Dict[str, Any]) -> str:
    """Format the analysis result for console display."""
    output = []
    
    # Assessment
    output.append("=" * 50)
    output.append("MEDIREASON CLINICAL ANALYSIS")
    output.append("=" * 50)
    output.append(f"Assessment: {result.get('assessment', 'Not available')}")
    output.append("")
    
    # Pattern applied
    if "primary_pattern_applied" in result:
        output.append(f"Primary Pattern Applied: {result['primary_pattern_applied']}")
        output.append("")
    
    # Reasoning steps
    if "reasoning_steps" in result and result["reasoning_steps"]:
        output.append("REASONING STEPS:")
        output.append("-" * 50)
        for idx, step in enumerate(result["reasoning_steps"]):
            output.append(f"{idx+1}. {step.get('step', 'Unknown step')}")
            output.append(f"   Observation: {step.get('observation', 'None')}")
            output.append(f"   Reasoning: {step.get('reasoning', 'None')}")
            output.append(f"   Conclusion: {step.get('conclusion', 'None')}")
            output.append("")
    
    # Differential diagnosis
    if "differential_diagnosis" in result and result["differential_diagnosis"]:
        output.append("DIFFERENTIAL DIAGNOSIS:")
        output.append("-" * 50)
        for dx in result["differential_diagnosis"]:
            output.append(f"Diagnosis: {dx.get('diagnosis', 'Unknown')}")
            output.append(f"Likelihood: {dx.get('likelihood', 'Unknown')}")
            if "supporting_evidence" in dx:
                output.append("Supporting Evidence:")
                for evidence in dx["supporting_evidence"]:
                    output.append(f"- {evidence}")
            output.append("")
    
    # Recommendations
    if "recommendations" in result and result["recommendations"]:
        output.append("RECOMMENDATIONS:")
        output.append("-" * 50)
        for rec in result["recommendations"]:
            output.append(f"- {rec}")
        output.append("")
    
    # Conclusion
    if "conclusion" in result:
        output.append("CONCLUSION:")
        output.append("-" * 50)
        output.append(result["conclusion"])
        output.append("")
    
    # New pattern
    if "new_pattern_added" in result:
        output.append(f"Note: New reasoning pattern {result['new_pattern_added']} added to library.")
        output.append("")
    
    return "\n".join(output)

def main():
    """Main function to run the MediReason agent."""
    parser = argparse.ArgumentParser(description="Run MediReason agent on a clinical case")
    parser.add_argument("--case", required=True, help="Path to case JSON file")
    parser.add_argument("--output", help="Path to save output JSON (optional)")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    
    args = parser.parse_args()
    
    # Load the case
    print(f"Loading case from {args.case}...")
    case_data = load_case(args.case)
    
    # Initialize the agent
    print("Initializing MediReason agent...")
    agent = MediReasonAgent()
    
    # Analyze the case
    print("Analyzing case...")
    result = agent.analyze_case(case_data)
    
    # Display result
    print("\n" + format_result_for_display(result))
    
    # Save result if output path provided
    if args.output:
        save_result(result, args.output)
    
    print("Analysis complete.")

if __name__ == "__main__":
    main() 