#!/usr/bin/env python
"""
Script to run a single test case through the MediReason system.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.agents.medireason_agent import MediReasonAgent
    from src.evaluation.metrics import evaluate_case
except ImportError as e:
    print(f"Error importing MediReason modules: {e}")
    sys.exit(1)

def load_case(case_file: str) -> Dict[str, Any]:
    """
    Load a test case from a JSON file.
    
    Args:
        case_file: Path to the case JSON file
        
    Returns:
        Case data as a dictionary
    """
    try:
        with open(case_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading case file {case_file}: {e}")
        sys.exit(1)

def load_reference(reference_file: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Load a reference result from a JSON file if specified.
    
    Args:
        reference_file: Path to the reference result JSON file, or None
        
    Returns:
        Reference data as a dictionary, or None if no file specified
    """
    if not reference_file:
        return None
        
    try:
        with open(reference_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load reference file {reference_file}: {e}")
        return None

def save_result(result: Dict[str, Any], output_file: str):
    """
    Save the analysis result to a JSON file.
    
    Args:
        result: Analysis result to save
        output_file: Path to the output file
    """
    try:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {output_file}")
    except Exception as e:
        print(f"Error saving result: {e}")

def print_evaluation_summary(evaluation: Dict[str, Any]):
    """
    Print a summary of the evaluation metrics.
    
    Args:
        evaluation: Evaluation results dictionary
    """
    print("\n=== EVALUATION SUMMARY ===")
    
    # Print case ID
    print(f"Case ID: {evaluation.get('case_id', 'unknown')}")
    
    # Print completeness metrics
    if "metrics" in evaluation and "completeness" in evaluation["metrics"]:
        completeness = evaluation["metrics"]["completeness"]
        print(f"\nCompleteness Score: {completeness.get('overall_completeness', 0.0):.2f}")
        print(f"Reasoning Steps: {completeness.get('reasoning_step_count', 0)}")
        print(f"Step Detail Level: {completeness.get('avg_step_completeness', 0.0):.2f}")
    
    # Print pattern application metrics
    if "metrics" in evaluation and "pattern_application" in evaluation["metrics"]:
        pattern = evaluation["metrics"]["pattern_application"]
        if pattern.get("has_pattern", False):
            print(f"\nPattern Applied: {pattern.get('pattern_id', 'unknown')}")
            print(f"Confidence: {pattern.get('confidence', 0.0):.2f}")
        else:
            print("\nNo pattern was applied")
    
    # Print accuracy metrics if available
    if "metrics" in evaluation and "accuracy" in evaluation["metrics"]:
        accuracy = evaluation["metrics"]["accuracy"]
        print(f"\nAccuracy Score: {accuracy.get('overall_accuracy', 0.0):.2f}")
        print(f"Conclusion Similarity: {accuracy.get('conclusion_similarity', 0.0):.2f}")
        print(f"Recommendation Match: {accuracy.get('recommendation_match_ratio', 0.0):.2f}")
    
    print("\n=========================")

def print_analysis_summary(result: Dict[str, Any]):
    """
    Print a summary of the analysis result.
    
    Args:
        result: Analysis result dictionary
    """
    print("\n=== ANALYSIS SUMMARY ===")
    
    # Print pattern used
    if "primary_pattern_applied" in result:
        print(f"Pattern Applied: {result['primary_pattern_applied']}")
    
    # Print differential diagnoses
    if "differential_diagnosis" in result and isinstance(result["differential_diagnosis"], list):
        print("\nDifferential Diagnosis:")
        for i, dx in enumerate(result["differential_diagnosis"]):
            if isinstance(dx, dict):
                print(f"  {i+1}. {dx.get('diagnosis', 'Unknown')} - {dx.get('likelihood', 'Unknown')}")
            else:
                print(f"  {i+1}. {dx}")
    
    # Print recommendations
    if "recommendations" in result and isinstance(result["recommendations"], list):
        print("\nRecommendations:")
        for i, rec in enumerate(result["recommendations"]):
            print(f"  {i+1}. {rec}")
    
    # Print conclusion
    if "conclusion" in result:
        print("\nConclusion:")
        print(f"  {result.get('conclusion', 'No conclusion provided')}")
    
    # Print confidence
    if "confidence" in result:
        print(f"\nConfidence: {result.get('confidence', 0.0):.2f}")
    
    print("\n=======================")

def main():
    parser = argparse.ArgumentParser(description="Run a test case through the MediReason system")
    parser.add_argument("case_file", help="Path to the case JSON file")
    parser.add_argument("--reference", "-r", help="Path to reference result JSON file (optional)")
    parser.add_argument("--output", "-o", help="Path to save the analysis result (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed output")
    
    args = parser.parse_args()
    
    # Load case and reference
    case_data = load_case(args.case_file)
    reference_data = load_reference(args.reference)
    
    # Create agent and analyze case
    agent = MediReasonAgent()
    result = agent.analyze_case(case_data)
    
    # Save result if output file specified
    if args.output:
        save_result(result, args.output)
    
    # Print summary
    print_analysis_summary(result)
    
    # Evaluate and print evaluation
    evaluation = evaluate_case(case_data, result, reference_data)
    print_evaluation_summary(evaluation)
    
    # Print details if verbose
    if args.verbose:
        print("\n=== DETAILED REASONING STEPS ===")
        if "reasoning_steps" in result and isinstance(result["reasoning_steps"], list):
            for i, step in enumerate(result["reasoning_steps"]):
                if isinstance(step, dict):
                    print(f"\nStep {i+1}: {step.get('step', 'Unknown')}")
                    print(f"Observation: {step.get('observation', 'None')}")
                    print(f"Reasoning: {step.get('reasoning', 'None')}")
                    print(f"Conclusion: {step.get('conclusion', 'None')}")
                else:
                    print(f"\nStep {i+1}: {step}")
        else:
            print("No detailed reasoning steps available")
    
if __name__ == "__main__":
    main() 