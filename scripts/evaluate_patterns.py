#!/usr/bin/env python
"""
Script to evaluate the performance of clinical reasoning patterns on test cases.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional
import datetime

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.agents.medireason_agent import MediReasonAgent
    from src.evaluation.metrics import evaluate_case, MediReasonEvaluator
    from src.integration.pattern_handler import PatternHandler
except ImportError as e:
    print(f"Error importing MediReason modules: {e}")
    sys.exit(1)

def load_cases(cases_dir: str, case_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Load test cases from the specified directory.
    
    Args:
        cases_dir: Directory containing case JSON files
        case_ids: Optional list of specific case IDs to load
        
    Returns:
        List of case data dictionaries
    """
    cases = []
    
    if not os.path.isdir(cases_dir):
        print(f"Error: Cases directory '{cases_dir}' does not exist")
        return cases
    
    for filename in os.listdir(cases_dir):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(cases_dir, filename), "r") as f:
                    case_data = json.load(f)
                    
                    # Skip if we have specific case IDs and this one isn't in the list
                    if case_ids and "case_id" in case_data and case_data["case_id"] not in case_ids:
                        continue
                        
                    cases.append(case_data)
            except Exception as e:
                print(f"Error loading case from {filename}: {e}")
    
    return cases

def run_evaluation(
    agent: MediReasonAgent, 
    cases: List[Dict[str, Any]], 
    references_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run evaluation on multiple cases.
    
    Args:
        agent: MediReasonAgent instance
        cases: List of case data dictionaries
        references_dir: Optional directory with reference results
        
    Returns:
        Dictionary with evaluation results
    """
    evaluator = MediReasonEvaluator()
    case_results = []
    
    for case in cases:
        case_id = case.get("case_id", "unknown")
        print(f"Analyzing case {case_id}...")
        
        # Get analysis from agent
        result = agent.analyze_case(case)
        
        # Load reference result if available
        reference = None
        if references_dir:
            reference_path = os.path.join(references_dir, f"reference_{case_id}.json")
            if os.path.exists(reference_path):
                try:
                    with open(reference_path, "r") as f:
                        reference = json.load(f)
                except Exception as e:
                    print(f"Error loading reference for case {case_id}: {e}")
        
        # Evaluate
        evaluation = evaluator.evaluate_case_analysis(case, result, reference)
        case_results.append({
            "case_id": case_id,
            "result": result,
            "evaluation": evaluation
        })
        
        print(f"Completed evaluation for case {case_id}")
    
    # Aggregate results
    aggregate = evaluator.aggregate_evaluations()
    
    # Format overall results
    final_results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "num_cases": len(cases),
        "aggregate_metrics": aggregate.get("metrics", {}),
        "case_results": case_results
    }
    
    return final_results

def analyze_pattern_performance(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze pattern performance to identify strengths and weaknesses.
    
    Args:
        results: Evaluation results from run_evaluation
        
    Returns:
        Dictionary with pattern performance analysis
    """
    pattern_handler = PatternHandler()
    pattern_performance = {}
    
    # Extract pattern usage from case results
    for case_result in results.get("case_results", []):
        result = case_result.get("result", {})
        pattern_id = result.get("primary_pattern_applied")
        
        if not pattern_id:
            continue
            
        # Initialize pattern stats if needed
        if pattern_id not in pattern_performance:
            pattern = pattern_handler.get_pattern(pattern_id)
            pattern_performance[pattern_id] = {
                "name": pattern_id,
                "trigger": pattern.get("trigger", "") if pattern else "",
                "usage_count": 0,
                "confidence_scores": [],
                "completeness_scores": [],
                "cases_applied": []
            }
        
        # Update pattern stats
        pattern_stats = pattern_performance[pattern_id]
        pattern_stats["usage_count"] += 1
        pattern_stats["cases_applied"].append(case_result.get("case_id", "unknown"))
        
        # Track confidence
        confidence = result.get("confidence", 0.0)
        pattern_stats["confidence_scores"].append(confidence)
        
        # Track completeness
        evaluation = case_result.get("evaluation", {})
        metrics = evaluation.get("metrics", {})
        completeness = metrics.get("completeness", {}).get("overall_completeness", 0.0)
        pattern_stats["completeness_scores"].append(completeness)
    
    # Calculate average metrics for each pattern
    for pattern_id, stats in pattern_performance.items():
        if stats["confidence_scores"]:
            stats["avg_confidence"] = sum(stats["confidence_scores"]) / len(stats["confidence_scores"])
        else:
            stats["avg_confidence"] = 0.0
            
        if stats["completeness_scores"]:
            stats["avg_completeness"] = sum(stats["completeness_scores"]) / len(stats["completeness_scores"])
        else:
            stats["avg_completeness"] = 0.0
            
        # Clean up temporary lists
        del stats["confidence_scores"]
        del stats["completeness_scores"]
    
    return {
        "pattern_performance": pattern_performance,
        "recommendations": generate_improvement_recommendations(pattern_performance)
    }

def generate_improvement_recommendations(pattern_performance: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate recommendations for improving patterns based on performance analysis.
    
    Args:
        pattern_performance: Pattern performance statistics
        
    Returns:
        List of improvement recommendations
    """
    recommendations = []
    
    for pattern_id, stats in pattern_performance.items():
        # Low usage patterns
        if stats["usage_count"] < 2:
            recommendations.append({
                "pattern_id": pattern_id,
                "issue": "Low usage",
                "recommendation": "Consider reviewing the trigger keywords to ensure they match common clinical presentations.",
                "priority": "Medium"
            })
        
        # Low confidence patterns
        if stats.get("avg_confidence", 0.0) < 0.6:
            recommendations.append({
                "pattern_id": pattern_id,
                "issue": "Low confidence",
                "recommendation": "Review the strategy steps to ensure they provide clear and specific guidance for clinical reasoning.",
                "priority": "High"
            })
        
        # Low completeness patterns
        if stats.get("avg_completeness", 0.0) < 0.7:
            recommendations.append({
                "pattern_id": pattern_id,
                "issue": "Low completeness",
                "recommendation": "Enhance the pattern to ensure it generates comprehensive differential diagnoses and specific recommendations.",
                "priority": "High"
            })
    
    return recommendations

def save_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to a JSON file."""
    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

def format_results_for_display(results: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    """Format evaluation results for console display."""
    output = []
    
    # Overall statistics
    output.append("=" * 60)
    output.append("MEDIREASON PATTERN EVALUATION RESULTS")
    output.append("=" * 60)
    output.append(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append(f"Cases analyzed: {results.get('num_cases', 0)}")
    output.append("")
    
    # Aggregate metrics
    aggregate_metrics = results.get("aggregate_metrics", {})
    output.append("AGGREGATE METRICS:")
    output.append("-" * 60)
    if "avg_completeness" in aggregate_metrics:
        output.append(f"Average completeness score: {aggregate_metrics['avg_completeness']:.2f}")
    if "avg_accuracy" in aggregate_metrics:
        output.append(f"Average accuracy score: {aggregate_metrics['avg_accuracy']:.2f}")
    output.append("")
    
    # Pattern utilization
    pattern_utilization = aggregate_metrics.get("pattern_utilization", {})
    if pattern_utilization:
        output.append("PATTERN UTILIZATION:")
        output.append("-" * 60)
        for pattern_id, count in pattern_utilization.items():
            output.append(f"{pattern_id}: {count} cases")
        output.append("")
    
    # Pattern performance
    pattern_performance = analysis.get("pattern_performance", {})
    if pattern_performance:
        output.append("PATTERN PERFORMANCE:")
        output.append("-" * 60)
        for pattern_id, stats in pattern_performance.items():
            output.append(f"Pattern: {pattern_id}")
            output.append(f"  Usage count: {stats.get('usage_count', 0)}")
            output.append(f"  Average confidence: {stats.get('avg_confidence', 0.0):.2f}")
            output.append(f"  Average completeness: {stats.get('avg_completeness', 0.0):.2f}")
            output.append(f"  Cases: {', '.join(stats.get('cases_applied', []))}")
            output.append("")
    
    # Improvement recommendations
    recommendations = analysis.get("recommendations", [])
    if recommendations:
        output.append("IMPROVEMENT RECOMMENDATIONS:")
        output.append("-" * 60)
        for rec in recommendations:
            output.append(f"Pattern: {rec.get('pattern_id', 'Unknown')}")
            output.append(f"  Issue: {rec.get('issue', 'Unknown')}")
            output.append(f"  Recommendation: {rec.get('recommendation', 'None')}")
            output.append(f"  Priority: {rec.get('priority', 'Low')}")
            output.append("")
    
    return "\n".join(output)

def main():
    """Main function to evaluate clinical reasoning patterns."""
    parser = argparse.ArgumentParser(description="Evaluate clinical reasoning patterns on test cases")
    parser.add_argument("--cases_dir", default="src/cases", help="Directory containing case JSON files")
    parser.add_argument("--case_id", action="append", help="Specific case ID(s) to evaluate (can be used multiple times)")
    parser.add_argument("--references_dir", help="Directory containing reference results for comparison")
    parser.add_argument("--output", help="Path to save output JSON (optional)")
    
    args = parser.parse_args()
    
    # Set up paths
    cases_dir = os.path.abspath(args.cases_dir)
    references_dir = os.path.abspath(args.references_dir) if args.references_dir else None
    
    # Load cases
    print(f"Loading cases from {cases_dir}...")
    cases = load_cases(cases_dir, args.case_id)
    
    if not cases:
        print("No cases found!")
        sys.exit(1)
    
    print(f"Found {len(cases)} cases")
    
    # Initialize agent
    print("Initializing MediReason agent...")
    agent = MediReasonAgent()
    
    # Run evaluation
    print("Running evaluation...")
    results = run_evaluation(agent, cases, references_dir)
    
    # Analyze pattern performance
    print("Analyzing pattern performance...")
    analysis = analyze_pattern_performance(results)
    
    # Display results
    print("\n" + format_results_for_display(results, analysis))
    
    # Save results if output path provided
    if args.output:
        save_results({**results, "analysis": analysis}, args.output)
    
    print("Evaluation complete.")

if __name__ == "__main__":
    main() 