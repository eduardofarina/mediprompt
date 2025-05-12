"""
Evaluation Metrics for MediReason

This module provides functions for evaluating the performance of the MediReason system.
"""

import json
import os
from typing import Dict, Any, List, Tuple, Optional
import datetime

class MediReasonEvaluator:
    """Evaluator for the MediReason clinical reasoning system."""
    
    def __init__(self, results_dir: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            results_dir: Directory to store evaluation results. If None, use default.
        """
        if results_dir is None:
            self.results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation", "results")
        else:
            self.results_dir = results_dir
            
        os.makedirs(self.results_dir, exist_ok=True)
        
    def evaluate_case_analysis(
        self, 
        case_data: Dict[str, Any], 
        system_result: Dict[str, Any], 
        reference_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a case analysis against reference data if available.
        
        Args:
            case_data: The original case data
            system_result: The system's analysis result
            reference_result: Optional reference analysis for comparison
            
        Returns:
            Dictionary with evaluation metrics
        """
        evaluation = {
            "case_id": case_data.get("case_id", "unknown"),
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": {}
        }
        
        # Basic completeness metrics
        evaluation["metrics"]["completeness"] = self._evaluate_completeness(system_result)
        
        # Pattern application metrics
        evaluation["metrics"]["pattern_application"] = self._evaluate_pattern_application(system_result)
        
        # Compare against reference if available
        if reference_result:
            evaluation["metrics"]["accuracy"] = self._evaluate_accuracy(system_result, reference_result)
        
        # Save evaluation results
        self._save_evaluation(evaluation)
        
        return evaluation
    
    def _evaluate_completeness(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the completeness of the analysis."""
        metrics = {}
        
        # Check for key components
        key_components = ["reasoning_steps", "conclusion", "differential_diagnosis", "recommendations"]
        for component in key_components:
            has_component = component in result and result[component]
            metrics[f"has_{component}"] = has_component
            
        # Check reasoning steps depth
        if "reasoning_steps" in result and isinstance(result["reasoning_steps"], list):
            metrics["reasoning_step_count"] = len(result["reasoning_steps"])
            
            # Check detail in reasoning steps
            step_completeness = []
            for step in result["reasoning_steps"]:
                # Count non-empty fields in step
                if isinstance(step, dict):
                    fields = ["step", "observation", "reasoning", "conclusion"]
                    non_empty = sum(1 for field in fields if field in step and step[field])
                    step_completeness.append(non_empty / len(fields))
            
            if step_completeness:
                metrics["avg_step_completeness"] = sum(step_completeness) / len(step_completeness)
            else:
                metrics["avg_step_completeness"] = 0.0
        else:
            metrics["reasoning_step_count"] = 0
            metrics["avg_step_completeness"] = 0.0
            
        # Overall completeness score (weighted average)
        completeness_score = 0.0
        weights = {
            "has_reasoning_steps": 0.3,
            "has_conclusion": 0.2,
            "has_differential_diagnosis": 0.2,
            "has_recommendations": 0.2,
            "avg_step_completeness": 0.1
        }
        
        for metric, weight in weights.items():
            if metric in metrics:
                if isinstance(metrics[metric], bool):
                    completeness_score += float(metrics[metric]) * weight
                else:
                    completeness_score += metrics[metric] * weight
        
        metrics["overall_completeness"] = completeness_score
        
        return metrics
    
    def _evaluate_pattern_application(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the pattern application in the analysis."""
        metrics = {}
        
        # Check for pattern application
        if "primary_pattern_applied" in result:
            metrics["has_pattern"] = True
            metrics["pattern_id"] = result["primary_pattern_applied"]
        else:
            metrics["has_pattern"] = False
            
        # Check if pattern application seems appropriate
        # In a real implementation, this would use more sophisticated logic
        metrics["appropriate_pattern"] = metrics.get("has_pattern", False)
        
        # Check confidence
        metrics["confidence"] = result.get("confidence", 0.0)
        
        return metrics
    
    def _evaluate_accuracy(self, system_result: Dict[str, Any], reference_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the accuracy of the analysis by comparing to reference.
        
        In a real implementation, this would use more sophisticated comparison logic.
        """
        metrics = {}
        
        # Compare conclusions (in a real implementation, this would use semantic similarity)
        system_conclusion = system_result.get("conclusion", "").lower()
        reference_conclusion = reference_result.get("conclusion", "").lower()
        
        if system_conclusion and reference_conclusion:
            # Very basic similarity: shared words divided by total words
            system_words = set(system_conclusion.split())
            reference_words = set(reference_conclusion.split())
            
            shared_words = len(system_words.intersection(reference_words))
            total_words = len(system_words.union(reference_words))
            
            if total_words > 0:
                metrics["conclusion_similarity"] = shared_words / total_words
            else:
                metrics["conclusion_similarity"] = 0.0
        else:
            metrics["conclusion_similarity"] = 0.0
            
        # Compare recommended actions
        # In a real implementation, this would use more sophisticated matching
        system_recommendations = system_result.get("recommendations", [])
        reference_recommendations = reference_result.get("recommendations", [])
        
        if system_recommendations and reference_recommendations:
            match_count = 0
            for sys_rec in system_recommendations:
                for ref_rec in reference_recommendations:
                    # Basic substring matching
                    if isinstance(sys_rec, str) and isinstance(ref_rec, str):
                        if sys_rec.lower() in ref_rec.lower() or ref_rec.lower() in sys_rec.lower():
                            match_count += 1
                            break
            
            metrics["recommendation_match_ratio"] = match_count / len(reference_recommendations) if reference_recommendations else 0.0
        else:
            metrics["recommendation_match_ratio"] = 0.0
            
        # Overall accuracy score
        metrics["overall_accuracy"] = (
            metrics.get("conclusion_similarity", 0.0) * 0.6 + 
            metrics.get("recommendation_match_ratio", 0.0) * 0.4
        )
        
        return metrics
    
    def _save_evaluation(self, evaluation: Dict[str, Any]):
        """Save the evaluation results to a file."""
        case_id = evaluation.get("case_id", "unknown")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{case_id}_{timestamp}.json"
        
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(evaluation, f, indent=2)
            
    def aggregate_evaluations(self, latest_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Aggregate evaluations to generate summary metrics.
        
        Args:
            latest_n: If provided, only consider the latest n evaluations
            
        Returns:
            Dictionary with aggregate metrics
        """
        # Load evaluation files
        eval_files = [f for f in os.listdir(self.results_dir) if f.startswith("eval_") and f.endswith(".json")]
        
        # Sort by timestamp (most recent first)
        eval_files.sort(reverse=True)
        
        # Limit to latest_n if specified
        if latest_n is not None:
            eval_files = eval_files[:latest_n]
        
        if not eval_files:
            return {"error": "No evaluation files found"}
        
        # Load and aggregate
        evaluations = []
        for file in eval_files:
            try:
                with open(os.path.join(self.results_dir, file), "r") as f:
                    evaluations.append(json.load(f))
            except Exception as e:
                print(f"Error loading evaluation file {file}: {e}")
        
        # Calculate aggregate metrics
        aggregate = {
            "count": len(evaluations),
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": {}
        }
        
        # Skip if no valid evaluations
        if not evaluations:
            return {"error": "No valid evaluation files found"}
        
        # Aggregate completeness metrics
        completeness_scores = [
            e["metrics"]["completeness"]["overall_completeness"] 
            for e in evaluations 
            if "metrics" in e and "completeness" in e["metrics"] and "overall_completeness" in e["metrics"]["completeness"]
        ]
        
        if completeness_scores:
            aggregate["metrics"]["avg_completeness"] = sum(completeness_scores) / len(completeness_scores)
        
        # Aggregate accuracy metrics if available
        accuracy_scores = [
            e["metrics"]["accuracy"]["overall_accuracy"] 
            for e in evaluations 
            if "metrics" in e and "accuracy" in e["metrics"] and "overall_accuracy" in e["metrics"]["accuracy"]
        ]
        
        if accuracy_scores:
            aggregate["metrics"]["avg_accuracy"] = sum(accuracy_scores) / len(accuracy_scores)
        
        # Pattern utilization
        pattern_counts = {}
        for e in evaluations:
            if "metrics" in e and "pattern_application" in e["metrics"] and "pattern_id" in e["metrics"]["pattern_application"]:
                pattern_id = e["metrics"]["pattern_application"]["pattern_id"]
                if pattern_id in pattern_counts:
                    pattern_counts[pattern_id] += 1
                else:
                    pattern_counts[pattern_id] = 1
        
        aggregate["metrics"]["pattern_utilization"] = pattern_counts
        
        return aggregate


def evaluate_case(
    case_data: Dict[str, Any], 
    result: Dict[str, Any], 
    reference: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluate a single case analysis.
    
    Args:
        case_data: The original case data
        result: The system's analysis result
        reference: Optional reference analysis for comparison
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluator = MediReasonEvaluator()
    return evaluator.evaluate_case_analysis(case_data, result, reference) 