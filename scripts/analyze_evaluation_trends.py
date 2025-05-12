#!/usr/bin/env python
"""
Script to analyze trends in evaluation results over time and generate a report.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional
import datetime
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from io import BytesIO
import base64

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_evaluation_results(results_dir: str) -> List[Dict[str, Any]]:
    """
    Load all evaluation results from the specified directory.
    
    Args:
        results_dir: Directory containing evaluation result JSON files
        
    Returns:
        List of evaluation result dictionaries, sorted by timestamp
    """
    results = []
    
    if not os.path.isdir(results_dir):
        print(f"Error: Results directory '{results_dir}' does not exist")
        return results
    
    for filename in os.listdir(results_dir):
        if filename.startswith("eval_") and filename.endswith(".json"):
            try:
                with open(os.path.join(results_dir, filename), "r") as f:
                    result = json.load(f)
                    results.append(result)
            except Exception as e:
                print(f"Error loading result from {filename}: {e}")
    
    # Sort by timestamp (oldest first)
    results.sort(key=lambda r: r.get("timestamp", ""))
    
    return results

def extract_pattern_metrics(evaluations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract and organize metrics by pattern ID.
    
    Args:
        evaluations: List of evaluation results
        
    Returns:
        Dictionary mapping pattern IDs to lists of metrics
    """
    pattern_metrics = {}
    
    for eval_result in evaluations:
        if "metrics" not in eval_result or "pattern_application" not in eval_result["metrics"]:
            continue
            
        pattern_data = eval_result["metrics"]["pattern_application"]
        pattern_id = pattern_data.get("pattern_id")
        
        if not pattern_id:
            continue
            
        if pattern_id not in pattern_metrics:
            pattern_metrics[pattern_id] = []
            
        # Extract relevant metrics
        metrics = {
            "timestamp": eval_result.get("timestamp", ""),
            "case_id": eval_result.get("case_id", "unknown"),
            "confidence": pattern_data.get("confidence", 0.0)
        }
        
        # Add completeness metrics if available
        if "completeness" in eval_result["metrics"]:
            completeness = eval_result["metrics"]["completeness"]
            metrics["overall_completeness"] = completeness.get("overall_completeness", 0.0)
            metrics["reasoning_step_count"] = completeness.get("reasoning_step_count", 0)
            metrics["avg_step_completeness"] = completeness.get("avg_step_completeness", 0.0)
            
        # Add accuracy metrics if available
        if "accuracy" in eval_result["metrics"]:
            accuracy = eval_result["metrics"]["accuracy"]
            metrics["overall_accuracy"] = accuracy.get("overall_accuracy", 0.0)
            metrics["conclusion_similarity"] = accuracy.get("conclusion_similarity", 0.0)
            metrics["recommendation_match_ratio"] = accuracy.get("recommendation_match_ratio", 0.0)
            
        pattern_metrics[pattern_id].append(metrics)
    
    return pattern_metrics

def generate_trend_graph(metrics: List[Dict[str, Any]], metric_name: str, title: str) -> Figure:
    """
    Generate a trend graph for a specific metric.
    
    Args:
        metrics: List of metrics dictionaries
        metric_name: Name of the metric to plot
        title: Title for the graph
        
    Returns:
        Matplotlib Figure object
    """
    # Extract data points
    data_points = [(m.get("timestamp", ""), m.get(metric_name, 0.0)) for m in metrics if metric_name in m]
    
    # Skip if no data
    if not data_points:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"No data available for {metric_name}", ha="center", va="center")
        return fig
    
    # Parse timestamps and sort by date
    try:
        data_points = [(datetime.datetime.fromisoformat(ts), val) for ts, val in data_points if ts]
        data_points.sort(key=lambda x: x[0])
    except ValueError:
        # If timestamp parsing fails, just use indices
        data_points = [(i, val) for i, (_, val) in enumerate(data_points)]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the data
    if isinstance(data_points[0][0], datetime.datetime):
        dates = [dp[0] for dp in data_points]
        values = [dp[1] for dp in data_points]
        ax.plot(dates, values, marker='o')
        fig.autofmt_xdate()
    else:
        indices = [dp[0] for dp in data_points]
        values = [dp[1] for dp in data_points]
        ax.plot(indices, values, marker='o')
    
    # Add labels and title
    ax.set_xlabel("Date" if isinstance(data_points[0][0], datetime.datetime) else "Evaluation Number")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # If metric is a ratio or score, set y limits to 0-1
    if "ratio" in metric_name or "completeness" in metric_name or "accuracy" in metric_name or "confidence" in metric_name or "similarity" in metric_name:
        ax.set_ylim(0, 1)
    
    return fig

def figure_to_base64(fig: Figure) -> str:
    """Convert a matplotlib figure to a base64 string for HTML embedding."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str

def generate_html_report(pattern_metrics: Dict[str, List[Dict[str, Any]]], output_path: str):
    """
    Generate an HTML report with trend graphs and analysis.
    
    Args:
        pattern_metrics: Dictionary mapping pattern IDs to lists of metrics
        output_path: Path to save the HTML report
    """
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>MediReason Evaluation Trends Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .pattern-section {{ margin-bottom: 40px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
        .graph {{ margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .recommendation {{ background-color: #e8f4f8; padding: 10px; border-left: 4px solid #4a86e8; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>MediReason Evaluation Trends Report</h1>
        <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total patterns evaluated: {len(pattern_metrics)}</p>
    </div>
    """
    
    # Add a section for each pattern
    for pattern_id, metrics in pattern_metrics.items():
        html_content += f"""
    <div class="pattern-section">
        <h2>Pattern: {pattern_id}</h2>
        <p>Total evaluations: {len(metrics)}</p>
        
        <h3>Performance Trends</h3>
        """
        
        # Generate graphs for key metrics
        metric_configs = [
            ("overall_completeness", f"{pattern_id} Completeness Trend"),
            ("overall_accuracy", f"{pattern_id} Accuracy Trend"),
            ("confidence", f"{pattern_id} Confidence Trend"),
            ("recommendation_match_ratio", f"{pattern_id} Recommendation Match Trend")
        ]
        
        for metric_name, title in metric_configs:
            if any(metric_name in m for m in metrics):
                fig = generate_trend_graph(metrics, metric_name, title)
                img_data = figure_to_base64(fig)
                html_content += f"""
        <div class="graph">
            <h4>{title}</h4>
            <img src="data:image/png;base64,{img_data}" alt="{title}" width="800">
        </div>
                """
        
        # Add a table with the latest metrics
        if metrics:
            latest_metrics = metrics[-1]
            html_content += """
        <h3>Latest Metrics</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            """
            
            for key, value in latest_metrics.items():
                if key not in ["timestamp", "case_id"]:
                    html_content += f"""
            <tr>
                <td>{key.replace("_", " ").title()}</td>
                <td>{value:.2f if isinstance(value, float) else value}</td>
            </tr>
                    """
            
            html_content += """
        </table>
            """
        
        # Add recommendations based on metrics
        html_content += """
        <h3>Improvement Recommendations</h3>
        """
        
        # Calculate average metrics
        avg_metrics = {}
        for metric_name in ["overall_completeness", "overall_accuracy", "confidence"]:
            values = [m.get(metric_name, 0.0) for m in metrics if metric_name in m]
            avg_metrics[metric_name] = sum(values) / len(values) if values else 0.0
        
        # Generate recommendations
        recommendations = []
        
        if avg_metrics.get("overall_completeness", 0.0) < 0.7:
            recommendations.append(f"""
        <div class="recommendation">
            <p><strong>Improve Completeness:</strong> Pattern {pattern_id} shows low completeness scores (average: {avg_metrics.get("overall_completeness", 0.0):.2f}).
            Consider adding more detailed reasoning steps and ensuring all key components (differential diagnosis, recommendations, conclusion) are thoroughly addressed.</p>
        </div>
            """)
            
        if avg_metrics.get("overall_accuracy", 0.0) < 0.7:
            recommendations.append(f"""
        <div class="recommendation">
            <p><strong>Improve Accuracy:</strong> Pattern {pattern_id} shows lower than optimal accuracy scores (average: {avg_metrics.get("overall_accuracy", 0.0):.2f}) when compared to reference results.
            Review the strategy steps and ensure they align with current clinical guidelines and expert reasoning.</p>
        </div>
            """)
            
        if avg_metrics.get("confidence", 0.0) < 0.7:
            recommendations.append(f"""
        <div class="recommendation">
            <p><strong>Address Confidence Issues:</strong> Pattern {pattern_id} exhibits low confidence scores (average: {avg_metrics.get("confidence", 0.0):.2f}).
            This may indicate the pattern is being applied to cases for which it's not optimally suited, or that the pattern lacks specificity in its strategy steps.</p>
        </div>
            """)
            
        if not recommendations:
            html_content += """
        <div class="recommendation">
            <p><strong>Pattern Performing Well:</strong> No critical issues identified for this pattern. Continue monitoring performance.</p>
        </div>
            """
        else:
            html_content += "".join(recommendations)
        
        html_content += """
    </div>
        """
    
    # Add overall summary
    html_content += """
    <div class="pattern-section">
        <h2>System-Wide Recommendations</h2>
        """
    
    # Calculate overall system metrics
    all_metrics = [m for metrics_list in pattern_metrics.values() for m in metrics_list]
    overall_avg_metrics = {}
    for metric_name in ["overall_completeness", "overall_accuracy", "confidence"]:
        values = [m.get(metric_name, 0.0) for m in all_metrics if metric_name in m]
        overall_avg_metrics[metric_name] = sum(values) / len(values) if values else 0.0
    
    html_content += f"""
        <p>Overall system completeness: {overall_avg_metrics.get("overall_completeness", 0.0):.2f}</p>
        <p>Overall system accuracy: {overall_avg_metrics.get("overall_accuracy", 0.0):.2f}</p>
        <p>Overall system confidence: {overall_avg_metrics.get("confidence", 0.0):.2f}</p>
        
        <h3>Next Steps</h3>
        <ol>
            <li>Focus on patterns with the lowest performance metrics first</li>
            <li>Regularly add new test cases to ensure comprehensive evaluation</li>
            <li>Update reference results as medical guidelines evolve</li>
            <li>Consider adding more specialized patterns for complex clinical scenarios</li>
        </ol>
    </div>
    """
    
    # Close HTML
    html_content += """
</body>
</html>
    """
    
    # Write to file
    try:
        with open(output_path, "w") as f:
            f.write(html_content)
        print(f"Report saved to {output_path}")
    except Exception as e:
        print(f"Error saving report: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation trends and generate a report")
    parser.add_argument("--results-dir", "-r", default="src/evaluation/results", help="Directory containing evaluation results")
    parser.add_argument("--output", "-o", default="evaluation_report.html", help="Path to save the HTML report")
    
    args = parser.parse_args()
    
    # Load evaluation results
    evaluations = load_evaluation_results(args.results_dir)
    
    if not evaluations:
        print("No evaluation results found")
        return
    
    print(f"Loaded {len(evaluations)} evaluation results")
    
    # Extract metrics by pattern
    pattern_metrics = extract_pattern_metrics(evaluations)
    
    if not pattern_metrics:
        print("No pattern metrics found in evaluation results")
        return
    
    print(f"Found metrics for {len(pattern_metrics)} patterns")
    
    # Generate HTML report
    generate_html_report(pattern_metrics, args.output)

if __name__ == "__main__":
    main() 