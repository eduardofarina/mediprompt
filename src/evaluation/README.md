# MediReason Evaluation System

The MediReason Evaluation System provides tools and metrics for assessing the performance of the clinical reasoning system. It allows for rigorous testing and continuous improvement of reasoning patterns.

## Key Components

### Evaluation Metrics

The `metrics.py` module contains the `MediReasonEvaluator` class, which provides metrics for:

- **Completeness**: Assesses whether the reasoning process includes all necessary components (reasoning steps, differential diagnosis, recommendations, etc.)
- **Pattern Application**: Evaluates whether the correct pattern was applied and with what confidence
- **Accuracy**: When reference results are available, compares the system's output to expert-created reference results

### Reference Results

The `references/` directory contains reference result JSON files for test cases. These expert-created results serve as a gold standard for comparing system outputs.

### Results Storage

The `results/` directory stores evaluation results from test runs, enabling tracking of system performance over time.

## Usage

```python
from src.evaluation.metrics import evaluate_case, MediReasonEvaluator

# Load case and system output
case_data = {...}  # Case data
system_result = {...}  # MediReason system output
reference_result = {...}  # Optional reference result

# Evaluate a single case
evaluation = evaluate_case(case_data, system_result, reference_result)

# Or use the evaluator class for more options
evaluator = MediReasonEvaluator()
evaluation = evaluator.evaluate_case_analysis(case_data, system_result, reference_result)

# Get aggregate metrics from multiple evaluations
aggregate = evaluator.aggregate_evaluations()
```

## Scripts

The `scripts/` directory contains utilities for running evaluations:

- `evaluate_patterns.py`: Evaluates multiple patterns across a set of test cases
- `run_test_case.py`: Runs and evaluates a single test case with detailed output

## Adding New Reference Results

To add a new reference result:

1. Create a JSON file named `reference_CASEXXX.json` in the `references/` directory
2. Include the following fields:
   - `primary_pattern_applied`: The ID of the pattern that should be applied
   - `reasoning_steps`: A list of reasoning steps with observations, reasoning, and conclusions
   - `differential_diagnosis`: A ranked list of possible diagnoses with supporting evidence
   - `recommendations`: A list of recommended actions
   - `conclusion`: A summary of the case analysis
   - `confidence`: A confidence score for the analysis (0.0-1.0)

## Metrics Explanation

### Completeness Metrics

- **Overall Completeness**: A weighted score (0.0-1.0) assessing the presence and quality of all necessary components
- **Reasoning Step Count**: Number of reasoning steps in the analysis
- **Average Step Completeness**: Average completeness of individual reasoning steps (0.0-1.0)

### Accuracy Metrics (when reference available)

- **Conclusion Similarity**: Similarity between system and reference conclusions (0.0-1.0)
- **Recommendation Match Ratio**: Proportion of reference recommendations matched by the system
- **Overall Accuracy**: Weighted combination of accuracy metrics (0.0-1.0) 