# MediReason Evaluation Guide

This guide provides instructions for evaluating the MediReason clinical reasoning system and interpreting the results.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- MediReason codebase with dependencies installed

### Evaluation Workflow

1. **Prepare Test Cases**: Create or select clinical cases in JSON format
2. **Create Reference Results**: Generate expert reference results for comparison
3. **Run Evaluations**: Execute evaluation scripts to analyze system performance
4. **Review Results**: Analyze metrics and identify improvement opportunities
5. **Update Patterns**: Refine clinical reasoning patterns based on evaluation results

## Running Evaluations

### Single Case Evaluation

To evaluate a single case:

```bash
python scripts/run_test_case.py src/cases/sample_pneumonia_case.json --reference src/evaluation/references/reference_CASE001.json --verbose
```

Command options:
- The first argument is the path to the case file
- `--reference` (or `-r`): Path to the reference result file (optional)
- `--output` (or `-o`): Path to save the analysis result (optional)
- `--verbose` (or `-v`): Print detailed reasoning steps

### Batch Evaluation

To evaluate multiple cases against a pattern library:

```bash
python scripts/evaluate_patterns.py --cases src/cases --references src/evaluation/references --output evaluation_results.json
```

Command options:
- `--cases` (or `-c`): Directory containing case files
- `--references` (or `-r`): Directory containing reference results
- `--output` (or `-o`): Output file for evaluation results
- `--pattern-ids` (or `-p`): Comma-separated list of pattern IDs to evaluate (optional)
- `--case-ids` (or `-i`): Comma-separated list of case IDs to evaluate (optional)

## Understanding Evaluation Metrics

### Completeness Metrics

These metrics assess how thorough and comprehensive the system's analysis is:

| Metric | Description | Good Score |
|--------|-------------|------------|
| Overall Completeness | Weighted score of all completeness factors | > 0.8 |
| Reasoning Step Count | Number of detailed reasoning steps | > 4 |
| Step Detail Level | Average completeness of reasoning steps | > 0.7 |

### Pattern Application Metrics

These metrics assess how well the system selects and applies reasoning patterns:

| Metric | Description | Good Score |
|--------|-------------|------------|
| Appropriate Pattern | Whether the system applied a suitable pattern | True |
| Confidence | System's confidence in its pattern application | > 0.7 |

### Accuracy Metrics

When reference results are available, these metrics compare system outputs to expert references:

| Metric | Description | Good Score |
|--------|-------------|------------|
| Conclusion Similarity | Similarity between system and reference conclusions | > 0.6 |
| Recommendation Match | Proportion of reference recommendations matched | > 0.7 |
| Overall Accuracy | Weighted combination of accuracy metrics | > 0.7 |

## Creating Reference Results

Reference results should represent expert clinical reasoning for test cases. Each reference should include:

1. **Primary Pattern Applied**: The most appropriate reasoning pattern for the case
2. **Reasoning Steps**: Detailed steps with observations, reasoning, and conclusions
3. **Differential Diagnosis**: Ranked list of possible diagnoses with supporting evidence
4. **Recommendations**: Clinical actions recommended for the case
5. **Conclusion**: Summary of the overall clinical assessment
6. **Confidence**: Confidence level in the assessment (0.0-1.0)

Example reference structure:

```json
{
  "primary_pattern_applied": "CRP-001",
  "reasoning_steps": [
    {
      "step": "Assess vital signs for evidence of respiratory distress",
      "observation": "Patient has tachypnea (RR 24), fever (38.7°C), and mild hypoxemia (O2 sat 91%)",
      "reasoning": "These vital sign abnormalities are consistent with respiratory distress",
      "conclusion": "Patient exhibits moderate respiratory distress"
    },
    // Additional steps...
  ],
  "differential_diagnosis": [
    {
      "diagnosis": "Community-acquired pneumonia",
      "likelihood": "High",
      "supporting_evidence": ["Fever", "Productive cough", "Decreased breath sounds", "Infiltrate on CXR"]
    },
    // Additional diagnoses...
  ],
  "recommendations": [
    "Admission to hospital for treatment of community-acquired pneumonia",
    "Empiric antibiotics with respiratory fluoroquinolone given penicillin allergy",
    // Additional recommendations...
  ],
  "conclusion": "This patient presents with clinical and radiographic evidence of community-acquired pneumonia...",
  "confidence": 0.85
}
```

## Continuous Improvement Process

1. **Identify Patterns with Low Scores**: Review patterns that consistently score poorly
2. **Analyze Failure Modes**: Determine common issues (incomplete reasoning, missed diagnoses, etc.)
3. **Update Pattern Strategies**: Refine reasoning steps to address identified issues
4. **Add Cautions**: Include specific cautions for atypical presentations
5. **Expand Evidence Base**: Add relevant clinical guidelines and literature to support reasoning
6. **Re-evaluate**: Run evaluations again to measure improvement

## Advanced Analytics

For deeper analysis of evaluation results:

```bash
python scripts/analyze_evaluation_trends.py --results-dir src/evaluation/results --output analytics_report.html
```

This will generate a report with:
- Performance trends over time
- Pattern-specific metrics
- Identified improvement opportunities
- Comparative analysis against reference standards 