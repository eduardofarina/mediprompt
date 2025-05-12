# MediReason: Clinical Reasoning Learning System

MediReason is an advanced clinical reasoning system built with Python and the Agno agent framework. It's designed to not only answer medical queries but also document and improve its own reasoning processes, implementing Karpathy's concept of "system prompt learning" - capturing explicit problem-solving strategies rather than relying solely on parameter updates.

## Core Capabilities

1. **Medical Knowledge**: Deep knowledge of medical literature, clinical guidelines, diagnostic criteria, and treatment protocols.

2. **Clinical Reasoning**: Analyzes patient cases step-by-step, considers differential diagnoses, and recommends appropriate workups and treatments.

3. **Learning System**: Maintains a structured "Clinical Reasoning Library" of strategies and heuristics that update based on interactions.

## Agno Agent Framework Integration

As an implementation using the Agno agent framework, MediReason has these specific capabilities:

1. **Tool Integration**: Uses Agno's tool-calling capabilities to access medical databases, literature search, and clinical guidelines in real-time.

2. **State Persistence**: The Clinical Reasoning Library is maintained as a persistent knowledge store using Agno's state management.

3. **Multi-step Reasoning**: Leverages Agno's ReAct pattern to perform step-by-step reasoning with observation and reflection at each step.

4. **Python Implementation**: Backend implemented in Python, allowing for integration with medical data processing libraries and healthcare APIs.

## Project Structure

```
MediReason/
├── src/
│   ├── agents/             # Agno agents for clinical reasoning
│   ├── tools/              # Tools for accessing medical resources
│   ├── patterns/           # Clinical reasoning pattern library
│   ├── cases/              # Example medical cases
│   ├── evaluation/         # Evaluation metrics and validation
│   └── integration/        # Integration with external systems
├── scripts/                # Utility scripts
├── resources/              # Medical terminology and reference data
└── tests/                  # Test suite
```

## Installation

```bash
# Clone the repository
git clone https://github.com/eduardofarina/medireason.git
cd medireason

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from src.agents.medireason_agent import MediReasonAgent

# Initialize the agent
agent = MediReasonAgent()

# Process a clinical case
case_data = {
    "presenting_symptoms": "Fever, cough, shortness of breath for 3 days",
    "patient_demographics": "65-year-old male with history of COPD",
    "vital_signs": "Temperature 38.5°C, HR 110, BP 135/85, RR 22, O2 sat 92% on RA",
    "physical_exam": "Decreased breath sounds in right lower lobe, no wheezing",
    "past_medical_history": "COPD, hypertension, type 2 diabetes"
}

# Get the clinical reasoning
result = agent.analyze_case(case_data)
print(result.conclusion)
```

## Clinical Reasoning Library

The Clinical Reasoning Library uses the following format:

```
CLINICAL REASONING PATTERN #[number]
TRIGGER: [When to apply this pattern]
STRATEGY: [Step-by-step approach]
EVIDENCE BASE: [Supporting medical literature/guidelines]
CAUTIONS: [When this approach might be misleading]
EXAMPLE APPLICATION: [Brief case example]
```

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) for details on how to submit patterns, improvements, or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by Andrej Karpathy's system prompt learning concept
- Built with the Agno agent framework
- Special thanks to the medical experts who contributed to the reasoning patterns 