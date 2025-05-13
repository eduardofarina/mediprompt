# MediPrompt: Medical Reasoning with System Prompt Learning

MediPrompt is an advanced medical reasoning application built with Streamlit that implements Andrej Karpathy's concept of "system prompt learning". It integrates multiple specialized agents to provide medical reasoning, management/treatment advice, literature search, and continuous learning capabilities.

## Key Features

- **Medical Reasoning Agent**: Analyzes clinical cases using structured, flexible reasoning patterns
- **Management/Treatment Agent**: Provides evidence-based management and treatment advice for diagnoses and clinical scenarios
- **Literature Search Agent**: Searches medical literature from PubMed and ArXiv and synthesizes findings
- **System Prompt Learning**: Continuously improves reasoning strategies by extracting explicit learning from successful and unsuccessful reasoning cases, automatically after each case is closed
- **Multi-agent Architecture**: Seamlessly integrates multiple specialized agents (diagnosis, management, literature)
- **Robust Orchestration**: Fallback logic ensures a valid agent is always called, even if the LLM output is incomplete or ambiguous
- **Interactive UI**: Clean Streamlit interface with chat, system prompt visualization, and settings

## System Prompt Learning

This application implements the concept introduced by Andrej Karpathy about system prompt learning:

> "We're missing (at least one) major paradigm for LLM learning. Not sure what to call it, possibly it has a name - system prompt learning?
>
> Pretraining is for knowledge.
> Finetuning (SL/RL) is for habitual behavior.
>
> Both of these involve a change in parameters but a lot of human learning feels more like a change in system prompt."

In MediPrompt, we implement this concept by:

1. Starting with base system prompts for different agents (diagnosis, management, literature)
2. Analyzing clinical reasoning cases to extract explicit problem-solving strategies
3. Adding these strategies to the system prompts to improve future reasoning
4. Tracking and visualizing the growth of the system prompt library over time
5. **System prompt learning is automatic after each case is closed**—no manual feedback required

## Architecture

The application is built with the following components:

- **Streamlit App**: Main user interface and application logic
- **LLM Integration**: Uses OpenAI's models for reasoning and analysis
- **Agents**:
  - MediReason: Clinical reasoning and diagnosis
  - ManagementAgent: Evidence-based management and treatment advice
  - LiteratureSearch: Medical literature search and synthesis
  - SystemPromptLearning: Extracts learning from clinical cases
- **System Prompt Library**: Stores and manages evolving system prompts
- **External Tools**: Integration with PubMed and ArXiv APIs

## Installation

1. Clone the repository:
```bash
git clone https://github.com/eduardofarina/mediprompt.git
cd mediprompt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
# On Windows
set OPENAI_API_KEY=your_api_key_here

# On Linux/MacOS
export OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run src/app.py
```

2. Use the chat interface to:
   - Enter medical cases for analysis
   - Ask for management/treatment advice
   - Ask general medical questions
   - View literature search results

3. Explore the System Prompt Learning tab to see how the system improves over time

4. Configure settings in the Settings tab

## Example Queries

- **Clinical Case Analysis**: "A 67-year-old male with hypertension presents with sudden onset of crushing chest pain radiating to the left arm, diaphoresis, and shortness of breath. Vital signs show BP 160/95, HR 110, RR 22, O2 sat 94% on room air. ECG shows ST elevation in leads V2-V5."

- **Management/Treatment Advice**: "How should you manage a patient with confirmed influenza during an epidemic when no rapid tests are available?"

- **Literature Search**: "What is the current evidence for SGLT2 inhibitors in heart failure with preserved ejection fraction?"

- **General Medical Question**: "What are the main differences between type 1 and type 2 diabetes in terms of pathophysiology and treatment?"

## Project Structure

```
mediprompt/
├── src/
│   ├── app.py                  # Main Streamlit application
│   ├── prompts.py              # System prompt library implementation
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── medireason_agent.py # Medical reasoning agent
│   │   ├── management_agent.py # Management/treatment agent
│   │   └── literature_agent.py # Literature search agent
│   ├── patterns/               # Clinical reasoning patterns
│   ├── system_prompts/         # System prompts for learning
│   └── tools/                  # External tools and integrations
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Andrej Karpathy for the system prompt learning concept
- OpenAI for providing the LLM capabilities
- Streamlit for the UI framework 