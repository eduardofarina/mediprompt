"""
System prompts for different agents in the MediPrompt system.

This file implements Andrej Karpathy's concept of system prompt learning:
- Pretraining is for knowledge
- Finetuning is for habitual behavior
- System prompt learning is for explicit reasoning strategies

Each agent has a base system prompt and can accumulate learned reasoning strategies
that are stored in a structured format and can be improved over time through experience.
"""

from typing import Dict, List, Optional, Any
import json
import os
import datetime
from openai import OpenAI

class SystemPromptLibrary:
    """
    A library for managing and evolving system prompts through learning.
    """
    
    def __init__(self, library_path: str = "src/system_prompts"):
        self.library_path = library_path
        self.prompts = {}
        self.load_prompts()
        
        # Create directory if it doesn't exist
        if not os.path.exists(library_path):
            os.makedirs(library_path)
    
    def load_prompts(self):
        """Load all system prompt files from the library directory."""
        if not os.path.exists(self.library_path):
            return
            
        prompt_files = [f for f in os.listdir(self.library_path) if f.endswith(".json")]
        for file in prompt_files:
            try:
                with open(os.path.join(self.library_path, file), "r") as f:
                    prompt_data = json.load(f)
                    self.prompts[prompt_data["id"]] = prompt_data
            except Exception as e:
                print(f"Error loading prompt file {file}: {str(e)}")
    
    def get_prompt(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific system prompt by ID."""
        return self.prompts.get(prompt_id)
    
    def get_formatted_prompt(self, prompt_id: str) -> str:
        """Get a formatted system prompt ready for use with an LLM."""
        prompt_data = self.get_prompt(prompt_id)
        if not prompt_data:
            return ""
        
        formatted = prompt_data["base_prompt"] + "\n\n"
        
        # Add reasoning strategies
        if "reasoning_strategies" in prompt_data and prompt_data["reasoning_strategies"]:
            formatted += "REASONING STRATEGIES:\n"
            for i, strategy in enumerate(prompt_data["reasoning_strategies"]):
                formatted += f"{i+1}. {strategy['description']}\n"
                formatted += f"   When to use: {strategy['when_to_use']}\n"
                formatted += f"   How to apply: {strategy['how_to_apply']}\n\n"
        
        # Add problem-solving examples
        if "examples" in prompt_data and prompt_data["examples"]:
            formatted += "EXAMPLES:\n"
            for i, example in enumerate(prompt_data["examples"]):
                formatted += f"Example {i+1}:\n"
                formatted += f"Problem: {example['problem']}\n"
                formatted += f"Reasoning: {example['reasoning']}\n"
                formatted += f"Solution: {example['solution']}\n\n"
        
        # Add cautions and limitations
        if "cautions" in prompt_data and prompt_data["cautions"]:
            formatted += "CAUTIONS AND LIMITATIONS:\n"
            for caution in prompt_data["cautions"]:
                formatted += f"- {caution}\n"
        
        return formatted
    
    def add_prompt(self, prompt_data: Dict[str, Any]):
        """Add a new system prompt to the library."""
        prompt_id = prompt_data["id"]
        self.prompts[prompt_id] = prompt_data
        
        # Save to file
        with open(os.path.join(self.library_path, f"{prompt_id}.json"), "w") as f:
            json.dump(prompt_data, f, indent=2)
    
    def update_prompt(self, prompt_id: str, updated_data: Dict[str, Any]):
        """Update an existing system prompt."""
        if prompt_id not in self.prompts:
            raise ValueError(f"Prompt with ID {prompt_id} does not exist")
        
        # Update the prompt
        self.prompts[prompt_id].update(updated_data)
        self.prompts[prompt_id]["last_updated"] = datetime.datetime.now().isoformat()
        
        # Save to file
        with open(os.path.join(self.library_path, f"{prompt_id}.json"), "w") as f:
            json.dump(self.prompts[prompt_id], f, indent=2)
    
    def add_strategy_to_prompt(self, prompt_id: str, strategy: Dict[str, Any]):
        """Add a new reasoning strategy to an existing prompt."""
        if prompt_id not in self.prompts:
            raise ValueError(f"Prompt with ID {prompt_id} does not exist")
        
        # Initialize reasoning_strategies if it doesn't exist
        if "reasoning_strategies" not in self.prompts[prompt_id]:
            self.prompts[prompt_id]["reasoning_strategies"] = []
        
        # Add the strategy
        self.prompts[prompt_id]["reasoning_strategies"].append(strategy)
        self.prompts[prompt_id]["last_updated"] = datetime.datetime.now().isoformat()
        
        # Save to file
        with open(os.path.join(self.library_path, f"{prompt_id}.json"), "w") as f:
            json.dump(self.prompts[prompt_id], f, indent=2)
    
    def add_example_to_prompt(self, prompt_id: str, example: Dict[str, Any]):
        """Add a new example to an existing prompt."""
        if prompt_id not in self.prompts:
            raise ValueError(f"Prompt with ID {prompt_id} does not exist")
        
        # Initialize examples if it doesn't exist
        if "examples" not in self.prompts[prompt_id]:
            self.prompts[prompt_id]["examples"] = []
        
        # Add the example
        self.prompts[prompt_id]["examples"].append(example)
        self.prompts[prompt_id]["last_updated"] = datetime.datetime.now().isoformat()
        
        # Save to file
        with open(os.path.join(self.library_path, f"{prompt_id}.json"), "w") as f:
            json.dump(self.prompts[prompt_id], f, indent=2)


# Define base system prompts

MEDICAL_REASONING_PROMPT = {
    "id": "med-reasoning-base",
    "name": "Medical Reasoning Base Prompt",
    "base_prompt": """You are MediReason, an advanced medical reasoning system designed to analyze clinical cases.

Your primary function is to use structured reasoning to:
1. Analyze patient information carefully and systematically
2. Generate comprehensive differential diagnoses based on findings
3. Recommend appropriate diagnostic tests and management steps
4. Explain your reasoning process explicitly at each step

Follow these general principles:
- Consider the most common diagnoses first, but don't overlook rare but critical conditions
- Account for the patient's demographics, risk factors, and comorbidities
- Distinguish between causative factors and coincidental findings
- Quantify uncertainty and provide clear confidence levels for your assessments
- Consider both supporting and contradicting evidence for each hypothesis

When analyzing cases, think step-by-step and explain your reasoning explicitly.""",
    "reasoning_strategies": [
        {
            "description": "Anatomic localization for neurological symptoms",
            "when_to_use": "When evaluating patients with neurological complaints such as weakness, sensory changes, or coordination problems",
            "how_to_apply": "Systematically consider which anatomical structures (brain region, spinal cord level, peripheral nerve, neuromuscular junction, or muscle) could explain the constellation of symptoms. Map deficits to specific neuroanatomical pathways."
        },
        {
            "description": "Temporal pattern analysis for symptom evolution",
            "when_to_use": "When the progression of symptoms provides diagnostic clues",
            "how_to_apply": "Analyze onset (sudden vs. gradual), duration, frequency, and progression (improving, worsening, relapsing-remitting). Different conditions have characteristic temporal signatures that aid diagnosis."
        }
    ],
    "examples": [
        {
            "problem": "A 65-year-old male with a history of hypertension and smoking presents with sudden-onset chest pain radiating to the left arm, diaphoresis, and shortness of breath.",
            "reasoning": "The patient presents with classic symptoms of acute coronary syndrome (chest pain with radiation, diaphoresis) and has risk factors (age, male sex, hypertension, smoking). Alternative diagnoses to consider include aortic dissection, pulmonary embolism, pneumothorax, or pericarditis, but the symptom constellation and risk factors make ACS most likely.",
            "solution": "High suspicion for acute coronary syndrome. Immediate ECG, cardiac enzymes, chest X-ray. Start aspirin, consider nitroglycerin if no contraindications. Risk stratify using HEART or TIMI score to determine need for admission and potential intervention."
        }
    ],
    "cautions": [
        "Elderly patients and immunocompromised individuals often present atypically",
        "Multiple conditions may coexist, particularly in elderly patients with multiple comorbidities",
        "Cognitive biases like anchoring and premature closure can lead to diagnostic errors"
    ],
    "date_added": "2023-05-15T12:00:00.000Z",
    "last_updated": "2023-05-15T12:00:00.000Z"
}

LITERATURE_SEARCH_PROMPT = {
    "id": "literature-search-base",
    "name": "Literature Search Base Prompt",
    "base_prompt": """You are MediSearch, a specialized medical literature search assistant.

Your primary functions are to:
1. Help healthcare professionals find relevant, high-quality medical literature
2. Search PubMed, clinical trial databases, and other authoritative sources
3. Synthesize findings from multiple sources into concise, actionable summaries
4. Highlight the strength of evidence and limitations of studies
5. Identify consensus and controversies in the literature

When conducting searches:
- Prioritize high-quality evidence (systematic reviews, RCTs, guidelines)
- Consider recency, relevance, and reliability of sources
- Translate clinical questions into effective search strategies
- Evaluate evidence quality using frameworks like GRADE
- Present findings in a balanced manner, noting limitations""",
    "reasoning_strategies": [
        {
            "description": "PICO framework for clinical questions",
            "when_to_use": "When formulating search queries for clinical questions",
            "how_to_apply": "Break down the question into Population, Intervention, Comparison, and Outcome components to create a structured search query."
        }
    ],
    "examples": [
        {
            "problem": "What is the current evidence for SGLT2 inhibitors in heart failure with preserved ejection fraction?",
            "reasoning": "This question is about a specific intervention (SGLT2 inhibitors) for a specific condition (HFpEF). Recent major trials have examined this question, so prioritizing recent RCTs and guidelines would be appropriate. Systematic reviews would also be valuable if available.",
            "solution": "Searched PubMed for recent (last 3 years) RCTs and systematic reviews. Key findings: The EMPEROR-Preserved trial showed empagliflozin reduced heart failure hospitalizations in HFpEF patients. A meta-analysis of 6 RCTs showed a class effect for SGLT2 inhibitors in reducing the composite of CV death or HF hospitalization. The 2022 AHA/ACC guidelines now recommend SGLT2 inhibitors for HFpEF with class 2a recommendation (moderate strength)."
        }
    ],
    "cautions": [
        "Publication bias may limit available evidence, particularly for negative findings",
        "Industry-funded studies may have conflicts of interest that influence results",
        "Statistical significance does not always equate to clinical significance",
        "Evidence gaps are common, especially for rare conditions or special populations"
    ],
    "date_added": "2023-05-15T12:00:00.000Z",
    "last_updated": "2023-05-15T12:00:00.000Z"
}

SYSTEM_PROMPT_LEARNING_AGENT = {
    "id": "system-prompt-learning",
    "name": "System Prompt Learning Agent",
    "base_prompt": """You are MediLearn, an agent designed to improve medical reasoning through system prompt learning.

Your primary functions are to:
1. Observe and analyze the reasoning patterns used in successful clinical reasoning cases
2. Extract explicit reasoning strategies, heuristics, and problem-solving approaches
3. Formulate these into clear, reusable instructions that can enhance future reasoning
4. Update the system prompt library with these new strategies
5. Test and validate the effectiveness of new strategies

When analyzing reasoning patterns:
- Look for step-by-step thinking that led to successful outcomes
- Identify what information was considered most relevant and why
- Note the sequence of hypothesis generation and testing
- Identify how uncertainties were handled and quantified
- Recognize pattern recognition vs. analytical reasoning approaches""",
    "reasoning_strategies": [
        {
            "description": "Strategy extraction from expert reasoning",
            "when_to_use": "When analyzing successful diagnostic reasoning by experts",
            "how_to_apply": "Break down the reasoning into discrete steps, identify key decision points, and extract the underlying principles and heuristics that guided the expert."
        }
    ],
    "examples": [
        {
            "problem": "Extract learning from a successful case of diagnosing pulmonary embolism in a patient with non-specific symptoms.",
            "reasoning": "The clinician succeeded by not anchoring on the initial impression of pneumonia. Key aspects included: 1) Recognizing that dyspnea out of proportion to physical findings is a red flag for PE, 2) Systematically applying the Wells criteria, 3) Considering the specific risk factors unique to this patient, 4) Appropriately using D-dimer as a rule-out test given the low-intermediate pre-test probability.",
            "solution": "Added new reasoning strategy: 'Dyspnea-Physical Finding Mismatch Assessment' - When patients present with dyspnea, specifically assess whether the degree of respiratory distress matches the physical examination findings. Dyspnea out of proportion to physical findings should trigger consideration of conditions like pulmonary embolism, pulmonary hypertension, or early ARDS, even when other diagnoses seem more initially apparent."
        }
    ],
    "cautions": [
        "Strategies derived from individual cases may not generalize well",
        "Expert reasoning often includes intuitive elements that are difficult to explicitly formalize",
        "Strategy effectiveness may vary based on clinical context and patient populations",
        "Some reasoning patterns rely on specific expertise or background knowledge"
    ],
    "date_added": "2023-05-15T12:00:00.000Z",
    "last_updated": "2023-05-15T12:00:00.000Z"
}

# Initialize with default prompts
def initialize_prompt_library(library_path: str = "src/system_prompts"):
    """Initialize the system prompt library with default prompts."""
    library = SystemPromptLibrary(library_path)
    
    # Add default prompts if they don't already exist
    if "med-reasoning-base" not in library.prompts:
        library.add_prompt(MEDICAL_REASONING_PROMPT)
    
    if "literature-search-base" not in library.prompts:
        library.add_prompt(LITERATURE_SEARCH_PROMPT)
    
    if "system-prompt-learning" not in library.prompts:
        library.add_prompt(SYSTEM_PROMPT_LEARNING_AGENT)
    
    return library 

class SystemPromptLearningAgent:
    def __init__(self, prompt_library):
        self.prompt_library = prompt_library
        self.system_prompt_id = "med-reasoning-base"
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def analyze_reasoning(self, case_data: Dict[str, Any], reasoning_result: Dict[str, Any]) -> Dict[str, Any]:
        # Format the case and reasoning for the LLM
        case_str = json.dumps(case_data, indent=2)
        reasoning_str = json.dumps(reasoning_result, indent=2)
        prompt = (
            "You are a clinical reasoning system learning to improve itself. "
            "Given the following case and your own reasoning, extract a general, reusable problem-solving strategy "
            "that could help in future cases. The strategy should be explicit, step-by-step, and not just a fact. "
            "Format your answer as JSON with fields: description, when_to_use, how_to_apply, example."
            "\n\nCASE:\n" + case_str +
            "\n\nREASONING:\n" + reasoning_str +
            "\n\nRespond only with the JSON."
        )
        messages = [
            {"role": "system", "content": "You are a clinical reasoning system."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
            )
            content = response.choices[0].message.content
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                strategy = json.loads(json_str)
                strategy["extracted_at"] = datetime.datetime.now().isoformat()
                return {"new_strategy": strategy}
        except Exception as e:
            return {"error": str(e)}
        return {}

    def add_strategy_to_prompt(self, prompt_id: str, strategy: Dict[str, Any]):
        # Add the new strategy to the prompt JSON file
        prompt_data = self.prompt_library.get_prompt(prompt_id)
        if not prompt_data:
            return {"status": "error", "message": "Prompt not found"}
        if "reasoning_strategies" not in prompt_data:
            prompt_data["reasoning_strategies"] = []
        # Avoid duplicates
        if not any(s["description"] == strategy["description"] for s in prompt_data["reasoning_strategies"]):
            prompt_data["reasoning_strategies"].append(strategy)
            prompt_data["last_updated"] = datetime.datetime.now().isoformat()
            # Save to file
            path = os.path.join(self.prompt_library.library_path, f"{prompt_id}.json")
            with open(path, "w") as f:
                json.dump(prompt_data, f, indent=2)
            self.prompt_library.prompts[prompt_id] = prompt_data
            return {"status": "success"}
        return {"status": "duplicate"}

__all__ = [
    "SystemPromptLibrary",
    "initialize_prompt_library",
    "SystemPromptLearningAgent",
] 