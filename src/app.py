import streamlit as st
import json
import os
import datetime
from typing import Dict, List, Any, Optional
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
from agno.tools.pubmed import PubmedTools
from agno.tools.arxiv import ArxivTools

# Import system prompt library
from prompts import SystemPromptLibrary, initialize_prompt_library, SystemPromptLearningAgent
from agents.medireason_agent import MediReasonAgent

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize system prompt library
prompt_library = initialize_prompt_library()

# Initialize tools
pubmed_tools = PubmedTools()
arxiv_tools = ArxivTools()

# Initialize the MediReason agent
medireason_agent = MediReasonAgent()

# Initialize the learning agent
learning_agent = SystemPromptLearningAgent(prompt_library)

# Add a mapping function to interpret agent's ask_user
FIELD_KEYWORDS = {
    "age": "patient_demographics",
    "sex": "patient_demographics",
    "demographic": "patient_demographics",
    "vital": "vital_signs",
    "symptom": "presenting_symptoms",
    "exam": "physical_exam",
    "history": "past_medical_history",
    "medication": "medications",
    "allerg": "allergies",
    "social": "social_history",
    "family": "family_history",
    "lab": "lab_results",
}

def is_info_already_provided(ask_user, case_data):
    ask = ask_user.lower() if ask_user else ""
    for key, field in FIELD_KEYWORDS.items():
        if key in ask and case_data.get(field):
            return True
    return False

def update_case_data_from_user_input(user_input, last_ask_user, case_data):
    # Try to map the last question to a field and update it
    for key, field in FIELD_KEYWORDS.items():
        if last_ask_user and key in last_ask_user.lower():
            case_data[field] = user_input
            return
    # If no mapping, just append to a generic notes field
    case_data["notes"] = case_data.get("notes", "") + "\n" + user_input

def get_llm_response(messages, model="gpt-4o", temperature=0.7):
    """Get a response from the specified LLM model"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting response from OpenAI: {str(e)}")
        return "I'm having trouble connecting to the AI service. Please try again later."

class LiteratureSearchAgent:
    """Agent for conducting medical literature searches and synthesizing results."""
    
    def __init__(self, prompt_library):
        self.prompt_library = prompt_library
        self.system_prompt = prompt_library.get_formatted_prompt("literature-search-base")
    
    def search_literature(self, query: str) -> Dict[str, Any]:
        """
        Search medical literature based on the query.
        
        Args:
            query: The search query
            
        Returns:
            Dictionary with search results
        """
        # Use PubMed tool
        try:
            pubmed_results = pubmed_tools.search_pubmed(query, max_results=5)
        except Exception as e:
            pubmed_results = f"PubMed search error: {str(e)}"
        
        # Use ArXiv tool
        try:
            arxiv_results = arxiv_tools.search_arxiv(query, max_results=5)
        except Exception as e:
            arxiv_results = f"ArXiv search error: {str(e)}"
        
        # Use LLM to synthesize results
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
            Please search for and synthesize literature on the following query:
            
            Query: {query}
            
            PubMed results:
            {pubmed_results}
            
            ArXiv results:
            {arxiv_results}
            
            Synthesize these results into a comprehensive but concise summary.
            Focus on the most relevant findings, the strength of evidence,
            and any consensus or controversies in the literature.
            """}
        ]
        
        synthesis = get_llm_response(messages)
        
        return {
            "query": query,
            "pubmed_results": pubmed_results,
            "arxiv_results": arxiv_results,
            "synthesis": synthesis,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def formulate_pico_query(self, clinical_question: str) -> str:
        """
        Use PICO framework to formulate a better search query.
        
        Args:
            clinical_question: The clinical question
            
        Returns:
            Formulated PICO query
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
            Please reformulate the following clinical question using the PICO framework
            (Population, Intervention, Comparison, Outcome):
            
            Clinical Question: {clinical_question}
            
            Extract the PICO elements and formulate an optimized search query for medical literature databases.
            """}
        ]
        
        response = get_llm_response(messages)
        return response

class SystemPromptLearningAgent:
    """Agent that learns from clinical reasoning cases to improve system prompts."""
    
    def __init__(self, prompt_library):
        self.prompt_library = prompt_library
        self.system_prompt = prompt_library.get_formatted_prompt("system-prompt-learning")
    
    def analyze_reasoning(self, case_data: Dict[str, Any], reasoning_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze clinical reasoning to extract new strategies.
        
        Args:
            case_data: The clinical case data
            reasoning_result: The reasoning process and results
            
        Returns:
            Dictionary with new strategies or improvements
        """
        # Format the case and reasoning for the LLM
        case_str = json.dumps(case_data, indent=2)
        reasoning_str = json.dumps(reasoning_result, indent=2)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
            Please analyze the following clinical case and the reasoning process applied to it.
            Extract any valuable reasoning strategies, heuristics, or problem-solving approaches
            that could be generalized and added to a system prompt library.
            
            CLINICAL CASE:
            {case_str}
            
            REASONING PROCESS AND RESULTS:
            {reasoning_str}
            
            Based on this analysis, formulate at least one explicit reasoning strategy that could improve
            future clinical reasoning. Format the strategy as follows:
            
            ```json
            {{
                "description": "Brief name of the strategy",
                "when_to_use": "Situations where this strategy is applicable",
                "how_to_apply": "Step-by-step instructions for applying this strategy",
                "example": "A brief example of this strategy in action"
            }}
            ```
            """}
        ]
        
        response = get_llm_response(messages)
        
        # Extract the JSON strategy from the response
        try:
            # Find JSON block in the response
            start_idx = response.find("```json") + 7 if "```json" in response else response.find("{")
            end_idx = response.rfind("}") + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                new_strategy = json.loads(json_str)
                
                # Add timestamp
                new_strategy["extracted_at"] = datetime.datetime.now().isoformat()
                
                return {
                    "new_strategy": new_strategy,
                    "analysis": response
                }
            else:
                return {
                    "error": "Could not extract strategy",
                    "analysis": response
                }
        except Exception as e:
            return {
                "error": f"Error parsing strategy: {str(e)}",
                "analysis": response
            }
    
    def add_strategy_to_prompt(self, prompt_id: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new strategy to an existing prompt.
        
        Args:
            prompt_id: The ID of the prompt to update
            strategy: The strategy to add
            
        Returns:
            Status of the operation
        """
        try:
            self.prompt_library.add_strategy_to_prompt(prompt_id, strategy)
            return {
                "status": "success",
                "message": f"Strategy added to prompt {prompt_id}",
                "strategy": strategy
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error adding strategy: {str(e)}"
            }

def process_medical_case(case_text: str, incorporate_system_prompt_learning=True):
    """
    Process a medical case through both agents.
    
    Args:
        case_text: The text description of the medical case
        incorporate_system_prompt_learning: Whether to use system prompt learning
        
    Returns:
        Results from both agents
    """
    # Parse case text into structured data using an LLM
    system_prompt = """
    You are a medical AI assistant. Your task is to extract structured data from a medical case description.
    Extract the following information and return it as a JSON object:
    {
        "presenting_symptoms": "",
        "patient_demographics": "",
        "vital_signs": "",
        "physical_exam": "",
        "past_medical_history": "",
        "medications": "",
        "allergies": "",
        "social_history": "",
        "family_history": "",
        "lab_results": ""
    }
    Only include fields that are mentioned in the description. If a field is not mentioned, leave it as an empty string.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": case_text}
    ]
    
    structured_case_text = get_llm_response(messages)
    
    try:
        # Extract JSON from the response
        start_idx = structured_case_text.find("{")
        end_idx = structured_case_text.rfind("}") + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = structured_case_text[start_idx:end_idx]
            case_data = json.loads(json_str)
        else:
            case_data = {
                "error": "Could not parse structured case data",
                "original_text": case_text
            }
    except Exception as e:
        case_data = {
            "error": f"Error parsing case data: {str(e)}",
            "original_text": case_text
        }
    
    # Process with MediReason agent
    medireason_result = medireason_agent.analyze_case(case_data)
    
    # Extract key terms for literature search
    lit_search_prompt = """
    You are a medical literature search specialist. Based on the clinical case below, extract 3-5 key search terms
    that would be most useful for finding relevant medical literature about this case. Format your response as a
    comma-separated list of terms.
    
    Clinical case:
    """
    
    messages = [
        {"role": "system", "content": lit_search_prompt},
        {"role": "user", "content": case_text}
    ]
    
    search_terms_text = get_llm_response(messages)
    search_terms = [term.strip() for term in search_terms_text.split(",")]
    
    # Process with Literature Search agent
    lit_search_agent = LiteratureSearchAgent(prompt_library)
    search_query = " ".join(search_terms[:3])  # Use top 3 terms
    lit_search_result = lit_search_agent.search_literature(search_query)
    
    # If system prompt learning is enabled, use it to improve prompts
    if incorporate_system_prompt_learning:
        learning_result = learning_agent.analyze_reasoning(case_data, medireason_result)
        
        # If a new strategy was extracted, add it to the medical reasoning prompt
        if "new_strategy" in learning_result and "error" not in learning_result:
            learning_agent.add_strategy_to_prompt("med-reasoning-base", learning_result["new_strategy"])
            
            # Log the learning
            if "system_prompt_learning_log" not in st.session_state:
                st.session_state.system_prompt_learning_log = []
                
            st.session_state.system_prompt_learning_log.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "case": case_text[:100] + "..." if len(case_text) > 100 else case_text,
                "strategy": learning_result["new_strategy"]["description"],
                "prompt_id": "med-reasoning-base"
            })
    else:
        learning_result = None
    
    return {
        "case_data": case_data,
        "medireason_result": medireason_result,
        "literature_search": lit_search_result,
        "system_prompt_learning": learning_result
    }

def format_medireason_result(result: Dict[str, Any]) -> str:
    """Format the MediReason analysis result for display"""
    formatted_result = "## Clinical Analysis Results\n\n"
    
    # Add assessment
    formatted_result += f"### Assessment\n{result.get('assessment', 'No assessment available')}\n\n"
    
    # Add confidence
    confidence = result.get('confidence', 0.0)
    formatted_result += f"**Confidence**: {confidence:.2f}\n\n"
    
    # Add primary pattern applied
    if 'primary_pattern_applied' in result:
        formatted_result += f"**Primary Reasoning Pattern**: {result['primary_pattern_applied']}\n\n"
    
    # Add conclusion
    if 'conclusion' in result:
        formatted_result += f"### Conclusion\n{result['conclusion']}\n\n"
    
    # Add differential diagnosis
    if 'differential_diagnosis' in result and result['differential_diagnosis']:
        formatted_result += "### Differential Diagnosis\n"
        for diagnosis in result['differential_diagnosis']:
            formatted_result += f"- **{diagnosis.get('diagnosis', 'Unknown')}** ({diagnosis.get('likelihood', 'Unknown')} likelihood)\n"
            if 'supporting_evidence' in diagnosis:
                formatted_result += "  - Supporting evidence: " + ", ".join(diagnosis['supporting_evidence']) + "\n"
        formatted_result += "\n"
    
    # Add recommendations
    if 'recommendations' in result and result['recommendations']:
        formatted_result += "### Recommendations\n"
        for rec in result['recommendations']:
            formatted_result += f"- {rec}\n"
        formatted_result += "\n"
    
    # Add reasoning steps (collapsed for readability)
    if 'reasoning_steps' in result and result['reasoning_steps']:
        formatted_result += "### Detailed Reasoning Steps\n"
        for i, step in enumerate(result['reasoning_steps']):
            formatted_result += f"<details><summary>Step {i+1}: {step.get('step', 'Reasoning step')}</summary>\n\n"
            formatted_result += f"**Observation**: {step.get('observation', '')}\n\n"
            formatted_result += f"**Reasoning**: {step.get('reasoning', '')}\n\n"
            formatted_result += f"**Conclusion**: {step.get('conclusion', '')}\n\n"
            formatted_result += "</details>\n\n"
    
    return formatted_result

def format_literature_search_result(result: Dict[str, Any]) -> str:
    """Format the literature search result for display"""
    formatted_result = "## Literature Search Results\n\n"
    
    # Add query
    formatted_result += f"**Search Query**: {result.get('query', 'No query available')}\n\n"
    
    # Add synthesis
    formatted_result += f"### Synthesis\n{result.get('synthesis', 'No synthesis available')}\n\n"
    
    # Add PubMed results
    formatted_result += f"<details><summary>PubMed Results</summary>\n\n{result.get('pubmed_results', 'No PubMed results available')}\n\n</details>\n\n"
    
    # Add ArXiv results
    formatted_result += f"<details><summary>ArXiv Results</summary>\n\n{result.get('arxiv_results', 'No ArXiv results available')}\n\n</details>\n\n"
    
    return formatted_result

def display_system_prompt_learning(learning_result: Dict[str, Any]) -> str:
    """Format the system prompt learning result for display"""
    if not learning_result or "error" in learning_result:
        return ""
        
    formatted_result = "## System Prompt Learning\n\n"
    
    # Add new strategy
    if "new_strategy" in learning_result:
        strategy = learning_result["new_strategy"]
        formatted_result += f"### New Reasoning Strategy Extracted\n\n"
        formatted_result += f"**Description**: {strategy.get('description', '')}\n\n"
        formatted_result += f"**When to use**: {strategy.get('when_to_use', '')}\n\n"
        formatted_result += f"**How to apply**: {strategy.get('how_to_apply', '')}\n\n"
        
        if "example" in strategy:
            formatted_result += f"**Example**: {strategy.get('example', '')}\n\n"
    
    # Add analysis
    formatted_result += f"<details><summary>Full Analysis</summary>\n\n{learning_result.get('analysis', '')}\n\n</details>\n\n"
    
    return formatted_result

def display_system_prompt_learning_log():
    """Display the log of system prompt learning instances"""
    if "system_prompt_learning_log" not in st.session_state or not st.session_state.system_prompt_learning_log:
        st.info("No system prompt learning history available yet.")
        return
    
    log = st.session_state.system_prompt_learning_log
    
    # Convert to DataFrame for easier display
    df = pd.DataFrame(log)
    
    # Display as table
    st.dataframe(df[["timestamp", "case", "strategy", "prompt_id"]])
    
    # Display growth chart
    fig, ax = plt.subplots()
    strategy_counts = range(1, len(log) + 1)
    timestamps = [datetime.datetime.fromisoformat(entry["timestamp"]) for entry in log]
    
    ax.plot(timestamps, strategy_counts)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Strategies Learned")
    ax.set_title("Growth of System Prompt Learning")
    
    st.pyplot(fig)

def classify_user_intent(user_input: str) -> str:
    """Classify user input as diagnosis, treatment, followup, or question."""
    text = user_input.lower()
    if any(word in text for word in ["treat", "treatment", "manage", "medication", "therapy", "drug", "dose", "antibiotic", "antiviral", "how do you treat", "what is the treatment"]):
        return "treatment"
    if any(word in text for word in ["follow-up", "prognosis", "outcome", "complication", "monitor", "recovery", "risk", "chance", "long-term"]):
        return "followup"
    if any(word in text for word in ["diagnosis", "differential", "what could this be", "what is the diagnosis", "what are the causes"]):
        return "diagnosis"
    # If the user gives a statement like "scabies was the diagnosis" or "the diagnosis was ..."
    if "was the diagnosis" in text or "diagnosed with" in text:
        return "treatment"  # move to management
    # If the user asks a direct question (ends with ?)
    if text.strip().endswith("?"):
        return "question"
    # Default to diagnosis if the input is a case description
    return "diagnosis"

# Streamlit UI
def main():
    st.set_page_config(
        page_title="MediPrompt - Medical Reasoning with System Prompt Learning",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 MediPrompt: Medical Reasoning with System Prompt Learning")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "case_data" not in st.session_state:
        st.session_state.case_data = {}
    if "last_ask_user" not in st.session_state:
        st.session_state.last_ask_user = None
    if "awaiting_agent" not in st.session_state:
        st.session_state.awaiting_agent = False
    if "dialogue_mode" not in st.session_state:
        st.session_state.dialogue_mode = "diagnosis"

    # Initialize session state for learning toggle and log
    if "enable_learning" not in st.session_state:
        st.session_state.enable_learning = True
    if "system_prompt_learning_log" not in st.session_state:
        st.session_state.system_prompt_learning_log = []

    tabs = st.tabs(["Chat Interface", "System Prompt Learning", "Settings"])
    
    with tabs[0]:
        if st.button("🔄 Start New Case"):
            st.session_state.case_data = {}
            st.session_state.chat_history = [{"role": "assistant", "content": "Starting a new case analysis. Please describe the case or ask a question."}]
            st.session_state.last_ask_user = None
            st.session_state.awaiting_agent = False
            st.session_state.dialogue_mode = "diagnosis"
            st.rerun()

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        user_input = st.chat_input("Enter a medical case, answer a question, or ask...")
        
        if user_input:
            # Classify user intent and set dialogue mode
            st.session_state.dialogue_mode = classify_user_intent(user_input)
            if st.session_state.last_ask_user:
                update_case_data_from_user_input(user_input, st.session_state.last_ask_user, st.session_state.case_data)
            else:
                if not st.session_state.chat_history:
                    st.session_state.case_data = {}
                st.session_state.case_data["initial_input"] = st.session_state.case_data.get("initial_input", "") + "\n" + user_input
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.awaiting_agent = True
            st.rerun()
        
        if st.session_state.awaiting_agent:
            st.session_state.awaiting_agent = False
            mode = st.session_state.dialogue_mode
            output = ""
            if mode == "diagnosis":
                result = medireason_agent.analyze_case(st.session_state.case_data)
                differentials = result.get("differential_diagnosis", [])
                recommendations = result.get("recommendations", [])
                ask_user = result.get("ask_user", None)
                if differentials:
                    output += "**Differential Diagnosis:**\n"
                    for d in differentials:
                        output += f"- {d.get('diagnosis')}: {d.get('reasoning', d.get('likelihood', ''))}\n"
                if recommendations:
                    output += "\n**Next Steps / Recommendations:**\n"
                    for r in recommendations:
                        output += f"- {r}\n"
                needs_more_info = ask_user and not is_info_already_provided(ask_user, st.session_state.case_data)
                if needs_more_info:
                    output += f"\n**Question:** {ask_user}"
                    st.session_state.last_ask_user = ask_user
                else:
                    st.session_state.last_ask_user = None
                    # Only run system prompt learning if not in diagnosis mode
                    if st.session_state.enable_learning:
                        try:
                            learning_result = learning_agent.analyze_reasoning(st.session_state.case_data, result)
                            if "new_strategy" in learning_result and "error" not in learning_result:
                                strategy = learning_result["new_strategy"]
                                add_status = learning_agent.add_strategy_to_prompt("med-reasoning-base", strategy)
                                log_entry = {
                                    "timestamp": datetime.datetime.now().isoformat(),
                                    "case": json.dumps(st.session_state.case_data.get("initial_input", "N/A"))[:100] + "...",
                                    "strategy": strategy.get("description", "N/A"),
                                    "prompt_id": "med-reasoning-base",
                                    "status": add_status.get("status", "unknown")
                                }
                                st.session_state.system_prompt_learning_log.append(log_entry)
                                output += f"\n\n*(Learned new strategy: {strategy.get('description', 'N/A')})*"
                        except Exception as e:
                            st.warning(f"System prompt learning failed: {str(e)}")
            elif mode == "treatment":
                # Give management advice for the most likely/confirmed diagnosis
                diagnosis = st.session_state.case_data.get("diagnosis")
                if not diagnosis:
                    # Try to infer from last differential
                    diagnosis = None
                    if st.session_state.chat_history:
                        for msg in reversed(st.session_state.chat_history):
                            if "Differential Diagnosis" in msg["content"]:
                                lines = msg["content"].split("\n")
                                for line in lines:
                                    if line.startswith("- "):
                                        diagnosis = line[2:].split(":")[0]
                                        break
                            if diagnosis:
                                break
                if diagnosis:
                    output += f"**Treatment/Management for {diagnosis}:**\n"
                else:
                    output += "**Treatment/Management:**\n"
                # Use LLM to generate management advice
                prompt = f"You are a clinical reasoning assistant. Provide concise, evidence-based management and treatment advice for the following diagnosis and case context:\nDiagnosis: {diagnosis}\nCase: {json.dumps(st.session_state.case_data)}"
                messages = [
                    {"role": "system", "content": "You are a clinical reasoning assistant."},
                    {"role": "user", "content": prompt}
                ]
                advice = get_llm_response(messages)
                output += advice
                st.session_state.last_ask_user = None
            elif mode == "followup":
                # Give follow-up/prognosis advice
                diagnosis = st.session_state.case_data.get("diagnosis")
                if not diagnosis:
                    diagnosis = None
                    if st.session_state.chat_history:
                        for msg in reversed(st.session_state.chat_history):
                            if "Differential Diagnosis" in msg["content"]:
                                lines = msg["content"].split("\n")
                                for line in lines:
                                    if line.startswith("- "):
                                        diagnosis = line[2:].split(":")[0]
                                        break
                            if diagnosis:
                                break
                output += f"**Follow-up/Prognosis for {diagnosis or 'this case'}:**\n"
                prompt = f"You are a clinical reasoning assistant. Provide concise, evidence-based follow-up, prognosis, and monitoring advice for the following diagnosis and case context:\nDiagnosis: {diagnosis}\nCase: {json.dumps(st.session_state.case_data)}"
                messages = [
                    {"role": "system", "content": "You are a clinical reasoning assistant."},
                    {"role": "user", "content": prompt}
                ]
                advice = get_llm_response(messages)
                output += advice
                st.session_state.last_ask_user = None
            else:  # mode == "question"
                # Just answer the question
                prompt = f"You are a clinical reasoning assistant. Answer the following question in the context of this case:\nCase: {json.dumps(st.session_state.case_data)}\nQuestion: {st.session_state.chat_history[-1]['content']}"
                messages = [
                    {"role": "system", "content": "You are a clinical reasoning assistant."},
                    {"role": "user", "content": prompt}
                ]
                answer = get_llm_response(messages)
                output += answer
                st.session_state.last_ask_user = None
            st.session_state.chat_history.append({"role": "assistant", "content": output})
            st.rerun()
    
    # Tab 2: System Prompt Learning
    with tabs[1]:
        st.header("System Prompt Learning")
        st.markdown("""
        This tab shows how the system is learning from clinical reasoning cases to improve its reasoning strategies.
        Each time the system analyzes a case, it extracts new reasoning strategies that can be applied to future cases.
        """)
        with st.expander("Current System Prompts"):
            st.subheader("Medical Reasoning Prompt")
            prompt_library.load_prompts()  # Always reload
            med_reasoning_prompt = prompt_library.get_formatted_prompt("med-reasoning-base")
            st.text_area("Medical Reasoning Prompt", med_reasoning_prompt, height=300, key="med_reasoning_display")
            st.subheader("Literature Search Prompt")
            lit_search_prompt = prompt_library.get_formatted_prompt("literature-search-base")
            st.text_area("Literature Search Prompt", lit_search_prompt, height=300, key="lit_search_display")
        st.subheader("Learning History")
        # Show most recent first
        if st.session_state.system_prompt_learning_log:
            df = pd.DataFrame(st.session_state.system_prompt_learning_log)
            df = df.sort_values("timestamp", ascending=False)
            st.dataframe(df[["timestamp", "case", "strategy", "prompt_id", "status"]])
        else:
            st.info("No system prompt learning history available yet.")
    
    # Tab 3: Settings
    with tabs[2]:
        st.header("Settings")
        
        # API key input
        api_key = st.text_input("OpenAI API Key", type="password",
                               help="Enter your OpenAI API key to use the service")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("API key set successfully!")
        
        # System prompt learning toggle
        enable_learning = st.toggle("Enable System Prompt Learning", value=st.session_state.enable_learning,
                                  help="Turn on/off the system's ability to learn new reasoning strategies")
        if enable_learning != st.session_state.enable_learning:
            st.session_state.enable_learning = enable_learning
            st.success(f"System prompt learning {'enabled' if enable_learning else 'disabled'}!")
            st.rerun() # Rerun to reflect change immediately if needed
        
        # Model selection
        model = st.selectbox("LLM Model", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
                            help="Select which OpenAI model to use")
        if model != st.session_state.get("model", "gpt-4o"):
            st.session_state.model = model
            st.success(f"Model changed to {model}!")
        
        # Clear chat history
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")

if __name__ == "__main__":
    main() 