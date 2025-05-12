import os
import json
import datetime
from openai import OpenAI
from typing import List, Dict, Any

class LLMOrchestrator:
    def __init__(self, agents, prompt_library, learning_agent):
        self.agents = agents  # dict: {"diagnosis": agent, "literature": agent, ...}
        self.prompt_library = prompt_library
        self.learning_agent = learning_agent
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def orchestrate(self, conversation: List[Dict[str, str]], case_data: Dict[str, Any]) -> Dict[str, Any]:
        debug_log = {}
        # Compose a prompt for the LLM to decide what to do next
        conv_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation[-8:]])
        case_str = json.dumps(case_data, indent=2)
        orchestrator_prompt = (
            "You are a clinical reasoning orchestrator. Given the conversation so far and the case data, "
            "decide what the user wants next (diagnosis, management, follow-up, literature, or just acknowledgment). "
            "Return a JSON with fields: 'mode' (diagnosis, treatment, followup, literature, acknowledge, question), "
            "'agent_calls' (list of which agents/functions to call), and 'thinking' (your reasoning for this step)."
            "\n\nConversation:\n" + conv_str +
            "\n\nCase Data:\n" + case_str +
            "\n\nRespond only with the JSON."
        )
        messages = [
            {"role": "system", "content": "You are a clinical reasoning orchestrator."},
            {"role": "user", "content": orchestrator_prompt}
        ]
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.2,
            )
            content = response.choices[0].message.content
            debug_log["llm_raw_output"] = content
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                orchestration = json.loads(json_str)
            else:
                orchestration = {"mode": "diagnosis", "agent_calls": ["diagnosis"], "thinking": "Defaulted to diagnosis due to malformed LLM output."}
                debug_log["llm_parse_error"] = "Malformed LLM output, defaulted to diagnosis."
        except Exception as e:
            orchestration = {"mode": "diagnosis", "agent_calls": ["diagnosis"], "thinking": f"LLM error: {str(e)}"}
            debug_log["llm_exception"] = str(e)

        # Only keep agent calls that are actually registered
        orchestration["agent_calls"] = [a for a in orchestration.get("agent_calls", []) if a in self.agents]
        # Fallbacks for all supported modes
        if (not orchestration.get("agent_calls")):
            if orchestration.get("mode") in ["diagnosis"]:
                orchestration["agent_calls"] = ["diagnosis"]
            elif orchestration.get("mode") in ["management", "treatment"]:
                orchestration["agent_calls"] = ["management"]
            elif orchestration.get("mode") == "literature":
                orchestration["agent_calls"] = ["literature"]

        debug_log["orchestration"] = orchestration
        agent_outputs = {}
        agent_errors = {}
        for call in orchestration.get("agent_calls", []):
            try:
                if call == "diagnosis":
                    agent_outputs["diagnosis"] = self.agents["diagnosis"].analyze_case(case_data)
                elif call == "management":
                    # Use the most likely diagnosis from case_data or last diagnosis result
                    diagnosis = case_data.get("diagnosis")
                    if not diagnosis and "diagnosis" in agent_outputs:
                        diffs = agent_outputs["diagnosis"].get("differential_diagnosis", [])
                        if diffs:
                            diagnosis = diffs[0].get("diagnosis")
                    advice = self.agents["management"].get_management_advice(diagnosis, case_data)
                    agent_outputs["management"] = advice
                elif call == "literature":
                    query = conversation[-1]["content"] if conversation else ""
                    agent_outputs["literature"] = self.agents["literature"].search_literature(query)
                elif call == "acknowledge":
                    agent_outputs["acknowledge"] = "Thank you! If you have more questions or want to discuss management, follow-up, or another case, let me know."
                elif call == "treatment":
                    diagnosis = case_data.get("diagnosis")
                    if not diagnosis and "diagnosis" in agent_outputs:
                        diffs = agent_outputs["diagnosis"].get("differential_diagnosis", [])
                        if diffs:
                            diagnosis = diffs[0].get("diagnosis")
                    prompt = f"You are a clinical reasoning assistant. Provide concise, evidence-based management and treatment advice for the following diagnosis and case context:\nDiagnosis: {diagnosis}\nCase: {json.dumps(case_data)}"
                    messages = [
                        {"role": "system", "content": "You are a clinical reasoning assistant."},
                        {"role": "user", "content": prompt}
                    ]
                    advice = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0.3,
                    ).choices[0].message.content
                    agent_outputs["treatment"] = advice
                elif call == "followup":
                    diagnosis = case_data.get("diagnosis")
                    if not diagnosis and "diagnosis" in agent_outputs:
                        diffs = agent_outputs["diagnosis"].get("differential_diagnosis", [])
                        if diffs:
                            diagnosis = diffs[0].get("diagnosis")
                    prompt = f"You are a clinical reasoning assistant. Provide concise, evidence-based follow-up, prognosis, and monitoring advice for the following diagnosis and case context:\nDiagnosis: {diagnosis}\nCase: {json.dumps(case_data)}"
                    messages = [
                        {"role": "system", "content": "You are a clinical reasoning assistant."},
                        {"role": "user", "content": prompt}
                    ]
                    advice = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0.3,
                    ).choices[0].message.content
                    agent_outputs["followup"] = advice
                elif call == "question":
                    prompt = f"You are a clinical reasoning assistant. Answer the following question in the context of this case:\nCase: {json.dumps(case_data)}\nQuestion: {conversation[-1]['content']}"
                    messages = [
                        {"role": "system", "content": "You are a clinical reasoning assistant."},
                        {"role": "user", "content": prompt}
                    ]
                    answer = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0.3,
                    ).choices[0].message.content
                    agent_outputs["question"] = answer
            except Exception as e:
                agent_errors[call] = str(e)
                agent_outputs[call] = f"[Agent error: {str(e)}]"
        debug_log["agent_outputs"] = agent_outputs
        debug_log["agent_errors"] = agent_errors

        # Compose the response for the user
        response = ""
        if agent_errors:
            response += "\n**Agent Errors:**\n" + json.dumps(agent_errors, indent=2)
        if "diagnosis" in agent_outputs:
            result = agent_outputs["diagnosis"]
            differentials = result.get("differential_diagnosis", [])
            recommendations = result.get("recommendations", [])
            ask_user = result.get("ask_user", None)
            if differentials:
                response += "**Differential Diagnosis:**\n"
                for d in differentials:
                    response += f"- {d.get('diagnosis')}: {d.get('reasoning', d.get('likelihood', ''))}\n"
            if recommendations:
                response += "\n**Next Steps / Recommendations:**\n"
                for r in recommendations:
                    response += f"- {r}\n"
            needs_more_info = ask_user and not is_info_already_provided(ask_user, case_data)
            if needs_more_info:
                response += f"\n**Question:** {ask_user}"
        if "management" in agent_outputs:
            response += f"\n**Management/Advice:**\n{agent_outputs['management']}"
        if "treatment" in agent_outputs:
            response += f"\n**Treatment/Management:**\n{agent_outputs['treatment']}"
        if "followup" in agent_outputs:
            response += f"\n**Follow-up/Prognosis:**\n{agent_outputs['followup']}"
        if "literature" in agent_outputs:
            response += f"\n**Literature Search:**\n{agent_outputs['literature'].get('synthesis', '')}"
        if "question" in agent_outputs:
            response += f"\n{agent_outputs['question']}"
        if "acknowledge" in agent_outputs:
            response += f"\n{agent_outputs['acknowledge']}"

        # System Prompt Learning: Only if diagnosis was called and no more info needed
        learning_log = None
        try:
            if "diagnosis" in agent_outputs and (not result.get("ask_user") or not needs_more_info):
                learning_result = self.learning_agent.analyze_reasoning(case_data, result)
                if "new_strategy" in learning_result and "error" not in learning_result:
                    strategy = learning_result["new_strategy"]
                    add_status = self.learning_agent.add_strategy_to_prompt("med-reasoning-base", strategy)
                    learning_log = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "case": json.dumps(case_data.get("initial_input", "N/A"))[:100] + "...",
                        "strategy": strategy.get("description", "N/A"),
                        "prompt_id": "med-reasoning-base",
                        "status": add_status.get("status", "unknown")
                    }
                    response += f"\n\n*(Learned new strategy: {strategy.get('description', 'N/A')})*"
        except Exception as e:
            response += f"\n[System Prompt Learning Error: {str(e)}]"
            debug_log["system_prompt_learning_error"] = str(e)

        # Always include debug log in the thinking output
        thinking = {
            "orchestrator": orchestration,
            "agent_outputs": agent_outputs,
            "debug_log": debug_log
        }
        if learning_log:
            thinking["system_prompt_learning"] = learning_log

        # If the response is empty, show a warning
        if not response.strip():
            response = "[No response generated. See debug log below.]"

        return {
            "response": response,
            "thinking": thinking,
            "learning_log": learning_log
        }

def is_info_already_provided(ask_user, case_data):
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
    ask = ask_user.lower() if ask_user else ""
    for key, field in FIELD_KEYWORDS.items():
        if key in ask and case_data.get(field):
            return True
    return False 